"""Tests for the MatchClaimField columnar claim store.

MatchClaimField is a genuine ``SemanticField[MatchClaim]`` holding a set of
synchronous-instant pairwise MatchClaims as Arrow columns, one per
MatchClaim property, instead of one frozen MatchClaim object per claim,
materialising individual claims only on demand. Field-level shared metadata
is injected on read.
Validation strategy and gold values are documented in
``tests/alignment/README.md`` (the "MatchClaimField Tests" section).
"""

from __future__ import annotations

import copy
import json
import re
import time
from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import timetoshare

from timetoalign.alignment import (
    Agent,
    AlignmentAnchor,
    AlignmentBundle,
    MatchClaim,
    MatchClaimField,
    MatchMetadata,
)
from timetoalign.core import AgentType, Coordinate, TimeUnit
from timetoalign.core.fields import SemanticField, derive_arrow_struct

# region Fixtures


@pytest.fixture
def meta() -> MatchMetadata:
    """The canonical shared metadata for the gold vector."""
    return MatchMetadata(
        agent=Agent(name="test", type=AgentType.software, identifier="manual")
    )


@pytest.fixture
def gold_field(meta: MatchMetadata) -> MatchClaimField:
    """The canonical 3-claim gold vector (A->B, A->C, B->C)."""
    return MatchClaimField.from_columns(
        timeline_a_ids=["A", "A", "B"],
        timeline_b_ids=["B", "C", "C"],
        coordinate_a=[0.0, 0.0, 1.0],
        coordinate_b=[10.0, 20.0, 21.0],
        unit_a=TimeUnit.number,
        unit_b=TimeUnit.number,
        metadata=meta,
    )


# endregion


# region Construction & schema


class TestFromColumns:
    """The vectorized constructor and the backing struct column's schema."""

    def test_from_columns_length(self, gold_field: MatchClaimField) -> None:
        assert len(gold_field) == 3

    def test_is_semantic_field(self) -> None:
        assert issubclass(MatchClaimField, SemanticField)
        assert MatchClaimField.scalar_cls is MatchClaim

    def test_pa_schema_is_derived(self) -> None:
        assert MatchClaimField.pa_schema == derive_arrow_struct(MatchClaim)

        coordinate_type = (
            MatchClaimField.pa_schema.field("start_anchor")
            .type.field("coordinate_a")
            .type
        )
        assert coordinate_type == pa.struct(
            [
                pa.field("value", pa.float64()),
                pa.field("numerator", pa.int64()),
                pa.field("denominator", pa.int64()),
                pa.field("unit", pa.string()),
            ]
        )

    def test_table_one_column_per_property(self, gold_field: MatchClaimField) -> None:
        derived = derive_arrow_struct(MatchClaim)
        table = gold_field.table
        assert [(f.name, f.type) for f in table.schema] == [
            (f.name, f.type) for f in derived
        ]
        assert table.column_names == [
            "timeline_a_id",
            "timeline_b_id",
            "start_anchor",
            "end_anchor",
            "is_synchronous",
            "is_explicit",
            "metadata",
            "source_claim_id",
            "event_a_id",
            "event_a_name",
            "event_b_id",
            "event_b_name",
            "source_coordinate",
            "id",
            "id_minted",
        ]

    def test_timeline_ids(self, gold_field: MatchClaimField) -> None:
        assert gold_field.timeline_ids == {"A", "B", "C"}

    def test_metadata_property(
        self, gold_field: MatchClaimField, meta: MatchMetadata
    ) -> None:
        assert gold_field.metadata is meta

    def test_from_columns_accepts_pyarrow_arrays(self, meta: MatchMetadata) -> None:
        field = MatchClaimField.from_columns(
            timeline_a_ids=pa.array(["A", "B"], type=pa.string()),
            timeline_b_ids=pa.array(["B", "C"], type=pa.string()),
            coordinate_a=pa.array([0.0, 1.0], type=pa.float64()),
            coordinate_b=pa.array([10.0, 11.0], type=pa.float64()),
            unit_a=TimeUnit.number,
            unit_b=TimeUnit.number,
            metadata=meta,
        )
        assert len(field) == 2
        assert field.table.schema.field("start_anchor").type == (
            MatchClaimField.pa_schema.field("start_anchor").type
        )

    def test_from_columns_applies_units(self) -> None:
        field = MatchClaimField.from_columns(
            ["score"],
            ["audio"],
            [Fraction(7, 3)],
            [1.25],
            unit_a=TimeUnit.quarters,
            unit_b=TimeUnit.seconds,
        )
        anchor = field[0].start_anchor
        assert anchor.coordinate_a.value == Fraction(7, 3)
        assert anchor.coordinate_a.unit is TimeUnit.quarters
        assert anchor.coordinate_b.value == 1.25
        assert anchor.coordinate_b.unit is TimeUnit.seconds

    def test_from_columns_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            MatchClaimField.from_columns(
                timeline_a_ids=["A"],
                timeline_b_ids=["B", "C"],
                coordinate_a=[0.0],
                coordinate_b=[1.0],
                unit_a=TimeUnit.number,
                unit_b=TimeUnit.number,
            )

    def test_translator_strenum_to_string(self) -> None:
        """A StrEnum field stores as pa.string() inside the derived struct.

        ``MatchMetadata.agent.type`` is an ``AgentType`` (a ``FancyStrEnum``);
        its derived sub-field must be ``pa.string()``.
        """
        meta_struct = derive_arrow_struct(MatchMetadata)
        agent_struct = meta_struct.field("agent").type
        assert agent_struct.field("type").type == pa.string()


# endregion


# region Materialisation


class TestGetItem:
    """Row materialisation into MatchClaim objects."""

    def test_getitem_first_row(
        self, gold_field: MatchClaimField, meta: MatchMetadata
    ) -> None:
        claim = gold_field[0]
        assert isinstance(claim, MatchClaim)
        assert claim.timeline_a_id == "A"
        assert claim.timeline_b_id == "B"
        assert claim.is_synchronous is True
        assert claim.is_interval is False
        assert claim.start_anchor.coordinate_a.value == 0.0
        assert claim.start_anchor.coordinate_b.value == 10.0
        assert claim.metadata is meta

    def test_getitem_injects_metadata(
        self, gold_field: MatchClaimField, meta: MatchMetadata
    ) -> None:
        """Metadata lives at field level (null in the struct) and is injected."""
        # The rows carry no per-row metadata.
        assert gold_field.table.column("metadata").to_pylist() == [None, None, None]
        # The materialised claim nonetheless carries the field-level metadata.
        assert gold_field[1].metadata is meta

    def test_getitem_negative_index(self, gold_field: MatchClaimField) -> None:
        last = gold_field[-1]
        assert last.timeline_a_id == "B"
        assert last.timeline_b_id == "C"
        assert last.start_anchor.coordinate_a.value == 1.0
        assert last.start_anchor.coordinate_b.value == 21.0

    def test_getitem_out_of_range_positive(self, gold_field: MatchClaimField) -> None:
        with pytest.raises(IndexError):
            _ = gold_field[3]

    def test_getitem_out_of_range_negative(self, gold_field: MatchClaimField) -> None:
        with pytest.raises(IndexError):
            _ = gold_field[-4]

    def test_iter(self, gold_field: MatchClaimField) -> None:
        claims = list(gold_field)
        assert len(claims) == 3
        assert all(isinstance(c, MatchClaim) for c in claims)
        assert all(c.is_synchronous and not c.is_interval for c in claims)
        assert [(c.timeline_a_id, c.timeline_b_id) for c in claims] == [
            ("A", "B"),
            ("A", "C"),
            ("B", "C"),
        ]

    def test_to_claims(self, gold_field: MatchClaimField) -> None:
        claims = gold_field.to_claims()
        assert len(claims) == 3
        assert all(c.is_synchronous and c.start_anchor is not None for c in claims)
        assert all(c.end_anchor is None for c in claims)


# endregion


# region Filters


class TestFilters:
    """Vectorized connecting / filter views."""

    def test_connecting(self, gold_field: MatchClaimField) -> None:
        connected = gold_field.connecting("C")
        assert len(connected) == 2
        pairs = {
            (connected[i].timeline_a_id, connected[i].timeline_b_id)
            for i in range(len(connected))
        }
        assert pairs == {("A", "C"), ("B", "C")}

    def test_connecting_carries_metadata(
        self, gold_field: MatchClaimField, meta: MatchMetadata
    ) -> None:
        assert gold_field.connecting("C").metadata is meta

    def test_connecting_no_match(self, gold_field: MatchClaimField) -> None:
        assert len(gold_field.connecting("Z")) == 0

    def test_filter_timeline_ids(self, gold_field: MatchClaimField) -> None:
        filtered = gold_field.filter(timeline_ids={"A"})
        assert len(filtered) == 2
        pairs = {
            (filtered[i].timeline_a_id, filtered[i].timeline_b_id)
            for i in range(len(filtered))
        }
        assert pairs == {("A", "B"), ("A", "C")}

    def test_filter_timeline_id_equals_connecting(
        self, gold_field: MatchClaimField
    ) -> None:
        by_filter = gold_field.filter(timeline_id="C")
        by_connecting = gold_field.connecting("C")
        assert by_filter.table.equals(by_connecting.table)

    def test_filter_both_none_copies(self, gold_field: MatchClaimField) -> None:
        copy = gold_field.filter()
        assert len(copy) == 3
        assert copy.table.equals(gold_field.table)

    def test_filter_and_combination(self, gold_field: MatchClaimField) -> None:
        # timeline_id="A" -> rows 0,1; timeline_ids={"B"} -> rows 0,2.
        # AND -> only row 0 (A<->B).
        filtered = gold_field.filter(timeline_id="A", timeline_ids={"B"})
        assert len(filtered) == 1
        assert (filtered[0].timeline_a_id, filtered[0].timeline_b_id) == ("A", "B")

    def test_filter_timeline_ids_multiple(self, gold_field: MatchClaimField) -> None:
        filtered = gold_field.filter(timeline_ids={"A", "B"})
        assert len(filtered) == 3

    def test_filter_id_pattern(self, gold_field: MatchClaimField) -> None:
        filtered = gold_field.filter(id_pattern=r"^A$")
        assert len(filtered) == 2

    def test_filter_id_pattern_no_match(self, gold_field: MatchClaimField) -> None:
        assert len(gold_field.filter(id_pattern=r"^Z")) == 0

    def test_filter_between_order_independent(
        self, gold_field: MatchClaimField
    ) -> None:
        forward = gold_field.filter(between=("A", "B"))
        reverse = gold_field.filter(between=("B", "A"))
        assert len(forward) == 1
        assert forward.table.equals(reverse.table)

    def test_filter_within_requires_both_sides(
        self, gold_field: MatchClaimField
    ) -> None:
        # A<->C and B<->C each have one side outside the set, so only A<->B
        # survives. This is the semantics of a domain/unit-restricted query.
        filtered = gold_field.filter(within={"A", "B"})
        assert len(filtered) == 1
        assert (filtered[0].timeline_a_id, filtered[0].timeline_b_id) == ("A", "B")

    def test_filter_synchronous_only_is_noop(self, gold_field: MatchClaimField) -> None:
        assert len(gold_field.filter(synchronous_only=True)) == 3

    def test_filter_nomatch_only_is_empty(self, gold_field: MatchClaimField) -> None:
        empty = gold_field.filter(nomatch_only=True)
        assert len(empty) == 0
        assert empty.metadata is gold_field.metadata

    def test_filter_mutually_exclusive_raises(
        self, gold_field: MatchClaimField
    ) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            gold_field.filter(synchronous_only=True, nomatch_only=True)

    def test_filter_combines_with_and(self, gold_field: MatchClaimField) -> None:
        # timeline_id="A" -> rows 0,1; between=("A","C") -> row 1.
        filtered = gold_field.filter(timeline_id="A", between=("A", "C"))
        assert len(filtered) == 1
        assert (filtered[0].timeline_a_id, filtered[0].timeline_b_id) == ("A", "C")


class TestAt:
    """Exact-coordinate row selection."""

    def test_at_matches_a_side(self, gold_field: MatchClaimField) -> None:
        selected = gold_field.at("A", 0.0)
        assert len(selected) == 2
        pairs = {
            (selected[i].timeline_a_id, selected[i].timeline_b_id)
            for i in range(len(selected))
        }
        assert pairs == {("A", "B"), ("A", "C")}

    def test_at_matches_b_side(self, gold_field: MatchClaimField) -> None:
        selected = gold_field.at("C", 20.0)
        assert len(selected) == 1
        assert (selected[0].timeline_a_id, selected[0].timeline_b_id) == ("A", "C")

    def test_at_exact_equality_only(self, gold_field: MatchClaimField) -> None:
        # No tolerance, no nearest-value fallback.
        assert len(gold_field.at("A", 0.0000001)) == 0

    def test_at_unknown_timeline_empty(
        self, gold_field: MatchClaimField, meta: MatchMetadata
    ) -> None:
        empty = gold_field.at("Z", 0.0)
        assert len(empty) == 0
        assert empty.metadata is meta

    def test_at_on_empty_field(self) -> None:
        field = MatchClaimField.from_columns(
            [], [], [], [], unit_a=TimeUnit.number, unit_b=TimeUnit.number
        )
        assert len(field.at("A", 0.0)) == 0


class TestConnectsGroups:
    """Vectorized cross-set connectivity."""

    def test_connects_groups_true(self, gold_field: MatchClaimField) -> None:
        assert gold_field.connects_groups({"A"}, {"C"}) is True

    def test_connects_groups_order_independent(
        self, gold_field: MatchClaimField
    ) -> None:
        assert gold_field.connects_groups({"C"}, {"A"}) is True

    def test_connects_groups_false(self, gold_field: MatchClaimField) -> None:
        assert gold_field.connects_groups({"A"}, {"Z"}) is False

    def test_connects_groups_empty_field(self) -> None:
        field = MatchClaimField.from_columns(
            [], [], [], [], unit_a=TimeUnit.number, unit_b=TimeUnit.number
        )
        assert field.connects_groups({"A"}, {"B"}) is False


class TestBulkReads:
    """Column reads that never materialise a claim."""

    def test_max_coordinate_a_side(self, gold_field: MatchClaimField) -> None:
        assert gold_field.max_coordinate("A") == 0.0

    def test_max_coordinate_b_side(self, gold_field: MatchClaimField) -> None:
        assert gold_field.max_coordinate("C") == 21.0

    def test_max_coordinate_spans_both_sides(self, gold_field: MatchClaimField) -> None:
        # B appears as coordinate_b=10.0 (row 0) and coordinate_a=1.0 (row 2).
        assert gold_field.max_coordinate("B") == 10.0

    def test_max_coordinate_unknown_timeline(self, gold_field: MatchClaimField) -> None:
        assert gold_field.max_coordinate("Z") is None

    def test_coordinate_pairs(self, gold_field: MatchClaimField) -> None:
        assert gold_field.coordinate_pairs() == (
            ["A", "A", "B"],
            ["B", "C", "C"],
            [0.0, 0.0, 1.0],
            [10.0, 20.0, 21.0],
        )

    def test_coordinate_pairs_empty(self) -> None:
        field = MatchClaimField.from_columns(
            [], [], [], [], unit_a=TimeUnit.number, unit_b=TimeUnit.number
        )
        assert field.coordinate_pairs() == ([], [], [], [])


# endregion


# region from_claims & round-trips


class TestFromClaims:
    """Building from existing claims and round-tripping."""

    def test_roundtrip_from_claims(self, gold_field: MatchClaimField) -> None:
        claims = gold_field.to_claims()
        rebuilt = MatchClaimField.from_claims(claims)
        assert rebuilt.to_claims() == claims
        assert [claim.id for claim in claims] == ["claim_1", "claim_2", "claim_3"]
        assert rebuilt.table.equals(gold_field.table, check_metadata=True)

    def test_fraction_and_mixed_units_roundtrip(self) -> None:
        claim = MatchClaim.from_projection(
            event={"start": Fraction(7, 3)},
            source_tl_id="score",
            target_tl_id="audio",
            target_coord=1.25,
            source_unit=TimeUnit.quarters,
            target_unit=TimeUnit.seconds,
        )
        restored = MatchClaimField.from_claims([claim])[0]
        assert restored.start_anchor.coordinate_a.value == Fraction(7, 3)
        assert restored.start_anchor.coordinate_a.unit is TimeUnit.quarters
        assert restored.start_anchor.coordinate_b.value == 1.25
        assert restored.start_anchor.coordinate_b.unit is TimeUnit.seconds
        assert MatchClaimField.from_claims([claim])._repr_html_() == (
            "<div style='font-family: monospace;'><strong>MatchClaimField</strong> "
            "<span style='color: #555;'>claims=1, timelines=2</span><table "
            "style='border-collapse: collapse; margin-top: 4px;'><thead><tr><th>#</th>"
            "<th>timeline A</th><th>coord A</th><th>timeline B</th><th>coord B</th>"
            "</tr></thead><tbody><tr><td>0</td><td>score</td><td>7/3 quarters</td>"
            "<td>audio</td><td>1.25 seconds</td></tr></tbody></table></div>"
        )

    def test_from_claims_adopts_common_metadata(self) -> None:
        meta = MatchMetadata(
            agent=Agent(name="dtw", type=AgentType.software, identifier="warp")
        )
        claims = [
            MatchClaim.from_projection(
                event={"id": "e", "start": 0.0},
                source_tl_id="A",
                target_tl_id="B",
                target_coord=10.0,
                source_unit=TimeUnit.number,
                target_unit=TimeUnit.number,
                metadata=meta,
            )
            for _ in range(3)
        ]
        field = MatchClaimField.from_claims(claims)
        assert field.metadata == meta

    def test_from_claims_mixed_metadata_stays_none(self) -> None:
        m1 = MatchMetadata(
            agent=Agent(name="a1", type=AgentType.software, identifier="c1")
        )
        m2 = MatchMetadata(
            agent=Agent(name="a2", type=AgentType.software, identifier="c2")
        )
        claims = [
            MatchClaim.from_projection(
                event={"id": "e", "start": 0.0},
                source_tl_id="A",
                target_tl_id="B",
                target_coord=10.0,
                source_unit=TimeUnit.number,
                target_unit=TimeUnit.number,
                metadata=m,
            )
            for m in (m1, m2)
        ]
        field = MatchClaimField.from_claims(claims)
        assert field.metadata is None
        assert [claim.metadata for claim in field] == [m1, m2]

    def test_from_claims_explicit_metadata_overrides(self) -> None:
        per_claim = MatchMetadata(
            agent=Agent(name="a1", type=AgentType.software, identifier="c1")
        )
        override = MatchMetadata(
            agent=Agent(name="override", type=AgentType.software, identifier="c2")
        )
        claims = [
            MatchClaim.from_projection(
                event={"id": "e", "start": 0.0},
                source_tl_id="A",
                target_tl_id="B",
                target_coord=10.0,
                source_unit=TimeUnit.number,
                target_unit=TimeUnit.number,
                metadata=per_claim,
            )
        ]
        field = MatchClaimField.from_claims(claims, metadata=override)
        assert field.metadata is override

    def test_from_claims_rejects_nomatch(self) -> None:
        nomatch = MatchClaim.nomatch(
            event={"id": "orphan", "start": 5.0},
            source_tl_id="A",
            target_tl_id="B",
            unit=TimeUnit.number,
        )
        with pytest.raises(ValueError, match="non-synchronous"):
            MatchClaimField.from_claims([nomatch])

    def test_from_claims_preserves_interval(self) -> None:
        interval = MatchClaim.from_events(
            event_a={"id": "a", "start": 0.0, "end": 1.0},
            tl_a_id="A",
            event_b={"id": "b", "start": 10.0, "end": 11.0},
            tl_b_id="B",
            unit_a=TimeUnit.number,
            unit_b=TimeUnit.number,
            end_coord_key="end",
        )
        assert interval.is_interval
        field = MatchClaimField.from_claims([interval])
        assert field[0].get_interval_for("A") == interval.get_interval_for("A")
        assert field[0].get_interval_for("B") == interval.get_interval_for("B")

    def test_from_claims_empty(self) -> None:
        field = MatchClaimField.from_claims([])
        assert len(field) == 0
        assert field.metadata is None


class TestDictRoundTrip:
    """to_dict / from_dict serialisation."""

    def test_roundtrip_from_dict(self, gold_field: MatchClaimField) -> None:
        rebuilt = MatchClaimField.from_dict(gold_field.to_dict())
        assert rebuilt.table.equals(gold_field.table)

    def test_roundtrip_preserves_metadata(
        self, gold_field: MatchClaimField, meta: MatchMetadata
    ) -> None:
        rebuilt = MatchClaimField.from_dict(gold_field.to_dict())
        assert rebuilt.metadata == meta

    def test_to_dict_column_contents(self, gold_field: MatchClaimField) -> None:
        data = gold_field.to_dict()
        assert list(data) == ["metadata", "rows"]
        rows = data["rows"]
        assert [row["timeline_a_id"] for row in rows] == ["A", "A", "B"]
        assert [row["timeline_b_id"] for row in rows] == ["B", "C", "C"]
        assert [row["start_anchor"]["coordinate_a"]["value"] for row in rows] == [
            0.0,
            0.0,
            1.0,
        ]
        assert [row["start_anchor"]["coordinate_b"]["value"] for row in rows] == [
            10.0,
            20.0,
            21.0,
        ]
        assert [row["id"] for row in rows] == ["claim_1", "claim_2", "claim_3"]

    def test_to_dict_renders_the_table(self, gold_field: MatchClaimField) -> None:
        data = gold_field.to_dict()
        assert all(list(row) == gold_field.table.column_names for row in data["rows"])
        assert [_without_coordinates(row) for row in data["rows"]] == [
            _without_coordinates(row) for row in gold_field.table.to_pylist()
        ]
        assert data["metadata"] == {
            "agent": {"name": "test", "type": "software", "identifier": "manual"},
            "certainty": 1.0,
        }
        claims = _round_trip_claims([None, None, None])
        field = MatchClaimField.from_claims(claims)
        rows = field.to_dict()["rows"]
        assert [_without_coordinates(row) for row in rows] == [
            _without_coordinates(row) for row in field.table.to_pylist()
        ]
        for index, row in enumerate(rows):
            claim = field[index].to_dict()
            assert row["start_anchor"] == claim["start_anchor"]
            assert row["end_anchor"] == claim["end_anchor"]
        assert rows[0]["end_anchor"] is not None

    def test_to_dict_literal(self) -> None:
        field = _two_claim_field()
        assert field.to_dict() == {
            "metadata": None,
            "rows": [
                _literal_row(
                    "claim_1",
                    ("score", "quarters", "fraction", [1, 3]),
                    ("audio", "seconds", "float", 0.5),
                ),
                _literal_row(
                    "claim_2",
                    ("midi", "ticks", "int", 96),
                    ("pcm", "samples", "int", 44100),
                ),
            ],
        }
        rebuilt = MatchClaimField.from_dict(field.to_dict())
        assert rebuilt.to_claims() == field.to_claims()
        assert rebuilt.table.equals(field.table, check_metadata=True)

    def test_from_dict_reads_an_edited_value(self) -> None:
        data = _two_claim_field().to_dict()
        anchor = data["rows"][0]["start_anchor"]
        anchor["coordinate_a"]["value"] = [2, 3]
        anchor["coordinate_b"]["value"] = 0.75
        claim = MatchClaimField.from_dict(data)[0]
        assert claim.start_anchor.coordinate_a == Coordinate(
            Fraction(2, 3), TimeUnit.quarters
        )
        assert claim.start_anchor.coordinate_b == Coordinate(0.75, TimeUnit.seconds)

    def test_roundtrip_from_dict_keeps_every_row_value(self) -> None:
        human = MatchMetadata(
            agent=Agent(name="ann", type=AgentType.human, identifier="ann"),
            certainty=0.5,
        )
        software = MatchMetadata(
            agent=Agent(name="dtw", type=AgentType.software, identifier="v1")
        )
        claims = _round_trip_claims([None, human, software])
        field = MatchClaimField.from_claims(claims)
        data = field.to_dict()
        rows = data["rows"]
        assert [row["id"] for row in rows] == ["claim_1", "midi 96", "claim_2"]
        assert [row["id_minted"] for row in rows] == [True, False, True]
        assert [row["is_explicit"] for row in rows] == [True, True, False]
        assert [row["source_claim_id"] for row in rows] == [
            None,
            None,
            "claim_source",
        ]
        assert [row["event_a_id"] for row in rows] == ["note_7", "tick_96", "onset_1"]
        assert [row["event_b_id"] for row in rows] == ["onset_7", None, "note_1"]
        assert [row["event_b_name"] for row in rows] == [None, None, "C♯4"]
        assert [row["metadata"] for row in rows] == [
            None,
            {
                "agent": {"name": "ann", "type": "human", "identifier": "ann"},
                "certainty": 0.5,
            },
            {
                "agent": {"name": "dtw", "type": "software", "identifier": "v1"},
                "certainty": 1.0,
            },
        ]
        assert rows[1] == _PROJECTION_ROW
        assert data["metadata"] is None
        rebuilt = MatchClaimField.from_dict(data)
        assert rebuilt.table.equals(field.table, check_metadata=True)
        assert rebuilt.to_claims() == claims

    def test_to_dict_no_metadata(self) -> None:
        field = MatchClaimField.from_columns(
            timeline_a_ids=["A"],
            timeline_b_ids=["B"],
            coordinate_a=[0.0],
            coordinate_b=[1.0],
            unit_a=TimeUnit.number,
            unit_b=TimeUnit.number,
        )
        assert field.to_dict()["metadata"] is None
        assert MatchClaimField.from_dict(field.to_dict()).metadata is None

    @pytest.mark.parametrize("shared", [False, True])
    def test_yaml_round_trip(self, shared: bool) -> None:
        human = MatchMetadata(
            agent=Agent(name="ann", type=AgentType.human, identifier="ann"),
            certainty=0.5,
        )
        claims = _round_trip_claims([human, human, human if shared else None])
        field = MatchClaimField.from_claims(claims)
        assert (field.metadata == human) is shared
        data = field.to_dict()
        loaded = timetoshare.loads(timetoshare.dumps(data, "yaml"))
        assert loaded == data
        rebuilt = MatchClaimField.from_dict(loaded)
        assert rebuilt.to_claims() == claims
        assert rebuilt.table.equals(field.table, check_metadata=True)

    def test_from_dict_refuses_other_keys(self, gold_field: MatchClaimField) -> None:
        data = gold_field.to_dict()
        columns = gold_field.table.column_names
        lacking = {key: value for key, value in data["rows"][1].items()}
        del lacking["event_a_id"]
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"In a MatchClaimField dictionary, row 1 must hold the keys "
                f"{columns}; it lacks ['event_a_id'] and has the unknown keys []."
            ),
        ):
            MatchClaimField.from_dict(data | {"rows": [data["rows"][0], lacking]})
        extra = data["rows"][0] | {"coordinate_a": 0.0}
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"In a MatchClaimField dictionary, row 0 must hold the keys "
                f"{columns}; it lacks [] and has the unknown keys ['coordinate_a']."
            ),
        ):
            MatchClaimField.from_dict(data | {"rows": [extra]})

    @pytest.mark.parametrize(
        ("path", "edit", "message"),
        [
            (
                ("start_anchor", "coordinate_a"),
                lambda mapping: mapping.pop("value"),
                "row 0 start_anchor.coordinate_a must hold the keys "
                "['unit', 'number_type', 'value']; it lacks ['value'] and has the "
                "unknown keys [].",
            ),
            (
                ("start_anchor", "coordinate_b"),
                lambda mapping: mapping.update(extra=1),
                "row 0 start_anchor.coordinate_b must hold the keys "
                "['unit', 'number_type', 'value']; it lacks [] and has the "
                "unknown keys ['extra'].",
            ),
            (
                ("start_anchor",),
                lambda mapping: mapping.update(unit="seconds"),
                "row 0 start_anchor must hold the keys ['timeline_a_id', "
                "'coordinate_a', 'timeline_b_id', 'coordinate_b']; it lacks [] "
                "and has the unknown keys ['unit'].",
            ),
            (
                ("metadata",),
                lambda mapping: mapping.pop("certainty"),
                "row 0 metadata must hold the keys ['agent', 'certainty']; it "
                "lacks ['certainty'] and has the unknown keys [].",
            ),
        ],
    )
    def test_from_dict_refuses_other_nested_keys(
        self,
        path: tuple[str, ...],
        edit: object,
        message: str,
    ) -> None:
        human = MatchMetadata(
            agent=Agent(name="ann", type=AgentType.human, identifier="ann"),
            certainty=0.5,
        )
        field = MatchClaimField.from_claims(_round_trip_claims([human, None, None]))
        data = field.to_dict()
        mapping = data["rows"][0]
        for key in path:
            mapping = mapping[key]
        edit(mapping)  # type: ignore[operator]
        with pytest.raises(
            ValueError,
            match=re.escape(f"In a MatchClaimField dictionary, {message}"),
        ):
            MatchClaimField.from_dict(data)


#: The projection of :func:`_round_trip_claims` as its ``to_dict`` row: the
#: ``ticks`` / ``samples`` coordinates as int coordinate dicts, the given id
#: kept.
_PROJECTION_ROW = {
    "timeline_a_id": "midi",
    "timeline_b_id": "pcm",
    "start_anchor": {
        "timeline_a_id": "midi",
        "coordinate_a": {"unit": "ticks", "number_type": "int", "value": 96},
        "timeline_b_id": "pcm",
        "coordinate_b": {"unit": "samples", "number_type": "int", "value": 44100},
    },
    "end_anchor": None,
    "is_synchronous": True,
    "is_explicit": True,
    "metadata": {
        "agent": {"name": "ann", "type": "human", "identifier": "ann"},
        "certainty": 0.5,
    },
    "source_claim_id": None,
    "event_a_id": "tick_96",
    "event_a_name": None,
    "event_b_id": None,
    "event_b_name": None,
    "source_coordinate": None,
    "id": "midi 96",
    "id_minted": False,
}


#: The coordinate cells of a ``MatchClaimField`` row, as key paths.
_COORDINATE_PATHS = (
    ("start_anchor", "coordinate_a"),
    ("start_anchor", "coordinate_b"),
    ("end_anchor", "coordinate_a"),
    ("end_anchor", "coordinate_b"),
    ("source_coordinate",),
)


def _without_coordinates(row: dict) -> dict:
    """*row* with every coordinate cell set to ``None`` (a deep copy)."""
    row = copy.deepcopy(row)
    for path in _COORDINATE_PATHS:
        holder = row
        for key in path[:-1]:
            holder = holder[key]
            if holder is None:
                break
        else:
            holder[path[-1]] = None
    return row


def _two_claim_field() -> MatchClaimField:
    """``score`` 1/3 quarters ↔ ``audio`` 0.5 s; ``midi`` 96 ticks ↔ ``pcm`` 44100 samples."""
    pairs = [
        (
            ("score", Coordinate(Fraction(1, 3), TimeUnit.quarters)),
            ("audio", Coordinate(0.5, TimeUnit.seconds)),
        ),
        (
            ("midi", Coordinate(96, TimeUnit.ticks)),
            ("pcm", Coordinate(44100, TimeUnit.samples)),
        ),
    ]
    return MatchClaimField.from_claims(
        [
            MatchClaim(
                timeline_a_id=a_id,
                timeline_b_id=b_id,
                start_anchor=AlignmentAnchor(
                    timeline_a_id=a_id,
                    coordinate_a=a,
                    timeline_b_id=b_id,
                    coordinate_b=b,
                ),
            )
            for (a_id, a), (b_id, b) in pairs
        ]
    )


def _literal_row(claim_id: str, a: tuple, b: tuple) -> dict:
    """The ``to_dict`` row of an anonymous instant claim between *a* and *b*.

    *a* and *b* are ``(timeline id, unit, number type, value)``.
    """
    return {
        "timeline_a_id": a[0],
        "timeline_b_id": b[0],
        "start_anchor": {
            "timeline_a_id": a[0],
            "coordinate_a": {"unit": a[1], "number_type": a[2], "value": a[3]},
            "timeline_b_id": b[0],
            "coordinate_b": {"unit": b[1], "number_type": b[2], "value": b[3]},
        },
        "end_anchor": None,
        "is_synchronous": True,
        "is_explicit": True,
        "metadata": None,
        "source_claim_id": None,
        "event_a_id": None,
        "event_a_name": None,
        "event_b_id": None,
        "event_b_name": None,
        "source_coordinate": None,
        "id": claim_id,
        "id_minted": True,
    }


def _round_trip_claims(metadata: list[MatchMetadata | None]) -> list[MatchClaim]:
    """Three claims covering every stored representation once.

    An exact-fraction interval claim with event ids on both sides, an
    int-coordinate projection, and an implicit float/fraction claim with an
    event name and a ``source_claim_id``; ``metadata`` gives each its own.
    Their ids are distinct, as a scope leaves them: the interval keeps its
    minted ``claim_1``, the projection is given ``"midi 96"``, the implicit
    claim holds the minted ``claim_2``.
    """
    interval = MatchClaim.from_events(
        event_a={"id": "note_7", "start": Fraction(7, 3), "end": Fraction(10, 3)},
        tl_a_id="score",
        event_b={"id": "onset_7", "start": 1.25, "end": 2.5},
        tl_b_id="audio",
        unit_a=TimeUnit.quarters,
        unit_b=TimeUnit.seconds,
        end_coord_key="end",
        metadata=metadata[0],
    )
    projection = MatchClaim.from_projection(
        event={"id": "tick_96", "start": 96},
        source_tl_id="midi",
        target_tl_id="pcm",
        target_coord=44100,
        source_unit=TimeUnit.ticks,
        target_unit=TimeUnit.samples,
        metadata=metadata[1],
    ).model_copy(update={"id": "midi 96", "id_minted": False})
    implicit = MatchClaim(
        timeline_a_id="audio",
        timeline_b_id="score",
        start_anchor=AlignmentAnchor(
            timeline_a_id="audio",
            coordinate_a=Coordinate(0.5, TimeUnit.seconds),
            timeline_b_id="score",
            coordinate_b=Coordinate(Fraction(1, 3), TimeUnit.quarters),
        ),
        is_explicit=False,
        event_a_id="onset_1",
        event_b_id="note_1",
        event_b_name="C♯4",
        source_claim_id="claim_source",
        metadata=metadata[2],
    ).model_copy(update={"id": "claim_2"})
    return [interval, projection, implicit]


_OUT_OF_SCOPE = (
    "MatchClaimField holds synchronous claims only; claim at index {} is "
    "non-synchronous (NOMATCH) and is out of scope."
)


def _with_child(table: pa.Table, name: str, child: pa.Array) -> pa.Table:
    """Return *table* with one of its property columns replaced."""
    return table.set_column(table.column_names.index(name), name, child)


#: The refusal of a table whose columns are not the MatchClaim properties.
_NOT_THE_PROPERTIES = (
    "A MatchClaimField table holds one column per MatchClaim property, {}; "
    "this table has the columns {}."
)


def _through_parquet(field: MatchClaimField, path: Path) -> MatchClaimField:
    """Write ``field.table`` to a Parquet file and read it back.

    The file's own schema is checked on the way: one column per property.
    """
    pq.write_table(field.table, path)
    assert [(f.name, f.type) for f in pq.read_schema(path)] == [
        (f.name, f.type) for f in MatchClaimField.pa_schema
    ]
    return MatchClaimField.from_table(pq.read_table(path))


class TestTableRoundTrip:
    """``.table`` / ``from_table`` through an actual Parquet file."""

    def test_claims_are_what_they_say(self) -> None:
        interval, projection, implicit = _round_trip_claims([None, None, None])
        assert interval.is_interval
        assert interval.start_anchor.coordinate_a.value == Fraction(7, 3)
        assert interval.end_anchor.coordinate_a.value == Fraction(10, 3)
        assert interval.start_anchor.coordinate_b.value == 1.25
        assert interval.end_anchor.coordinate_b.value == 2.5
        assert (interval.event_a_id, interval.event_b_id) == ("note_7", "onset_7")
        assert projection.start_anchor.coordinate_a.value == 96
        assert type(projection.start_anchor.coordinate_a.value) is int
        assert projection.start_anchor.coordinate_b.value == 44100
        assert type(projection.start_anchor.coordinate_b.value) is int
        assert implicit.is_explicit is False
        assert implicit.start_anchor.coordinate_b.value == Fraction(1, 3)
        assert [(c.id, c.id_minted) for c in (interval, projection, implicit)] == [
            ("claim_1", True),
            ("midi 96", False),
            ("claim_2", True),
        ]

    def test_shared_metadata_round_trip(self, tmp_path: Path) -> None:
        shared = MatchMetadata(
            agent=Agent(name="dtw", type=AgentType.software, identifier="v1"),
            certainty=0.75,
        )
        claims = _round_trip_claims([shared, shared, shared])
        field = MatchClaimField.from_claims(claims)
        assert field.metadata == shared
        assert field.table.schema.metadata == {
            b"timetoalign:match_metadata": (
                b'{"agent": {"name": "dtw", "type": "software", '
                b'"identifier": "v1"}, "certainty": 0.75}'
            )
        }
        restored = _through_parquet(field, tmp_path / "claims.parquet")
        assert restored.metadata == shared
        assert restored.to_claims() == claims

    def test_mixed_metadata_round_trip(self, tmp_path: Path) -> None:
        human = MatchMetadata(
            agent=Agent(name="ann", type=AgentType.human, identifier="ann"),
            certainty=0.5,
        )
        software = MatchMetadata(
            agent=Agent(name="dtw", type=AgentType.software, identifier="v1")
        )
        claims = _round_trip_claims([None, human, software])
        field = MatchClaimField.from_claims(claims)
        assert field.metadata is None
        assert field.table.schema.metadata is None
        restored = _through_parquet(field, tmp_path / "claims.parquet")
        assert restored.metadata is None
        assert restored.to_claims() == claims
        assert [claim.metadata for claim in restored] == [None, human, software]

    def test_columnar_field_round_trip(
        self, gold_field: MatchClaimField, meta: MatchMetadata, tmp_path: Path
    ) -> None:
        restored = _through_parquet(gold_field, tmp_path / "claims.parquet")
        assert restored.metadata == meta
        assert restored.table.equals(gold_field.table, check_metadata=True)

    def test_empty_field_round_trip(self, tmp_path: Path) -> None:
        restored = _through_parquet(
            MatchClaimField.from_claims([]), tmp_path / "claims.parquet"
        )
        assert len(restored) == 0
        assert restored.metadata is None

    def test_chunked_columns(self, gold_field: MatchClaimField) -> None:
        table = gold_field.table
        chunked = pa.concat_tables([table.slice(0, 1), table.slice(1)])
        assert chunked.column("start_anchor").num_chunks == 2
        restored = MatchClaimField.from_table(chunked)
        assert restored.table.equals(gold_field.table, check_metadata=True)

    def test_refuses_extra_column(self, gold_field: MatchClaimField) -> None:
        columns = gold_field.table.column_names
        table = gold_field.table.append_column("extra", pa.array([1, 2, 3]))
        with pytest.raises(
            ValueError,
            match=re.escape(_NOT_THE_PROPERTIES.format(columns, [*columns, "extra"])),
        ):
            MatchClaimField.from_table(table)

    def test_refuses_missing_column(self, gold_field: MatchClaimField) -> None:
        columns = gold_field.table.column_names
        table = gold_field.table.drop_columns(["id_minted"])
        with pytest.raises(
            ValueError,
            match=re.escape(_NOT_THE_PROPERTIES.format(columns, columns[:-1])),
        ):
            MatchClaimField.from_table(table)

    def test_refuses_single_column_layout(
        self, gold_field: MatchClaimField, tmp_path: Path
    ) -> None:
        """The former layout, the whole claim in one struct column, is refused."""
        table = gold_field.table
        single = pa.table(
            {"match_claim": table.to_struct_array()}, metadata=table.schema.metadata
        )
        path = tmp_path / "single.parquet"
        pq.write_table(single, path)
        with pytest.raises(
            ValueError,
            match=re.escape(
                _NOT_THE_PROPERTIES.format(table.column_names, ["match_claim"])
            ),
        ):
            MatchClaimField.from_table(pq.read_table(path))

    def test_refuses_non_synchronous_row(self, gold_field: MatchClaimField) -> None:
        table = _with_child(
            gold_field.table, "is_synchronous", pa.array([True, False, True])
        )
        with pytest.raises(ValueError, match=re.escape(_OUT_OF_SCOPE.format(1))):
            MatchClaimField.from_table(table)

    def test_refuses_row_without_start_anchor(
        self, gold_field: MatchClaimField
    ) -> None:
        anchor = gold_field.table.column("start_anchor").combine_chunks()
        without = pa.StructArray.from_arrays(
            [anchor.field(sub.name) for sub in anchor.type],
            fields=list(anchor.type),
            mask=pa.array([False, False, True]),
        )
        table = _with_child(gold_field.table, "start_anchor", without)
        with pytest.raises(ValueError, match=re.escape(_OUT_OF_SCOPE.format(2))):
            MatchClaimField.from_table(table)

    def test_from_claims_refuses_alike(self) -> None:
        nomatch = MatchClaim.nomatch(
            event={"start": 1.0},
            source_tl_id="A",
            target_tl_id="B",
            unit=TimeUnit.number,
        )
        with pytest.raises(ValueError, match=re.escape(_OUT_OF_SCOPE.format(0))):
            MatchClaimField.from_claims([nomatch])

    def test_refuses_metadata_stated_twice(self) -> None:
        human = MatchMetadata(
            agent=Agent(name="ann", type=AgentType.human, identifier="ann")
        )
        field = MatchClaimField.from_claims(_round_trip_claims([None, human, human]))
        assert field.metadata is None
        table = field.table.replace_schema_metadata(
            {b"timetoalign:match_metadata": json.dumps(human.to_dict())}
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "A MatchClaimField table states its metadata once: this table "
                "carries it in its schema metadata ('timetoalign:match_metadata') "
                "and in 2 row(s)."
            ),
        ):
            MatchClaimField.from_table(table)

    def test_from_dict_refuses_non_synchronous_row(
        self, gold_field: MatchClaimField
    ) -> None:
        data = gold_field.to_dict()
        data["rows"][2]["is_synchronous"] = False
        with pytest.raises(ValueError, match=re.escape(_OUT_OF_SCOPE.format(2))):
            MatchClaimField.from_dict(data)

    def test_from_dict_refuses_metadata_stated_twice(self) -> None:
        human = MatchMetadata(
            agent=Agent(name="ann", type=AgentType.human, identifier="ann")
        )
        field = MatchClaimField.from_claims(_round_trip_claims([None, human, human]))
        data = field.to_dict() | {"metadata": human.to_dict()}
        with pytest.raises(
            ValueError,
            match=re.escape(
                "A MatchClaimField dictionary states its metadata once: this "
                "dictionary carries it under its key 'metadata' and in 2 row(s)."
            ),
        ):
            MatchClaimField.from_dict(data)

    def test_refuses_other_type(self, gold_field: MatchClaimField) -> None:
        table = _with_child(gold_field.table, "is_explicit", pa.array([1, 1, 1]))
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Column 'is_explicit' has type int64, which is not the type of "
                "the MatchClaim property, bool."
            ),
        ):
            MatchClaimField.from_table(table)


# endregion


# region Equality


class TestEquality:
    """Claim equality is over the claim, never over the bundle holding it."""

    def test_copy_in_a_bundle_is_equal(self) -> None:
        claim = _round_trip_claims([None, None, None])[0]
        copy = claim.model_copy()
        copy.set_bundle(AlignmentBundle())
        assert copy.bundle is not None and claim.bundle is None
        assert claim == copy
        assert hash(claim) == hash(copy)

    def test_round_trip_outside_a_bundle(self) -> None:
        claims = MatchClaimField.from_claims(
            _round_trip_claims([None, None, None])
        ).to_claims()
        assert [claim.bundle for claim in claims] == [None, None, None]
        assert MatchClaimField.from_claims(claims).to_claims() == claims

    def test_round_trip_of_a_bundle_s_claims(self) -> None:
        bundle = AlignmentBundle()
        bundle.add_match_claims(_round_trip_claims([None, None, None]))
        claims = bundle.cross_group_claims
        assert [claim.bundle is bundle for claim in claims] == [True, True, True]
        restored = MatchClaimField.from_claims(claims).to_claims()
        assert [claim.bundle for claim in restored] == [None, None, None]
        assert restored == claims


# endregion


# region Edge cases, repr & scale


class TestEdgeCases:
    """Empty fields, repr, exports."""

    def test_empty_field(self) -> None:
        field = MatchClaimField.from_columns(
            [], [], [], [], unit_a=TimeUnit.number, unit_b=TimeUnit.number
        )
        assert len(field) == 0
        assert field.timeline_ids == set()
        assert field.to_claims() == []
        assert list(field) == []

    def test_repr(self, gold_field: MatchClaimField) -> None:
        assert repr(gold_field) == "MatchClaimField(claims=3, timelines=3)"

    def test_repr_html_contains_summary(self, gold_field: MatchClaimField) -> None:
        html = gold_field._repr_html_()
        assert "MatchClaimField" in html
        assert "claims=3" in html
        assert "timelines=3" in html

    def test_top_level_export(self) -> None:
        import timetoalign
        from timetoalign.alignment import MatchClaimField as AliasFromAlignment

        assert timetoalign.MatchClaimField is MatchClaimField
        assert AliasFromAlignment is MatchClaimField


class TestScale:
    """The vectorized path scales without building Python objects."""

    def test_scale_builds_columnar(self) -> None:
        n = 100_000
        start = time.perf_counter()
        field = MatchClaimField.from_columns(
            timeline_a_ids=["A"] * n,
            timeline_b_ids=["B"] * n,
            coordinate_a=[float(i) for i in range(n)],
            coordinate_b=[float(i) + 0.5 for i in range(n)],
            unit_a=TimeUnit.number,
            unit_b=TimeUnit.number,
        )
        elapsed = time.perf_counter() - start

        assert len(field) == n
        # A generous ceiling: the vectorized path must not build N objects.
        assert elapsed < 1.0
        claim = field[50_000]
        assert claim.start_anchor.coordinate_a.value == 50_000.0
        assert claim.start_anchor.coordinate_b.value == 50_000.5
        assert field.timeline_ids == {"A", "B"}


# endregion
