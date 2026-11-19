"""Tests for the MatchClaimField columnar claim store.

MatchClaimField is a genuine ``SemanticField[MatchClaim]`` holding a set of
synchronous-instant pairwise MatchClaims as a single derived struct column
instead of one frozen MatchClaim object per claim, materialising individual
claims only on demand. Field-level shared metadata is injected on read.
Validation strategy and gold values are documented in
``tests/alignment/README.md`` (the "MatchClaimField Tests" section).
"""

from __future__ import annotations

import json
import re
import time
from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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

    def test_table_single_struct_column(self, gold_field: MatchClaimField) -> None:
        table = gold_field.table
        assert table.num_columns == 1
        assert table.column_names == ["match_claim"]
        assert table.schema.field("match_claim").type == MatchClaimField.pa_schema

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
        assert field.table.schema.field("match_claim").type == MatchClaimField.pa_schema

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
        # The struct row carries no per-row metadata.
        struct = gold_field.table.column("match_claim").combine_chunks()
        assert struct.field("metadata").to_pylist() == [None, None, None]
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
        assert data["timeline_a_id"] == ["A", "A", "B"]
        assert data["timeline_b_id"] == ["B", "C", "C"]
        assert data["coordinate_a"] == [0.0, 0.0, 1.0]
        assert data["coordinate_b"] == [10.0, 20.0, 21.0]

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
        assert data["id"] == ["claim_1", "midi 96", "claim_2"]
        assert data["id_minted"] == [True, False, True]
        assert data["is_explicit"] == [True, True, False]
        assert data["source_claim_id"] == [None, None, "claim_source"]
        assert data["event_a_id"] == ["note_7", "tick_96", "onset_1"]
        assert data["event_b_id"] == ["onset_7", None, "note_1"]
        assert data["event_b_name"] == [None, None, "C♯4"]
        assert data["row_metadata"] == [
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
    """Return *table* with one sub-field of its claim column replaced."""
    column = table.column("match_claim").combine_chunks()
    children = [
        child if sub.name == name else column.field(sub.name) for sub in column.type
    ]
    rebuilt = pa.StructArray.from_arrays(children, fields=list(column.type))
    return pa.table({"match_claim": rebuilt}, metadata=table.schema.metadata)


def _through_parquet(field: MatchClaimField, path: Path) -> MatchClaimField:
    """Write ``field.table`` to a Parquet file and read it back."""
    pq.write_table(field.table, path)
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

    def test_chunked_column(self, gold_field: MatchClaimField) -> None:
        column = gold_field.table.column("match_claim").combine_chunks()
        chunked = pa.chunked_array([column.slice(0, 1), column.slice(1)])
        table = pa.table(
            {"match_claim": chunked}, metadata=gold_field.table.schema.metadata
        )
        assert table.column("match_claim").num_chunks == 2
        restored = MatchClaimField.from_table(table)
        assert restored.table.equals(gold_field.table, check_metadata=True)

    def test_refuses_extra_column(self, gold_field: MatchClaimField) -> None:
        table = gold_field.table.append_column("extra", pa.array([1, 2, 3]))
        with pytest.raises(
            ValueError,
            match=re.escape(
                "A MatchClaimField table holds exactly one column named "
                "'match_claim'; this table has the columns ['match_claim', 'extra']."
            ),
        ):
            MatchClaimField.from_table(table)

    def test_refuses_renamed_column(self, gold_field: MatchClaimField) -> None:
        table = gold_field.table.rename_columns(["claims"])
        with pytest.raises(
            ValueError,
            match=re.escape(
                "A MatchClaimField table holds exactly one column named "
                "'match_claim'; this table has the columns ['claims']."
            ),
        ):
            MatchClaimField.from_table(table)

    def test_refuses_non_synchronous_row(self, gold_field: MatchClaimField) -> None:
        table = _with_child(
            gold_field.table, "is_synchronous", pa.array([True, False, True])
        )
        with pytest.raises(ValueError, match=re.escape(_OUT_OF_SCOPE.format(1))):
            MatchClaimField.from_table(table)

    def test_refuses_row_without_start_anchor(
        self, gold_field: MatchClaimField
    ) -> None:
        column = gold_field.table.column("match_claim").combine_chunks()
        anchor = column.field("start_anchor")
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

    def test_refuses_other_type(self) -> None:
        table = pa.table({"match_claim": pa.array([1, 2])})
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Column 'match_claim' has type int64, which is not the "
                "MatchClaim struct type "
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
