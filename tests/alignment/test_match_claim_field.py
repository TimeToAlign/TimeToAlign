"""Tests for MatchClaimField, the MatchClaim table.

MatchClaimField is a genuine ``SemanticField[MatchClaim]`` holding every kind
of MatchClaim as Arrow columns, one per MatchClaim property, materialising
individual claims only on demand; a provenance shared by every row is held
once and injected on read. Validation strategy and gold values are documented
in ``tests/alignment/README.md`` (the "MatchClaimField Tests" section).
"""

from __future__ import annotations

import copy
import json
import re
from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import timetoshare
from timetoshare import DEFAULT_BELIEFS

from timetoalign.alignment import (
    AlignmentAnchor,
    AlignmentBundle,
    Belief,
    MatchClaim,
    MatchClaimField,
    Provenance,
)
from timetoalign.core import Coordinate, NumberType, TimeUnit
from timetoalign.core.fields import SemanticField, StructField, derive_arrow_struct
from timetoalign.core.time import COORDINATE_CELL_TYPE, RATIONAL_STRUCT_TYPE
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteLogicalTimeline,
    DiscretePhysicalTimeline,
)

# region Fixtures

SCORE = ContinuousLogicalTimeline(length=16, uid="score")
AUDIO = ContinuousPhysicalTimeline(length=60.0, uid="audio")
MIDI = DiscreteLogicalTimeline(length=10_000, uid="midi")
PCM = DiscretePhysicalTimeline(length=44_100 * 60, uid="pcm")
FLOAT_SCORE = ContinuousLogicalTimeline(
    length=16, uid="float_score", number_type="float"
)
TIMELINES = [SCORE, AUDIO, MIDI, PCM, FLOAT_SCORE]

#: The gold vector's axes.
GOLD_AXES = {
    "A": ("quarters", "fraction"),
    "B": ("seconds", "float"),
    "C": ("seconds", "float"),
}

SHARED = Provenance(service="https://example.org/aligner", method="inferred")


def _belief(actor: str, belief: str) -> Belief:
    return Belief(actor=actor, belief=belief, date="2026-11-21")


@pytest.fixture
def gold_field() -> MatchClaimField:
    """The canonical 3-claim gold vector (A->B, A->C, B->C)."""
    return MatchClaimField.from_columns(
        timeline_a_ids=["A", "A", "B"],
        timeline_b_ids=["B", "C", "C"],
        event_a_ids=["a0", "a0", "b1"],
        event_b_ids=["b0", "c0", "c1"],
        coordinate_a=[0, 0, 1],
        coordinate_b=[10.0, 20.0, 21.0],
        timelines=GOLD_AXES,
        provenance=SHARED,
    )


def _every_kind(provenance: list[Provenance | None]) -> list[MatchClaim]:
    """Five claims covering every kind and every stored representation once.

    An exact-fraction interval claim (``c1``), an int claim between two
    discrete timelines (``midi 96``, event names), a float on a ``quarters``
    timeline declaring ``float`` (``c3``), a NOMATCH claim (``c4``) and a
    conceptual claim (``c5``); ``provenance`` gives each its own.
    """
    interval = MatchClaim.from_events(
        {"id": "note_7", "start": Fraction(7, 3), "end": Fraction(10, 3)},
        SCORE,
        {"id": "onset_7", "start": 1.25, "end": 2.5},
        AUDIO,
        end_coord_key="end",
        provenance=provenance[0],
        id="c1",
    )
    discrete = MatchClaim.from_events(
        {"id": "tick_96", "name": "C4", "start": 96},
        MIDI,
        {"id": "sample_44100", "name": "onset", "start": 44100},
        PCM,
        certainty=0.5,
        provenance=provenance[1],
        id="midi 96",
    )
    float_quarters = MatchClaim.from_events(
        {"id": "n2", "start": Coordinate(1.5, TimeUnit.quarters, number_type="float")},
        FLOAT_SCORE,
        {"id": "o2", "start": 0.75},
        AUDIO,
        provenance=provenance[2],
        id="c3",
    )
    nomatch = MatchClaim.nomatch(
        {"id": "note_9", "start": Fraction(27, 2)},
        SCORE,
        AUDIO,
        provenance=provenance[3],
        id="c4",
    )
    conceptual = MatchClaim.from_events(
        {"id": "n10", "start": 4},
        SCORE,
        {"id": "o10", "start": 2.0},
        AUDIO,
        is_synchronous=False,
        provenance=provenance[4],
        id="c5",
    )
    return [interval, discrete, float_quarters, nomatch, conceptual]


NONE5: list[Provenance | None] = [None] * 5

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
                pa.field("number_type", pa.string()),
            ]
        )
        assert coordinate_type == COORDINATE_CELL_TYPE
        # An anchor is its two coordinates: its timelines are its row's.
        assert MatchClaimField.pa_schema.field("start_anchor").type == pa.struct(
            [
                pa.field("coordinate_a", COORDINATE_CELL_TYPE),
                pa.field("coordinate_b", COORDINATE_CELL_TYPE),
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
            "provenance",
            "certainty",
            "event_a_id",
            "event_a_name",
            "event_b_id",
            "event_b_name",
            "id",
        ]

    def test_float_cells_carry_their_exact_ratio(
        self, gold_field: MatchClaimField
    ) -> None:
        anchors = gold_field.table.column("start_anchor").to_pylist()
        assert [
            (cell["value"], cell["numerator"], cell["denominator"], cell["number_type"])
            for cell in (anchor["coordinate_b"] for anchor in anchors)
        ] == [(10.0, 10, 1, "float"), (20.0, 20, 1, "float"), (21.0, 21, 1, "float")]
        tenth = MatchClaimField.from_columns(
            timeline_a_ids=["A"],
            timeline_b_ids=["B"],
            event_a_ids=["a"],
            event_b_ids=["b"],
            coordinate_a=[0],
            coordinate_b=[0.1],
            timelines=GOLD_AXES,
        )
        cell = tenth.table.column("start_anchor").to_pylist()[0]["coordinate_b"]
        assert (cell["numerator"], cell["denominator"]) == (
            3602879701896397,
            36028797018963968,
        )
        assert tenth[0].start_anchor.coordinate_b == Coordinate(0.1, TimeUnit.seconds)

    def test_timeline_ids(self, gold_field: MatchClaimField) -> None:
        assert gold_field.timeline_ids == {"A", "B", "C"}

    def test_provenance_property(self, gold_field: MatchClaimField) -> None:
        assert gold_field.provenance is SHARED

    def test_from_columns_accepts_pyarrow_arrays(self) -> None:
        field = MatchClaimField.from_columns(
            timeline_a_ids=pa.array(["A", "B"]),
            timeline_b_ids=pa.array(["B", "C"]),
            event_a_ids=pa.array(["a", "b"]),
            event_b_ids=pa.chunked_array([["b"], ["c"]]),
            coordinate_a=pa.array([0, 1], type=pa.int64()),
            coordinate_b=pa.array([10.0, 11.0], type=pa.float64()),
            timelines=GOLD_AXES,
        )
        assert [claim.event_b_id for claim in field] == ["b", "c"]
        assert field[1].start_anchor.coordinate_a == Coordinate(1.0, TimeUnit.seconds)

    def test_from_columns_writes_each_row_on_its_axis(
        self, gold_field: MatchClaimField
    ) -> None:
        first, _, last = gold_field.to_claims()
        assert first.start_anchor.coordinate_a == Coordinate(
            Fraction(0, 1), TimeUnit.quarters
        )
        assert first.start_anchor.coordinate_a.number_type is NumberType.fraction
        assert last.start_anchor.coordinate_a == Coordinate(1.0, TimeUnit.seconds)
        assert type(last.start_anchor.coordinate_a.value) is float
        assert [(c.event_a_id, c.event_b_id) for c in (first, last)] == [
            ("a0", "b0"),
            ("b1", "c1"),
        ]
        assert [c.certainty for c in gold_field] == [1.0, 1.0, 1.0]
        assert [c.claim_type.name for c in gold_field] == ["event_match"] * 3

    def test_from_columns_end_columns(self) -> None:
        field = MatchClaimField.from_columns(
            ["score", "score"],
            ["audio", "audio"],
            ["n1", "n2"],
            ["o1", "o2"],
            [Fraction(1, 3), 2],
            [0.5, 1.0],
            timelines=[SCORE, AUDIO],
            end_coordinate_a=[Fraction(2, 3), None],
            end_coordinate_b=[0.75, None],
            certainty=0.25,
        )
        interval, instant = field.to_claims()
        assert interval.get_interval_for("score").end == Coordinate(
            Fraction(2, 3), TimeUnit.quarters
        )
        assert interval.get_interval_for("audio").end == Coordinate(
            0.75, TimeUnit.seconds
        )
        assert instant.end_anchor is None
        assert [interval.certainty, instant.certainty] == [0.25, 0.25]

    def test_from_columns_refusals(self) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "from_columns requires equal-length columns; got lengths "
                "timeline_a_id=1, timeline_b_id=2, event_a_id=1, event_b_id=1, "
                "coordinate_a=1, coordinate_b=1."
            ),
        ):
            MatchClaimField.from_columns(
                ["A"], ["B", "C"], ["a"], ["b"], [0], [1.0], timelines=GOLD_AXES
            )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Row 1 names no event on side b: a MatchClaim names the events it "
                "matches. Give a side without an event of its own a Projection event "
                "first, never a bare coordinate."
            ),
        ):
            MatchClaimField.from_columns(
                ["A", "A"],
                ["B", "B"],
                ["a", "a"],
                ["b", None],
                [0, 1],
                [1.0, 2.0],
                timelines=GOLD_AXES,
            )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "The rows name timeline 'Z', which is not among the timelines given."
            ),
        ):
            MatchClaimField.from_columns(
                ["A"], ["Z"], ["a"], ["z"], [0], [1.0], timelines=GOLD_AXES
            )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "from_columns takes the end columns of both sides, or neither."
            ),
        ):
            MatchClaimField.from_columns(
                ["A"],
                ["B"],
                ["a"],
                ["b"],
                [0],
                [1.0],
                timelines=GOLD_AXES,
                end_coordinate_a=[1],
            )


# endregion


# region Materialisation


class TestGetItem:
    """Row materialisation into MatchClaim objects."""

    def test_getitem_first_row(self, gold_field: MatchClaimField) -> None:
        claim = gold_field[0]
        assert isinstance(claim, MatchClaim)
        assert (claim.timeline_a_id, claim.timeline_b_id) == ("A", "B")
        assert (claim.is_synchronous, claim.is_interval) == (True, False)
        assert claim.start_anchor.coordinate_b == Coordinate(10.0, TimeUnit.seconds)
        assert (claim.event_a_id, claim.event_b_id) == ("a0", "b0")
        assert claim.provenance is SHARED

    def test_getitem_injects_provenance(self, gold_field: MatchClaimField) -> None:
        assert gold_field.table.column("provenance").to_pylist() == [None, None, None]
        assert gold_field[1].provenance is SHARED

    def test_getitem_negative_index(self, gold_field: MatchClaimField) -> None:
        last = gold_field[-1]
        assert (last.timeline_a_id, last.timeline_b_id) == ("B", "C")
        assert last.start_anchor.coordinate_b == Coordinate(21.0, TimeUnit.seconds)

    def test_getitem_out_of_range(self, gold_field: MatchClaimField) -> None:
        with pytest.raises(IndexError):
            _ = gold_field[3]
        with pytest.raises(IndexError):
            _ = gold_field[-4]

    def test_iter_and_to_claims(self, gold_field: MatchClaimField) -> None:
        claims = list(gold_field)
        assert claims == gold_field.to_claims()
        assert [(c.timeline_a_id, c.timeline_b_id) for c in claims] == [
            ("A", "B"),
            ("A", "C"),
            ("B", "C"),
        ]
        assert all(c.end_anchor is None for c in claims)


# endregion


# region Filters and queries


class TestFilters:
    """Vectorized connecting / filter views."""

    def test_connecting(self, gold_field: MatchClaimField) -> None:
        connected = gold_field.connecting("C")
        assert [(c.timeline_a_id, c.timeline_b_id) for c in connected] == [
            ("A", "C"),
            ("B", "C"),
        ]
        assert connected.provenance is SHARED
        assert len(gold_field.connecting("Z")) == 0

    def test_filter_timeline_ids(self, gold_field: MatchClaimField) -> None:
        filtered = gold_field.filter(timeline_ids={"A"})
        assert [(c.timeline_a_id, c.timeline_b_id) for c in filtered] == [
            ("A", "B"),
            ("A", "C"),
        ]
        assert len(gold_field.filter(timeline_ids={"A", "B"})) == 3

    def test_filter_timeline_id_equals_connecting(
        self, gold_field: MatchClaimField
    ) -> None:
        assert gold_field.filter(timeline_id="C").table.equals(
            gold_field.connecting("C").table
        )

    def test_filter_without_criteria_copies(self, gold_field: MatchClaimField) -> None:
        assert gold_field.filter().table.equals(gold_field.table)

    def test_filter_combines_with_and(self, gold_field: MatchClaimField) -> None:
        both = gold_field.filter(timeline_id="A", timeline_ids={"B"})
        assert [(c.timeline_a_id, c.timeline_b_id) for c in both] == [("A", "B")]
        both = gold_field.filter(timeline_id="A", between=("A", "C"))
        assert [(c.timeline_a_id, c.timeline_b_id) for c in both] == [("A", "C")]

    def test_filter_id_pattern(self, gold_field: MatchClaimField) -> None:
        assert len(gold_field.filter(id_pattern=r"^A$")) == 2
        assert len(gold_field.filter(id_pattern=r"^Z")) == 0

    def test_filter_between_order_independent(
        self, gold_field: MatchClaimField
    ) -> None:
        forward = gold_field.filter(between=("A", "B"))
        assert len(forward) == 1
        assert forward.table.equals(gold_field.filter(between=("B", "A")).table)

    def test_filter_within_requires_both_sides(
        self, gold_field: MatchClaimField
    ) -> None:
        filtered = gold_field.filter(within={"A", "B"})
        assert [(c.timeline_a_id, c.timeline_b_id) for c in filtered] == [("A", "B")]

    def test_filter_mutually_exclusive_raises(
        self, gold_field: MatchClaimField
    ) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            gold_field.filter(synchronous_only=True, nomatch_only=True)


class TestAt:
    """Exact-coordinate row selection."""

    def test_at_matches_a_side(self, gold_field: MatchClaimField) -> None:
        selected = gold_field.at("A", 0)
        assert [(c.timeline_a_id, c.timeline_b_id) for c in selected] == [
            ("A", "B"),
            ("A", "C"),
        ]

    def test_at_matches_b_side(self, gold_field: MatchClaimField) -> None:
        selected = gold_field.at("C", 20.0)
        assert [(c.timeline_a_id, c.timeline_b_id) for c in selected] == [("A", "C")]

    def test_at_exact_equality_only(self, gold_field: MatchClaimField) -> None:
        assert len(gold_field.at("C", 20.0000001)) == 0

    def test_at_unknown_timeline_empty(self, gold_field: MatchClaimField) -> None:
        empty = gold_field.at("Z", 0)
        assert len(empty) == 0
        assert empty.provenance is SHARED

    def test_at_on_empty_field(self) -> None:
        assert len(MatchClaimField.from_claims([]).at("A", 0)) == 0


class TestConnectsGroups:
    def test_connects_groups(self, gold_field: MatchClaimField) -> None:
        assert gold_field.connects_groups({"A"}, {"C"}) is True
        assert gold_field.connects_groups({"C"}, {"A"}) is True
        assert gold_field.connects_groups({"A"}, {"Z"}) is False
        assert MatchClaimField.from_claims([]).connects_groups({"A"}, {"B"}) is False


def _ids_row_by_row(field: MatchClaimField) -> set[str]:
    """The timeline ids as they were read before: every row, A column first."""
    data = field._raw.data
    ids: set[str] = set()
    if data is None or len(data) == 0:
        return ids
    for name in ("timeline_a_id", "timeline_b_id"):
        ids.update(v for v in data.field(name).to_pylist() if v is not None)
    return ids


def _counting_reads(monkeypatch: pytest.MonkeyPatch) -> list[pa.Array]:
    """Record the array of every read of a field's timeline ids."""
    reads: list[pa.Array] = []
    read = MatchClaimField._distinct_timeline_ids

    def counted(data: pa.StructArray) -> tuple[str, ...]:
        reads.append(data)
        return read(data)

    monkeypatch.setattr(
        MatchClaimField, "_distinct_timeline_ids", staticmethod(counted)
    )
    return reads


class TestTimelineIdsReadOnce:
    """``timeline_ids`` answers as the row-by-row reading did, read once per table."""

    def _fields(
        self, gold_field: MatchClaimField, thoresen: list[MatchClaim]
    ) -> list[MatchClaimField]:
        b_first = MatchClaimField.from_columns(
            timeline_a_ids=["B", "C", "A"],
            timeline_b_ids=["A", "B", "C"],
            event_a_ids=["b0", "c0", "a0"],
            event_b_ids=["a1", "b1", "c1"],
            coordinate_a=[1.0, 2.0, 3],
            coordinate_b=[4, 5.0, 6.0],
            timelines=GOLD_AXES,
        )
        return [
            gold_field,
            MatchClaimField.from_claims(_every_kind(NONE5)),
            MatchClaimField.from_claims(thoresen),
            b_first,
            gold_field.filter(timeline_ids={"C"}),
            MatchClaimField.from_claims([]),
        ]

    def test_same_ids_in_the_same_order(
        self,
        gold_field: MatchClaimField,
        thoresen_segment_claims: list[MatchClaim],
    ) -> None:
        expected = [
            ["A", "B", "C"],
            ["score", "midi", "float_score", "audio", "pcm"],
            ["dgt_1", "dgt_2"],
            ["B", "C", "A"],
            ["A", "B", "C"],
            [],
        ]
        fields = self._fields(gold_field, thoresen_segment_claims)
        for field, ids in zip(fields, expected, strict=True):
            assert field.timeline_ids == set(ids)
            assert list(field.timeline_ids) == list(_ids_row_by_row(field))

    def test_one_field_reads_once(
        self, gold_field: MatchClaimField, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads = _counting_reads(monkeypatch)
        for _ in range(5):
            assert gold_field.timeline_ids == {"A", "B", "C"}
        assert len(reads) == 1

    def test_the_returned_set_is_the_caller_s(
        self, gold_field: MatchClaimField
    ) -> None:
        ids = gold_field.timeline_ids
        ids.add("Z")
        assert gold_field.timeline_ids == {"A", "B", "C"}

    def test_a_bundle_reads_its_table_once_per_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        score = ContinuousLogicalTimeline(length=16, uid="score")
        audio = ContinuousPhysicalTimeline(length=60.0, uid="audio")
        bundle = AlignmentBundle(id="reads")
        bundle.add_timeline(score)
        bundle.add_timeline(audio)
        bundle.add_match_claims(
            [
                MatchClaim.from_events(
                    {"id": f"n{q}", "start": q},
                    score,
                    {"id": f"o{q}", "start": q / 2},
                    audio,
                )
                for q in (0, 4, 8)
            ]
        )
        reads = _counting_reads(monkeypatch)

        first = bundle.match_claim_table
        for q in (0, 4, 8, 4, 0):
            bundle.get_matchstamp_at(q, "score")
        assert sum(data is first._raw.data for data in reads) == 1

        bundle.add_match_claims(
            [
                MatchClaim.from_events(
                    {"id": "n12", "start": 12},
                    score,
                    {"id": "o12", "start": 6.0},
                    audio,
                )
            ]
        )
        second = bundle.match_claim_table
        assert second is not first
        for q in (0, 4, 8, 12, 0):
            bundle.get_matchstamp_at(q, "score")
        assert sum(data is first._raw.data for data in reads) == 1
        assert sum(data is second._raw.data for data in reads) == 1

    def test_an_assigned_table_answers_its_own_ids(
        self, gold_field: MatchClaimField
    ) -> None:
        bundle = AlignmentBundle(id="assigned")
        assert bundle.match_claim_table.timeline_ids == set()
        bundle.match_claim_table = gold_field
        assert bundle.match_claim_table.timeline_ids == {"A", "B", "C"}


# endregion


# region Every kind of MatchClaim


class TestEveryKind:
    """NOMATCH and conceptual claims are rows like any other."""

    def test_round_trip_of_every_kind(self) -> None:
        claims = _every_kind(NONE5)
        field = MatchClaimField.from_claims(claims)
        assert field.to_claims() == claims
        assert [claim.claim_type.name for claim in field] == [
            "event_match",
            "event_match",
            "event_match",
            "nomatch",
            "conceptual",
        ]
        assert [claim.id for claim in field] == ["c1", "midi 96", "c3", "c4", "c5"]
        # Each anchor is rebuilt on its row's timelines.
        assert field[0].end_anchor == claims[0].end_anchor
        assert field[0].end_anchor.timelines == ("score", "audio")
        assert MatchClaimField.from_dict(field.to_dict()).to_claims() == claims

    def test_queries_read_the_kinds(self) -> None:
        field = MatchClaimField.from_claims(_every_kind(NONE5))
        assert [c.id for c in field.filter(synchronous_only=True)] == [
            "c1",
            "midi 96",
            "c3",
        ]
        assert [c.id for c in field.filter(nomatch_only=True)] == ["c4"]
        # c4 is a NOMATCH of the event at 27/2 on score: no anchor, never relevant.
        assert [c.id for c in field.at("score", Fraction(27, 2))] == []
        assert [c.id for c in field.at("score", Fraction(3, 1))] == ["c1"]
        assert field.connects_groups({"score"}, {"audio"}) is True
        anchorless = MatchClaimField.from_claims(_every_kind(NONE5)[3:5])
        assert anchorless.connects_groups({"score"}, {"audio"}) is False

    def test_repr_html_of_an_anchorless_row(self) -> None:
        field = MatchClaimField.from_claims(_every_kind(NONE5)[3:4])
        assert field._repr_html_() == (
            "<div style='font-family: monospace;'><strong>MatchClaimField</strong> "
            "<span style='color: #555;'>claims=1, timelines=2</span><table "
            "style='border-collapse: collapse; margin-top: 4px;'><thead><tr><th>#</th>"
            "<th>timeline A</th><th>coord A</th><th>timeline B</th><th>coord B</th>"
            "</tr></thead><tbody><tr><td>0</td><td>score</td><td>—</td>"
            "<td>audio</td><td>—</td></tr></tbody></table></div>"
        )


# endregion


# region The belief filter


class TestBeliefFilter:
    def _field(self) -> MatchClaimField:
        claims = _every_kind(
            [
                None,
                Provenance(method="manual", beliefs=(_belief("jo", "false"),)),
                Provenance(method="manual", beliefs=(_belief("jo", "true"),)),
                None,
                None,
            ]
        )[:3]
        return MatchClaimField.from_claims(claims)

    @pytest.mark.parametrize(
        ("believer", "beliefs", "kept"),
        [
            (None, DEFAULT_BELIEFS, ["c1", "midi 96", "c3"]),
            ("jo", DEFAULT_BELIEFS, ["c1", "c3"]),
            ("jo", ["true"], ["c3"]),
            ("jo", ["undecided"], ["c1"]),
            ("jo", ["false"], ["midi 96"]),
            ("ana", DEFAULT_BELIEFS, ["c1", "midi 96", "c3"]),
            ("ana", ["true"], []),
        ],
    )
    def test_rows(
        self, believer: str | None, beliefs: list[str], kept: list[str]
    ) -> None:
        filtered = self._field().filter(believer=believer, beliefs=beliefs)
        assert [claim.id for claim in filtered] == kept

    def test_shared_provenance_decides_for_every_row(
        self, gold_field: MatchClaimField
    ) -> None:
        doubted = MatchClaimField.from_claims(
            gold_field.to_claims(),
            provenance=SHARED.with_belief(_belief("jo", "false")),
        )
        assert len(doubted.filter(believer="jo")) == 0
        assert len(doubted.filter(believer="ana")) == 3
        assert len(doubted.filter(believer="jo", timeline_id="C")) == 0

    @pytest.mark.parametrize("believer", ["jo", None])
    def test_unknown_state(
        self, gold_field: MatchClaimField, believer: str | None
    ) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "'not_false' is no belief state: write 'true', 'undecided', "
                "'unlikely' or 'false'"
            ),
        ):
            gold_field.filter(believer=believer, beliefs=["not_false"])


# endregion


# region Coordinates on the timelines, vectorized


def _off_axis(where: str, unit: str, number_type: str, timeline: str, axis: str) -> str:
    return (
        f"{where} is in {unit} ({number_type}), but timeline '{timeline}' counts in "
        f"{axis}: a MatchClaim's coordinates are on its timelines' own axes. "
        f"Convert a position through the timeline's C-Map before the MatchClaim "
        f"is made."
    )


class TestCheckOnTimelines:
    def _field(self) -> MatchClaimField:
        good = MatchClaim.from_events(
            {"id": "n1", "start": 1}, SCORE, {"id": "t1", "start": 96}, MIDI, id="good"
        )
        seconds_on_samples = MatchClaim(
            timeline_a_id="score",
            timeline_b_id="pcm",
            start_anchor=AlignmentAnchor(
                timeline_a_id="score",
                coordinate_a=Coordinate(Fraction(1, 2), TimeUnit.quarters),
                timeline_b_id="pcm",
                coordinate_b=Coordinate(3.16952381, TimeUnit.seconds),
            ),
            event_a_id="n2",
            event_b_id="s2",
            id="off",
        )
        return MatchClaimField.from_claims([good, seconds_on_samples])

    def test_refused_and_accepted(self) -> None:
        field = self._field()
        with pytest.raises(
            ValueError,
            match=re.escape(
                _off_axis(
                    "Row 1 start_anchor.coordinate_b",
                    "seconds",
                    "float",
                    "pcm",
                    "samples (int)",
                )
            ),
        ):
            field.check_on_timelines(TIMELINES)
        accepted = {"score": SCORE, "midi": MIDI, "pcm": ("seconds", "float")}
        assert field.check_on_timelines(accepted) is field
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Row 0 start_anchor.coordinate_b is on timeline 'midi', which is not "
                "among the timelines given."
            ),
        ):
            field.check_on_timelines([SCORE, PCM])

    def test_a_nomatch_row_holds_no_coordinate(self) -> None:
        nomatch = MatchClaim.nomatch(
            {"id": "x", "start": Fraction(27, 2)}, SCORE, AUDIO, id="x"
        )
        field = MatchClaimField.from_claims([nomatch])
        assert field.check_on_timelines([SCORE, AUDIO]) is field
        # No cell of the row is on any axis, so no axis can refuse it.
        assert (
            field.check_on_timelines({"score": ("quarters", "float"), "audio": AUDIO})
            is field
        )

    def test_from_parquet_runs_the_check(self, tmp_path: Path) -> None:
        path = self._field().to_parquet(tmp_path)
        assert len(MatchClaimField.from_parquet(path)) == 2
        with pytest.raises(
            ValueError, match=re.escape("Row 1 start_anchor.coordinate_b")
        ):
            MatchClaimField.from_parquet(path, timelines=TIMELINES)


# endregion


# region Dictionary form


#: The coordinate cells of a ``MatchClaimField`` row, as key paths.
_COORDINATE_PATHS = (
    ("start_anchor", "coordinate_a"),
    ("start_anchor", "coordinate_b"),
    ("end_anchor", "coordinate_a"),
    ("end_anchor", "coordinate_b"),
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
    return MatchClaimField.from_columns(
        ["score", "midi"],
        ["audio", "pcm"],
        ["n1", "t96"],
        ["o1", "s44100"],
        [Fraction(1, 3), 96],
        [0.5, 44100],
        timelines=TIMELINES,
    )


def _literal_row(claim_id: str | None, a: tuple, b: tuple) -> dict:
    """The ``to_dict`` row of an instant event match between *a* and *b*.

    *a* and *b* are ``(timeline id, event id, unit, number type, value)``;
    the anchor holds the two coordinates, its timelines being the row's.
    """
    return {
        "timeline_a_id": a[0],
        "timeline_b_id": b[0],
        "start_anchor": {
            "coordinate_a": {"unit": a[2], "number_type": a[3], "value": a[4]},
            "coordinate_b": {"unit": b[2], "number_type": b[3], "value": b[4]},
        },
        "end_anchor": None,
        "is_synchronous": True,
        "provenance": None,
        "certainty": 1.0,
        "event_a_id": a[1],
        "event_a_name": None,
        "event_b_id": b[1],
        "event_b_name": None,
        "id": claim_id,
    }


#: The ``midi 96`` claim of :func:`_every_kind` as its ``to_dict`` row, with
#: its own provenance stored in the row.
_DISCRETE_ROW = {
    "timeline_a_id": "midi",
    "timeline_b_id": "pcm",
    "start_anchor": {
        "coordinate_a": {"unit": "ticks", "number_type": "int", "value": 96},
        "coordinate_b": {"unit": "samples", "number_type": "int", "value": 44100},
    },
    "end_anchor": None,
    "is_synchronous": True,
    "provenance": {
        "stated_by": ["ana"],
        "date": None,
        "method": "manual",
        "service": None,
        "basis": None,
        "match": None,
        "note": None,
        "beliefs": [
            {"actor": "ana", "belief": "true", "date": "2026-11-21", "note": None}
        ],
    },
    "certainty": 0.5,
    "event_a_id": "tick_96",
    "event_a_name": "C4",
    "event_b_id": "sample_44100",
    "event_b_name": "onset",
    "id": "midi 96",
}

HUMAN = Provenance(
    stated_by=("ana",), method="manual", beliefs=(_belief("ana", "true"),)
)
SOFTWARE = Provenance(service="https://example.org/dtw", method="inferred")


class TestDictRoundTrip:
    """to_dict / from_dict serialisation."""

    def test_roundtrip_from_dict(self, gold_field: MatchClaimField) -> None:
        rebuilt = MatchClaimField.from_dict(gold_field.to_dict())
        assert rebuilt.table.equals(gold_field.table, check_metadata=True)
        assert rebuilt.provenance == SHARED

    def test_to_dict_renders_the_table(self, gold_field: MatchClaimField) -> None:
        data = gold_field.to_dict()
        assert list(data) == ["provenance", "rows"]
        assert data["provenance"] == {
            "method": "inferred",
            "service": "https://example.org/aligner",
        }
        assert all(list(row) == gold_field.table.column_names for row in data["rows"])
        field = MatchClaimField.from_claims(
            _every_kind([None, HUMAN, SOFTWARE, None, HUMAN])
        )
        rows = field.to_dict()["rows"]
        assert [_without_coordinates(row) for row in rows] == [
            _without_coordinates(row) for row in field.table.to_pylist()
        ]
        for index, row in enumerate(rows):
            claim = field[index].to_dict()
            assert row["start_anchor"] == claim["start_anchor"]
            assert row["end_anchor"] == claim["end_anchor"]
        assert list(rows[0]["end_anchor"]) == ["coordinate_a", "coordinate_b"]
        assert rows[3]["start_anchor"] is None

    def test_to_dict_literal(self) -> None:
        field = _two_claim_field()
        assert field.to_dict() == {
            "provenance": None,
            "rows": [
                _literal_row(
                    None,
                    ("score", "n1", "quarters", "fraction", [1, 3]),
                    ("audio", "o1", "seconds", "float", 0.5),
                ),
                _literal_row(
                    None,
                    ("midi", "t96", "ticks", "int", 96),
                    ("pcm", "s44100", "samples", "int", 44100),
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
        claims = _every_kind([None, HUMAN, SOFTWARE, None, HUMAN])
        field = MatchClaimField.from_claims(claims)
        data = field.to_dict()
        rows = data["rows"]
        assert [row["id"] for row in rows] == ["c1", "midi 96", "c3", "c4", "c5"]
        assert [row["is_synchronous"] for row in rows] == [
            True,
            True,
            True,
            False,
            False,
        ]
        assert [row["event_b_id"] for row in rows] == [
            "onset_7",
            "sample_44100",
            "o2",
            None,
            "o10",
        ]
        assert rows[1] == _DISCRETE_ROW
        assert data["provenance"] is None
        rebuilt = MatchClaimField.from_dict(data)
        assert rebuilt.table.equals(field.table, check_metadata=True)
        assert rebuilt.to_claims() == claims

    @pytest.mark.parametrize("shared", [False, True])
    def test_yaml_round_trip(self, shared: bool) -> None:
        claims = _every_kind([HUMAN, HUMAN, HUMAN, HUMAN, HUMAN if shared else None])
        field = MatchClaimField.from_claims(claims)
        assert (field.provenance == HUMAN) is shared
        data = field.to_dict()
        loaded = timetoshare.loads(timetoshare.dumps(data, "yaml"))
        assert loaded == data
        rebuilt = MatchClaimField.from_dict(loaded)
        assert rebuilt.to_claims() == claims
        assert rebuilt.table.equals(field.table, check_metadata=True)

    def test_from_dict_refuses_other_keys(self, gold_field: MatchClaimField) -> None:
        data = gold_field.to_dict()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "A MatchClaimField dictionary holds the keys ['provenance', 'rows'], "
                "not ['metadata', 'rows']."
            ),
        ):
            MatchClaimField.from_dict({"metadata": None, "rows": data["rows"]})
        columns = gold_field.table.column_names
        lacking = dict(data["rows"][1])
        del lacking["event_a_id"]
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"In a MatchClaimField dictionary, row 1 must hold the keys "
                f"{columns}; it lacks ['event_a_id'] and has the unknown keys []."
            ),
        ):
            MatchClaimField.from_dict(data | {"rows": [data["rows"][0], lacking]})
        extra = data["rows"][0] | {"metadata": None}
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"In a MatchClaimField dictionary, row 0 must hold the keys "
                f"{columns}; it lacks [] and has the unknown keys ['metadata']."
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
                "row 0 start_anchor must hold the keys ['coordinate_a', "
                "'coordinate_b']; it lacks [] and has the unknown keys ['unit'].",
            ),
            (
                ("start_anchor",),
                lambda mapping: mapping.update(timeline_a_id="score"),
                "row 0 start_anchor must hold the keys ['coordinate_a', "
                "'coordinate_b']; it lacks [] and has the unknown keys "
                "['timeline_a_id'].",
            ),
            (
                ("provenance",),
                lambda mapping: mapping.pop("note"),
                "row 0 provenance must hold the keys ['stated_by', 'date', 'method', "
                "'service', 'basis', 'match', 'note', 'beliefs']; it lacks ['note'] "
                "and has the unknown keys [].",
            ),
            (
                ("provenance", "beliefs", 0),
                lambda mapping: mapping.update(weight=1),
                "row 0 provenance.beliefs.0 must hold the keys ['actor', 'belief', "
                "'date', 'note']; it lacks [] and has the unknown keys ['weight'].",
            ),
        ],
    )
    def test_from_dict_refuses_other_nested_keys(
        self, path: tuple, edit: object, message: str
    ) -> None:
        field = MatchClaimField.from_claims(
            _every_kind([HUMAN, None, None, None, None])
        )
        data = field.to_dict()
        mapping = data["rows"][0]
        for key in path:
            mapping = mapping[key]
        edit(mapping)  # type: ignore[operator]
        with pytest.raises(
            ValueError, match=re.escape(f"In a MatchClaimField dictionary, {message}")
        ):
            MatchClaimField.from_dict(data)

    def test_from_dict_refuses_provenance_stated_twice(self) -> None:
        field = MatchClaimField.from_claims(
            _every_kind([None, HUMAN, HUMAN, None, None])
        )
        data = field.to_dict() | {"provenance": HUMAN.to_dict()}
        with pytest.raises(
            ValueError,
            match=re.escape(
                "A MatchClaimField dictionary states its provenance once: this "
                "dictionary carries it under its key 'provenance' and in 2 row(s)."
            ),
        ):
            MatchClaimField.from_dict(data)


# endregion


# region Table and Parquet file


def _with_child(table: pa.Table, name: str, child: pa.Array) -> pa.Table:
    """Return *table* with one of its property columns replaced."""
    return table.set_column(table.column_names.index(name), name, child)


#: The refusal of a table lacking columns of the MatchClaim properties.
_LACKS_PROPERTIES = (
    "A MatchClaim table holds one column per MatchClaim property, {}; "
    "this table lacks {}."
)


class TestTableRoundTrip:
    """``.table`` / ``from_table``."""

    def test_claims_are_what_they_say(self) -> None:
        interval, discrete, float_quarters, nomatch, conceptual = _every_kind(NONE5)
        assert interval.end_anchor.coordinate_a.value == Fraction(10, 3)
        assert interval.end_anchor.coordinate_b.value == 2.5
        assert type(discrete.start_anchor.coordinate_a.value) is int
        assert type(discrete.start_anchor.coordinate_b.value) is int
        assert float_quarters.start_anchor.coordinate_a.number_type is NumberType.float
        assert (nomatch.event_a_id, nomatch.anchors) == ("note_9", [])
        assert (conceptual.is_synchronous, conceptual.anchors) == (False, [])

    def test_shared_provenance_in_schema_metadata(self) -> None:
        claims = _every_kind([SOFTWARE] * 5)
        field = MatchClaimField.from_claims(claims)
        assert field.provenance == SOFTWARE
        assert field.table.schema.metadata == {
            b"timetoshare:provenance": (
                b'{"method": "inferred", "service": "https://example.org/dtw"}'
            )
        }
        assert MatchClaimField.from_table(field.table).to_claims() == claims

    def test_mixed_provenance_round_trip(self) -> None:
        claims = _every_kind([None, HUMAN, SOFTWARE, None, HUMAN])
        field = MatchClaimField.from_claims(claims)
        assert field.provenance is None
        assert field.table.schema.metadata is None
        restored = MatchClaimField.from_table(field.table)
        assert [claim.provenance for claim in restored] == [
            None,
            HUMAN,
            SOFTWARE,
            None,
            HUMAN,
        ]
        assert restored.to_claims() == claims

    def test_chunked_columns(self, gold_field: MatchClaimField) -> None:
        table = gold_field.table
        chunked = pa.concat_tables([table.slice(0, 1), table.slice(1)])
        assert chunked.column("start_anchor").num_chunks == 2
        restored = MatchClaimField.from_table(chunked)
        assert restored.table.equals(gold_field.table, check_metadata=True)

    def test_keeps_further_columns(self, gold_field: MatchClaimField) -> None:
        extra = pa.array([1, 2, 3])
        table = gold_field.table.append_column("extra", extra)
        field = MatchClaimField.from_table(table)
        assert field.table.equals(table, check_metadata=True)
        assert field.package_table.equals(
            gold_field.package_table.append_column("extra", extra), check_metadata=True
        )
        assert field.filter(timeline_id="C").table.column("extra").to_pylist() == [2, 3]
        assert field != gold_field

    def test_refuses_missing_columns(self, gold_field: MatchClaimField) -> None:
        columns = gold_field.table.column_names
        with pytest.raises(
            ValueError,
            match=re.escape(_LACKS_PROPERTIES.format(columns, ["id"])),
        ):
            MatchClaimField.from_table(gold_field.table.drop_columns(["id"]))
        single = pa.table({"match_claim": gold_field.table.to_struct_array()})
        with pytest.raises(
            ValueError,
            match=re.escape(_LACKS_PROPERTIES.format(columns, columns)),
        ):
            MatchClaimField.from_table(single)

    def test_refuses_other_type(self, gold_field: MatchClaimField) -> None:
        table = _with_child(gold_field.table, "is_synchronous", pa.array([1, 1, 1]))
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Column 'is_synchronous' has type int64, which is not the type of "
                "the MatchClaim property, bool."
            ),
        ):
            MatchClaimField.from_table(table)

    @pytest.mark.parametrize(
        ("column", "values", "message"),
        [
            (
                "is_synchronous",
                [True, False, True],
                "MatchClaim table row 1 is not synchronous and holds an anchor.",
            ),
            (
                "event_b_id",
                ["b0", "c0", None],
                "MatchClaim table row 2 is synchronous and does not name an event on "
                "each side; give a side without an event of its own a Projection "
                "event first.",
            ),
            (
                "certainty",
                [1.0, 1.5, 1.0],
                "MatchClaim table row 1 has a certainty outside [0, 1].",
            ),
            (
                "timeline_a_id",
                ["B", "A", "B"],
                "MatchClaim table row 0 connects a timeline with itself.",
            ),
        ],
    )
    def test_refuses_rows_no_claim_could_be(
        self, gold_field: MatchClaimField, column: str, values: list, message: str
    ) -> None:
        table_type = gold_field.table.schema.field(column).type
        table = _with_child(gold_field.table, column, pa.array(values, type=table_type))
        with pytest.raises(ValueError, match=re.escape(message)):
            MatchClaimField.from_table(table)

    def test_swapped_timelines_refused(self, gold_field: MatchClaimField) -> None:
        # An anchor's coordinates are on its row's timelines: swapping a row's
        # timelines puts each coordinate on the other's axis.
        table = gold_field.table
        table = _with_child(table, "timeline_a_id", pa.array(["B", "A", "B"]))
        table = _with_child(table, "timeline_b_id", pa.array(["A", "C", "C"]))
        with pytest.raises(
            ValueError,
            match=re.escape(
                "The MatchClaim table holds coordinates of timeline 'B' in quarters "
                "(fraction) and seconds (float): the coordinates of one timeline are "
                "on its one axis."
            ),
        ):
            MatchClaimField.from_table(table)

    def test_null_certainty_read(self, gold_field: MatchClaimField) -> None:
        table = _with_child(
            gold_field.table, "certainty", pa.array([None, 0.5, 1.0], pa.float64())
        )
        assert [claim.certainty for claim in MatchClaimField.from_table(table)] == [
            None,
            0.5,
            1.0,
        ]

    def test_one_timeline_on_two_axes_refused(self) -> None:
        field = MatchClaimField.from_columns(
            timeline_a_ids=["A", "A"],
            timeline_b_ids=["B", "C"],
            event_a_ids=["a0", "a1"],
            event_b_ids=["b0", "c0"],
            coordinate_a=[0, 1],
            coordinate_b=[10.0, 44100],
            timelines={**GOLD_AXES, "C": ("samples", "int")},
        )
        # Renaming the row's timeline moves its anchor's coordinate with it.
        table = _with_child(field.table, "timeline_b_id", pa.array(["B", "B"]))
        with pytest.raises(
            ValueError,
            match=re.escape(
                "The MatchClaim table holds coordinates of timeline 'B' in seconds "
                "(float) and samples (int): the coordinates of one timeline are on "
                "its one axis."
            ),
        ):
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
        with pytest.raises(
            ValueError,
            match=re.escape(
                "MatchClaim table row 2 is synchronous and holds no start anchor."
            ),
        ):
            MatchClaimField.from_table(table)

    def test_refuses_anchorless_row_naming_no_event(self) -> None:
        field = MatchClaimField.from_claims(_every_kind(NONE5)[3:4])
        table = _with_child(
            field.table, "event_a_id", pa.array([None], type=pa.string())
        )
        with pytest.raises(
            ValueError, match=re.escape("MatchClaim table row 0 names no event.")
        ):
            MatchClaimField.from_table(table)

    def test_refuses_provenance_stated_twice(self) -> None:
        field = MatchClaimField.from_claims(
            _every_kind([None, HUMAN, HUMAN, None, None])
        )
        table = field.table.replace_schema_metadata(
            {b"timetoshare:provenance": json.dumps(HUMAN.to_dict())}
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "A MatchClaimField table states its provenance once: this table "
                "carries it in its schema metadata ('timetoshare:provenance') "
                "and in 2 row(s)."
            ),
        ):
            MatchClaimField.from_table(table)


class TestParquetFile:
    """``to_parquet`` / ``from_parquet`` and the succession rule."""

    def test_round_trip_through_the_file(self, tmp_path: Path) -> None:
        claims = _every_kind([None, HUMAN, SOFTWARE, None, HUMAN])[:5]
        field = MatchClaimField.from_claims(claims)
        path = field.to_parquet(tmp_path)
        assert path == tmp_path / "match_claims.parquet"
        written = pq.read_table(path)
        assert written.schema.equals(field.package_table.schema)

        def numbers(anchor: dict | None) -> tuple | None:
            if anchor is None:
                return None
            return tuple(
                tuple(anchor[side][m] for m in ("numerator", "denominator", "value"))
                for side in ("coordinate_a", "coordinate_b")
            )

        assert [numbers(a) for a in written.column("start_anchor").to_pylist()] == [
            ((7, 3, None), (None, None, 1.25)),
            ((96, None, None), (44100, None, None)),
            ((None, None, 1.5), (None, None, 0.75)),
            None,
            None,
        ]
        assert numbers(written.column("end_anchor")[0].as_py()) == (
            (10, 3, None),
            (None, None, 2.5),
        )
        restored = MatchClaimField.from_parquet(tmp_path)
        assert restored.to_claims() == claims
        assert restored.table.equals(field.table, check_metadata=True)
        assert (
            MatchClaimField.from_parquet(path, timelines=TIMELINES).to_claims()
            == claims
        )

    def test_a_cell_not_in_the_package_form_is_refused(self, tmp_path: Path) -> None:
        field = MatchClaimField.from_claims(_every_kind(NONE5))
        table = field.package_table
        anchors = table.column("start_anchor").to_pylist()
        anchors[1]["coordinate_b"]["denominator"] = 1
        index = table.schema.get_field_index("start_anchor")
        path = tmp_path / "match_claims.parquet"
        pq.write_table(
            table.set_column(
                index,
                table.schema.field(index),
                pa.array(anchors, table.schema.field(index).type),
            ),
            path,
        )
        with pytest.raises(ValueError) as raised:
            MatchClaimField.from_parquet(path)
        assert str(raised.value) == (
            "MatchClaim table start_anchor.coordinate_b row 1: the int cell "
            "(numerator, denominator, value) holds (44100, 1, null), where an int "
            "is (n, null, null), a fraction (n, d >= 1, null) and a float "
            "(null, null, x) with x finite"
        )

    def test_shared_provenance_through_the_file(
        self, gold_field: MatchClaimField, tmp_path: Path
    ) -> None:
        # The gold vector holds no ids; the bundle it joins mints them.
        bundle = AlignmentBundle()
        for uid, timeline_class in (
            ("A", ContinuousLogicalTimeline),
            ("B", ContinuousPhysicalTimeline),
            ("C", ContinuousPhysicalTimeline),
        ):
            bundle.add_timeline(timeline_class(length=32, uid=uid))
        held = bundle.add_match_claim_field(gold_field).match_claim_table
        path = held.to_parquet(tmp_path / "claims.parquet")
        restored = MatchClaimField.from_parquet(path)
        assert restored.provenance == SHARED
        assert restored.table.column("id").to_pylist() == [
            "claim_1",
            "claim_2",
            "claim_3",
        ]
        assert restored.table.equals(held.table, check_metadata=True)

    def test_empty_field_through_the_file(self, tmp_path: Path) -> None:
        restored = MatchClaimField.from_parquet(
            MatchClaimField.from_claims([]).to_parquet(tmp_path)
        )
        assert (len(restored), restored.provenance) == (0, None)

    def test_succession(self, tmp_path: Path) -> None:
        claims = _every_kind([None, HUMAN, SOFTWARE, None, HUMAN])[:5]
        MatchClaimField.from_claims(claims).to_parquet(tmp_path)

        believed = list(claims)
        believed[2] = claims[2].model_copy(
            update={"provenance": SOFTWARE.with_belief(_belief("jo", "false"))}
        )
        extra = MatchClaim.from_events(
            {"id": "n11", "start": 5},
            SCORE,
            {"id": "o11", "start": 3.0},
            AUDIO,
            id="c7",
        )
        MatchClaimField.from_claims([*believed, extra]).to_parquet(tmp_path)
        written = MatchClaimField.from_parquet(tmp_path).to_claims()
        assert written == [*believed, extra]

        lacking = [claim for claim in written if claim.id != "c3"]
        with pytest.raises(
            ValueError,
            match=re.escape(
                "The new MatchClaim table lacks the MatchClaim 'c3' of the existing "
                "one: a MatchClaim is never removed; record the belief 'false' in it "
                "instead (AlignmentBundle.add_match_claim_belief)."
            ),
        ):
            MatchClaimField.from_claims(lacking).to_parquet(tmp_path)

        def changed(index: int, update: dict) -> list[MatchClaim]:
            edited = list(written)
            edited[index] = written[index].model_copy(update=update)
            return edited

        moved = written[0].start_anchor.model_copy(
            update={"coordinate_b": Coordinate(1.5, TimeUnit.seconds)}
        )
        for update, column, claim_id in (
            ({"certainty": 0.4}, "certainty", "midi 96"),
            ({"start_anchor": moved}, "start_anchor.coordinate_b.value", "c1"),
        ):
            index = [claim.id for claim in written].index(claim_id)
            with pytest.raises(
                ValueError,
                match=re.escape(
                    f"The new MatchClaim table changes the MatchClaim '{claim_id}' of "
                    f"the existing one in '{column}': only a MatchClaim's provenance "
                    f"may change; a correction is a new MatchClaim under a new id."
                ),
            ):
                MatchClaimField.from_claims(changed(index, update)).to_parquet(tmp_path)
        assert MatchClaimField.from_parquet(tmp_path).to_claims() == written
        assert sorted(path.name for path in tmp_path.iterdir()) == [
            "match_claims.parquet"
        ]


# endregion


# region A null coordinate


class TestNullCoordinateRefused:
    """An anchor holding a null coordinate is refused by every reader.

    A non-finite number becomes a null cell; TimeToShare refuses such a table
    (``invalid_coordinate``), so no reader admits it and no writer writes it.
    """

    START_A = (
        "MatchClaim table row 0 start_anchor.coordinate_a: the coordinate is null."
    )

    @staticmethod
    def _nulled(table: pa.Table, anchor: str, side: str, row: int) -> pa.Table:
        """*table* with one coordinate cell of one anchor set to null."""
        column = table.column(anchor).combine_chunks()
        children = [column.field(i) for i in range(column.type.num_fields)]
        index = column.type.get_field_index(f"coordinate_{side}")
        child = children[index]
        children[index] = pc.if_else(
            pa.array([i == row for i in range(len(child))]),
            pa.nulls(len(child), child.type),
            child,
        )
        nulled = pa.StructArray.from_arrays(
            children, fields=list(column.type), mask=column.is_null()
        )
        return _with_child(table, anchor, nulled)

    @staticmethod
    def _nan_claim(**kwargs) -> MatchClaim:
        return MatchClaim(
            timeline_a_id="B",
            timeline_b_id="C",
            is_synchronous=True,
            event_a_id="b",
            event_b_id="c",
            start_anchor=AlignmentAnchor(
                timeline_a_id="B",
                coordinate_a=Coordinate(float("nan"), TimeUnit.seconds),
                timeline_b_id="C",
                coordinate_b=Coordinate(1.5, TimeUnit.seconds),
            ),
            **kwargs,
        )

    def test_from_claims(self) -> None:
        with pytest.raises(ValueError, match=re.escape(self.START_A)):
            MatchClaimField.from_claims([self._nan_claim(id="x")])

    def test_from_claims_an_end_anchor(self) -> None:
        interval = MatchClaim(
            id="y",
            timeline_a_id="A",
            timeline_b_id="B",
            is_synchronous=True,
            event_a_id="a",
            event_b_id="b",
            start_anchor=AlignmentAnchor(
                timeline_a_id="A",
                coordinate_a=Coordinate(Fraction(1), TimeUnit.quarters),
                timeline_b_id="B",
                coordinate_b=Coordinate(1.0, TimeUnit.seconds),
            ),
            end_anchor=AlignmentAnchor(
                timeline_a_id="A",
                coordinate_a=Coordinate(Fraction(2), TimeUnit.quarters),
                timeline_b_id="B",
                coordinate_b=Coordinate(float("nan"), TimeUnit.seconds),
            ),
        )
        nomatch = MatchClaim(
            id="z",
            timeline_a_id="A",
            timeline_b_id="B",
            is_synchronous=False,
            event_a_id="a",
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "MatchClaim table row 1 end_anchor.coordinate_b: the coordinate "
                "is null."
            ),
        ):
            MatchClaimField.from_claims([nomatch, interval])

    def test_from_columns(self) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "MatchClaim table row 1 start_anchor.coordinate_b: the coordinate "
                "is null."
            ),
        ):
            MatchClaimField.from_columns(
                ["A", "A", "B"],
                ["B", "C", "C"],
                ["a0", "a0", "b1"],
                ["b0", "c0", "c1"],
                [0, 0, 1],
                [10.0, float("nan"), 21.0],
                timelines=GOLD_AXES,
            )

    @pytest.mark.parametrize(
        ("end_a", "end_b", "where"),
        [
            ([Fraction(2, 3), 1], [0.75, None], "end_anchor.coordinate_b"),
            ([Fraction(2, 3), None], [0.75, 2.0], "end_anchor.coordinate_a"),
            ([Fraction(2, 3), 1], [0.75, float("nan")], "end_anchor.coordinate_b"),
            (
                [Fraction(2, 3), float("nan")],
                [0.75, float("nan")],
                "end_anchor.coordinate_a",
            ),
        ],
    )
    def test_from_columns_an_end_on_one_side_or_a_nan_end(
        self, end_a: list, end_b: list, where: str
    ) -> None:
        """Never an instant claim made of an interval claim's broken end."""
        with pytest.raises(ValueError) as raised:
            MatchClaimField.from_columns(
                ["score", "score"],
                ["audio", "audio"],
                ["n1", "n2"],
                ["o1", "o2"],
                [Fraction(1, 3), Fraction(1, 2)],
                [0.5, 1.0],
                timelines=[SCORE, AUDIO],
                end_coordinate_a=end_a,
                end_coordinate_b=end_b,
            )
        assert str(raised.value) == (
            f"MatchClaim table row 1 {where}: the coordinate is null."
        )

    def test_from_dict(self, gold_field: MatchClaimField) -> None:
        data = gold_field.to_dict()
        data["rows"][0]["start_anchor"]["coordinate_a"] = None
        with pytest.raises(ValueError, match=re.escape(self.START_A)):
            MatchClaimField.from_dict(data)

    def test_from_table(self, gold_field: MatchClaimField) -> None:
        table = self._nulled(gold_field.table, "start_anchor", "a", 0)
        with pytest.raises(ValueError, match=re.escape(self.START_A)):
            MatchClaimField.from_table(table)

    def test_from_package_table_and_parquet(
        self, gold_field: MatchClaimField, tmp_path: Path
    ) -> None:
        table = self._nulled(gold_field.package_table, "start_anchor", "a", 0)
        with pytest.raises(ValueError, match=re.escape(self.START_A)):
            MatchClaimField.from_package_table(table)
        pq.write_table(table, tmp_path / "match_claims.parquet")
        with pytest.raises(ValueError, match=re.escape(self.START_A)):
            MatchClaimField.from_parquet(tmp_path)

    def test_a_nomatch_row_is_no_problem(self) -> None:
        nomatch = MatchClaim(
            id="z",
            timeline_a_id="A",
            timeline_b_id="B",
            is_synchronous=False,
            event_a_id="a",
        )
        assert MatchClaimField.from_claims([nomatch]).to_claims() == [nomatch]


# endregion


# region One spelling of a value


class TestNullRowRefused:
    """A null row is no MatchClaim: the constructor and every reader refuse it."""

    CONSTRUCTOR = (
        "MatchClaimField row 1 is null: a MatchClaim table holds no null rows."
    )

    @staticmethod
    def _field() -> MatchClaimField:
        return MatchClaimField.from_columns(
            ["score", "score"],
            ["audio", "audio"],
            ["n1", "n2"],
            ["o1", "o2"],
            [Fraction(1, 3), Fraction(2, 3)],
            [0.5, 0.7],
            timelines={
                "score": ("quarters", "fraction"),
                "audio": ("seconds", "float"),
            },
        )

    @classmethod
    def _rows(cls, mask: list[bool]) -> pa.StructArray:
        """The field's rows as one struct array, with *mask* marking null rows."""
        rows = cls._field().table.to_struct_array().combine_chunks()
        return pa.StructArray.from_arrays(
            [rows.field(i) for i in range(rows.type.num_fields)],
            fields=list(rows.type),
            mask=pa.array(mask),
        )

    @staticmethod
    def _wrap(rows: pa.StructArray | pa.ChunkedArray) -> MatchClaimField:
        return MatchClaimField(StructField(rows, pa.field("matchclaim", rows.type)))

    def test_the_rows_without_a_null_one_are_accepted(self) -> None:
        assert len(self._wrap(self._rows([False, False]))) == 2

    def test_the_constructor_refuses_a_null_row(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            self._wrap(self._rows([False, True]))
        assert str(excinfo.value) == self.CONSTRUCTOR

    def test_the_constructor_refuses_a_null_row_of_a_chunked_array(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            self._wrap(pa.chunked_array([self._rows([False, True])]))
        assert str(excinfo.value) == self.CONSTRUCTOR

    def test_from_table_keeps_its_message(self) -> None:
        table = self._field().table
        nulled = pa.Table.from_arrays(
            [
                pa.array([column[0].as_py(), None], type=column.type)
                for column in table.columns
            ],
            names=table.column_names,
        )
        with pytest.raises(ValueError) as excinfo:
            MatchClaimField.from_table(nulled)
        assert str(excinfo.value) == (
            "MatchClaim table row 1 names no timeline_a_id or no timeline_b_id."
        )

    def test_from_dict_keeps_its_message(self) -> None:
        data = self._field().to_dict()
        data["rows"][1] = None
        with pytest.raises(ValueError) as excinfo:
            MatchClaimField.from_dict(data)
        assert str(excinfo.value) == (
            "In a MatchClaimField dictionary, row 1 must be a mapping holding the "
            "keys ['timeline_a_id', 'timeline_b_id', 'start_anchor', 'end_anchor', "
            "'is_synchronous', 'provenance', 'certainty', 'event_a_id', "
            "'event_a_name', 'event_b_id', 'event_b_name', 'id'], not None."
        )


class TestOneSpelling:
    """An unreduced fraction is held reduced, so fields compare by value."""

    AXES = {"score": ("quarters", "fraction"), "audio": ("seconds", "float")}

    def _field(self, coordinate_a: pa.Array | list) -> MatchClaimField:
        return MatchClaimField.from_columns(
            ["score"],
            ["audio"],
            ["n1"],
            ["o1"],
            coordinate_a,
            [1.0],
            timelines=self.AXES,
        )

    def _reduced(self) -> MatchClaimField:
        return self._field([Fraction(1, 2)])

    def _unreduced(self) -> MatchClaimField:
        return self._field(
            pa.array(
                [{"value": 0.5, "numerator": 2, "denominator": 4}],
                type=RATIONAL_STRUCT_TYPE,
            )
        )

    @staticmethod
    def _held(field: MatchClaimField) -> MatchClaimField:
        """*field* as a bundle holds it, its claim given the id ``claim_1``."""
        bundle = AlignmentBundle()
        bundle.add_timeline(ContinuousLogicalTimeline(length=16, uid="score"))
        bundle.add_timeline(ContinuousPhysicalTimeline(length=60.0, uid="audio"))
        return bundle.add_match_claim_field(field).match_claim_table

    def test_from_columns_holds_the_reduced_ratio(self) -> None:
        unreduced = self._unreduced()
        assert unreduced == self._reduced()
        assert unreduced.table.column("start_anchor").to_pylist()[0][
            "coordinate_a"
        ] == {
            "value": 0.5,
            "numerator": 1,
            "denominator": 2,
            "unit": "quarters",
            "number_type": "fraction",
        }

    def test_from_table_holds_the_reduced_ratio(self) -> None:
        reduced = self._reduced()
        table = reduced.table
        anchors = table.column("start_anchor").to_pylist()
        anchors[0]["coordinate_a"].update(numerator=2, denominator=4)
        index = table.schema.get_field_index("start_anchor")
        rewritten = table.set_column(
            index,
            table.schema.field(index),
            pa.array(anchors, table.schema.field(index).type),
        )
        assert MatchClaimField.from_table(rewritten) == reduced

    def test_both_spellings_succeed_each_other_in_the_file(
        self, tmp_path: Path
    ) -> None:
        unreduced = self._held(self._unreduced())
        reduced = self._held(self._reduced())
        for order, fields in (
            ("unreduced_first", (unreduced, unreduced, reduced)),
            ("reduced_first", (reduced, unreduced)),
        ):
            directory = tmp_path / order
            directory.mkdir()
            for field in fields:
                field.to_parquet(directory)
            read = MatchClaimField.from_parquet(directory)
            assert read == unreduced
            assert read == reduced


# endregion


# region Equality


class TestEquality:
    """Claim equality is over the claim, never over the bundle holding it."""

    def test_copy_in_a_bundle_is_equal(self) -> None:
        claim = _every_kind(NONE5)[0]
        held = claim.model_copy()
        held.set_bundle(AlignmentBundle())
        assert held.bundle is not None and claim.bundle is None
        assert claim == held
        assert hash(claim) == hash(held)

    def test_round_trip_outside_a_bundle(self) -> None:
        claims = MatchClaimField.from_claims(_every_kind(NONE5)).to_claims()
        assert [claim.bundle for claim in claims] == [None] * 5
        assert MatchClaimField.from_claims(claims).to_claims() == claims

    def test_tables_of_equal_rows_are_equal(self, gold_field: MatchClaimField) -> None:
        provenance = [None, HUMAN, SOFTWARE, None, HUMAN]
        field = MatchClaimField.from_claims(_every_kind(provenance))
        assert field == MatchClaimField.from_dict(field.to_dict())
        assert gold_field == MatchClaimField.from_dict(gold_field.to_dict())
        assert field != gold_field
        assert MatchClaimField.from_claims([]) == MatchClaimField.from_claims([])

    def test_a_shared_provenance_counts_as_every_row_s(
        self, gold_field: MatchClaimField
    ) -> None:
        per_row = MatchClaimField._from_struct_array(gold_field._rows_with_provenance())
        assert (per_row.provenance, gold_field.provenance) == (None, SHARED)
        assert per_row.table.column("provenance").null_count == 0
        assert per_row == gold_field
        claims = gold_field.to_claims()
        one_without = MatchClaimField.from_claims(
            [claims[0].model_copy(update={"provenance": None}), *claims[1:]]
        )
        assert one_without != gold_field

    def test_a_belief_added_is_unequal(self) -> None:
        claims = _every_kind([None, HUMAN, SOFTWARE, None, HUMAN])
        believed = claims[1].model_copy(
            update={"provenance": HUMAN.with_belief(_belief("jo", "false"))}
        )
        assert MatchClaimField.from_claims(claims) != MatchClaimField.from_claims(
            [claims[0], believed, *claims[2:]]
        )


# endregion


# region Edge cases, repr & scale


class TestEdgeCases:
    """Empty fields, repr, exports."""

    def test_empty_field(self) -> None:
        field = MatchClaimField.from_columns([], [], [], [], [], [], timelines={})
        assert len(field) == 0
        assert field.timeline_ids == set()
        assert field.to_claims() == []

    def test_repr(self, gold_field: MatchClaimField) -> None:
        assert repr(gold_field) == "MatchClaimField(claims=3, timelines=3)"

    def test_repr_html_contains_summary(self, gold_field: MatchClaimField) -> None:
        assert gold_field._repr_html_() == (
            "<div style='font-family: monospace;'><strong>MatchClaimField</strong> "
            "<span style='color: #555;'>claims=3, timelines=3</span><table "
            "style='border-collapse: collapse; margin-top: 4px;'><thead><tr><th>#</th>"
            "<th>timeline A</th><th>coord A</th><th>timeline B</th><th>coord B</th>"
            "</tr></thead><tbody><tr><td>0</td><td>A</td><td>0 quarters</td><td>B</td>"
            "<td>10 seconds</td></tr><tr><td>1</td><td>A</td><td>0 quarters</td><td>C"
            "</td><td>20 seconds</td></tr><tr><td>2</td><td>B</td><td>1 seconds</td>"
            "<td>C</td><td>21 seconds</td></tr></tbody></table></div>"
        )

    def test_top_level_export(self) -> None:
        import timetoalign
        import timetoalign.alignment as alignment

        assert (
            timetoalign.MatchClaimField is alignment.MatchClaimField is MatchClaimField
        )
        assert timetoalign.Provenance is alignment.Provenance is Provenance
        assert timetoalign.Belief is alignment.Belief is Belief
        for gone in ("Agent", "MatchMetadata"):
            assert not hasattr(alignment, gone)
            assert gone not in timetoalign.__all__


class _CountingValidator:
    """A model's validator that records every validation it runs."""

    def __init__(self, inner: object, name: str, built: list[str]) -> None:
        self._inner = inner
        self._name = name
        self._built = built

    def validate_python(self, *args: object, **kwargs: object) -> object:
        self._built.append(self._name)
        return self._inner.validate_python(*args, **kwargs)

    def validate_json(self, *args: object, **kwargs: object) -> object:
        self._built.append(self._name)
        return self._inner.validate_json(*args, **kwargs)

    def validate_strings(self, *args: object, **kwargs: object) -> object:
        self._built.append(self._name)
        return self._inner.validate_strings(*args, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


def _count_constructions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Count every MatchClaim and AlignmentAnchor built from then on.

    Every validating construction (``__init__``, ``model_validate``,
    ``model_validate_json``, ``model_validate_strings``) runs the class's
    ``__pydantic_validator__``; ``model_construct`` validates nothing and is
    counted on its own.
    """
    built: list[str] = []
    for model in (MatchClaim, AlignmentAnchor):
        monkeypatch.setattr(
            model,
            "__pydantic_validator__",
            _CountingValidator(model.__pydantic_validator__, model.__name__, built),
        )

        def counting_construct(
            *args: object,
            _construct=model.model_construct,
            _name=model.__name__,
            **kwargs: object,
        ) -> object:
            built.append(_name)
            return _construct(*args, **kwargs)

        monkeypatch.setattr(model, "model_construct", staticmethod(counting_construct))
    return built


class TestScale:
    """The vectorized path scales without building Python objects."""

    def test_the_count_sees_every_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built = _count_constructions(monkeypatch)
        values = {
            "timeline_a_id": "A",
            "timeline_b_id": "B",
            "coordinate_a": Coordinate(1, TimeUnit.quarters),
            "coordinate_b": Coordinate(1.5, TimeUnit.seconds),
        }
        anchor = AlignmentAnchor(**values)
        AlignmentAnchor.model_validate(values)
        AlignmentAnchor.model_construct(**values)
        AlignmentAnchor.model_validate_json(anchor.model_dump_json())
        AlignmentAnchor.model_validate_strings(
            {"timeline_a_id": "A", "timeline_b_id": "B"}
            | {
                key: anchor.model_dump(mode="json")[key]
                for key in ("coordinate_a", "coordinate_b")
            }
        )
        assert built == ["AlignmentAnchor"] * 5

        built.clear()
        claim = MatchClaimField.from_columns(
            timeline_a_ids=["A"],
            timeline_b_ids=["B"],
            event_a_ids=["a0"],
            event_b_ids=["b0"],
            coordinate_a=[1],
            coordinate_b=[1.5],
            timelines=GOLD_AXES,
        )[0]
        built.clear()
        MatchClaim.from_dict(claim.to_dict())
        assert built == ["AlignmentAnchor", "MatchClaim"]

    def test_scale_builds_columnar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        n = 100_000
        built = _count_constructions(monkeypatch)

        field = MatchClaimField.from_columns(
            timeline_a_ids=["A"] * n,
            timeline_b_ids=["B"] * n,
            event_a_ids=[f"a{i}" for i in range(n)],
            event_b_ids=[f"b{i}" for i in range(n)],
            coordinate_a=list(range(n)),
            coordinate_b=[float(i) + 0.5 for i in range(n)],
            timelines=GOLD_AXES,
        )

        assert len(field) == n
        # The vectorized path builds no MatchClaim and no AlignmentAnchor.
        assert built == []
        claim = field[50_000]
        # Reading one row back builds exactly its AlignmentAnchor and its MatchClaim.
        assert built == ["AlignmentAnchor", "MatchClaim"]
        assert claim.start_anchor.coordinate_a == Coordinate(50_000, TimeUnit.quarters)
        assert claim.start_anchor.coordinate_b == Coordinate(50_000.5, TimeUnit.seconds)
        assert field.timeline_ids == {"A", "B"}


# endregion
