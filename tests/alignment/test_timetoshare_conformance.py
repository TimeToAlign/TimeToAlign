"""The MatchClaim table is TimeToShare's table type.

Validation strategy and exact values: ``tests/alignment/README.md`` (the
section "The MatchClaim table is TimeToShare's").
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from timetoshare.tables import (
    COORDINATE_TYPE,
    MATCH_CLAIM_SCHEMA,
    PROVENANCE_METADATA_KEY,
    fixture_path,
    match_claim_problems,
    row_provenance,
)

from timetoalign.alignment import (
    AlignmentAnchor,
    AlignmentBundle,
    MatchClaim,
    MatchClaimField,
    Provenance,
)
from timetoalign.core import CHUNK, NOMATCH, Coordinate, TimeUnit
from timetoalign.core.time import COORDINATE_CELL_TYPE, PACKAGE_COORDINATE_CELL_TYPE
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscretePhysicalTimeline,
)

#: TimeToShare's fixture MatchClaim table, shipped with the installed package.
_FIXTURE = fixture_path("match_claims.parquet")

SCORE = ContinuousLogicalTimeline(length=16, uid="score")
AUDIO = ContinuousPhysicalTimeline(length=60.0, uid="audio")
PCM = DiscretePhysicalTimeline(length=44_100 * 60, uid="pcm")


def _bundle() -> AlignmentBundle:
    """A bundle of fresh twins of the three timelines: the scope that mints ids."""
    bundle = AlignmentBundle(id="conformance")
    bundle.add_timeline(ContinuousLogicalTimeline(length=16, uid="score"))
    bundle.add_timeline(ContinuousPhysicalTimeline(length=60.0, uid="audio"))
    bundle.add_timeline(DiscretePhysicalTimeline(length=44_100 * 60, uid="pcm"))
    return bundle


class TestSchema:
    def test_the_table_type_is_timetoshare_s(self) -> None:
        written = MatchClaimField.from_claims([]).package_table.schema
        assert written.remove_metadata().equals(MATCH_CLAIM_SCHEMA)
        assert PACKAGE_COORDINATE_CELL_TYPE == COORDINATE_TYPE
        in_memory = pa.schema(list(MatchClaimField.pa_schema))
        assert in_memory.names == MATCH_CLAIM_SCHEMA.names
        anchor = in_memory.field("start_anchor").type
        assert [anchor.field(i).type for i in range(2)] == [COORDINATE_CELL_TYPE] * 2
        assert MatchClaimField._PROVENANCE_KEY == PROVENANCE_METADATA_KEY.encode()


class TestFixtureTable:
    def test_read_and_written_back_unchanged(self, tmp_path: Path) -> None:
        fixture = pq.read_table(_FIXTURE)
        field = MatchClaimField.from_parquet(_FIXTURE)
        assert [claim.id for claim in field] == [f"claim_{n}" for n in range(1, 17)]
        kinds = [claim.claim_type.name for claim in field]
        assert kinds == ["event_match"] * 11 + ["nomatch", "event_match"] + [
            "conceptual",
            "event_match",
            "event_match",
        ]
        nomatch = field[11]
        assert (nomatch.event_a_id, nomatch.event_b_id, nomatch.anchors) == (
            "note_10",
            NOMATCH,
            [],
        )
        # The sentinels are strings in the event ID columns: CHUNK on both
        # sides of claim_11, on side A of claim_16.
        assert [
            (claim.id, claim.event_a_id, claim.event_b_id, claim.chunk_sides)
            for claim in field
            if claim.chunk_sides
        ] == [
            ("claim_11", CHUNK, CHUNK, ("dpt_1", "cpt_1")),
            ("claim_16", CHUNK, "note_2", ("cpt_1",)),
        ]
        # An anchor cell holds two coordinates; reading rebuilds the anchor on
        # the row's timelines.
        assert [anchor.timelines for claim in field for anchor in claim.anchors] == [
            (claim.timeline_a_id, claim.timeline_b_id)
            for claim in field
            for _ in claim.anchors
        ]
        assert field[1].start_anchor == AlignmentAnchor(
            timeline_a_id="clt_1",
            coordinate_a=Coordinate(1, TimeUnit.quarters),
            timeline_b_id="dpt_1",
            coordinate_b=Coordinate(44100, TimeUnit.samples),
        )
        assert field[1].provenance == Provenance(
            date="2026-11-20", method="imported", service="https://example.org/aligner"
        )

        written = pq.read_table(field.to_parquet(tmp_path))
        assert written.equals(fixture, check_metadata=True)
        assert written.schema.metadata == fixture.schema.metadata
        # In memory a list's item is named "item", in Parquet "element": the
        # values and types are compared, and the schema metadata apart.
        stored_again = MatchClaimField.from_claims(field.to_claims()).package_table
        assert stored_again.equals(fixture)
        assert stored_again.schema.metadata == fixture.schema.metadata

    def test_columns_read_by_name(self) -> None:
        fixture = pq.read_table(_FIXTURE)
        reversed_table = fixture.select(list(reversed(fixture.column_names)))
        assert (
            MatchClaimField.from_package_table(reversed_table).to_claims()
            == MatchClaimField.from_package_table(fixture).to_claims()
        )


class TestWhatTheLibraryWrites:
    def test_from_columns_passes_timetoshare_s_check(self, tmp_path: Path) -> None:
        shared = Provenance(service="https://example.org/aligner", method="inferred")
        field = MatchClaimField.from_columns(
            timeline_a_ids=["score", "score", "pcm"],
            timeline_b_ids=["audio", "pcm", "audio"],
            event_a_ids=["n0", "n1", "s2"],
            event_b_ids=["o0", "s1", "o2"],
            coordinate_a=[Fraction(1, 3), 4, 44100],
            coordinate_b=[0.1, 88200, 1.0],
            timelines=[SCORE, AUDIO, PCM],
            provenance=shared,
        )
        minted = field.table.column("id").to_pylist()
        assert None not in minted
        assert len(set(minted)) == 3
        bundle = _bundle()
        bundle.add_match_claim_field(field)
        claims = bundle.match_claim_table
        assert claims.table.column("id").to_pylist() == minted
        table = pq.read_table(claims.to_parquet(tmp_path))
        assert match_claim_problems(table) == []
        members = ("numerator", "denominator", "value")
        cells = [
            tuple(
                tuple(anchor[side][member] for member in members)
                for side in ("coordinate_a", "coordinate_b")
            )
            for anchor in table.column("start_anchor").to_pylist()
        ]
        assert cells == [
            ((1, 3, None), (None, None, 0.1)),
            ((4, 1, None), (88200, None, None)),
            ((44100, None, None), (None, None, 1.0)),
        ]
        expected = {
            "stated_by": None,
            "date": None,
            "method": "inferred",
            "service": "https://example.org/aligner",
            "basis": None,
            "match": None,
            "note": None,
            "beliefs": None,
        }
        assert row_provenance(table) == [expected] * 3

    def test_every_stated_kind_passes_timetoshare_s_check(self) -> None:
        claims = [
            MatchClaim.from_events(
                {"id": "note_7", "start": Fraction(7, 3), "end": Fraction(10, 3)},
                SCORE,
                {"id": "onset_7", "start": 1.25, "end": 2.5},
                AUDIO,
                end_coord_key="end",
                certainty=0.5,
                id="c1",
            ),
            MatchClaim.nomatch(
                {"id": "note_9", "start": Fraction(27, 2)}, SCORE, AUDIO, id="c2"
            ),
            MatchClaim.from_events(
                {"id": "n10", "start": 4},
                SCORE,
                {"id": "o10", "start": 2.0},
                AUDIO,
                is_synchronous=False,
            ),
        ]
        field = MatchClaimField.from_claims(claims)
        ids = ["c1", "c2", claims[2].id]
        assert claims[2].id not in (None, "c1", "c2")
        assert field.table.column("id").to_pylist() == ids
        bundle = _bundle()
        bundle.add_match_claims(claims)
        table = bundle.match_claim_table.package_table
        assert match_claim_problems(table) == []
        assert table.column("id").to_pylist() == ids
