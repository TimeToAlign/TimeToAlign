"""Beliefs in held MatchClaims: the only change a MatchClaim admits.

Validation strategy and exact values: ``tests/alignment/README.md`` (the
section "Beliefs in Held MatchClaims").
"""

from __future__ import annotations

import datetime
import re
from fractions import Fraction
from pathlib import Path

import pytest

from timetoalign.alignment import AlignmentBundle, ClaimFilter, MatchClaimField
from timetoalign.alignment.filters import belief_context, believed
from timetoalign.core import Belief, IdCoordinate, Provenance, TimeUnit
from timetoalign.timelines import statements
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    DiscretePhysicalTimeline,
)

#: One actor, written absolutely and compactly (``wd`` is a catalogue prefix).
Q42 = "http://www.wikidata.org/entity/Q42"
WD_Q42 = "wd:Q42"

SHARED = Provenance(method="imported", service="https://example.org/aligner")
REFUSED_WORD = (
    "The belief filter keeps the states ['true', 'undecided', 'unlikely', "
    "'false'], not ['maybe']"
)


def _timelines(
    prefix: str = "",
) -> tuple[ContinuousLogicalTimeline, DiscretePhysicalTimeline]:
    score = ContinuousLogicalTimeline(length=Fraction(16), uid=f"{prefix}score")
    score.add_events(
        [
            {"id": f"n{number}", "start": Fraction(start), "event_type": "Note"}
            for number, start in ((1, 0), (2, 2), (3, 4))
        ]
    )
    pcm = DiscretePhysicalTimeline(length=705600, uid=f"{prefix}pcm")
    pcm.add_events(
        [
            {"id": f"o{number}", "start": start, "event_type": "Note"}
            for number, start in ((1, 0), (2, 100000), (3, 176400))
        ]
    )
    return score, pcm


def _bundle() -> AlignmentBundle:
    """``claim_1``–``claim_3``: n1↔o1, n2↔o2, n3↔o3, one shared provenance."""
    bundle = AlignmentBundle(id="beliefs")
    for timeline in _timelines():
        bundle.add_timeline(timeline)
    bundle.create_match_claims(
        [(f"n{i}", "score", f"o{i}", "pcm") for i in (1, 2, 3)], provenance=SHARED
    )
    return bundle


def _ids(bundle: AlignmentBundle, **filters: object) -> list[str | None]:
    return [claim.id for claim in bundle.get_match_claims(**filters)]  # type: ignore[arg-type]


QUARTER_2 = IdCoordinate(2, TimeUnit.quarters, "score")


class TestAddBelief:
    def test_only_the_provenance_of_the_claim_changes(self) -> None:
        bundle = _bundle()
        before = bundle.match_claim_table.table.drop_columns(["provenance"])

        returned = bundle.add_match_claim_belief(
            "claim_2", actor=Q42, belief="false", date="2026-11-21"
        )

        assert returned is bundle
        assert [claim.provenance for claim in bundle.get_match_claims()] == [
            SHARED,
            SHARED.model_copy(
                update={
                    "beliefs": (Belief(actor=Q42, belief="false", date="2026-11-21"),)
                }
            ),
            SHARED,
        ]
        assert bundle.match_claim_table.table.drop_columns(["provenance"]).equals(
            before
        )
        assert bundle.match_claim_table.provenance is None

    def test_one_belief_per_actor_compared_as_iris(self) -> None:
        bundle = _bundle()
        bundle.add_match_claim_belief(
            "claim_2", actor=Q42, belief="false", date="2026-11-21"
        )

        bundle.add_match_claim_belief(
            "claim_2", actor=WD_Q42, belief="true", date="2026-11-22"
        )

        assert [claim.provenance.beliefs for claim in bundle.get_match_claims()] == [
            (),
            (Belief(actor=WD_Q42, belief="true", date="2026-11-22"),),
            (),
        ]

    def test_the_belief_filter_reads_it_at_once(self) -> None:
        bundle = _bundle()
        assert _ids(bundle, believer=Q42) == ["claim_1", "claim_2", "claim_3"]

        bundle.add_match_claim_belief(
            "claim_2", actor=Q42, belief="false", date="2026-11-21"
        )
        assert _ids(bundle, believer=Q42) == ["claim_1", "claim_3"]
        assert _ids(bundle, believer=WD_Q42) == ["claim_1", "claim_3"]
        assert _ids(bundle, believer=Q42, beliefs=["true"]) == []
        assert _ids(bundle, believer=Q42, beliefs=["false"]) == ["claim_2"]
        assert _ids(bundle, believer=Q42, beliefs=["undecided"]) == [
            "claim_1",
            "claim_3",
        ]

        bundle.add_match_claim_belief(
            "claim_2", actor=WD_Q42, belief="true", date="2026-11-22"
        )
        assert _ids(bundle, believer=Q42, beliefs=["true"]) == ["claim_2"]

    def test_refusals_change_nothing(self) -> None:
        bundle = _bundle()
        before = bundle.match_claim_table

        with pytest.raises(
            KeyError,
            match=re.escape(
                "The scope 0 ('beliefs') of bundle 'beliefs' holds no MatchClaim "
                "'claim_9' (3 MatchClaims)."
            ),
        ):
            bundle.add_match_claim_belief("claim_9", actor=Q42, belief="true")
        with pytest.raises(
            KeyError,
            match=re.escape("No scope has the key 3. Available scopes: {0: 'beliefs'}"),
        ):
            bundle.add_match_claim_belief(
                "claim_1", actor=Q42, belief="true", scope_key=3
            )
        with pytest.raises(ValueError):
            bundle.add_match_claim_belief("claim_1", actor=Q42, belief="maybe")  # type: ignore[arg-type]
        assert bundle.match_claim_table == before
        assert bundle.match_claim_table.provenance == SHARED

    def test_the_date_defaults_to_today(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(statements, "_today", lambda: datetime.date(2026, 11, 21))
        bundle = _bundle()

        bundle.add_match_claim_belief(
            "claim_1", actor=Q42, belief="unlikely", note="by ear"
        )

        assert bundle.get_match_claims()[0].provenance.beliefs == (
            Belief(actor=Q42, belief="unlikely", date="2026-11-21", note="by ear"),
        )

    def test_a_statement_and_a_claim_each_take_their_belief(self) -> None:
        bundle = _bundle()
        bundle.add_statement("dcterms:title", "Beliefs")

        bundle.add_belief(
            "dcterms:title", "Beliefs", actor=Q42, belief="unlikely", date="2026-11-21"
        )
        bundle.add_match_claim_belief(
            "claim_2", actor=Q42, belief="false", date="2026-11-21"
        )

        assert bundle.statements == {"dcterms:title": ["Beliefs"]}
        assert bundle.provenance == [
            {
                "beliefs": [{"actor": Q42, "belief": "unlikely", "date": "2026-11-21"}],
                "property": "dcterms:title",
                "value": "Beliefs",
            }
        ]
        assert [claim.provenance for claim in bundle.get_match_claims()] == [
            SHARED,
            SHARED.model_copy(
                update={
                    "beliefs": (Belief(actor=Q42, belief="false", date="2026-11-21"),)
                }
            ),
            SHARED,
        ]

    def test_the_actor_context_is_every_prefix_the_manifest_declares(self) -> None:
        data = _bundle().to_dict()
        data["@context"] = {**data["@context"], "orcid": "https://orcid.org/"}
        bundle = AlignmentBundle.from_dict(data)

        bundle.create_match_claims(
            [({"start": 0}, "score", {"start": 0}, "pcm")],
            actor="orcid:0000-0001",
            date="2026-11-21",
        )

        assert bundle.prefixes == {"orcid": "https://orcid.org/"}
        assert belief_context(bundle)["orcid"] == "https://orcid.org/"  # type: ignore[index]
        assert _ids(
            bundle, believer="https://orcid.org/0000-0001", beliefs=["true"]
        ) == ["claim_1"]

    def test_a_claim_is_addressed_in_its_scope(self) -> None:
        sources = []
        for prefix in ("a_", "b_"):
            source = AlignmentBundle(id=f"source_{prefix[0]}")
            for timeline in _timelines(prefix):
                source.add_timeline(timeline)
            source.create_match_claims([("n1", f"{prefix}score", "o1", f"{prefix}pcm")])
            sources.append(source)
        merged = AlignmentBundle.from_bundles(sources, id="merged")

        merged.add_match_claim_belief(
            "claim_1", actor=Q42, belief="false", date="2026-11-21", scope_key=2
        )

        assert [claim.provenance for claim in merged.get_match_claims()] == [
            None,
            Provenance(beliefs=(Belief(actor=Q42, belief="false", date="2026-11-21"),)),
        ]
        assert [
            claim.timeline_a_id for claim in merged.get_match_claims(believer=Q42)
        ] == ["a_score"]

    def test_the_package_written_again_succeeds_itself(self, tmp_path: Path) -> None:
        bundle = _bundle()
        bundle.to_file(tmp_path / "package.yaml")

        bundle.add_match_claim_belief(
            "claim_2", actor=Q42, belief="false", date="2026-11-21"
        )
        bundle.to_file(tmp_path / "package.yaml")

        read = AlignmentBundle.from_file(tmp_path / "package.yaml")
        assert read.match_claim_table == bundle.match_claim_table
        assert read.get_match_claims()[1].provenance.beliefs == (
            Belief(actor=Q42, belief="false", date="2026-11-21"),
        )


class TestCoordinateGettersTakeTheBeliefFilter:
    def test_a_claim_believed_false_no_longer_moves_a_coordinate(self) -> None:
        bundle = _bundle()
        assert bundle.get_coordinate_at(QUARTER_2, "pcm", format="int") == 100000
        assert (
            bundle.get_coordinate_at(QUARTER_2, "pcm", format="int", believer=Q42)
            == 100000
        )

        bundle.add_match_claim_belief(
            "claim_2", actor=Q42, belief="false", date="2026-11-21"
        )

        for believer in (Q42, WD_Q42):
            assert (
                bundle.get_coordinate_at(
                    QUARTER_2, "pcm", format="int", believer=believer
                )
                == 88200
            )
        assert bundle.get_coordinate_at(QUARTER_2, "pcm", format="int") == 100000
        assert (
            bundle.get_coordinate_for("n2", "pcm", format="int", believer=Q42) == 88200
        )
        assert bundle.get_coordinates_at(
            [QUARTER_2, IdCoordinate(3, TimeUnit.quarters, "score")],
            "pcm",
            believer=Q42,
        ) == [
            IdCoordinate(88200, TimeUnit.samples, "pcm"),
            IdCoordinate(132300, TimeUnit.samples, "pcm"),
        ]
        assert bundle.get_coordinate(QUARTER_2, "pcm", believer=Q42) == IdCoordinate(
            88200, TimeUnit.samples, "pcm"
        )
        assert bundle.transfer(2, "score", "pcm", believer=Q42) == 88200.0
        assert bundle.transfer(2, "score", "pcm") == 100000.0
        assert bundle.transfer_interval(1, 3, "score", "pcm", believer=Q42) == (
            44100.0,
            132300.0,
        )
        assert bundle.transfer_interval(1, 3, "score", "pcm") == (50000.0, 138200.0)

    def test_an_unknown_state_is_refused(self) -> None:
        bundle = _bundle()
        refused = re.escape(REFUSED_WORD)
        maybe = ["maybe"]
        calls = [
            lambda: bundle.get_coordinate_at(QUARTER_2, "pcm", beliefs=maybe),
            lambda: bundle.get_coordinates_at([QUARTER_2], "pcm", beliefs=maybe),
            lambda: bundle.get_coordinate_for("n2", "pcm", beliefs=maybe),
            lambda: bundle.get_coordinates_for(["n2"], "pcm", beliefs=maybe),
            lambda: bundle.get_coordinate(QUARTER_2, "pcm", beliefs=maybe),
            lambda: bundle.transfer(2, "score", "pcm", beliefs=maybe),
            lambda: bundle.transfer_interval(1, 3, "score", "pcm", beliefs=maybe),
            lambda: bundle.get_match_claims(beliefs=maybe),
            lambda: bundle.get_matchstamp_at(QUARTER_2, beliefs=maybe),
            lambda: ClaimFilter(beliefs=maybe),  # type: ignore[arg-type]
            lambda: believed([], None, maybe),  # type: ignore[arg-type]
        ]
        for call in calls:
            with pytest.raises(ValueError, match=refused):
                call()


class TestCreateMatchClaimsWithAnActor:
    def test_the_person_stating_believes_true(self) -> None:
        bundle = _bundle()

        (claim,) = bundle.create_match_claims(
            [("n1", "score", "o2", "pcm")], actor=WD_Q42, date="2026-11-21"
        )

        assert claim.provenance == Provenance(
            beliefs=(Belief(actor=WD_Q42, belief="true", date="2026-11-21"),)
        )

    def test_the_actor_s_belief_replaces_theirs_in_the_provenance(self) -> None:
        bundle = _bundle()
        given = SHARED.model_copy(
            update={"beliefs": (Belief(actor=Q42, belief="false", date="2026-11-20"),)}
        )

        (claim,) = bundle.create_match_claims(
            [("n1", "score", "o2", "pcm")],
            provenance=given,
            actor=WD_Q42,
            date="2026-11-21",
        )

        assert claim.provenance == SHARED.model_copy(
            update={
                "beliefs": (Belief(actor=WD_Q42, belief="true", date="2026-11-21"),)
            }
        )

    def test_without_an_actor_the_provenance_stands(self) -> None:
        (claim,) = _bundle().create_match_claims(
            [("n1", "score", "o2", "pcm")], provenance=SHARED
        )

        assert claim.provenance == SHARED

    def test_a_date_without_an_actor_is_refused(self) -> None:
        bundle = _bundle()

        with pytest.raises(
            ValueError,
            match=re.escape(
                "a date dates the belief of the actor stating the MatchClaims; "
                "give actor="
            ),
        ):
            bundle.create_match_claims(
                [("n1", "score", "o2", "pcm")], date="2026-11-21"
            )
        assert len(bundle.match_claim_table) == 3


class TestCopyOnWrite:
    def test_the_field_s_with_belief(self) -> None:
        bundle = _bundle()
        table = bundle.match_claim_table
        held = Belief(actor=Q42, belief="false", date="2026-11-21")

        believed_table = table.with_belief("claim_2", held)

        assert [claim.provenance.beliefs for claim in believed_table] == [
            (),
            (held,),
            (),
        ]
        assert table == bundle.match_claim_table
        assert table.provenance == SHARED
        with pytest.raises(
            KeyError,
            match=re.escape(
                "The MatchClaim table holds no MatchClaim 'claim_9' (3 MatchClaims)."
            ),
        ):
            table.with_belief("claim_9", held)

    def test_a_one_row_field_keeps_its_provenance_shared(self) -> None:
        claim = _bundle().get_match_claims()[0]
        field = MatchClaimField.from_claims([claim])
        held = Belief(actor=Q42, belief="true", date="2026-11-21")

        believed_field = field.with_belief("claim_1", held)

        assert believed_field.provenance == SHARED.model_copy(
            update={"beliefs": (held,)}
        )

    def test_the_claim_s_with_belief(self) -> None:
        bundle = _bundle()
        claim = bundle.get_match_claims()[0]
        held = Belief(actor=Q42, belief="unlikely", date="2026-11-21")

        copy = claim.with_belief(held)

        assert copy.provenance == SHARED.model_copy(update={"beliefs": (held,)})
        assert copy.bundle is None
        assert claim.provenance == SHARED
        assert claim.bundle is bundle
        bare = claim.model_copy(update={"provenance": None})
        assert bare.with_belief(held).provenance == Provenance(beliefs=(held,))
