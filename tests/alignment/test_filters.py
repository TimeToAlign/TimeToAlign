"""Tests for the Unified Filter API (ClaimFilter).

Tests for ``timetoalign.alignment.filters.ClaimFilter``.

Covers:
- Exact single-ID matching
- Set-of-IDs matching
- Regex (id_pattern) matching
- Between (pair) matching
- Within (both timelines in a set) matching
- synchronous_only / nomatch_only filters
- Mutual exclusion of synchronous_only and nomatch_only
- Combined filters
- The belief filter (believer / belief)
- Timeline-level filtering (matches_timeline)
- The list form (select)
- Domain / unit filtering
- ClaimFilter.from_kwargs convenience constructor
- __repr__
"""

from __future__ import annotations

import re

import pytest
from timetoshare import DEFAULT_BELIEFS

from timetoalign.alignment.claims import MatchClaim, MatchClaimField
from timetoalign.alignment.filters import ClaimFilter
from timetoalign.core import NOMATCH, Belief, Coordinate, Provenance, TimeUnit

from .helpers import event_id, match_claim

# region Fixtures


def _claim(
    timeline_a: str, value_a: float, timeline_b: str, value_b: float, **kwargs
) -> MatchClaim:
    """A synchronous instant claim between two fixture events.

    A filter reads a claim's timelines, kind and provenance, never its
    timelines' axes, so dimensionless positions serve.
    """
    return match_claim(
        timeline_a,
        Coordinate(value_a, TimeUnit.number),
        timeline_b,
        Coordinate(value_b, TimeUnit.number),
        **kwargs,
    )


@pytest.fixture
def claim_score_perf1() -> MatchClaim:
    """Synchronous instant claim: score:clt_1 <-> perf:dlt_1."""
    return _claim("score:clt_1", 0.0, "perf:dlt_1", 0.0)


@pytest.fixture
def claim_score_perf5() -> MatchClaim:
    """Synchronous instant claim: score:clt_1 <-> perf:dlt_5."""
    return _claim("score:clt_1", 10.0, "perf:dlt_5", 1280.0)


@pytest.fixture
def claim_nomatch() -> MatchClaim:
    """NOMATCH claim: an event of score:clt_1 has no counterpart on perf:dlt_5."""
    return MatchClaim(
        timeline_a_id="score:clt_1",
        timeline_b_id="perf:dlt_5",
        is_synchronous=False,
        event_a_id=event_id("score:clt_1", Coordinate(20.0, TimeUnit.number)),
        event_b_id=NOMATCH,
    )


@pytest.fixture
def claim_conceptual() -> MatchClaim:
    """Conceptual claim: two events of score:clt_1 and perf:dlt_5, not synchronous."""
    return MatchClaim(
        timeline_a_id="score:clt_1",
        timeline_b_id="perf:dlt_5",
        is_synchronous=False,
        event_a_id="score:clt_1@section",
        event_b_id="perf:dlt_5@section",
    )


@pytest.fixture
def claim_audio_video() -> MatchClaim:
    """Synchronous instant claim: audio:cpt_1 <-> video:dgt_1."""
    return _claim("audio:cpt_1", 45.0, "video:dgt_1", 1350.0)


@pytest.fixture
def all_claims(
    claim_score_perf1, claim_score_perf5, claim_nomatch, claim_audio_video
) -> list[MatchClaim]:
    """All four test claims."""
    return [claim_score_perf1, claim_score_perf5, claim_nomatch, claim_audio_video]


# endregion


# region Test ClaimFilter basic creation


class TestClaimFilterCreation:
    """Tests for ClaimFilter instantiation and validation."""

    def test_empty_filter(self):
        """Empty filter matches everything."""
        f = ClaimFilter()
        assert f.timeline_id is None
        assert f.timeline_ids is None
        assert f.id_pattern is None
        assert f.between is None
        assert f.synchronous_only is False
        assert f.nomatch_only is False

    def test_mutual_exclusion(self):
        """synchronous_only and nomatch_only cannot both be True."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            ClaimFilter(synchronous_only=True, nomatch_only=True)

    def test_from_kwargs(self):
        """from_kwargs creates equivalent filter."""
        f = ClaimFilter.from_kwargs(
            timeline_id="score:clt_1",
            synchronous_only=True,
        )
        assert f.timeline_id == "score:clt_1"
        assert f.synchronous_only is True

    def test_repr_empty(self):
        """Empty filter has clean repr."""
        f = ClaimFilter()
        assert repr(f) == "ClaimFilter()"

    def test_repr_with_fields(self):
        """Repr shows only non-default fields."""
        f = ClaimFilter(timeline_id="score:clt_1", synchronous_only=True)
        r = repr(f)
        assert "timeline_id='score:clt_1'" in r
        assert "synchronous_only=True" in r
        assert "nomatch_only" not in r


# endregion


# region Test exact ID matching


class TestClaimFilterExactId:
    """Tests for timeline_id (single exact ID) filter."""

    def test_matches_claim_a_side(self, claim_score_perf1):
        """Matches when timeline_id is on side A."""
        f = ClaimFilter(timeline_id="score:clt_1")
        assert f.matches_claim(claim_score_perf1) is True

    def test_matches_claim_b_side(self, claim_score_perf1):
        """Matches when timeline_id is on side B."""
        f = ClaimFilter(timeline_id="perf:dlt_1")
        assert f.matches_claim(claim_score_perf1) is True

    def test_no_match(self, claim_score_perf1):
        """Does not match unrelated timeline."""
        f = ClaimFilter(timeline_id="perf:dlt_99")
        assert f.matches_claim(claim_score_perf1) is False

    def test_matches_timeline(self):
        """matches_timeline for exact ID."""
        f = ClaimFilter(timeline_id="score:clt_1")
        assert f.matches_timeline("score:clt_1") is True
        assert f.matches_timeline("perf:dlt_1") is False


# endregion


# region Test ID set matching


class TestClaimFilterIdSet:
    """Tests for timeline_ids (set of IDs) filter."""

    def test_matches_any(self, claim_score_perf1, claim_audio_video):
        """Matches if any ID in the set is involved."""
        f = ClaimFilter(timeline_ids={"perf:dlt_1", "audio:cpt_1"})
        assert f.matches_claim(claim_score_perf1) is True
        assert f.matches_claim(claim_audio_video) is True

    def test_no_match(self, claim_score_perf5):
        """Does not match when neither timeline is in the set."""
        f = ClaimFilter(timeline_ids={"perf:dlt_1", "audio:cpt_1"})
        assert f.matches_claim(claim_score_perf5) is False

    def test_matches_timeline(self):
        """matches_timeline for set of IDs."""
        f = ClaimFilter(timeline_ids={"score:clt_1", "perf:dlt_1"})
        assert f.matches_timeline("score:clt_1") is True
        assert f.matches_timeline("perf:dlt_1") is True
        assert f.matches_timeline("perf:dlt_5") is False


# endregion


# region Test regex matching


class TestClaimFilterRegex:
    """Tests for id_pattern (regex) filter."""

    def test_prefix_pattern(self, claim_score_perf1, claim_audio_video):
        """Regex prefix pattern matches."""
        f = ClaimFilter(id_pattern=r"^perf:")
        assert f.matches_claim(claim_score_perf1) is True  # perf:dlt1 matches
        assert f.matches_claim(claim_audio_video) is False  # no perf: timeline

    def test_suffix_pattern(self, claim_score_perf1, claim_score_perf5):
        """Regex suffix pattern matches."""
        f = ClaimFilter(id_pattern=r"dlt_1$")
        assert f.matches_claim(claim_score_perf1) is True
        assert f.matches_claim(claim_score_perf5) is False

    def test_complex_pattern(self, claim_score_perf1, claim_score_perf5):
        """Regex range pattern."""
        f = ClaimFilter(id_pattern=r"dlt_[1-3]$")
        assert f.matches_claim(claim_score_perf1) is True  # dlt1
        assert f.matches_claim(claim_score_perf5) is False  # dlt5

    def test_matches_timeline(self):
        """matches_timeline with regex."""
        f = ClaimFilter(id_pattern=r"^perf:")
        assert f.matches_timeline("perf:dlt_1") is True
        assert f.matches_timeline("score:clt_1") is False


# endregion


# region Test between matching


class TestClaimFilterBetween:
    """Tests for between (pair) filter."""

    def test_matches_exact_pair(self, claim_score_perf1):
        """Matches when both timelines are the pair."""
        f = ClaimFilter(between=("score:clt_1", "perf:dlt_1"))
        assert f.matches_claim(claim_score_perf1) is True

    def test_matches_reversed_pair(self, claim_score_perf1):
        """Matches regardless of order (connects_both is set-based)."""
        f = ClaimFilter(between=("perf:dlt_1", "score:clt_1"))
        assert f.matches_claim(claim_score_perf1) is True

    def test_no_match(self, claim_score_perf1):
        """Does not match wrong pair."""
        f = ClaimFilter(between=("score:clt_1", "perf:dlt_5"))
        assert f.matches_claim(claim_score_perf1) is False


# endregion


# region Test synchronous / nomatch filters


class TestClaimFilterSynchronous:
    """Tests for synchronous_only and nomatch_only filters."""

    def test_synchronous_only_passes_sync(self, claim_score_perf1):
        """Synchronous claims pass synchronous_only."""
        f = ClaimFilter(synchronous_only=True)
        assert f.matches_claim(claim_score_perf1) is True

    def test_synchronous_only_rejects_nomatch(self, claim_nomatch):
        """NOMATCH claims fail synchronous_only."""
        f = ClaimFilter(synchronous_only=True)
        assert f.matches_claim(claim_nomatch) is False

    def test_nomatch_only_passes_nomatch(self, claim_nomatch):
        """NOMATCH claims pass nomatch_only."""
        f = ClaimFilter(nomatch_only=True)
        assert f.matches_claim(claim_nomatch) is True

    def test_nomatch_only_rejects_sync(self, claim_score_perf1):
        """Synchronous claims fail nomatch_only."""
        f = ClaimFilter(nomatch_only=True)
        assert f.matches_claim(claim_score_perf1) is False

    def test_nomatch_only_rejects_conceptual(self, claim_conceptual):
        """A conceptual claim is no NOMATCH claim: two events, no sentinel."""
        assert ClaimFilter(nomatch_only=True).matches_claim(claim_conceptual) is False

    def test_synchronous_only_rejects_conceptual(self, claim_conceptual):
        """A conceptual claim is not synchronous."""
        f = ClaimFilter(synchronous_only=True)
        assert f.matches_claim(claim_conceptual) is False


# endregion


# region Test within


class TestClaimFilterWithin:
    """``within``: both timelines of a claim are in the set."""

    def test_both_sides_in(self, claim_score_perf1):
        """A claim between two timelines of the set passes."""
        f = ClaimFilter(within={"score:clt_1", "perf:dlt_1"})
        assert f.matches_claim(claim_score_perf1) is True

    def test_one_side_out(self, claim_score_perf5):
        """A claim with one timeline outside the set fails."""
        f = ClaimFilter(within={"score:clt_1", "perf:dlt_1"})
        assert f.matches_claim(claim_score_perf5) is False

    def test_matches_timeline(self):
        """A timeline passes when it is in the set."""
        f = ClaimFilter(within={"score:clt_1", "perf:dlt_1"})
        assert f.matches_timeline("score:clt_1") is True
        assert f.matches_timeline("perf:dlt_5") is False


# endregion


# region Test combined filters


class TestClaimFilterCombined:
    """Tests for combining multiple filter criteria (AND logic)."""

    def test_id_and_synchronous(self, all_claims):
        """timeline_id + synchronous_only narrows results."""
        f = ClaimFilter(timeline_id="score:clt_1", synchronous_only=True)
        results = [c for c in all_claims if f.matches_claim(c)]
        # claim_score_perf1, claim_score_perf5 are synchronous + involve score
        # claim_nomatch involves score but is not synchronous
        assert len(results) == 2

    def test_id_and_nomatch(self, all_claims):
        """timeline_id + nomatch_only narrows results."""
        f = ClaimFilter(timeline_id="score:clt_1", nomatch_only=True)
        results = [c for c in all_claims if f.matches_claim(c)]
        # Only claim_nomatch: involves score + non-synchronous
        assert len(results) == 1

    def test_pattern_and_between(self, all_claims):
        """id_pattern + between narrows to specific pair with pattern."""
        f = ClaimFilter(id_pattern=r"^perf:", between=("score:clt_1", "perf:dlt_1"))
        results = [c for c in all_claims if f.matches_claim(c)]
        assert len(results) == 1
        assert results[0].timeline_b_id == "perf:dlt_1"

    def test_empty_filter_matches_all(self, all_claims):
        """Empty filter matches every claim."""
        f = ClaimFilter()
        results = [c for c in all_claims if f.matches_claim(c)]
        assert len(results) == 4

    def test_missing_timeline_metadata_fails_unit_filter(self) -> None:
        """A unit criterion rejects a timeline absent from the metadata lookup."""
        f = ClaimFilter(include_units={TimeUnit.seconds})

        assert f.matches_timeline("missing", timelines={}) is False


# endregion


# region Test the belief filter


def _believed_by(actor: str, belief: str) -> Provenance:
    """A provenance entry stating one actor's belief."""
    return Provenance(
        method="manual",
        beliefs=(Belief(actor=actor, belief=belief, date="2026-11-21"),),
    )


def _belief_claims() -> dict[str, MatchClaim]:
    """Four claims by name, one per belief case."""
    return {
        "unstated": _claim("score:clt_1", 1.0, "perf:dlt_1", 1.0),
        "ana_false": _claim(
            "score:clt_1",
            2.0,
            "perf:dlt_1",
            2.0,
            provenance=_believed_by("ana", "false"),
        ),
        "ana_true": _claim(
            "score:clt_1",
            3.0,
            "perf:dlt_1",
            3.0,
            provenance=_believed_by("ana", "true"),
        ),
        "ben_false": _claim(
            "score:clt_1",
            4.0,
            "perf:dlt_1",
            4.0,
            provenance=_believed_by("ben", "false"),
        ),
    }


class TestClaimFilterBelief:
    """Tests for the belief filter (believer, beliefs).

    Four claims, one per case: no provenance, believed false by ``ana``,
    believed true by ``ana``, believed false by ``ben``.
    """

    @pytest.fixture
    def claims(self) -> dict[str, MatchClaim]:
        """The four claims by name."""
        return _belief_claims()

    @pytest.mark.parametrize(
        ("believer", "beliefs", "kept"),
        [
            # No believer: nothing is filtered.
            (None, DEFAULT_BELIEFS, {"unstated", "ana_false", "ana_true", "ben_false"}),
            (None, ["true"], {"unstated", "ana_false", "ana_true", "ben_false"}),
            # ana's false belief drops its claim; ben's belief is not hers.
            ("ana", DEFAULT_BELIEFS, {"unstated", "ana_true", "ben_false"}),
            # Only what ana believes true.
            ("ana", ["true"], {"ana_true"}),
            # What ana holds no belief on: ben's belief is not hers.
            ("ana", ["undecided"], {"unstated", "ben_false"}),
            ("ana", ["false"], {"ana_false"}),
            ("ben", DEFAULT_BELIEFS, {"unstated", "ana_false", "ana_true"}),
            ("ben", ["true"], set()),
            ("ben", ["true", "false"], {"ben_false"}),
        ],
    )
    def test_matches_claim(
        self,
        claims: dict[str, MatchClaim],
        believer: str | None,
        beliefs: list[str],
        kept: set[str],
    ) -> None:
        """matches_claim keeps exactly the claims the belief filter keeps."""
        f = ClaimFilter(believer=believer, beliefs=beliefs)
        assert {name for name, c in claims.items() if f.matches_claim(c)} == kept

    def test_states_held_as_a_frozen_set(self) -> None:
        """The states are checked once and held as a frozen set; repr in order."""
        f = ClaimFilter(believer="ana", beliefs=["false", "true", "false"])
        assert f.beliefs == frozenset({"true", "false"})
        assert repr(f) == "ClaimFilter(believer='ana', beliefs=['true', 'false'])"
        assert repr(ClaimFilter(believer="ana")) == "ClaimFilter(believer='ana')"
        assert ClaimFilter(beliefs=("unlikely", "true", "undecided")) == ClaimFilter()

    def test_belief_combines_with_other_criteria(
        self, claims: dict[str, MatchClaim], claim_audio_video: MatchClaim
    ) -> None:
        """The belief filter is ANDed with the timeline criteria."""
        f = ClaimFilter(timeline_id="score:clt_1", believer="ana", beliefs=["true"])
        assert f.matches_claim(claims["ana_true"]) is True
        assert f.matches_claim(claims["unstated"]) is False
        assert f.matches_claim(claim_audio_video) is False

    def test_invalid_states_refused(self) -> None:
        """A word that is no state, or a bare string, is refused."""
        with pytest.raises(
            ValueError,
            match=re.escape(
                "'not_false' is no belief state: write 'true', 'undecided', "
                "'unlikely' or 'false'"
            ),
        ):
            ClaimFilter(believer="ana", beliefs=["not_false"])
        with pytest.raises(ValueError, match="not the string 'unlikely'"):
            ClaimFilter(believer="ana", beliefs="unlikely")

    def test_belief_ignored_by_matches_timeline(self) -> None:
        """The belief filter is claim-level: a timeline test ignores it."""
        f = ClaimFilter(believer="ana", beliefs=["true"])
        assert f.matches_timeline("score:clt_1") is True


# endregion


# region Test the list form


class TestClaimFilterSelect:
    """``select``, the list form of ``matches_claim``."""

    def test_nomatch_only(self, all_claims, claim_nomatch, claim_conceptual):
        """Only the NOMATCH claim, the same object."""
        selected = ClaimFilter(nomatch_only=True).select(
            [*all_claims, claim_conceptual]
        )
        assert len(selected) == 1
        assert selected[0] is claim_nomatch

    def test_keeps_what_matches_claim_keeps(self, claim_audio_video):
        """The same objects in the same order as ``matches_claim``."""
        claims = [
            *_belief_claims().values(),
            claim_audio_video,
        ]
        f = ClaimFilter(timeline_id="score:clt_1", believer="ana")
        selected = f.select(claims)
        expected = [claim for claim in claims if f.matches_claim(claim)]
        assert [id(claim) for claim in selected] == [id(claim) for claim in expected]
        assert len(selected) == 3


# endregion


# region Test the two forms of the filter agree


def _parity_claims() -> list[MatchClaim]:
    """One claim of every kind, each with its id, the interval one disbelieved."""
    return [
        _claim("score:clt_1", 0.0, "perf:dlt_1", 0.0, id="sync"),
        MatchClaim(
            id="nm_a",
            timeline_a_id="score:clt_1",
            timeline_b_id="perf:dlt_5",
            is_synchronous=False,
            event_a_id="score:clt_1@20.0",
            event_b_id=NOMATCH,
        ),
        MatchClaim(
            id="nm_b",
            timeline_a_id="score:clt_1",
            timeline_b_id="audio:cpt_1",
            is_synchronous=False,
            event_b_id="audio:cpt_1@3.0",
            event_a_id=NOMATCH,
        ),
        MatchClaim(
            id="conceptual",
            timeline_a_id="perf:dlt_1",
            timeline_b_id="perf:dlt_5",
            is_synchronous=False,
            event_a_id="perf:dlt_1@section",
            event_b_id="perf:dlt_5@section",
        ),
        match_claim(
            "score:clt_1",
            Coordinate(4.0, TimeUnit.number),
            "audio:cpt_1",
            Coordinate(2.0, TimeUnit.number),
            end_a=Coordinate(8.0, TimeUnit.number),
            end_b=Coordinate(4.0, TimeUnit.number),
            id="interval",
            provenance=_believed_by("ana", "false"),
        ),
    ]


class TestClaimFilterFieldParity:
    """``ClaimFilter.select`` and ``MatchClaimField.filter`` keep the same claims."""

    @pytest.mark.parametrize(
        ("criteria", "kept"),
        [
            ({}, ["sync", "nm_a", "nm_b", "conceptual", "interval"]),
            ({"timeline_id": "perf:dlt_5"}, ["nm_a", "conceptual"]),
            ({"timeline_ids": {"audio:cpt_1"}}, ["nm_b", "interval"]),
            ({"id_pattern": r"^perf:"}, ["sync", "nm_a", "conceptual"]),
            ({"between": ("perf:dlt_1", "score:clt_1")}, ["sync"]),
            (
                {"within": {"score:clt_1", "perf:dlt_1", "perf:dlt_5"}},
                ["sync", "nm_a", "conceptual"],
            ),
            ({"synchronous_only": True}, ["sync", "interval"]),
            ({"nomatch_only": True}, ["nm_a", "nm_b"]),
            ({"believer": "ana"}, ["sync", "nm_a", "nm_b", "conceptual"]),
            ({"believer": "ana", "beliefs": ["false"]}, ["interval"]),
            ({"timeline_id": "score:clt_1", "nomatch_only": True}, ["nm_a", "nm_b"]),
            (
                {"within": {"score:clt_1", "audio:cpt_1"}, "synchronous_only": True},
                ["interval"],
            ),
            ({"id_pattern": r"^audio:", "believer": "ana"}, ["nm_b"]),
            (
                {"between": ("score:clt_1", "audio:cpt_1"), "synchronous_only": True},
                ["interval"],
            ),
        ],
    )
    def test_both_forms_keep_the_same_claims(
        self, criteria: dict, kept: list[str]
    ) -> None:
        claims = _parity_claims()
        selected = ClaimFilter(**criteria).select(claims)
        filtered = MatchClaimField.from_claims(claims).filter(**criteria)
        assert [claim.id for claim in selected] == kept
        assert filtered.table.column("id").to_pylist() == kept


# endregion
