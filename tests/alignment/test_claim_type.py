"""Tests for the derived ``MatchClaim.claim_type`` classification.

A :class:`MatchClaim` never stores its semantic kind. ``claim_type`` reads the
kind back from two structural facts: whether it is synchronous, and whether
one of its two sides (``event_a_id`` / ``event_b_id``) is the NOMATCH
sentinel. Three kinds. Validation strategy:
``tests/alignment/README.md`` ("Claim Kinds").

======================================  =================  ===========================
Structure                               ``ClaimType``      Typical origin
======================================  =================  ===========================
synchronous (each side event or CHUNK)  ``event_match``    ``MatchClaim.from_events``
non-synchronous, NOMATCH on one side    ``nomatch``        ``MatchClaim.nomatch``
non-synchronous, an event on each side  ``conceptual``     ``from_events(..., is_synchronous=False)``
======================================  =================  ===========================

Display
-------

``__repr__`` shows an uppercase ``[KIND]`` badge for every kind except a plain
``event_match`` (which stays badge-free). ``__str__`` and ``_repr_html_`` report
the derived kind for non-synchronous claims, so a conceptual claim never reads
"NOMATCH".
"""

from __future__ import annotations

import pytest

from timetoalign.alignment import ClaimType, MatchClaim, MatchClaimField
from timetoalign.core import ClaimType as CoreClaimType
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
)

SCORE = ContinuousLogicalTimeline(length=200, uid="score:1")
RECORDING = ContinuousPhysicalTimeline(length=100.0, uid="recording:1")


def _event_match() -> MatchClaim:
    return MatchClaim.from_events(
        {"id": "e001", "name": "Note C4", "start": 100},
        SCORE,
        {"id": "e042", "name": "Note C4", "start": 45.5},
        RECORDING,
    )


def _conceptual() -> MatchClaim:
    return MatchClaim.from_events(
        {"id": "e001", "start": 100},
        SCORE,
        {"id": "e042", "start": 45.5},
        RECORDING,
        is_synchronous=False,
        id="conceptual",
    )


def _nomatch() -> MatchClaim:
    return MatchClaim.nomatch({"id": "x", "start": 0}, SCORE, RECORDING)


# region Classification by construction path


class TestClaimTypeClassification:
    """Exact ``claim_type`` for every construction path."""

    def test_from_events_is_event_match(self) -> None:
        assert _event_match().claim_type is ClaimType.event_match

    def test_chunk_sides_are_an_event_match(self) -> None:
        chunk = MatchClaim.from_events(
            {"start": 100}, SCORE, {"start": 45.5}, RECORDING
        )
        assert (chunk.claim_type, chunk.is_nomatch, chunk.is_conceptual) == (
            ClaimType.event_match,
            False,
            False,
        )

    def test_from_events_not_synchronous_is_conceptual(self) -> None:
        claim = _conceptual()
        assert claim.claim_type is ClaimType.conceptual
        assert claim.anchors == []
        assert (claim.event_a_id, claim.event_b_id) == ("e001", "e042")

    def test_nomatch_is_nomatch(self) -> None:
        claim = _nomatch()
        assert claim.claim_type is ClaimType.nomatch
        assert (claim.event_a_id, claim.event_b_id) == ("x", "NOMATCH_SENTINEL")

    def test_field_row_materialises_as_event_match(self) -> None:
        field = MatchClaimField.from_columns(
            timeline_a_ids=["A", "A"],
            timeline_b_ids=["B", "B"],
            event_a_ids=["a0", "a1"],
            event_b_ids=["b0", "b1"],
            coordinate_a=[0, 1],
            coordinate_b=[10.0, 11.0],
            timelines={"A": ("quarters", "fraction"), "B": ("seconds", "float")},
        )
        assert [claim.claim_type for claim in field] == [
            ClaimType.event_match,
            ClaimType.event_match,
        ]
        rows = MatchClaimField.from_claims([_nomatch(), _conceptual()])
        assert [claim.claim_type for claim in rows] == [
            ClaimType.nomatch,
            ClaimType.conceptual,
        ]


# endregion


# region Convenience booleans


class TestClaimTypeBooleans:
    """The thin ``is_*`` predicates over ``claim_type``."""

    def test_booleans(self) -> None:
        assert (_event_match().is_nomatch, _event_match().is_conceptual) == (
            False,
            False,
        )
        assert (_nomatch().is_nomatch, _nomatch().is_conceptual) == (True, False)
        assert (_conceptual().is_nomatch, _conceptual().is_conceptual) == (
            False,
            True,
        )

    def test_no_anonymous_predicate(self) -> None:
        assert not hasattr(MatchClaim, "is_anonymous")


# endregion


# region Display consequences


class TestClaimTypeDisplay:
    """``__repr__`` / ``__str__`` reflect the derived kind, not raw synchrony."""

    def test_repr_event_match_has_no_badge(self) -> None:
        assert repr(_event_match()) == (
            "MatchClaim(instant: score:1@100 quarters <-> recording:1@45.5 seconds)"
        )

    def test_repr_conceptual_says_conceptual_not_nomatch(self) -> None:
        assert repr(_conceptual()) == (
            "MatchClaim(score:1 <-> recording:1 [CONCEPTUAL])"
        )

    def test_str_conceptual_does_not_say_nomatch(self) -> None:
        assert str(_conceptual()) == (
            "MatchClaim (CONCEPTUAL)\n"
            "  Timeline A:  score:1\n"
            "  Event A:     e001\n"
            "  Timeline B:  recording:1\n"
            "  Event B:     e042"
        )

    def test_repr_nomatch_says_nomatch(self) -> None:
        assert repr(_nomatch()) == "MatchClaim(score:1 <-> recording:1 [NOMATCH])"


# endregion


# region Enum


class TestClaimTypeEnum:
    """The members of ``ClaimType`` and their spellings."""

    def test_members(self) -> None:
        assert [member.name for member in ClaimType] == [
            "event_match",
            "nomatch",
            "conceptual",
        ]

    def test_canonical_name_instantiation(self) -> None:
        for member in ClaimType:
            assert ClaimType(member.name) is member

    def test_alias_instantiation(self) -> None:
        assert ClaimType("event") is ClaimType.event_match
        assert ClaimType("nomat") is ClaimType.nomatch

    @pytest.mark.parametrize("name", ["projection", "anchor", "anon", "implicit"])
    def test_kinds_without_events_are_gone(self, name: str) -> None:
        with pytest.raises(ValueError):
            ClaimType(name)

    def test_core_and_alignment_exports_are_identical(self) -> None:
        assert ClaimType is CoreClaimType


# endregion
