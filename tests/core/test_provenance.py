"""Provenance: paradata and beliefs, one entry.

Validation strategy and exact values: ``tests/core/README.md`` (the
``test_provenance.py`` section).
"""

from __future__ import annotations

import re

import pyarrow as pa
import pytest
from pydantic import ValidationError

from timetoalign.core.fields import derive_arrow_struct
from timetoalign.core.provenance import (
    DEFAULT_BELIEFS,
    Belief,
    BeliefField,
    Provenance,
    ProvenanceField,
    actor_iri,
    belief_mask,
    check_belief_filter,
    provenance_array,
)


def _belief(actor: str, belief: str, date: str = "2026-11-21") -> Belief:
    return Belief(actor=actor, belief=belief, date=date)


#: The five rows of the belief-filter column.
_ROWS: list[Provenance | None] = [
    None,
    Provenance(service="https://example.org/aligner"),
    Provenance(method="manual", beliefs=(_belief("jo", "false"),)),
    Provenance(method="manual", beliefs=(_belief("jo", "true"),)),
    Provenance(
        method="manual",
        beliefs=(_belief("jo", "unlikely"), _belief("ana", "false")),
    ),
]


class TestKeys:
    def test_entry_keys_are_the_format_s(self) -> None:
        assert derive_arrow_struct(Provenance).names == [
            "stated_by",
            "date",
            "method",
            "service",
            "basis",
            "match",
            "note",
            "beliefs",
        ]
        assert derive_arrow_struct(Belief).names == ["actor", "belief", "date", "note"]
        assert ProvenanceField.pa_schema == derive_arrow_struct(Provenance)
        assert BeliefField.pa_schema == derive_arrow_struct(Belief)

    def test_unknown_key_refused(self) -> None:
        with pytest.raises(ValidationError, match="status"):
            Provenance.from_dict({"service": "s", "status": "confirmed"})
        with pytest.raises(ValidationError, match="weight"):
            Belief.from_dict(
                {"actor": "jo", "belief": "true", "date": "2026-11-21", "weight": 1}
            )

    def test_empty_entry_refused(self) -> None:
        with pytest.raises(
            ValidationError, match="A provenance entry states at least one key"
        ):
            Provenance()


class TestValues:
    @pytest.mark.parametrize(
        ("date", "message"),
        [
            ("2026-1-5", "A date is written YYYY-MM-DD, not '2026-1-5'"),
            ("2026-02-30", "'2026-02-30' is no date of the calendar"),
        ],
    )
    def test_dates(self, date: str, message: str) -> None:
        with pytest.raises(ValidationError, match=re.escape(message)):
            Provenance(date=date)
        with pytest.raises(ValidationError, match=re.escape(message)):
            _belief("jo", "true", date)
        assert Provenance(date="2026-02-28").date == "2026-02-28"

    def test_iri_lists(self) -> None:
        with pytest.raises(ValidationError, match="stated_by lists at least one IRI"):
            Provenance(stated_by=())
        with pytest.raises(ValidationError, match="basis lists each IRI once"):
            Provenance(basis=("a", "a"))
        assert Provenance(stated_by=["a", "b"]).stated_by == ("a", "b")

    def test_closed_vocabularies(self) -> None:
        for kind in ("exact", "close", "broad", "narrow", "related"):
            assert Provenance(match=kind).match == kind
        with pytest.raises(ValidationError):
            Provenance(match="similar")
        for value in ("true", "false", "unlikely"):
            assert _belief("jo", value).belief == value
        with pytest.raises(ValidationError):
            _belief("jo", "maybe")
        with pytest.raises(ValidationError, match="A belief names its actor"):
            _belief("", "true")

    def test_method_and_service_as_the_format_takes_them(self) -> None:
        for method in ("manual", "reconciled", "imported", "inferred"):
            assert Provenance(method=method).method == method
        assert (
            Provenance(method="https://example.org/m").method == "https://example.org/m"
        )
        with pytest.raises(
            ValidationError,
            match=re.escape(
                "A method is one of ['manual', 'reconciled', 'imported', "
                "'inferred'] or an IRI, not 'by hand'"
            ),
        ):
            Provenance(method="by hand")
        with pytest.raises(ValidationError, match="A service is named, never empty"):
            Provenance(service="")


class TestNullBeliefs:
    def test_null_beliefs_read_as_none_and_written_back_null(self) -> None:
        row = {
            "stated_by": None,
            "date": "2026-11-20",
            "method": "imported",
            "service": "https://example.org/aligner",
            "basis": None,
            "match": None,
            "note": None,
            "beliefs": None,
        }
        entry = Provenance.from_row(row)
        assert entry == Provenance(
            date="2026-11-20", method="imported", service="https://example.org/aligner"
        )
        assert entry.beliefs == ()
        assert provenance_array([entry]).to_pylist() == [row]


class TestOneBeliefPerActor:
    def test_two_beliefs_of_one_actor_refused(self) -> None:
        with pytest.raises(
            ValidationError,
            match=re.escape("An actor holds at most one belief; 'ana' holds 2"),
        ):
            Provenance(beliefs=(_belief("ana", "true"), _belief("ana", "false")))

    def test_with_belief_replaces_in_place_and_appends(self) -> None:
        entry = Provenance(
            beliefs=(
                _belief("ana", "unlikely", "2026-11-20"),
                _belief("jo", "true"),
            )
        )
        replaced = entry.with_belief(_belief("ana", "false", "2026-11-21"))
        assert replaced.beliefs == (
            _belief("ana", "false", "2026-11-21"),
            _belief("jo", "true"),
        )
        appended = replaced.with_belief(_belief("kim", "true"))
        assert [belief.actor for belief in appended.beliefs] == ["ana", "jo", "kim"]
        assert entry.beliefs == (
            _belief("ana", "unlikely", "2026-11-20"),
            _belief("jo", "true"),
        )
        assert replaced.get_belief("ana") == _belief("ana", "false", "2026-11-21")
        assert replaced.get_belief("kim") is None


class TestDictionary:
    def test_literal_round_trip(self) -> None:
        entry = Provenance(
            stated_by=("https://orcid.org/0000-0001",),
            date="2026-11-21",
            method="manual",
            basis=("https://example.org/score", "https://example.org/audio"),
            match="exact",
            note="checked by ear",
            beliefs=(
                Belief(
                    actor="https://orcid.org/0000-0001",
                    belief="true",
                    date="2026-11-21",
                    note="mine",
                ),
                _belief("jo", "unlikely"),
            ),
        )
        data = {
            "stated_by": ["https://orcid.org/0000-0001"],
            "date": "2026-11-21",
            "method": "manual",
            "basis": ["https://example.org/score", "https://example.org/audio"],
            "match": "exact",
            "note": "checked by ear",
            "beliefs": [
                {
                    "actor": "https://orcid.org/0000-0001",
                    "belief": "true",
                    "date": "2026-11-21",
                    "note": "mine",
                },
                {"actor": "jo", "belief": "unlikely", "date": "2026-11-21"},
            ],
        }
        assert entry.to_dict() == data
        assert Provenance.from_dict(data) == entry
        assert Provenance(service="s").to_dict() == {"service": "s"}
        assert Belief.from_dict(_belief("jo", "true").to_dict()) == _belief(
            "jo", "true"
        )


_ALL_STATES = ("true", "undecided", "unlikely", "false")


class TestBeliefFilter:
    @pytest.mark.parametrize(
        ("believer", "beliefs", "expected"),
        [
            ("jo", DEFAULT_BELIEFS, [True, True, False, True, True]),
            ("jo", ["true"], [False, False, False, True, False]),
            ("jo", ["undecided"], [True, True, False, False, False]),
            ("jo", ["unlikely"], [False, False, False, False, True]),
            ("jo", ["false"], [False, False, True, False, False]),
            ("jo", _ALL_STATES, [True, True, True, True, True]),
            ("jo", [], [False, False, False, False, False]),
            ("ana", DEFAULT_BELIEFS, [True, True, True, True, False]),
            ("ana", ["true"], [False, False, False, False, False]),
            ("ana", ["false"], [False, False, False, False, True]),
            (None, DEFAULT_BELIEFS, [True, True, True, True, True]),
            (None, ["true"], [True, True, True, True, True]),
        ],
    )
    def test_one_predicate(
        self, believer: str | None, beliefs: list[str], expected: list[bool]
    ) -> None:
        column = provenance_array(_ROWS)
        assert belief_mask(column, believer, beliefs).to_pylist() == expected
        assert (
            ProvenanceField.from_provenances(_ROWS)
            .passes_belief_filter(believer, beliefs)
            .to_pylist()
            == expected
        )
        assert [
            entry.passes_belief_filter(believer, beliefs)
            for entry in _ROWS
            if entry is not None
        ] == expected[1:]

    @pytest.mark.parametrize("believer", ["jo", "ana"])
    def test_the_four_states_partition_the_rows(self, believer: str) -> None:
        column = provenance_array(_ROWS)
        masks = [
            belief_mask(column, believer, [state]).to_pylist() for state in _ALL_STATES
        ]
        assert [sum(row) for row in zip(*masks, strict=True)] == [1] * len(_ROWS)

    def test_default_is_all_but_false(self) -> None:
        column = provenance_array(_ROWS)
        assert DEFAULT_BELIEFS == ("true", "undecided", "unlikely")
        assert belief_mask(column, "jo").equals(
            belief_mask(column, "jo", {"unlikely", "true", "undecided"})
        )
        assert belief_mask(column, "jo", ["true", "true"]).equals(
            belief_mask(column, "jo", ("true",))
        )

    def test_empty_column_and_unknown_states(self) -> None:
        assert belief_mask(provenance_array([]), "jo").to_pylist() == []
        with pytest.raises(
            ValueError,
            match=re.escape(
                "The belief filter keeps the states ['true', 'undecided', "
                "'unlikely', 'false'], not ['not_false']"
            ),
        ):
            belief_mask(provenance_array(_ROWS), "jo", ["not_false"])  # type: ignore[list-item]
        with pytest.raises(
            ValueError,
            match=re.escape(
                "The belief filter takes a collection of states, not the string "
                "'false': write ['false']."
            ),
        ):
            belief_mask(provenance_array(_ROWS), "jo", "false")

    def test_chunked_column(self) -> None:
        column = provenance_array(_ROWS)
        chunked = pa.chunked_array([column.slice(0, 2), column.slice(2)])
        assert belief_mask(chunked, "jo").to_pylist() == [True, True, False, True, True]


_ANA = "https://example.org/people/ana"
_CONTEXT = {"ex": "https://example.org/people/"}

#: One actor written compact and absolute, and an actor of an undeclared prefix.
_IRI_ROWS: list[Provenance | None] = [
    Provenance(method="manual", beliefs=(_belief("ex:ana", "false"),)),
    Provenance(method="manual", beliefs=(_belief(_ANA, "false"),)),
    Provenance(method="manual", beliefs=(_belief(_ANA, "true"),)),
    Provenance(method="manual", beliefs=(_belief("wd:Q1", "false"),)),
]


class TestActorsCompareAsIris:
    @pytest.mark.parametrize(
        ("believer", "beliefs", "context", "expected"),
        [
            ("ex:ana", DEFAULT_BELIEFS, _CONTEXT, [False, False, True, True]),
            (_ANA, DEFAULT_BELIEFS, _CONTEXT, [False, False, True, True]),
            (_ANA, ["true"], _CONTEXT, [False, False, True, False]),
            ("ex:ana", ["true"], _CONTEXT, [False, False, True, False]),
            ("ex:ana", DEFAULT_BELIEFS, None, [False, True, True, True]),
            (_ANA, DEFAULT_BELIEFS, None, [True, False, True, True]),
            ("wd:Q1", DEFAULT_BELIEFS, _CONTEXT, [True, True, True, False]),
        ],
    )
    def test_expanded_under_the_context(
        self,
        believer: str,
        beliefs: list[str],
        context: dict[str, str] | None,
        expected: list[bool],
    ) -> None:
        column = provenance_array(_IRI_ROWS)
        mask = belief_mask(column, believer, beliefs, context=context)
        assert mask.to_pylist() == expected

    def test_actor_iri(self) -> None:
        assert actor_iri("ex:ana", _CONTEXT) == _ANA
        assert actor_iri("ex:ana") == "ex:ana"
        assert actor_iri(_ANA) == _ANA
        assert actor_iri("jo", _CONTEXT) == "jo"

    def test_one_actor_however_spelled_holds_one_belief(self) -> None:
        entry = Provenance(
            method="manual",
            beliefs=(_belief(_ANA, "true"), _belief("jo", "unlikely")),
        )

        replaced = entry.with_belief(_belief("ex:ana", "false"), context=_CONTEXT)

        assert replaced.beliefs == (
            _belief("ex:ana", "false"),
            _belief("jo", "unlikely"),
        )
        assert entry.get_belief("ex:ana", context=_CONTEXT) == _belief(_ANA, "true")
        assert replaced.get_belief(_ANA, context=_CONTEXT) == _belief("ex:ana", "false")
        assert entry.get_belief("ex:ana") is None
        appended = entry.with_belief(_belief("ex:ana", "false"))
        assert appended.beliefs == (
            _belief(_ANA, "true"),
            _belief("jo", "unlikely"),
            _belief("ex:ana", "false"),
        )
        with pytest.raises(
            ValidationError,
            match=re.escape(f"An actor holds at most one belief; {_ANA!r} holds 2"),
        ):
            Provenance(beliefs=(_belief(_ANA, "true"), _belief(_ANA, "false")))

    def test_two_spellings_of_the_believer_read_as_the_most_critical(self) -> None:
        false_pair = provenance_array(
            [
                Provenance(beliefs=(_belief("ex:ana", "true"), _belief(_ANA, "false"))),
                Provenance(beliefs=(_belief(_ANA, "false"), _belief("ex:ana", "true"))),
            ]
        )
        unlikely_pair = provenance_array(
            [
                Provenance(
                    beliefs=(_belief("ex:ana", "true"), _belief(_ANA, "unlikely"))
                ),
                Provenance(
                    beliefs=(_belief(_ANA, "unlikely"), _belief("ex:ana", "true"))
                ),
            ]
        )

        def mask(column: pa.Array, beliefs, context=_CONTEXT) -> list[bool]:
            return belief_mask(column, "ex:ana", beliefs, context=context).to_pylist()

        assert mask(false_pair, DEFAULT_BELIEFS) == [False, False]
        assert mask(false_pair, ["false"]) == [True, True]
        assert mask(false_pair, ["true"]) == [False, False]
        assert mask(unlikely_pair, ["unlikely"]) == [True, True]
        assert mask(unlikely_pair, ["true"]) == [False, False]
        assert mask(false_pair, ["true"], context=None) == [True, True]

    def test_one_check_of_the_states(self) -> None:
        assert check_belief_filter(["true", "unlikely"]) == frozenset(
            {"true", "unlikely"}
        )
        assert check_belief_filter(("true", "true")) == frozenset({"true"})
        assert check_belief_filter([]) == frozenset()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "The belief filter takes a collection of states, not the string "
                "'true': write ['true']."
            ),
        ):
            check_belief_filter("true")
        with pytest.raises(
            ValueError,
            match=re.escape(
                "The belief filter keeps the states ['true', 'undecided', "
                "'unlikely', 'false'], not ['maybe']"
            ),
        ):
            check_belief_filter(["maybe", "true"])


class TestField:
    def test_rows_read_back(self) -> None:
        field = ProvenanceField.from_provenances(_ROWS)
        assert field.value.data.type == derive_arrow_struct(Provenance)
        assert [field[i] for i in range(len(_ROWS))] == _ROWS
