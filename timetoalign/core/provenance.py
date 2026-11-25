"""Provenance: how an item was made, and who believes it.

One provenance entry serves every item that can be stated by someone: a
MatchClaim holds one in its ``provenance`` property. It has two parts:

- **Paradata**, how the item was made: ``stated_by``, ``date``, ``method``,
  ``service``, ``basis``, ``match`` and ``note``, the keys the TimeToShare
  provenance entries use.
- **Beliefs**, who believes it: a list of :class:`Belief`, each an actor's
  ``true``, ``false`` or ``unlikely`` with its date. An actor holds at most
  one belief; a later one replaces the earlier (:meth:`Provenance.with_belief`).
  One actor's belief leaves another's standing, and nothing derives a truth
  value from them.

The **belief filter** reads the beliefs for one actor, the *believer*: it
keeps the items whose state for the believer is one of the states asked for
(``beliefs``), each of ``"true"``, ``"undecided"`` (the believer holds no
belief on the item), ``"unlikely"`` and ``"false"``; by default every state
but ``"false"`` (:data:`DEFAULT_BELIEFS`). Without a believer nothing is
filtered. :func:`belief_mask` is the one place the filter is decided; the
scalar and the Field both ask it.

Each scalar is followed by its paired Field: :class:`Belief` /
:class:`BeliefField`, :class:`Provenance` / :class:`ProvenanceField`.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)
from timetoshare import DEFAULT_BELIEFS, BeliefState, check_belief_filter
from timetoshare.statements import MATCHES, METHODS, expand_iri

from .fields import (
    SemanticField,
    _cast_column,
    _compared,
    _filled,
    _struct_array_of,
    _struct_member,
    data_shaped,
)

BeliefValue = Literal["true", "false", "unlikely"]
"""What an actor believes of an item."""

_CRITICAL_FIRST: tuple[BeliefValue, ...] = ("false", "unlikely", "true")
"""The beliefs, the most critical first: one row holding two beliefs of one
believer (two spellings of the actor) is read as the first of them."""

_DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")

#: An IRI as the format tells one from a method word: a scheme, a colon and
#: something after it.
_IRI_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_+.-]*:.")


def _checked_date(value: str) -> str:
    """Return *value* if it is a calendar date written ``YYYY-MM-DD``."""
    if not _DATE_PATTERN.fullmatch(value):
        raise ValueError(f"A date is written YYYY-MM-DD, not {value!r}")
    try:
        datetime.date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{value!r} is no date of the calendar") from None
    return value


def _without_nones(data: dict[str, Any]) -> dict[str, Any]:
    """The mapping without the keys whose value is ``None``."""
    return {key: value for key, value in data.items() if value is not None}


def actor_iri(actor: str, context: Mapping[str, Any] | None = None) -> str:
    """The IRI an actor stands for under *context*: how actors are compared.

    :func:`timetoshare.expand_iri` under *context*; an actor that expands to
    nothing (a compact IRI whose prefix *context* does not declare, or a name
    that is no IRI) is compared as written. Without a context absolute IRIs
    and the URI schemes still expand.

    Examples:
        >>> actor_iri("orcid:0000-0001", {"orcid": "https://orcid.org/"})
        'https://orcid.org/0000-0001'
        >>> actor_iri("orcid:0000-0001")
        'orcid:0000-0001'
    """
    return expand_iri(context, actor) or actor


# region Belief


class Belief(BaseModel):
    """One actor's belief in an item, at a date.

    Attributes:
        actor: Who holds the belief (an IRI or another identifier of a
            person).
        belief: ``"true"``, ``"false"`` or ``"unlikely"``.
        date: When the actor stated it, written ``YYYY-MM-DD``.
        note: Optional free text.

    Examples:
        >>> Belief(actor="https://orcid.org/0000-0001", belief="false", date="2026-11-21")
        Belief(actor='https://orcid.org/0000-0001', belief='false', date='2026-11-21', note=None)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    actor: str
    belief: BeliefValue
    date: str
    note: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Belief:
        """Read a belief from its dictionary (:meth:`to_dict`)."""
        return cls.model_validate(data)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Belief:
        """Read a belief from one row of its struct column."""
        return cls.model_validate(row)

    @field_validator("actor")
    @classmethod
    def _require_actor(cls, value: str) -> str:
        if not value:
            raise ValueError("A belief names its actor")
        return value

    @field_validator("date")
    @classmethod
    def _check_date(cls, value: str) -> str:
        return _checked_date(value)

    def to_dict(self) -> dict[str, Any]:
        """The belief as a JSON-safe dictionary, an absent ``note`` left out."""
        return _without_nones(self.model_dump(mode="json"))


class BeliefField(SemanticField[Belief]):
    """Paired Field for :class:`Belief`."""


# endregion


# region Provenance


class Provenance(BaseModel):
    """How an item was made (paradata) and who believes it.

    Every key is optional, but an entry states at least one. The format
    records a person who states an item with the belief ``"true"``, and a
    loader or a service with how the item was made and no belief.

    Attributes:
        stated_by: Who stated the item (IRIs), at least one, each once.
        date: When it was stated, written ``YYYY-MM-DD``.
        method: How: one of TimeToShare's method words (``"manual"``,
            ``"reconciled"``, ``"imported"``, ``"inferred"``) or an IRI
            naming another method.
        service: The service or software that made it (an IRI), never empty.
        basis: The sources it rests on (IRIs), at least one, each once.
        match: For a reconciled value, how closely it matches its source:
            one of TimeToShare's SKOS mapping words (``"exact"``,
            ``"close"``, ``"broad"``, ``"narrow"``, ``"related"``).
        note: Free text.
        beliefs: Who believes the item, at most one :class:`Belief` per
            actor, in the order given. Actors are compared as the IRIs they
            stand for (:func:`actor_iri`): on construction without a
            context, so ``https://orcid.org/0000-0001`` is one actor however
            it is spelled absolutely; :meth:`get_belief` and
            :meth:`with_belief` take the context that expands compact IRIs.

    Examples:
        >>> entry = Provenance(service="https://example.org/aligner", method="inferred")
        >>> entry = entry.with_belief(
        ...     Belief(actor="ana", belief="unlikely", date="2026-11-20")
        ... )
        >>> entry.with_belief(
        ...     Belief(actor="ana", belief="false", date="2026-11-21")
        ... ).get_belief("ana").belief
        'false'
        >>> entry.passes_belief_filter("ana"), entry.passes_belief_filter("ana", ["true"])
        (True, False)
        >>> entry.passes_belief_filter("bea", ["undecided"])
        True
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    stated_by: tuple[str, ...] | None = None
    date: str | None = None
    method: str | None = None
    service: str | None = None
    basis: tuple[str, ...] | None = None
    match: str | None = None
    note: str | None = None
    beliefs: tuple[Belief, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Provenance:
        """Read an entry from its dictionary (:meth:`to_dict`).

        Raises:
            ValueError: If a key is unknown or a value invalid.
        """
        return cls.model_validate(data)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Provenance:
        """Read an entry from one row of its struct column."""
        return cls.model_validate(row)

    @field_validator("stated_by", "basis")
    @classmethod
    def _check_iri_list(
        cls, value: tuple[str, ...] | None, info: ValidationInfo
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        if not value:
            raise ValueError(f"{info.field_name} lists at least one IRI")
        if len(set(value)) != len(value):
            raise ValueError(f"{info.field_name} lists each IRI once")
        return value

    @field_validator("date")
    @classmethod
    def _check_date(cls, value: str | None) -> str | None:
        return None if value is None else _checked_date(value)

    @field_validator("method")
    @classmethod
    def _check_method(cls, value: str | None) -> str | None:
        if value is None or value in METHODS or _IRI_PATTERN.match(value):
            return value
        raise ValueError(f"A method is one of {list(METHODS)} or an IRI, not {value!r}")

    @field_validator("service")
    @classmethod
    def _check_service(cls, value: str | None) -> str | None:
        if value == "":
            raise ValueError("A service is named, never empty")
        return value

    @field_validator("match")
    @classmethod
    def _check_match(cls, value: str | None) -> str | None:
        if value is None or value in MATCHES:
            return value
        raise ValueError(f"A match is one of {list(MATCHES)}, not {value!r}")

    @field_validator("beliefs", mode="before")
    @classmethod
    def _read_null_beliefs(cls, value: Any) -> Any:
        """Read a null list of beliefs, as a table row states none, as none."""
        return () if value is None else value

    @model_validator(mode="after")
    def _check_actors(self) -> Provenance:
        actors = [actor_iri(belief.actor) for belief in self.beliefs]
        repeated = sorted({actor for actor in actors if actors.count(actor) > 1})
        if repeated:
            raise ValueError(
                f"An actor holds at most one belief; {repeated[0]!r} holds "
                f"{actors.count(repeated[0])}"
            )
        return self

    @model_validator(mode="after")
    def _check_stated(self) -> Provenance:
        if not self.beliefs and all(
            getattr(self, name) is None
            for name in type(self).model_fields
            if name != "beliefs"
        ):
            raise ValueError("A provenance entry states at least one key")
        return self

    def get_belief(
        self, actor: str, *, context: Mapping[str, Any] | None = None
    ) -> Belief | None:
        """The belief *actor* holds, if any.

        Args:
            actor: The actor, an IRI (compact or absolute).
            context: The JSON-LD context under which actors compare as IRIs
                (:func:`actor_iri`); ``None`` declares no prefix.
        """
        wanted = actor_iri(actor, context)
        for belief in self.beliefs:
            if actor_iri(belief.actor, context) == wanted:
                return belief
        return None

    def with_belief(
        self, belief: Belief, *, context: Mapping[str, Any] | None = None
    ) -> Provenance:
        """A copy holding *belief*, which replaces its actor's earlier one.

        The actor's earlier belief is found as :meth:`get_belief` finds it,
        actors compared as IRIs under *context*, so a compact and an
        absolute spelling of one actor hold one belief; the new belief takes
        its place, as written. Every other actor's belief stands, in its
        place; a new actor's belief is appended.

        Args:
            belief: The belief.
            context: The JSON-LD context under which actors compare as IRIs
                (:func:`actor_iri`); ``None`` declares no prefix.
        """
        actor = actor_iri(belief.actor, context)
        held = [actor_iri(other.actor, context) == actor for other in self.beliefs]
        beliefs = [
            other for other, same in zip(self.beliefs, held, strict=True) if not same
        ]
        position = held.index(True) if True in held else len(beliefs)
        beliefs.insert(position, belief)
        return self.model_copy(update={"beliefs": tuple(beliefs)})

    @data_shaped
    def passes_belief_filter(
        self,
        believer: str | None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Whether the belief filter for *believer* keeps the item.

        Args:
            believer: The actor whose beliefs filter; ``None`` keeps all.
            beliefs: The states kept (:data:`BeliefState`): the item passes
                when the believer's belief in it, or ``"undecided"`` where
                the believer holds none, is one of them. Default: every
                state but ``"false"``.
            context: The JSON-LD context under which actors compare as IRIs
                (:func:`belief_mask`).

        Returns:
            Whether the item passes, decided by :func:`belief_mask`.
        """
        mask = belief_mask(provenance_array([self]), believer, beliefs, context=context)
        return bool(mask[0].as_py())

    def to_dict(self) -> dict[str, Any]:
        """The entry as a JSON-safe dictionary, absent keys left out."""
        data = _without_nones(self.model_dump(mode="json", exclude={"beliefs"}))
        if self.beliefs:
            data["beliefs"] = [belief.to_dict() for belief in self.beliefs]
        return data


class ProvenanceField(SemanticField[Provenance]):
    """Paired Field for :class:`Provenance`: one entry per row, null where none."""

    @classmethod
    def from_provenances(
        cls, values: Sequence[Provenance | None], *, name: str = "provenance"
    ) -> ProvenanceField:
        """Build a field from entries (``None`` for a row stating none)."""
        array = provenance_array(values)
        return cls.from_field((array, pa.field(name, array.type)))

    def passes_belief_filter(
        self,
        believer: str | None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> pa.BooleanArray:
        """Per row, whether the belief filter for *believer* keeps it.

        The vectorized mirror of :meth:`Provenance.passes_belief_filter`,
        decided by :func:`belief_mask`.
        """
        return belief_mask(self._raw.data, believer, beliefs, context=context)


def provenance_array(values: Sequence[Provenance | None]) -> pa.StructArray:
    """The struct array of *values*, of the type :class:`ProvenanceField` stores.

    An entry without beliefs has a null ``beliefs`` cell, as TimeToShare's
    tables write it.

    Args:
        values: One entry per row, ``None`` for a row stating none.

    Returns:
        A struct array of ``ProvenanceField.pa_schema``.
    """
    rows = [None if value is None else _row(value) for value in values]
    return _struct_array_of(rows, ProvenanceField.struct_schema())


def _row(value: Provenance) -> dict[str, Any]:
    """One entry as a row of its struct column; no beliefs is a null cell."""
    row = value.model_dump(mode="json")
    if not row["beliefs"]:
        row["beliefs"] = None
    return row


def belief_mask(
    provenance: pa.Array[Any] | pa.ChunkedArray[Any] | None,
    believer: str | None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    *,
    context: Mapping[str, Any] | None = None,
) -> pa.BooleanArray:
    """The belief filter: per row, whether it is kept for *believer*.

    The one place the filter is decided; every consumer of provenance (a
    claim, a :class:`ProvenanceField`, a MatchClaim table, the statements of
    a holder) asks it. Vectorized over the beliefs: no entry is materialised.

    Actors compare as the IRIs they stand for under *context*, as
    TimeToShare compares them (:func:`timetoshare.expand_iri`), so a compact
    and an absolute IRI of one actor name the same actor; an actor that
    expands to nothing (a compact IRI whose prefix *context* does not
    declare) compares as written. Where two spellings of the believer hold
    two beliefs on one row, the row is read as the most critical of them
    (``"false"``, then ``"unlikely"``, then ``"true"``).

    Args:
        provenance: A struct array of ``ProvenanceField.pa_schema`` (null
            where a row states no provenance).
        believer: The actor whose beliefs filter, an IRI (compact or
            absolute); ``None`` keeps every row.
        beliefs: The states kept (:data:`BeliefState`): a row passes when
            the believer's belief in it (``"true"``, ``"unlikely"``,
            ``"false"``), or ``"undecided"`` where the believer holds none,
            is one of them. Default: every state but ``"false"``.
        context: The JSON-LD context whose prefixes expand compact IRIs, the
            believer's and the actors' alike; ``None`` declares no prefix
            (absolute IRIs and the URI schemes still compare as IRIs).

    Returns:
        A boolean array, one value per row, never null.

    Raises:
        ValueError: If *beliefs* is not a collection of belief states
            (:func:`check_belief_filter`).

    Examples:
        >>> rows = provenance_array(
        ...     [
        ...         None,
        ...         Provenance(beliefs=(Belief(actor="ana", belief="false", date="2026-11-21"),)),
        ...         Provenance(beliefs=(Belief(actor="ana", belief="true", date="2026-11-21"),)),
        ...     ]
        ... )
        >>> belief_mask(rows, "ana").to_pylist()
        [True, False, True]
        >>> belief_mask(rows, "ana", ["undecided", "false"]).to_pylist()
        [True, True, False]
        >>> belief_mask(rows, None, ["true"]).to_pylist()
        [True, True, True]
    """
    states = check_belief_filter(beliefs)
    if isinstance(provenance, pa.ChunkedArray):
        provenance = provenance.combine_chunks()
    n = 0 if provenance is None else len(provenance)
    if believer is None:
        return pa.array(np.ones(n, dtype=bool), type=pa.bool_())
    if provenance is None or n == 0:
        return pa.array([], type=pa.bool_())
    # ``struct_field`` carries a null entry over to its beliefs.
    held = _struct_member(provenance, "beliefs")
    stated = pc.list_flatten(held)
    owners = _cast_column(pc.list_parent_indices(held), pa.int64())
    # Each row's state is the believer's belief where the row holds one,
    # else "undecided". An entry holds at most one belief per actor as
    # written, but two spellings of one actor (``ex:ana`` and the IRI it
    # expands to) name one believer only under *context*: the row then
    # takes the most critical of them, in :data:`_CRITICAL_FIRST` order.
    own = _filled(
        _names_actor(_struct_member(stated, "actor"), believer, context),
        pa.scalar(False),
    )
    beliefs_held = _struct_member(stated, "belief").filter(own)
    # One rank per row, the most critical belief held; a row holding none
    # keeps the rank past the last belief, which reads as "undecided".
    rank = np.full(n, len(_CRITICAL_FIRST), dtype=np.int64)
    np.minimum.at(
        rank,
        owners.filter(own).to_numpy(),
        pc.index_in(beliefs_held, value_set=pa.array(_CRITICAL_FIRST)).to_numpy(),
    )
    state = np.array([*_CRITICAL_FIRST, "undecided"], dtype=object)[rank]
    kept = np.isin(state, sorted(states))
    return pa.array(kept, type=pa.bool_())


def _names_actor(
    actors: pa.Array[Any], believer: str, context: Mapping[str, Any] | None
) -> pa.BooleanArray:
    """Per actor, whether it is *believer*, both compared as expanded IRIs.

    Each distinct actor is expanded once (:func:`actor_iri`), so the cost
    follows the number of actors, not of beliefs.
    """
    distinct = pc.unique(actors)
    names = pa.array(
        [
            None if actor is None else actor_iri(actor, context)
            for actor in distinct.to_pylist()
        ],
        pa.string(),
    )
    return _compared(
        "equal",
        pc.take(names, pc.index_in(actors, value_set=distinct)),
        pa.scalar(actor_iri(believer, context), pa.string()),
    )


# endregion
