"""Where MatchClaims are held: claim ids, admission and the claims in force.

Claim ids are minted, given or refused when a table or a bundle admits
claims; tables are joined and concatenated; an
:class:`~timetoalign.alignment.bundle.AlignmentBundle` keeps its MatchClaims
in force per belief filter and answers its claim queries from them.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Collection, Iterable, Sequence
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
from timetoshare import DEFAULT_BELIEFS, BeliefState, check_belief_filter

from timetoalign.alignment.filters import ClaimFilter, belief_context, meet
from timetoalign.core import TimeUnit
from timetoalign.core.enums import Domain
from timetoalign.core.fields import _compared, _filled, _null_column, _struct_child
from timetoalign.core.ids import (
    CLAIM_PREFIX,
    IdMinter,
    content_minter,
    quote_id,
    table_seed_part,
)
from timetoalign.core.provenance import Belief, BeliefValue
from timetoalign.timelines.statements import _day

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.alignment.claims import MatchClaim, MatchClaimField


def _refuse_held_claim_ids(ids: pa.Array, taken: pa.Array, *, holder: str) -> None:
    """Refuse claim ids already held, or that newcomers hold twice.

    A claim's id never changes once it holds one, and a MatchClaim is
    addressed by its id alone, so an id the holder already holds, or one
    held by two newcomers, is refused, whether it was given or minted.
    Vectorized: the cost is that of hashing the ids, whatever their number.

    Args:
        ids: The newcomers' ids (a string array).
        taken: The ids the holder already holds.
        holder: What holds the ids, for the message (``"bundle"``,
            ``"field"``).

    Raises:
        ValueError: If an id is already held or held twice.
    """
    ids = pc.cast(ids, pa.string())
    present = ids.filter(pc.is_valid(ids))
    held = present.filter(pc.is_in(present, value_set=pc.cast(taken, pa.string())))
    if len(held):
        raise ValueError(
            f"claim id {quote_id(held[0].as_py())} is already held by another "
            f"claim of this {holder}"
        )
    counts = pc.value_counts(present)
    repeated = counts.filter(_compared("greater", _struct_child(counts, "counts"), 1))
    if len(repeated):
        raise ValueError(
            f"claim id {quote_id(repeated.field('values')[0].as_py())} is held "
            f"by {repeated.field('counts')[0].as_py()} claims"
        )


def _minted_claim_ids(
    struct_array: pa.StructArray, minter: IdMinter | None
) -> pa.StructArray:
    """Return the claim rows *struct_array* with an id minted for each row without one.

    The ids are drawn from *minter* in row order (``claim_…``); without a
    minter, from the content minter of the rows
    (:func:`~timetoalign.core.ids.content_minter` of ``"claim"`` and the
    Arrow IPC bytes of every column of the rows but ``id``). Given ids are
    kept.
    """
    ids = pc.cast(struct_array.field("id"), pa.string())
    count = ids.null_count
    if count == 0:
        return struct_array
    if minter is None:
        rows = pa.Table.from_batches(
            [pa.RecordBatch.from_struct_array(struct_array)]
        ).drop_columns(["id"])
        minter = content_minter(CLAIM_PREFIX, table_seed_part(rows))
    filled = pc.replace_with_mask(
        ids, pc.is_null(ids), minter.mint_many(CLAIM_PREFIX, count)
    )
    fields = list(struct_array.type)
    return pa.StructArray.from_arrays(
        [
            filled if field.name == "id" else struct_array.field(field.name)
            for field in fields
        ],
        fields=fields,
    )


def _given_or_seeded_claim_id(claim_id: str | None, seed: int | None) -> str | None:
    """*claim_id*, else with a *seed* the first id of ``IdMinter(seed)``.

    None leaves the claim to mint its id from its content.
    """
    if claim_id is not None or seed is None:
        return claim_id
    return IdMinter(seed).mint(CLAIM_PREFIX)


def _join_claims(
    claims: Sequence["MatchClaim"], taken: pa.Array, *, holder: str
) -> list["MatchClaim"]:
    """Return *claims* as they join a holder of the ids *taken*.

    Every claim holds its id from construction, and joining mints none: a
    claim whose id the holder holds, or two claims under one id, are
    refused before anything changes (:func:`_refuse_held_claim_ids`). A
    claim that belongs to no bundle joins as itself; a claim a bundle
    already holds joins as a copy, so the bundle it came from keeps its
    back-reference.

    Args:
        claims: The newcomers.
        taken: The ids the holder already holds.
        holder: What holds the ids, for the message.

    Returns:
        The claims to store, in order.
    """
    _refuse_held_claim_ids(
        pa.array([claim.id for claim in claims], type=pa.string()),
        pc.cast(taken, pa.string()),
        holder=holder,
    )
    return [claim if claim._bundle is None else claim.model_copy() for claim in claims]


#: A belief filter as the bundle's caches key it: ``(believer, beliefs)``,
#: the states a frozen set (:func:`_belief_key`).
_BeliefKey = tuple["str | None", frozenset[BeliefState]]


#: The belief filter that filters nothing (no believer).
_ALL_BELIEFS: _BeliefKey = (None, frozenset(DEFAULT_BELIEFS))


def _belief_key(believer: str | None, beliefs: Collection[str]) -> _BeliefKey:
    """The cache key of a belief filter, its states checked.

    Without a believer nothing is filtered, whatever the states, so every
    such filter is :data:`_ALL_BELIEFS`.

    Raises:
        ValueError: If *beliefs* is not a collection of belief states.
    """
    states = check_belief_filter(beliefs)
    return _ALL_BELIEFS if believer is None else (believer, states)


def _refuse_shared_claim_ids(bundles: Sequence["AlignmentBundle"]) -> None:
    """Refuse bundles to merge two of which hold a MatchClaim under one id.

    A MatchClaim is addressed by its id alone, so the merge of two bundles
    each holding ``claim_1`` would hold two MatchClaims under one address.
    Vectorized: one hash count over every id.

    Raises:
        ValueError: Naming the first id held twice and the two bundles
            holding it, by id and by their place in *bundles*.
    """
    held = [bundle.match_claim_table._ids for bundle in bundles]
    if sum(len(ids) for ids in held) < 2:
        return
    ids = pc.cast(pa.concat_arrays(held), pa.string())
    counts = pc.value_counts(ids.filter(pc.is_valid(ids)))
    repeated = counts.filter(_compared("greater", _struct_child(counts, "counts"), 1))
    if not len(repeated):
        return
    claim_id = repeated.field("values")[0].as_py()
    places = [
        index
        for index, ids in enumerate(held)
        if pc.any(_filled(pc.equal(ids, claim_id), pa.scalar(False))).as_py()
    ]
    first, second = places[0], places[1]
    raise ValueError(
        f"MatchClaim id {quote_id(claim_id)} is held by bundle "
        f"{quote_id(bundles[first].id)} and by bundle "
        f"{quote_id(bundles[second].id)} (bundles {first} and {second} of the "
        f"merge): a MatchClaim is addressed by its id alone, so the two merge "
        f"once one of them states its MatchClaims under other ids."
    )


def _concatenated(self: MatchClaimField, other: "MatchClaimField") -> "MatchClaimField":
    """Implement :meth:`MatchClaimField._concatenated`."""
    if len(self) == 0:
        return other
    if len(other) == 0:
        return self
    if self._provenance == other._provenance:
        provenance = self._provenance
        parts = [self._struct_array(), other._struct_array()]
    else:
        provenance = None
        parts = [self._rows_with_provenance(), other._rows_with_provenance()]
    return self._from_struct_array(
        pa.concat_arrays(parts),
        provenance=provenance,
        further=self._further_concatenated(other),
    )


def _further_concatenated(
    self: MatchClaimField, other: "MatchClaimField"
) -> pa.Table | None:
    """Implement :meth:`MatchClaimField._further_concatenated`."""
    if self._further is None and other._further is None:
        return None
    names: dict[str, pa.DataType] = {}
    for further in (self._further, other._further):
        if further is None:
            continue
        for column in further.schema:
            held = names.setdefault(column.name, column.type)
            if held != column.type:
                raise ValueError(
                    f"The further column {column.name!r} of the MatchClaim "
                    f"tables joined has the types {held} and {column.type}."
                )

    def full(further: pa.Table | None, n: int) -> list[pa.Array]:
        """Return one array per joined further column, *n* nulls where *further* lacks it."""
        return [
            (
                further.column(name).combine_chunks()
                if further is not None and name in further.column_names
                else _null_column(n, kind)
            )
            for name, kind in names.items()
        ]

    columns = zip(
        full(self._further, len(self)),
        full(other._further, len(other)),
        strict=True,
    )
    return pa.table(
        {
            name: pa.concat_arrays(list(pair))
            for name, pair in zip(names, columns, strict=True)
        }
    )


def _add_match_claims(
    self: AlignmentBundle, claims: Iterable[MatchClaim]
) -> "AlignmentBundle":
    """Implement :meth:`AlignmentBundle.add_match_claims`."""
    from timetoalign.alignment.claims import MatchClaim, MatchClaimField

    claims = list(claims)
    for claim in claims:
        if not isinstance(claim, MatchClaim):
            raise TypeError(
                f"add_match_claims takes MatchClaims, not {type(claim).__name__}"
            )
    if not claims:
        return self
    self._check_rows(MatchClaimField.from_claims(claims))
    joined = _join_claims(claims, self.match_claim_table._ids, holder="bundle")
    for claim in joined:
        claim.set_bundle(self)
    self._append_rows(MatchClaimField.from_claims(joined))
    return self


def _add_match_claim_belief(
    self: AlignmentBundle,
    claim_id: str,
    *,
    actor: str,
    belief: BeliefValue,
    date: str | datetime.date | None = None,
    note: str | None = None,
) -> "AlignmentBundle":
    """Implement :meth:`AlignmentBundle.add_match_claim_belief`."""
    held = Belief(actor=actor, belief=belief, date=_day(date), note=note)
    table = self.match_claim_table
    held_rows = pc.indices_nonzero(
        _filled(_compared("equal", table._ids, claim_id), pa.scalar(False))
    )
    # An index is never null.
    rows = [row for row in held_rows.to_pylist() if row is not None]
    if not rows:
        raise KeyError(
            f"Bundle {quote_id(self.id)} holds no MatchClaim "
            f"{quote_id(claim_id)} ({len(table)} MatchClaims)."
        )
    self.match_claim_table = table._with_belief_at(rows[0], held, belief_context(self))
    self._invalidate_warp_cache()
    return self


def _check_rows(self: AlignmentBundle, rows: MatchClaimField) -> None:
    """Implement :meth:`AlignmentBundle._check_rows`."""
    known = self._known_timelines()
    named = rows.timeline_ids
    unknown = sorted(named.difference(known))
    if unknown:
        raise KeyError(
            f"MatchClaims name timeline(s) {unknown} that this bundle does "
            f"not hold. Available timelines: {list(known)}"
        )
    rows.check_on_timelines({timeline_id: known[timeline_id] for timeline_id in named})


def _claims_in_force(
    self: AlignmentBundle, belief_key: _BeliefKey = _ALL_BELIEFS
) -> MatchClaimField:
    """Implement :meth:`AlignmentBundle._claims_in_force`."""
    if belief_key == _ALL_BELIEFS:
        return self.match_claim_table
    in_force = self._in_force_cache.get(belief_key)
    if in_force is None:
        believer, beliefs = belief_key
        in_force = self.match_claim_table.filter(
            believer=believer,
            beliefs=beliefs,
            context=belief_context(self),
        )
        self._in_force_cache[belief_key] = in_force
    return in_force


def _known_id_pattern(self: AlignmentBundle, id_pattern: str | None) -> str | None:
    """Implement :meth:`AlignmentBundle._known_id_pattern`."""
    if id_pattern is None:
        return None

    pattern = re.compile(id_pattern)
    matched = {
        timeline_id
        for timeline_id in self._known_timelines()
        if pattern.search(timeline_id)
    }
    if not matched:
        return r"(?!)"
    alternatives = "|".join(re.escape(timeline_id) for timeline_id in matched)
    return rf"^(?:{alternatives})\Z"


def _get_match_claims(
    self: AlignmentBundle,
    *,
    timeline_id: str | None = None,
    timeline_ids: set[str] | None = None,
    id_pattern: str | None = None,
    between: tuple[str, str] | None = None,
    within: set[str] | None = None,
    synchronous_only: bool = False,
    nomatch_only: bool = False,
    include_domains: set["Domain"] | None = None,
    include_units: set["TimeUnit"] | None = None,
    believer: str | None = None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
) -> list[MatchClaim]:
    """Implement :meth:`AlignmentBundle.get_match_claims`."""
    return self._materialized(
        self.get_claim_field(
            timeline_id=timeline_id,
            timeline_ids=timeline_ids,
            id_pattern=id_pattern,
            between=between,
            within=within,
            synchronous_only=synchronous_only,
            nomatch_only=nomatch_only,
            include_domains=include_domains,
            include_units=include_units,
            believer=believer,
            beliefs=beliefs,
        )
    )


def _get_claim_field(
    self: AlignmentBundle,
    *,
    timeline_id: str | None = None,
    timeline_ids: set[str] | None = None,
    id_pattern: str | None = None,
    between: tuple[str, str] | None = None,
    within: set[str] | None = None,
    synchronous_only: bool = False,
    nomatch_only: bool = False,
    include_domains: set["Domain"] | None = None,
    include_units: set["TimeUnit"] | None = None,
    believer: str | None = None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
) -> MatchClaimField:
    """Implement :meth:`AlignmentBundle.get_claim_field`."""
    filt = self._claim_filter(
        timeline_id=timeline_id,
        timeline_ids=timeline_ids,
        id_pattern=id_pattern,
        between=between,
        within=within,
        synchronous_only=synchronous_only,
        nomatch_only=nomatch_only,
        include_domains=include_domains,
        include_units=include_units,
        believer=believer,
        beliefs=beliefs,
    )
    if filt is None:
        return self.match_claim_table._empty()
    # The belief filter is the one shared point's; the rest filters what
    # it keeps.
    return self._claims_in_force(_belief_key(filt.believer, filt.beliefs)).filter(
        timeline_id=filt.timeline_id,
        timeline_ids=filt.timeline_ids,
        id_pattern=filt.id_pattern,
        between=filt.between,
        within=meet(
            filt.within, filt.domain_unit_timeline_ids(self._known_timelines())
        ),
        synchronous_only=filt.synchronous_only,
        nomatch_only=filt.nomatch_only,
    )


def _claim_filter(
    self: AlignmentBundle,
    *,
    timeline_id: str | None,
    timeline_ids: set[str] | None,
    id_pattern: str | None,
    between: tuple[str, str] | None,
    within: set[str] | None,
    synchronous_only: bool,
    nomatch_only: bool,
    include_domains: set["Domain"] | None,
    include_units: set["TimeUnit"] | None,
    believer: str | None,
    beliefs: Collection[BeliefState],
) -> ClaimFilter | None:
    """Implement :meth:`AlignmentBundle._claim_filter`."""
    known = self._known_timelines()
    if timeline_id is not None and timeline_id not in known:
        return None
    if between is not None and any(between_id not in known for between_id in between):
        return None
    return ClaimFilter.from_kwargs(
        timeline_id=timeline_id,
        timeline_ids=(
            {selected for selected in timeline_ids if selected in known}
            if timeline_ids is not None
            else None
        ),
        id_pattern=self._known_id_pattern(id_pattern),
        between=between,
        within=within,
        synchronous_only=synchronous_only,
        nomatch_only=nomatch_only,
        include_domains=include_domains,
        include_units=include_units,
        believer=believer,
        beliefs=beliefs,
    )
