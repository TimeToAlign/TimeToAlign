"""Statements about what a bundle, a timeline or a region stands for.

A TimeToShare manifest describes the things its elements stand for — a
work, a recording, a printed edition, a performer — by property–value
pairs: the bundle (the manifest itself), every timeline at any depth and
every region may hold ``statements`` (``{property: [value, ...]}``) and
``provenance``, the entries saying how a pair was made (who stated it,
when, how, on what basis) and who believes it. A value naming an element
of the manifest is the node ``{"id": "#cpt_1"}``
(:func:`timetoshare.to_file_id` builds the reference); the element's own
``id`` stays ``cpt_1``. Statements are the same in a file and in memory.

A pair listed in ``statements`` is stated by the manifest; a pair held only
in ``provenance`` is a **candidate**, offered for someone to state. A
person who states a pair is recorded with the belief ``"true"``
(``actor=``); a loader or a service proposing a candidate records how it
was made and holds no belief. Every actor may record a belief in any pair,
``"true"``, ``"false"`` or ``"unlikely"``: believing a stated pair
``"false"`` records that belief and never removes the pair, and a candidate
an actor believes ``"false"`` is not offered to that actor again. The
belief filter (``believer=``, ``beliefs=``) is decided by
:func:`~timetoalign.core.provenance.belief_mask`, as for MatchClaims.

Everything about the pairs themselves is TimeToShare's
(:mod:`timetoshare.statements`, :mod:`timetoshare.catalogue`): matching,
adding, removing, candidates, beliefs, the rectangles a catalogue reads,
the prefixes used. This module adds where statements live:
:class:`StatementsMixin` is the one read surface of
:class:`~timetoalign.timelines.Timeline`,
:class:`~timetoalign.timelines.Region` and
:class:`~timetoalign.alignment.AlignmentBundle`; the mutable timeline and
bundle change their statements in place (:class:`MutableStatementsMixin`),
the frozen region returns a copy (:meth:`Region.with_statement
<timetoalign.timelines.Region.with_statement>` and its siblings).

A holder declares the prefixes its statements need beyond the catalogue's
(:meth:`MutableStatementsMixin.add_prefix`); a manifest written carries in
its inline ``@context`` exactly the prefixes its statements use. The
library keeps references true: an element given a new ID is named by it,
and a pair naming an element that leaves the manifest is dropped with its
provenance.

Examples:
    >>> from timetoalign.timelines import ContinuousPhysicalTimeline, DiscretePhysicalTimeline
    >>> video = ContinuousPhysicalTimeline(length=None, uid="cpt_1")
    >>> _ = video.add_statement("type", "schema:VideoObject")
    >>> _ = video.add_verb("performed_by", "wikidata", "wd:Q235650", label="Maria João Pires")
    >>> video.statements
    {'type': ['schema:VideoObject'], 'wdt:P175': [{'id': 'wd:Q235650', 'label': 'Maria João Pires'}]}
    >>> [(rectangle.verb, rectangle.label) for rectangle in video.get_rectangles()]
    [('is_a', None), ('performed_by', 'Maria João Pires')]
    >>> editor = "https://example.org/people/editor"
    >>> wav = DiscretePhysicalTimeline(length=1470868, uid="dpt_1")
    >>> _ = wav.add_candidate("dcterms:source", {"id": "#cpt_1"},
    ...     stated_by="https://example.org/agents/aligner", method="inferred", date="2026-11-18")
    >>> wav.statements, [entry["method"] for entry in wav.get_candidates()]
    ({}, ['inferred'])
    >>> _ = wav.add_statement("dcterms:source", {"id": "#cpt_1"}, actor=editor, date="2026-11-19")
    >>> wav.statements, wav.get_candidates()
    ({'dcterms:source': [{'id': '#cpt_1'}]}, [])
    >>> wav.provenance[0]["beliefs"]
    [{'actor': 'https://example.org/people/editor', 'belief': 'true', 'date': '2026-11-19'}]
    >>> _ = wav.add_belief("dcterms:source", {"id": "#cpt_1"}, actor="https://example.org/people/reviewer",
    ...     belief="false", date="2026-11-20")
    >>> wav.statements, wav.get_provenance(believer="https://example.org/people/reviewer")
    ({'dcterms:source': [{'id': '#cpt_1'}]}, [])
    >>> from timetoalign.timelines import ContinuousLogicalTimeline
    >>> score = ContinuousLogicalTimeline(length=27, uid="clt_6")
    >>> intro = score.create_region(0, 12, id="region_1")
    >>> described = intro.with_statement("rdfs:comment", "the opening")
    >>> intro.statements, described.statements
    ({}, {'rdfs:comment': ['the opening']})
    >>> _ = score.replace_region(described)
    >>> score.get_region("region_1").statements
    {'rdfs:comment': ['the opening']}
    >>> score.add_statement("foaf:maker", {"id": "wd:Q4700"})
    Traceback (most recent call last):
    ...
    ValueError: the prefix 'foaf' is declared neither by the catalogue nor on timeline 'clt_6'
    >>> _ = score.add_prefix("mbw", "http://musicbrainz.org/work/")
    >>> _ = score.add_statement("schema:exampleOfWork", {"id": "mbw:8d331505-4d88-39ae-81c7-bec5da77af96"})
    >>> score.prefixes
    {'mbw': 'http://musicbrainz.org/work/'}
"""

from __future__ import annotations

import copy
import datetime
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any, Final

import timetoshare
from timetoshare import (
    DEFAULT_BELIEFS,
    FORMAT_CONTEXT,
    BeliefState,
    Catalogue,
    Rectangle,
    check_belief_filter,
    load_catalogue,
)
from typing_extensions import Self

from ..core.provenance import (
    Belief,
    BeliefValue,
    Provenance,
    belief_mask,
    provenance_array,
)

__all__ = ["LIBRARY_AGENT", "MutableStatementsMixin", "NodePath", "StatementsMixin"]

LIBRARY_AGENT: Final[str] = "https://timetoalign.github.io/"
"""The agent an importer of this library names as ``stated_by`` by default."""

NodePath = tuple[str | int, ...]
"""A value node inside a holder's statements: property, index, property, ...

``()`` is the holder itself; ``("wdt:P629", 0)`` the first value of
``wdt:P629``; ``("wdt:P629", 0, "schema:sameAs", 1)`` a node inside that one.
"""

STATE_KEYS: Final[tuple[str, str]] = ("statements", "provenance")
"""The two keys of a manifest object that hold its statements."""

PAIR_KEYS: Final[tuple[str, str]] = ("property", "value")
"""The keys a statement's provenance entry adds to the entry of the format."""

# The day a candidate or a belief is dated when no date is given; a module
# attribute, so that a test can fix the day.
_today = datetime.date.today


# region State


def read_statement_state(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return the statements and provenance of a manifest object, as written."""
    return {key: copy.deepcopy(data[key]) for key in STATE_KEYS if key in data}


def declared_prefixes(declared: Mapping[str, Any]) -> dict[str, str]:
    """Return the prefixes of *declared* the catalogue lacks or gives another namespace.

    *declared* is a ``@context`` without the format's terms; an entry whose
    value is no namespace string declares no prefix.
    """
    catalogue = load_catalogue().prefixes
    return {
        prefix: namespace
        for prefix, namespace in declared.items()
        if isinstance(namespace, str) and namespace != catalogue.get(prefix)
    }


def needed_prefixes(
    state: Mapping[str, Any], declared: Mapping[str, str]
) -> dict[str, str]:
    """Return the prefixes of *declared* the statements of *state* use.

    Only those the catalogue lacks or gives another namespace
    (:func:`declared_prefixes`): a holder keeps the prefixes its pairs need
    beyond the catalogue's.
    """
    own = declared_prefixes(declared)
    return {
        prefix: own[prefix]
        for prefix in timetoshare.used_prefixes(dict(state))
        if prefix in own
    }


def renamed_state(
    state: Mapping[str, Any], renamed: Mapping[str, str | None]
) -> dict[str, Any]:
    """Return *state* with the elements it names renamed or dropped.

    *renamed* maps bare element IDs to their new IDs, or to None for an
    element that left the manifest: a value naming it leaves its list, an
    entry of such a value leaves the provenance, and a ``basis`` or
    ``stated_by`` item naming it leaves its list
    (:func:`timetoshare.with_references_renamed`).
    """
    if not state:
        return {}
    result = timetoshare.with_references_renamed(dict(state), renamed)
    return {key: result[key] for key in STATE_KEYS if key in result}


def statement_holders(roots: Iterable[Any], scopes: Iterable[Any]) -> list[Any]:
    """Return everything holding statements in the manifest of *roots* and *scopes*.

    Every timeline of each root's tree (each renaming its regions), the
    manifest each root was read from as a whole, and what holds the
    statements of each group's and bundle's own manifest; each once.
    """
    holders: list[Any] = []
    for root in roots:
        holders.extend((root, *(child for _, child in root._iter_descendants())))
        if root._manifest is not None:
            holders.append(root._manifest)
    holders.extend(
        holder for scope in scopes if (holder := scope._statements_holder()) is not None
    )
    return list({id(holder): holder for holder in holders}.values())


def rename_references(
    roots: Iterable[Any], scopes: Iterable[Any], renamed: Mapping[str, str | None]
) -> None:
    """Rename or drop the elements the statements of *roots* and *scopes* name.

    Every holder of :func:`statement_holders` applies *renamed* at once, so
    ``{"a": "b", "b": "c"}`` renames each reference once. Pairs mapping an
    id to itself are passed over.
    """
    renamed = {old: new for old, new in renamed.items() if old != new}
    if not renamed:
        return
    for holder in statement_holders(roots, scopes):
        holder._rename_statement_references(renamed)


def _subject(node: NodePath) -> tuple[str | int, ...]:
    """The path of a holder's value node in its one-subject document."""
    return ("statements", *node) if node else ()


def _agents(stated_by: str | Sequence[str]) -> list[str]:
    """``stated_by`` as the file writes it: a list of IRIs."""
    return [stated_by] if isinstance(stated_by, str) else list(stated_by)


def _day(date: str | datetime.date | None) -> str:
    """A date as the file writes it, today when None."""
    if date is None:
        return _today().isoformat()
    return date.isoformat() if isinstance(date, datetime.date) else date


def _entry_members(provenance: Provenance | Mapping[str, Any]) -> dict[str, Any]:
    """The members of a pair's entry *provenance* gives, beliefs included.

    A mapping is read as :class:`~timetoalign.core.provenance.Provenance`
    reads one, so a statement's entry is checked as a MatchClaim's is.

    Raises:
        ValueError: If a key is unknown or a value invalid.
    """
    if not isinstance(provenance, Provenance):
        provenance = Provenance.from_dict(dict(provenance))
    return provenance.to_dict()


def _actor_belief(actor: str | None, date: str | datetime.date | None) -> Belief | None:
    """The belief ``"true"`` of the person *actor* stating a pair, or None.

    Raises:
        ValueError: If *date* is given without *actor*, or is no calendar
            date.
    """
    if actor is None:
        if date is not None:
            raise ValueError(
                "a date dates the belief of the actor stating the pair; give actor="
            )
        return None
    return Belief(actor=actor, belief="true", date=_day(date))


def _believed(
    document: dict[str, Any],
    subject: tuple[str | int, ...],
    property: str,
    value: Any,
    held: Belief,
) -> dict[str, Any]:
    """*document* with *held* recorded in the pair's entry (:func:`timetoshare.with_belief`)."""
    return timetoshare.with_belief(
        document,
        subject,
        property,
        value,
        actor=held.actor,
        belief=held.belief,
        date=held.date,
        note=held.note,
    )


def _stating(
    document: dict[str, Any],
    subject: tuple[str | int, ...],
    property: str,
    value: Any,
    members: Mapping[str, Any] | None,
    believed: Belief | None,
) -> dict[str, Any]:
    """*document* with the pair stated (:func:`timetoshare.with_pair`).

    The pair's entry takes *members* (an entry's members, as
    :func:`_entry_members` gives them), their beliefs recorded in order, and
    then *believed*, the stating actor's belief, which replaces an earlier
    one of that actor.
    """
    document = timetoshare.with_pair(
        document, subject, property, value, provenance=members
    )
    beliefs = [Belief.from_dict(held) for held in (members or {}).get("beliefs", ())]
    for held in [*beliefs, *([believed] if believed is not None else [])]:
        document = _believed(document, subject, property, value, held)
    return document


def _believing(
    entries: list[dict[str, Any]],
    context: Mapping[str, Any],
    believer: str | None,
    beliefs: Collection[BeliefState],
) -> list[dict[str, Any]]:
    """The *entries* the belief filter keeps for *believer*.

    The filter is :func:`~timetoalign.core.provenance.belief_mask`'s,
    which compares actors as their IRIs under *context*, so ``believer``
    may be compact or absolute.
    """
    if believer is None:
        # Without a believer every entry is kept; the states are still checked.
        check_belief_filter(beliefs)
        return entries
    rows: list[Provenance | None] = []
    for entry in entries:
        members = {key: item for key, item in entry.items() if key not in PAIR_KEYS}
        # An entry holding the pair alone states no provenance: a null row.
        stated = members.get("beliefs") or members.keys() - {"beliefs"}
        rows.append(Provenance.from_dict(members) if stated else None)
    mask = belief_mask(provenance_array(rows), believer, beliefs, context=context)
    return [
        entry for entry, kept in zip(entries, mask.to_pylist(), strict=True) if kept
    ]


# endregion


# region Mixins


class StatementsMixin:
    """The statements of a bundle, a timeline or a region, and their provenance.

    A holder stores exactly the file's two keys (``_statement_state``:
    ``statements`` and ``provenance``, each only when not empty) and the
    prefixes its pairs need beyond the catalogue's (``_prefixes``). Reading
    and every change run TimeToShare's functions on a one-subject document
    built from the holder's own state: its ``@context`` holds the format's
    terms, the catalogue's prefixes and the holder's own, and the holder is
    the subject ``()``.
    """

    _statement_state: Mapping[str, Any]
    _prefixes: Mapping[str, str]

    # region Hooks

    def _statement_state_view(self) -> Mapping[str, Any]:
        """Return the holder's statements and provenance, not to be changed."""
        return self._statement_state

    def _own_prefixes(self) -> Mapping[str, str]:
        """Return the prefixes the holder declares beyond the catalogue."""
        return self._prefixes

    def _statement_holder_name(self) -> str:
        """Return the holder as a message names it (``timeline 'cpt_1'``)."""
        raise NotImplementedError

    def _with_statement_state(
        self,
        state: Mapping[str, Any] | None = None,
        prefixes: Mapping[str, str] | None = None,
    ) -> Self:
        """Return the holder with a new state and prefixes (None keeps them).

        In place on a timeline or a bundle, a copy for a region.
        """
        raise NotImplementedError

    # endregion

    # region The core

    def _statement_context(self) -> dict[str, Any]:
        """Return the holder's ``@context``: terms, catalogue prefixes, own prefixes."""
        return {**FORMAT_CONTEXT, **load_catalogue().prefixes, **self._own_prefixes()}

    def _statement_document(self) -> dict[str, Any]:
        """Return a one-subject manifest whose subject ``()`` is the holder."""
        return {
            "@context": self._statement_context(),
            **copy.deepcopy(dict(self._statement_state_view())),
        }

    def _applied(
        self, edit: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> dict[str, Any]:
        """Run *edit* on the holder's document and return the new state.

        Nothing is stored: the caller stores what this returns.

        Raises:
            ValueError: If the new state uses a prefix the holder's context
                lacks (``the prefix 'foaf' is declared neither by the
                catalogue nor on timeline 'clt_6'``), or the TimeToShare
                function refuses the edit, in its words.
        """
        result = edit(self._statement_document())
        context = result["@context"]
        for prefix in timetoshare.used_prefixes(result):
            if prefix not in context:
                raise ValueError(
                    f"the prefix '{prefix}' is declared neither by the catalogue nor "
                    f"on {self._statement_holder_name()}"
                )
        return {key: result[key] for key in STATE_KEYS if key in result}

    # endregion

    # region Read side

    @property
    def statements(self) -> dict[str, list[Any]]:
        """The pairs stated, ``{property: [value, ...]}``, keys in code-point order.

        A copy; ``{}`` when nothing is stated.
        """
        held = self._statement_state_view().get("statements", {})
        return {key: copy.deepcopy(held[key]) for key in sorted(held)}

    @property
    def provenance(self) -> list[dict[str, Any]]:
        """The provenance entries, in their order; a copy, ``[]`` when none."""
        return copy.deepcopy(list(self._statement_state_view().get("provenance", [])))

    @property
    def prefixes(self) -> dict[str, str]:
        """The prefixes declared on this holder beyond the catalogue's, sorted."""
        return dict(sorted(self._own_prefixes().items()))

    def get_pairs(self, *, node: NodePath = ()) -> list[tuple[str, Any]]:
        """Return the pairs ``(property, value)`` of the holder or of a value node.

        Keys in code-point order, values in list order; each ``type`` item
        is a pair ``("type", iri)`` (:func:`timetoshare.pairs`).

        Args:
            node: The value node whose pairs to return; ``()`` the holder.

        Raises:
            ValueError: If *node* addresses no value node.
        """
        return timetoshare.pairs(self._statement_document(), _subject(node))

    def get_provenance(
        self,
        *,
        node: NodePath = (),
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> list[dict[str, Any]]:
        """Return the provenance entries of the holder or of a value node.

        Entries of stated pairs and candidates alike, in their order
        (:func:`timetoshare.entries`), kept by the belief filter.

        Args:
            node: The value node whose entries to return; ``()`` the holder.
            believer: The actor whose beliefs filter, an IRI (compact or
                absolute); every entry when None.
            beliefs: The states kept: the entries of pairs whose state for
                the believer (``"true"``, ``"undecided"`` where the believer
                holds no belief, ``"unlikely"``, ``"false"``) is one of
                them; default all but ``"false"``.

        Raises:
            ValueError: If *node* addresses no value node, or *beliefs* is
                not a collection of belief states.
        """
        document = self._statement_document()
        return _believing(
            timetoshare.entries(document, _subject(node)),
            document["@context"],
            believer,
            beliefs,
        )

    def get_candidates(
        self,
        *,
        node: NodePath = (),
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> list[dict[str, Any]]:
        """Return the candidates: the entries of pairs the holder does not state.

        A candidate is offered for someone to state (:meth:`add_statement`
        with the same pair). With *believer* and the default *beliefs*, a
        candidate the believer believes ``"false"`` is left out: it is not
        offered to that actor again (:func:`timetoshare.candidates` with
        ``actor=``).

        Args:
            node: The value node whose candidates to return; ``()`` the
                holder.
            believer: The actor whose beliefs filter, an IRI (compact or
                absolute); every candidate when None.
            beliefs: The states kept, as for :meth:`get_provenance`.

        Raises:
            ValueError: If *node* addresses no value node, or *beliefs* is
                not a collection of belief states.
        """
        document = self._statement_document()
        return _believing(
            timetoshare.candidates(document, _subject(node)),
            document["@context"],
            believer,
            beliefs,
        )

    def get_rectangles(self, catalogue: Catalogue | None = None) -> list[Rectangle]:
        """Return the holder's pairs gathered by verb: one verb about one thing.

        Pairs a catalogue reads as one verb about one thing, in whatever
        vocabularies, form one rectangle; the ``is_a`` rectangle comes
        first (:func:`timetoshare.rectangles`).

        Args:
            catalogue: The catalogue to read the pairs with; the shipped one
                when None.
        """
        return timetoshare.rectangles(
            catalogue or load_catalogue(), self._statement_document(), ()
        )

    # endregion

    # region Changes, shared by both faces

    def _state_with_statement(
        self,
        property: str,
        value: Any,
        *,
        node: NodePath,
        provenance: Provenance | Mapping[str, Any] | None,
        actor: str | None,
        date: str | datetime.date | None,
    ) -> dict[str, Any]:
        members = None if provenance is None else _entry_members(provenance)
        believed = _actor_belief(actor, date)
        return self._applied(
            lambda document: _stating(
                document, _subject(node), property, value, members, believed
            )
        )

    def _state_without_statement(
        self, property: str, value: Any, *, node: NodePath
    ) -> dict[str, Any]:
        return self._applied(
            lambda document: timetoshare.without_pair(
                document, _subject(node), property, value
            )
        )

    def _state_with_candidate(
        self,
        property: str,
        value: Any,
        *,
        stated_by: str | Sequence[str],
        method: str,
        date: str | datetime.date | None,
        service: str | None,
        basis: Sequence[str],
        match: str | None,
        note: str | None,
        node: NodePath,
    ) -> dict[str, Any]:
        day = _day(date)
        entry = Provenance(
            stated_by=tuple(_agents(stated_by)),
            date=day,
            method=method,
            service=service,
            basis=tuple(basis) or None,
            match=match,
            note=note,
        )
        return self._applied(
            lambda document: timetoshare.with_candidate(
                document,
                _subject(node),
                property,
                value,
                stated_by=list(entry.stated_by or ()),
                date=day,
                method=method,
                service=entry.service,
                basis=list(entry.basis or ()),
                match=entry.match,
                note=entry.note,
            )
        )

    def _state_with_belief(
        self,
        property: str,
        value: Any,
        *,
        actor: str,
        belief: BeliefValue,
        date: str | datetime.date | None,
        note: str | None,
        node: NodePath,
    ) -> dict[str, Any]:
        held = Belief(actor=actor, belief=belief, date=_day(date), note=note)
        return self._applied(
            lambda document: _believed(document, _subject(node), property, value, held)
        )

    def _state_with_verb(
        self,
        verb: str,
        system: str,
        value: str,
        *,
        label: str | None,
        slots: Mapping[str, Any] | None,
        alternative: int,
        provenance: Provenance | Mapping[str, Any] | None,
        actor: str | None,
        date: str | datetime.date | None,
        catalogue: Catalogue | None,
    ) -> dict[str, Any]:
        members = None if provenance is None else _entry_members(provenance)
        believed = _actor_belief(actor, date)
        pairs = timetoshare.instantiate(
            catalogue or load_catalogue(),
            verb,
            system,
            value,
            alternative=alternative,
            label=label,
            slots=slots,
        )

        def edit(document: dict[str, Any]) -> dict[str, Any]:
            """Return *document* with each property–value pair of the verb stated on it."""
            for property, pair_value in pairs:
                document = _stating(
                    document, (), property, pair_value, members, believed
                )
            return document

        return self._applied(edit)

    def _prefixes_with(self, name: str, namespace: str) -> dict[str, str]:
        """Return the holder's prefixes with *name* declared as *namespace*.

        Raises:
            ValueError: If the format refuses the declaration, or the holder
                declares *name* already with another namespace.
        """
        problems = timetoshare.statement_problems(
            {"@context": {**FORMAT_CONTEXT, name: namespace}}
        )
        if problems:
            raise ValueError(
                f"cannot declare the prefix '{name}' as '{namespace}': {problems[0]}"
            )
        held = dict(self._own_prefixes())
        if name in held:
            if held[name] != namespace:
                raise ValueError(
                    f"the prefix '{name}' already stands for '{held[name]}' on "
                    f"{self._statement_holder_name()}"
                )
            return held
        if load_catalogue().prefixes.get(name) == namespace:
            return held
        return dict(sorted({**held, name: namespace}.items()))

    # endregion


class MutableStatementsMixin(StatementsMixin):
    """Statements changed in place: a timeline's and a bundle's.

    Every method returns the holder, for chaining. A pair whose property
    or value uses a prefix the holder's context lacks is refused before
    anything changes; :meth:`add_prefix` declares one.
    """

    def _with_statement_state(
        self,
        state: Mapping[str, Any] | None = None,
        prefixes: Mapping[str, str] | None = None,
    ) -> Self:
        """Store a new state and prefixes (None keeps them) and return the holder."""
        self._set_statement_state(
            self._statement_state if state is None else state,
            self._prefixes if prefixes is None else prefixes,
        )
        return self

    def _set_statement_state(
        self, state: Mapping[str, Any], prefixes: Mapping[str, str]
    ) -> None:
        """Store the holder's statements, provenance and prefixes."""
        self._statement_state = dict(state)
        self._prefixes = dict(prefixes)

    def _rename_statement_references(self, renamed: Mapping[str, str | None]) -> None:
        """Rename or drop the elements the holder's statements name, in place.

        See :func:`renamed_state`.
        """
        if self._statement_state:
            self._statement_state = renamed_state(self._statement_state, renamed)

    def add_statement(
        self,
        property: str,
        value: Any,
        *,
        node: NodePath = (),
        provenance: Provenance | Mapping[str, Any] | None = None,
        actor: str | None = None,
        date: str | datetime.date | None = None,
    ) -> Self:
        """State the pair ``(property, value)``.

        A candidate of the pair becomes the stated pair's entry, its
        beliefs kept (:func:`timetoshare.with_pair`). *provenance* sets the
        entry's members; *actor*, the person stating the pair, is recorded
        with the belief ``"true"``, replacing that actor's earlier belief.
        A loader or a service gives *provenance* alone and holds no belief.

        Args:
            property: ``type`` ("is a"), a compact IRI or an absolute IRI.
            value: A JSON value of the file's form: a literal, a value
                object, a node (``{"id": ...}``) or ``{"list": [...]}``;
                ``type`` takes a class IRI.
            node: The value node to state the pair of; ``()`` the holder.
            provenance: How the pair was made, a
                :class:`~timetoalign.core.provenance.Provenance` or its
                dictionary (``stated_by``, ``date``, ``method``, ...);
                its ``beliefs`` are recorded too.
            actor: The person stating the pair, an IRI.
            date: The day of *actor*'s belief, ``YYYY-MM-DD`` or a
                :class:`datetime.date`; today when None.

        Raises:
            ValueError: If the pair is stated already (``'wdt:P175' already
                has this value``), the property is no property, *node*
                addresses no value node, *provenance* is no provenance
                entry, *date* is given without *actor*, or a prefix is
                declared neither by the catalogue nor on the holder.
        """
        return self._with_statement_state(
            self._state_with_statement(
                property,
                value,
                node=node,
                provenance=provenance,
                actor=actor,
                date=date,
            )
        )

    def remove_statement(
        self, property: str, value: Any, *, node: NodePath = ()
    ) -> Self:
        """Remove the pair ``(property, value)`` and its entry.

        Removing is an edit of its own: no belief removes a pair.

        Raises:
            ValueError: If the pair is not stated (``'type' does not have
                this value``) or *node* addresses no value node.
        """
        return self._with_statement_state(
            self._state_without_statement(property, value, node=node)
        )

    def add_candidate(
        self,
        property: str,
        value: Any,
        *,
        stated_by: str | Sequence[str],
        method: str,
        date: str | datetime.date | None = None,
        service: str | None = None,
        basis: Sequence[str] = (),
        match: str | None = None,
        note: str | None = None,
        node: NodePath = (),
    ) -> Self:
        """Offer the pair ``(property, value)`` as a candidate, without stating it.

        The candidate is an entry of the pair alone; it takes the place of
        another entry of the pair, whose beliefs it keeps
        (:func:`timetoshare.with_candidate`). A loader or a service
        offering it holds no belief; :meth:`add_statement` states it.

        Args:
            property: The property, as for :meth:`add_statement`.
            value: The value, as for :meth:`add_statement`.
            stated_by: Who offers it: an IRI, or several.
            method: ``manual``, ``reconciled``, ``imported``, ``inferred``
                or an IRI.
            date: The day, ``YYYY-MM-DD`` or a :class:`datetime.date`;
                today when None.
            service: The service that offered it, an IRI.
            basis: The sources the candidate rests on, IRIs.
            match: ``exact``, ``close``, ``broad``, ``narrow`` or
                ``related``.
            note: A remark.
            node: The value node to offer the pair of.

        Raises:
            ValueError: If the pair is stated (``'wdt:P175' already has
                this value; a stated pair is no candidate``), a member is
                no valid provenance member, or a prefix is undeclared.
        """
        return self._with_statement_state(
            self._state_with_candidate(
                property,
                value,
                stated_by=stated_by,
                method=method,
                date=date,
                service=service,
                basis=basis,
                match=match,
                note=note,
                node=node,
            )
        )

    def add_belief(
        self,
        property: str,
        value: Any,
        *,
        actor: str,
        belief: BeliefValue,
        date: str | datetime.date | None = None,
        note: str | None = None,
        node: NodePath = (),
    ) -> Self:
        """Record *actor*'s belief in the pair ``(property, value)``.

        The pair's entry, made where the pair has none, records the belief,
        replacing the actor's earlier one (:func:`timetoshare.with_belief`).
        Whether the pair is stated is left as it is: believing a stated
        pair ``"false"`` records the belief and never removes the pair, and
        every other actor's belief stands. A candidate the actor believes
        ``"false"`` is left out of :meth:`get_candidates` for that actor.

        Args:
            property: The property, as for :meth:`add_statement`.
            value: The value, as for :meth:`add_statement`.
            actor: Who holds the belief, an IRI.
            belief: ``"true"``, ``"false"`` or ``"unlikely"``.
            date: The day, ``YYYY-MM-DD`` or a :class:`datetime.date`;
                today when None.
            note: A remark.
            node: The value node whose pair is meant.

        Raises:
            ValueError: If *belief* is none of the three, the date is no
                calendar date, *node* addresses no value node, or a prefix
                is undeclared.
        """
        return self._with_statement_state(
            self._state_with_belief(
                property,
                value,
                actor=actor,
                belief=belief,
                date=date,
                note=note,
                node=node,
            )
        )

    def add_verb(
        self,
        verb: str,
        system: str,
        value: str,
        *,
        label: str | None = None,
        slots: Mapping[str, Any] | None = None,
        alternative: int = 0,
        provenance: Provenance | Mapping[str, Any] | None = None,
        actor: str | None = None,
        date: str | datetime.date | None = None,
        catalogue: Catalogue | None = None,
    ) -> Self:
        """State a verb of the catalogue in one system's vocabulary.

        The pairs of the system's pattern for the verb are stated in their
        order, all or none (:func:`timetoshare.instantiate`), each as
        :meth:`add_statement` states a pair, with *provenance* and *actor*.

        Args:
            verb: The verb's ID in the catalogue (``performed_by``).
            system: The system's ID (``wikidata``, ``fallback``).
            value: What the pattern's slot ``?`` takes (``wd:Q235650``).
            label: The label of the node the value becomes, if any.
            slots: Values of the pattern's named slots (``{"role":
                "piano"}``).
            alternative: Which of the system's patterns to use.
            provenance: As for :meth:`add_statement`.
            actor: As for :meth:`add_statement`.
            date: As for :meth:`add_statement`.
            catalogue: The catalogue; the shipped one when None.

        Raises:
            ValueError: If the catalogue has no such verb or pattern, a
                named slot has no value, or a pair is refused as by
                :meth:`add_statement`.
        """
        return self._with_statement_state(
            self._state_with_verb(
                verb,
                system,
                value,
                label=label,
                slots=slots,
                alternative=alternative,
                provenance=provenance,
                actor=actor,
                date=date,
                catalogue=catalogue,
            )
        )

    def add_prefix(self, name: str, namespace: str) -> Self:
        """Declare a prefix the holder's statements may use.

        The catalogue's prefixes need no declaration. A manifest written
        holding the holder carries the prefix in its ``@context`` when the
        statements use it, a bundle's prefix also when its MatchClaim
        table's actors use it.

        Raises:
            ValueError: If the format refuses the declaration (``cannot
                declare the prefix 'unit' as ...``), or the holder declares
                the prefix already with another namespace (``the prefix
                'foaf' already stands for 'http://xmlns.com/foaf/0.1/' on
                timeline 'cpt_1'``). The same declaration again changes
                nothing.
        """
        return self._with_statement_state(prefixes=self._prefixes_with(name, namespace))


# endregion
