"""Statements about what a bundle, a timeline or a region stands for.

A TimeToShare manifest describes the things its elements stand for — a
work, a recording, a printed edition, a performer — by property–value
pairs: the bundle (the manifest itself), every timeline at any depth and
every region may hold ``statements`` (``{property: [value, ...]}``) and
``provenance``, the entries saying who stated, proposed, confirmed or
rejected a pair, when and how. A value naming an element of the manifest is
the node ``{"id": "#cpt_1"}`` (:func:`timetoshare.to_file_id` builds the
reference); the element's own ``id`` stays ``cpt_1``. Statements are the
same in a file and in memory.

Everything about the pairs themselves is TimeToShare's
(:mod:`timetoshare.statements`, :mod:`timetoshare.catalogue`): matching,
adding, removing, proposals, confirmation, rejection, the rectangles a
catalogue reads, the prefixes used. This module adds where statements live:
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
    >>> wav = DiscretePhysicalTimeline(length=1470868, uid="dpt_1")
    >>> _ = wav.propose_statement("dcterms:source", {"id": "#cpt_1"},
    ...     stated_by="https://example.org/people/editor", method="inferred", date="2026-11-18")
    >>> wav.statements, [entry["status"] for entry in wav.provenance]
    ({}, ['proposed'])
    >>> _ = wav.confirm_statement("dcterms:source", {"id": "#cpt_1"},
    ...     stated_by="https://example.org/people/editor", date="2026-11-19")
    >>> wav.statements, [entry["status"] for entry in wav.provenance]
    ({'dcterms:source': [{'id': '#cpt_1'}]}, ['confirmed'])
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
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Final, Literal

import timetoshare
from timetoshare import FORMAT_CONTEXT, Catalogue, Rectangle, load_catalogue
from typing_extensions import Self

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

Status = Literal["confirmed", "proposed", "rejected"]

# The day a proposal, confirmation or rejection is dated when no date is
# given; a module attribute, so that a test can fix the day.
_today = datetime.date.today


# region State


def read_statement_state(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return the statements and provenance of a manifest object, as written."""
    return {key: copy.deepcopy(data[key]) for key in STATE_KEYS if key in data}


def needed_prefixes(
    state: Mapping[str, Any], declared: Mapping[str, str]
) -> dict[str, str]:
    """Return the prefixes of *declared* the statements of *state* use.

    Only those the catalogue lacks or gives another namespace: a holder
    keeps the prefixes its pairs need beyond the catalogue's.
    """
    catalogue = load_catalogue().prefixes
    return {
        prefix: declared[prefix]
        for prefix in timetoshare.used_prefixes(dict(state))
        if prefix in declared and declared[prefix] != catalogue.get(prefix)
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
        self, status: Status | None = None, *, node: NodePath = ()
    ) -> list[dict[str, Any]]:
        """Return the provenance entries of the holder or of a value node.

        Args:
            status: Only the entries of this status; all of them when None.
            node: The value node whose entries to return; ``()`` the holder.

        Raises:
            ValueError: If *node* addresses no value node.
        """
        return timetoshare.entries(self._statement_document(), _subject(node), status)

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
        provenance: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return self._applied(
            lambda document: timetoshare.with_pair(
                document, _subject(node), property, value, provenance=provenance
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

    def _state_with_proposal(
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
        return self._applied(
            lambda document: timetoshare.with_proposal(
                document,
                _subject(node),
                property,
                value,
                stated_by=_agents(stated_by),
                date=_day(date),
                method=method,
                service=service,
                basis=basis,
                match=match,
                note=note,
            )
        )

    def _state_with_confirmation(
        self,
        property: str,
        value: Any,
        *,
        stated_by: str | Sequence[str],
        date: str | datetime.date | None,
        node: NodePath,
    ) -> dict[str, Any]:
        return self._applied(
            lambda document: timetoshare.with_confirmation(
                document,
                _subject(node),
                property,
                value,
                stated_by=_agents(stated_by),
                date=_day(date),
            )
        )

    def _state_with_rejection(
        self,
        property: str,
        value: Any,
        *,
        stated_by: str | Sequence[str],
        date: str | datetime.date | None,
        note: str | None,
        node: NodePath,
    ) -> dict[str, Any]:
        return self._applied(
            lambda document: timetoshare.with_rejection(
                document,
                _subject(node),
                property,
                value,
                stated_by=_agents(stated_by),
                date=_day(date),
                note=note,
            )
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
        provenance: Mapping[str, Any] | None,
        catalogue: Catalogue | None,
    ) -> dict[str, Any]:
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
            for property, pair_value in pairs:
                document = timetoshare.with_pair(
                    document, (), property, pair_value, provenance=provenance
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
        provenance: Mapping[str, Any] | None = None,
    ) -> Self:
        """State the pair ``(property, value)``.

        A ``proposed`` or ``rejected`` entry of the pair is removed. With
        *provenance*, a ``confirmed`` entry records who stated the pair.

        Args:
            property: ``type`` ("is a"), a compact IRI or an absolute IRI.
            value: A JSON value of the file's form: a literal, a value
                object, a node (``{"id": ...}``) or ``{"list": [...]}``;
                ``type`` takes a class IRI.
            node: The value node to state the pair of; ``()`` the holder.
            provenance: The members of the confirmed entry but
                ``property``, ``value`` and ``status``, as the file writes
                them (``stated_by``, ``date``, ``method``, ...).

        Raises:
            ValueError: If the pair is stated already (``'wdt:P175' already
                has this value``), the property is no property, *node*
                addresses no value node, or a prefix is declared neither by
                the catalogue nor on the holder.
        """
        return self._with_statement_state(
            self._state_with_statement(
                property, value, node=node, provenance=provenance
            )
        )

    def remove_statement(
        self, property: str, value: Any, *, node: NodePath = ()
    ) -> Self:
        """Remove the pair ``(property, value)`` and its ``confirmed`` entry.

        Raises:
            ValueError: If the pair is not stated (``'type' does not have
                this value``) or *node* addresses no value node.
        """
        return self._with_statement_state(
            self._state_without_statement(property, value, node=node)
        )

    def propose_statement(
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
        """Propose the pair ``(property, value)`` without stating it.

        A ``proposed`` entry takes the place of another entry of the pair.

        Args:
            property: The property, as for :meth:`add_statement`.
            value: The value, as for :meth:`add_statement`.
            stated_by: Who proposes: an IRI, or several.
            method: ``manual``, ``reconciled``, ``imported``, ``inferred``
                or an IRI.
            date: The day, ``YYYY-MM-DD`` or a :class:`datetime.date`;
                today when None.
            service: The service that proposed it, an IRI.
            basis: The sources the proposal rests on, IRIs.
            match: ``exact``, ``close``, ``broad``, ``narrow`` or
                ``related``.
            note: A remark.
            node: The value node to propose the pair of.

        Raises:
            ValueError: If the pair is stated (``'wdt:P175' already has
                this value; a stated pair is not proposed``), the date is
                no calendar date, or a prefix is undeclared.
        """
        return self._with_statement_state(
            self._state_with_proposal(
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

    def confirm_statement(
        self,
        property: str,
        value: Any,
        *,
        stated_by: str | Sequence[str],
        date: str | datetime.date | None = None,
        node: NodePath = (),
    ) -> Self:
        """Confirm a proposed or rejected pair: state it, its entry ``confirmed``.

        *value* is stated as given, its label included; the entry keeps its
        other members and takes *stated_by* and *date*.

        Raises:
            ValueError: If the pair has no proposed or rejected entry (``no
                proposal or rejection of this value of 'dcterms:source'``)
                or is stated already.
        """
        return self._with_statement_state(
            self._state_with_confirmation(
                property, value, stated_by=stated_by, date=date, node=node
            )
        )

    def reject_statement(
        self,
        property: str,
        value: Any,
        *,
        stated_by: str | Sequence[str],
        date: str | datetime.date | None = None,
        note: str | None = None,
        node: NodePath = (),
    ) -> Self:
        """Reject a pair: no longer stated, kept as a ``rejected`` entry.

        A stated pair is removed; the pair's entry, created where it has
        none, takes *stated_by*, *date* and, where given, *note*.

        Raises:
            ValueError: If the date is no calendar date or *node* addresses
                no value node.
        """
        return self._with_statement_state(
            self._state_with_rejection(
                property, value, stated_by=stated_by, date=date, note=note, node=node
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
        provenance: Mapping[str, Any] | None = None,
        catalogue: Catalogue | None = None,
    ) -> Self:
        """State a verb of the catalogue in one system's vocabulary.

        The pairs of the system's pattern for the verb are stated in their
        order, all or none (:func:`timetoshare.instantiate`), each with a
        ``confirmed`` entry when *provenance* is given.

        Args:
            verb: The verb's ID in the catalogue (``performed_by``).
            system: The system's ID (``wikidata``, ``fallback``).
            value: What the pattern's slot ``?`` takes (``wd:Q235650``).
            label: The label of the node the value becomes, if any.
            slots: Values of the pattern's named slots (``{"role":
                "piano"}``).
            alternative: Which of the system's patterns to use.
            provenance: As for :meth:`add_statement`.
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
                catalogue=catalogue,
            )
        )

    def add_prefix(self, name: str, namespace: str) -> Self:
        """Declare a prefix the holder's statements may use.

        The catalogue's prefixes need no declaration. A manifest written
        holding the holder carries the prefix in its ``@context`` when the
        statements use it.

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
