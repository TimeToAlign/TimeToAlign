"""Links from a manifest's elements to works and other entities.

The description layer of a TimeToShare manifest links the manifest, a
timeline or a region to external entities: a work, an expression, a
manifestation, an item or an agent. Its model is the ``timetoshare``
package's, re-exported here unchanged: :class:`Link` (one attributed reading
of an entity), :class:`PartSlot` and :class:`PartOf` (the parts of the
entity in that reading and the wholes it belongs to), :class:`Identifier`,
:class:`Assertion` (who stated a link, when, how), :class:`Scheme` (an
identifier scheme a manifest declares) and :class:`Descriptor` (a document
about the entity). ``Link.from_urls`` and ``PartSlot.from_url`` build them
from the URLs of catalogue pages.

What the library adds is where links live. :class:`LinksMixin` is the one
surface of every element carrying links — :class:`~timetoalign.timelines.Timeline`,
:class:`~timetoalign.timelines.Region` and
:class:`~timetoalign.alignment.AlignmentBundle` (the manifest itself) — with
``links`` and :meth:`LinksMixin.get_links`; the mutable timeline and bundle
add, replace and remove links in place (:meth:`MutableLinksMixin.add_link`,
:meth:`~MutableLinksMixin.replace_link`, :meth:`~MutableLinksMixin.remove_link`),
the frozen region returns a copy
(:meth:`Region.with_link <timetoalign.timelines.Region.with_link>`,
:meth:`Region.without_link <timetoalign.timelines.Region.without_link>`).
A link added without an assertion takes the default assertion, if one is set
(:func:`set_default_assertion`), dated the day it is added.

A part slot names the timelines and regions representing the part by id.
The library keeps those names true: an element given a new id is named by
its new id, and a slot whose elements all leave the manifest is ``absent``
as ``not_represented``, the part existing and the manifest leaving it out.

Examples:
    >>> import timetoalign as tta
    >>> from timetoalign.timelines import ContinuousPhysicalTimeline
    >>> take = ContinuousPhysicalTimeline(length=690.0, uid="take")
    >>> _ = take.add_link(
    ...     "https://musicbrainz.org/release/814c53fd-0732-4819-9753-4578f5ab992c"
    ... )
    >>> take.links[0].to_dict()
    {'relation': 'pertains_to', 'level': 'manifestation', \
'uri': ['http://musicbrainz.org/release/814c53fd-0732-4819-9753-4578f5ab992c']}
    >>> [link.level for link in take.get_links(scheme="musicbrainz_release")]
    ['manifestation']
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable, Mapping
from contextvars import ContextVar, Token
from dataclasses import replace
from typing import Any

from timetoshare import (
    Assertion,
    Descriptor,
    Identifier,
    Link,
    PartOf,
    PartSlot,
    Scheme,
    default_registry,
)
from typing_extensions import Self

__all__ = [
    "Assertion",
    "DefaultAssertionToken",
    "Descriptor",
    "Identifier",
    "Link",
    "LinksMixin",
    "MutableLinksMixin",
    "PartOf",
    "PartSlot",
    "Scheme",
    "get_default_assertion",
    "set_default_assertion",
]

# region The default assertion

_DEFAULT_ASSERTION: ContextVar[Assertion | None] = ContextVar(
    "timetoalign_default_assertion", default=None
)
"""The assertion a link added without one takes, undated; None when unset."""

_today = datetime.date.today
"""The day a link takes the default assertion on; one seam, so a test can pin it."""


def _uris(value: str | Iterable[str]) -> tuple[str, ...]:
    """A single URI, or several, as the tuple an assertion holds."""
    return (value,) if isinstance(value, str) else tuple(value)


class DefaultAssertionToken:
    """The default assertion that :func:`set_default_assertion` set.

    Used as a context manager, it restores the default it replaced on exit;
    used otherwise, the default stays set in the current context.

    Attributes:
        assertion: The default assertion set, without a date.
    """

    def __init__(self, assertion: Assertion, token: Token[Assertion | None]) -> None:
        """Hold the assertion set and the token that restores the previous one."""
        self.assertion = assertion
        self._token = token

    def __enter__(self) -> Assertion:
        """Return the default assertion set."""
        return self.assertion

    def __exit__(self, *exc_info: object) -> None:
        """Restore the default assertion this one replaced."""
        _DEFAULT_ASSERTION.reset(self._token)

    def __repr__(self) -> str:
        return f"DefaultAssertionToken({self.assertion!r})"


def set_default_assertion(
    *,
    asserted_by: str | Iterable[str],
    method: str | None = "manual",
    service: str | None = None,
    based_on: str | Iterable[str] = (),
    match: str | None = None,
    status: str | None = None,
) -> DefaultAssertionToken:
    """Set the assertion a link added without one takes.

    ``add_link`` and ``with_link`` give a link that states no assertion this
    one, dated the day the link is added (``asserted_on``); a link stating
    its own keeps it. The default is held in a context variable, so it is
    the current context's (a thread's, an asynchronous task's) and never
    leaks into another. The terms are written as given; the TimeToShare
    validator checks them when a manifest holding them is read.

    Args:
        asserted_by: The URI of the agent asserting, or several.
        method: How the links are reached, ``manual`` by default.
        service: The service reaching them, or None.
        based_on: The URI of a source the links rest on, or several.
        match: How closely the linked entities correspond, or None.
        status: Whether the links stand, or None.

    Returns:
        A token which, used as a context manager, restores the previous
        default on exit.

    Examples:
        >>> import timetoalign as tta
        >>> with tta.set_default_assertion(asserted_by="https://example.org/me"):
        ...     tta.get_default_assertion().asserted_by
        ('https://example.org/me',)
        >>> tta.get_default_assertion() is None
        True
    """
    assertion = Assertion(
        asserted_by=_uris(asserted_by),
        method=method,
        service=service,
        based_on=_uris(based_on),
        match=match,
        status=status,
    )
    return DefaultAssertionToken(assertion, _DEFAULT_ASSERTION.set(assertion))


def get_default_assertion() -> Assertion | None:
    """Return the default assertion of the current context, undated, or None."""
    return _DEFAULT_ASSERTION.get()


# endregion

# region What links name

_LEFT_OUT = "not_represented"
"""Why no element represents a part whose representing elements all left the manifest."""


def _named_schemes(links: Iterable[Link]) -> set[str]:
    """Return the schemes the identifiers of *links* name, at any depth.

    The identifiers of each link, of its part slots and of its ``part_of``
    entries. A scheme is named by the id of a scheme of the manifest or by
    the URI of a scheme.
    """
    return {
        identifier.scheme
        for link in links
        for owner in (link, *link.parts, *link.part_of)
        for identifier in owner.identifiers
    }


def _renamed_parts(link: Link, renamed: Mapping[str, str | None]) -> Link:
    """Return *link* with the elements its part slots name renamed.

    Each id of a slot's ``represented_by`` that *renamed* maps is replaced
    by the id it maps to, or left out where it maps to None; a slot left
    with none is ``absent`` as ``not_represented``. A link none of whose
    slots names a mapped id is returned as it is.
    """
    changed = False
    parts = []
    for slot in link.parts:
        kept = tuple(
            new
            for new in (
                renamed.get(element, element) for element in slot.represented_by
            )
            if new is not None
        )
        if kept == slot.represented_by:
            parts.append(slot)
            continue
        changed = True
        parts.append(
            replace(slot, represented_by=kept)
            if kept
            else replace(slot, represented_by=(), absent=_LEFT_OUT)
        )
    return replace(link, parts=tuple(parts)) if changed else link


# endregion

# region Elements carrying links


def _registry_schemes(scheme: str) -> set[str]:
    """The registry's scheme URIs that *scheme* names, as an id or as the URI."""
    return {
        known.uri
        for known in default_registry().schemes
        if scheme in (known.id, known.uri)
    }


def _in_scheme(link: Link, scheme: str, registry_uris: set[str]) -> bool:
    """Whether *link* names its entity in *scheme*.

    By an identifier whose scheme is *scheme* (a manifest scheme's id or a
    scheme URI) or the URI of the registry scheme *scheme* names, or by a URI
    the registry recognises in the scheme of that id or URI.
    """
    if any(
        identifier.scheme == scheme or identifier.scheme in registry_uris
        for identifier in link.identifiers
    ):
        return True
    registry = default_registry()
    for uri in link.uri:
        recognition = registry.recognize(uri)
        if recognition is not None and scheme in (
            recognition.scheme.id,
            recognition.scheme.uri,
        ):
            return True
    return False


class LinksMixin:
    """The one surface of every element carrying links.

    A timeline, a region and a bundle (the manifest) hold ``links``, the
    tuple of their :class:`Link` objects in the order given, each one
    reading of an entity the element pertains to. A subclass supplies
    ``links`` and the element's name in a message
    (:meth:`_links_holder_name`); this class answers queries over the
    links, builds the link an element adds (:meth:`_link_to_add`) and finds
    the one it replaces or removes (:meth:`_link_position`).
    """

    links: tuple[Link, ...]

    def _links_holder_name(self) -> str:
        """Return the element as a message names it (``timeline 'score'``)."""
        raise NotImplementedError

    def _link_position(self, link: Link) -> int:
        """Return the position of the first link equal to *link*.

        Raises:
            ValueError: If the element holds no link equal to *link*
                (``timeline 'score' holds no link equal to the link given``).
        """
        for position, held in enumerate(self.links):
            if held == link:
                return position
        raise ValueError(
            f"{self._links_holder_name()} holds no link equal to the link given"
        )

    def get_links(
        self,
        *,
        level: str | None = None,
        relation: str | None = None,
        scheme: str | None = None,
        asserted_by: str | None = None,
    ) -> list[Link]:
        """Return the links matching every filter given, in their order.

        Args:
            level: The level of the linked entity (``work``, ``expression``,
                ``manifestation``, ``item``, ``agent``).
            relation: The link's relation (``pertains_to``, an agent's role).
            scheme: A scheme the link names its entity in: the id of a scheme
                of the manifest or the URI of a scheme, matched by the
                link's identifiers, or the id or URI of a scheme of the
                TimeToShare registry, matched also by a URI of the link the
                registry recognises in that scheme.
            asserted_by: The URI of an agent among those asserting the link.

        Returns:
            The links matching every filter given; all links without one.

        Examples:
            >>> from timetoalign import Link
            >>> from timetoalign.timelines import ContinuousLogicalTimeline
            >>> score = ContinuousLogicalTimeline(length=8)
            >>> _ = score.add_link(
            ...     "https://musicbrainz.org/work/4a622b0d-b0c0-405e-b49e-05b70c108284"
            ... )
            >>> _ = score.add_link(
            ...     "https://imslp.org/wiki/Piano_Sonata_No.29,_Op.106_(Beethoven,_Ludwig_van)"
            ... )
            >>> len(score.get_links(level="work"))
            2
            >>> [link.identifiers[0].value for link in score.get_links(scheme="imslp")]
            ['Piano_Sonata_No.29,_Op.106_(Beethoven,_Ludwig_van)']
        """
        registry_uris = set() if scheme is None else _registry_schemes(scheme)
        return [
            link
            for link in self.links
            if (level is None or link.level == level)
            and (relation is None or link.relation == relation)
            and (scheme is None or _in_scheme(link, scheme, registry_uris))
            and (
                asserted_by is None
                or (
                    link.assertion is not None
                    and asserted_by in link.assertion.asserted_by
                )
            )
        ]

    @staticmethod
    def _link_to_add(link: Link | str, options: Mapping[str, Any]) -> Link:
        """Return the link an element adds for *link* and *options*.

        A string is the URL of the entity, read by ``Link.from_url`` with
        *options*. A link stating no assertion takes the default assertion,
        dated today, when one is set.

        Raises:
            TypeError: If *options* are given beside a :class:`Link`.
            ValueError: If ``Link.from_url`` refuses the URL or the options.
        """
        if isinstance(link, str):
            link = Link.from_url(link, **options)
        elif options:
            raise TypeError(
                f"keyword arguments {sorted(options)} build a link from a URL; "
                "a Link is added as it is"
            )
        default = _DEFAULT_ASSERTION.get()
        if link.assertion is None and default is not None:
            link = link.with_assertion(replace(default, asserted_on=_today()))
        return link


class MutableLinksMixin(LinksMixin):
    """A mutable element carrying links: a timeline, a bundle.

    The element holds its links in ``_links``, a list, and adds, replaces
    and removes a link in place.
    """

    _links: list[Link]

    @property
    def links(self) -> tuple[Link, ...]:  # type: ignore[override]
        """The element's links, in the order added."""
        return tuple(self._links)

    def add_link(self, link: Link | str, **options: Any) -> Self:
        """Add a link to the element and return the element.

        Args:
            link: The link, or the URL of the linked entity, which
                ``Link.from_url`` reads with *options* (``level``,
                ``relation``, ``identifiers``, ``label``, ``assertion``).
            **options: Passed to ``Link.from_url`` with a URL.

        Returns:
            The element, so that calls chain.

        Raises:
            TypeError: If *options* are given beside a :class:`Link`.
            ValueError: If ``Link.from_url`` refuses the URL or the options.

        Examples:
            >>> import timetoalign as tta
            >>> bundle = tta.AlignmentBundle()
            >>> _ = bundle.add_link(
            ...     "https://www.wikidata.org/wiki/Q146730", level="work", label="Op. 106"
            ... )
            >>> bundle.links[0].uri, bundle.links[0].label
            (('http://www.wikidata.org/entity/Q146730',), 'Op. 106')
        """
        self._links.append(self._link_to_add(link, options))
        return self

    def replace_link(self, old: Link, new: Link | str, **options: Any) -> Self:
        """Replace a link of the element with another, in its place.

        The first link equal to *old* is replaced. The new link is built as
        :meth:`add_link` builds it, so a link stating no assertion takes the
        default assertion, if one is set.

        Args:
            old: A link equal to the one to replace.
            new: The link taking its place, or the URL of the linked
                entity, which ``Link.from_url`` reads with *options*.
            **options: Passed to ``Link.from_url`` with a URL.

        Returns:
            The element, so that calls chain.

        Raises:
            TypeError: If *options* are given beside a :class:`Link`.
            ValueError: If the element holds no link equal to *old*
                (``timeline 'score' holds no link equal to the link
                given``), or ``Link.from_url`` refuses the URL or the
                options. The links are left as they were.

        Examples:
            >>> from timetoalign.timelines import ContinuousLogicalTimeline
            >>> score = ContinuousLogicalTimeline(length=8)
            >>> work = "https://musicbrainz.org/work/4a622b0d-b0c0-405e-b49e-05b70c108284"
            >>> _ = score.add_link(work)
            >>> _ = score.replace_link(score.links[0], work, label="Op. 106")
            >>> [link.label for link in score.links]
            ['Op. 106']
        """
        position = self._link_position(old)
        self._links[position] = self._link_to_add(new, options)
        return self

    def remove_link(self, link: Link) -> Self:
        """Remove a link from the element.

        Args:
            link: A link equal to the one to remove; the first equal link
                is removed.

        Returns:
            The element, so that calls chain.

        Raises:
            ValueError: If the element holds no link equal to *link*
                (``bundle 'study' holds no link equal to the link given``).
        """
        del self._links[self._link_position(link)]
        return self

    def _rename_parts(self, renamed: Mapping[str, str | None]) -> None:
        """Rename the elements the part slots of the element's links name.

        See :func:`_renamed_parts`.
        """
        self._links = [_renamed_parts(link, renamed) for link in self._links]


# endregion
