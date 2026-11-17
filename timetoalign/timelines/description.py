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
add links in place (:meth:`MutableLinksMixin.add_link`), the frozen region
returns a copy (:meth:`Region.with_link <timetoalign.timelines.Region.with_link>`).
A link added without an assertion takes the default assertion, if one is set
(:func:`set_default_assertion`), dated the day it is added.

Examples:
    >>> import timetoalign as tta
    >>> from timetoalign.timelines import ContinuousPhysicalTimeline
    >>> take = ContinuousPhysicalTimeline(length=690.0, uid="take")
    >>> _ = take.add_link(
    ...     "https://musicbrainz.org/release/814c53fd-0732-4819-9753-4578f5ab992c"
    ... )
    >>> take.links[0].to_dict(minimal=True)
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
    ``links``; this class answers queries over it and builds the link an
    element adds (:meth:`_link_to_add`).
    """

    links: tuple[Link, ...]

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

    The element holds its links in ``_links``, a list, and adds a link in
    place.
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


# endregion
