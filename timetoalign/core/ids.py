"""Identity management for TTA objects.

This module provides scoped IDs that preserve source provenance,
generators for creating unique IDs within a scope, and utilities
for resolving partial/regex ID patterns.

Includes:
- ScopedId: An identifier with scope:local format.
- IdGenerator: General-purpose unique ID generation within a scope.
- mint_id: The one generator of manifest element ids, ``{prefix}_{N}``
  (e.g., ``clt_1``, ``region_2``, ``score:clt_1``).
- validate_id: The rule every given id obeys.
- resolve_id: Resolve a partial string or regex to a single ID.
- resolve_ids: Resolve a pattern to all matching IDs.
"""

from __future__ import annotations

import re
from collections.abc import Container, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

from .enums import TimeUnit


@dataclass(frozen=True, slots=True)
class ScopedId:
    """An identifier with a scope prefix and local part.

    Format: "{scope}:{local}" or just "{local}" if scope is empty.

    The scope typically represents the source (e.g., "midi", "score")
    while the local part is the original ID from that source.

    Attributes:
        scope: The namespace/source prefix (e.g., "midi", "score:track1")
        local: The local identifier within that scope

    Examples:
        >>> ScopedId("midi", "n42")
        ScopedId(scope='midi', local='n42')

        >>> str(ScopedId("midi", "n42"))
        'midi:n42'

        >>> ScopedId.parse("midi:n42")
        ScopedId(scope='midi', local='n42')

        >>> ScopedId.parse("bare_id")
        ScopedId(scope='', local='bare_id')
    """

    scope: str
    local: str

    SEPARATOR: ClassVar[str] = ":"

    # Validation patterns
    _SCOPE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^[a-zA-Z_][a-zA-Z0-9_.\-]*$"
    )
    _LOCAL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^[^\s:]+$"
    )  # No whitespace or colons

    def __post_init__(self) -> None:
        # Allow empty scope
        if self.scope and not self._SCOPE_PATTERN.match(self.scope):
            raise ValueError(
                f"Invalid scope '{self.scope}': must start with letter/underscore, "
                f"contain only alphanumeric, underscore, dot, or hyphen"
            )
        if not self.local:
            raise ValueError("Local ID cannot be empty")
        if not self._LOCAL_PATTERN.match(self.local):
            raise ValueError(
                f"Invalid local ID '{self.local}': cannot contain whitespace or colons"
            )

    def __str__(self) -> str:
        if not self.scope:
            return self.local
        return f"{self.scope}{self.SEPARATOR}{self.local}"

    def __repr__(self) -> str:
        return f"ScopedId(scope={self.scope!r}, local={self.local!r})"

    @classmethod
    def parse(cls, id_str: str) -> ScopedId:
        """Parse a scoped ID string.

        Args:
            id_str: A string like "scope:local" or just "local"

        Returns:
            A ScopedId instance

        Examples:
            >>> ScopedId.parse("midi:n42")
            ScopedId(scope='midi', local='n42')

            >>> ScopedId.parse("bare_id")
            ScopedId(scope='', local='bare_id')
        """
        if cls.SEPARATOR in id_str:
            scope, local = id_str.split(cls.SEPARATOR, 1)
            return cls(scope=scope, local=local)
        return cls(scope="", local=id_str)

    def with_scope(self, new_scope: str) -> ScopedId:
        """Return a new ScopedId with a different scope."""
        return ScopedId(scope=new_scope, local=self.local)

    def with_local(self, new_local: str) -> ScopedId:
        """Return a new ScopedId with a different local part."""
        return ScopedId(scope=self.scope, local=new_local)

    def nested(self, child_scope: str) -> ScopedId:
        """Create a nested scope by appending to the current scope.

        Example:
            >>> ScopedId("midi", "note").nested("track1")
            ScopedId(scope='midi.track1', local='note')
        """
        if self.scope:
            new_scope = f"{self.scope}.{child_scope}"
        else:
            new_scope = child_scope
        return ScopedId(scope=new_scope, local=self.local)

    @property
    def is_scoped(self) -> bool:
        """Whether this ID has a non-empty scope."""
        return bool(self.scope)


@dataclass
class IdGenerator:
    """Generates unique IDs within a given scope.

    Supports both:
    1. Wrapping external IDs with the scope
    2. Generating new IDs when none provided

    Tracks seen IDs to detect collisions.

    Attributes:
        scope: The scope to apply to all generated/wrapped IDs

    Examples:
        >>> gen = IdGenerator("midi")

        >>> gen.get_or_create("n42")  # Wrap external ID
        'midi:n42'

        >>> gen.get_or_create(None, type_hint="note")  # Generate
        'midi:note_1'

        >>> gen.get_or_create(None, type_hint="note")  # Generate next
        'midi:note_2'
    """

    scope: str

    # Internal state
    _counters: dict[str, int] = field(default_factory=dict)
    _seen: set[str] = field(default_factory=set)

    def get_or_create(
        self,
        external_id: str | None = None,
        type_hint: str = "event",
    ) -> str:
        """Get a scoped ID string.

        Args:
            external_id: The ID from the source, if any. Will be wrapped
                        with the scope but otherwise preserved.
            type_hint: A prefix for generated IDs if external_id is None.

        Returns:
            A string in "scope:local_id" format (or just "local_id" if
            scope is empty).
        """
        if external_id is not None and str(external_id).strip():
            # Wrap external ID
            local = str(external_id).strip()
        else:
            # Generate new ID
            counter = self._counters.get(type_hint, 0) + 1
            self._counters[type_hint] = counter
            local = f"{type_hint}_{counter}"

        # Create scoped ID
        scoped = ScopedId(self.scope, local)
        full_id = str(scoped)

        # Track for collision detection (we don't raise, just track)
        self._seen.add(full_id)

        return full_id

    def create(self, type_hint: str = "event") -> str:
        """Generate a new unique ID (no external ID)."""
        return self.get_or_create(None, type_hint=type_hint)

    def wrap(self, external_id: str) -> str:
        """Wrap an external ID with the scope."""
        return self.get_or_create(external_id)

    def reset(self) -> None:
        """Reset all counters and seen IDs."""
        self._counters.clear()
        self._seen.clear()

    def reset_counters(self) -> None:
        """Reset counters but keep seen IDs (for continuation)."""
        self._counters.clear()

    @property
    def count(self) -> int:
        """Total number of IDs generated/wrapped."""
        return len(self._seen)

    def has_seen(self, id_str: str) -> bool:
        """Check if an ID has been generated/wrapped by this generator."""
        return id_str in self._seen


# region Manifest element ids

_REFUSED_ID_CHARACTERS = re.compile(r"[/\x00-\x1f]")
_PART_WORDS = ("start", "end", "name", "label", "card")
"""The words that follow an element's address as its parts, and so are no id."""
_BAND_ID = re.compile(
    r"(markers|conversion_maps|children|regions|flow_control)_[1-9][0-9]*"
)
"""The ids of a timeline's bands, which follow its address, and so are no id."""


def validate_id(element_id: str) -> str:
    """Return *element_id* if it is a valid given id, else raise.

    Any non-empty string is a valid id except one containing ``/`` (the path
    separator of composed addresses) or a control character
    (U+0000 to U+001F), the part words ``start``, ``end``, ``name``,
    ``label`` and ``card``, which follow an address as its parts, and the
    ids of a timeline's bands, which follow its address: a kind among
    ``markers``, ``conversion_maps``, ``children``, ``regions`` and
    ``flow_control``, ``_`` and a number from 1 without a leading zero
    (``markers_1``, ``regions_12``). Whitespace, colons, dots and unicode
    are permitted.

    Args:
        element_id: The id to check.

    Returns:
        The same id.

    Raises:
        ValueError: If the id is empty, contains a refused character, is a
            part word or is the id of a band.

    Examples:
        >>> validate_id("Bass Kick")
        'Bass Kick'
        >>> validate_id("a/b")
        Traceback (most recent call last):
        ...
        ValueError: Invalid id 'a/b': an id must be non-empty and contain neither '/' nor a control character
    """
    if (
        not isinstance(element_id, str)
        or not element_id
        or (_REFUSED_ID_CHARACTERS.search(element_id))
    ):
        raise ValueError(
            f"Invalid id {element_id!r}: an id must be non-empty and contain "
            "neither '/' nor a control character"
        )
    if element_id in _PART_WORDS:
        raise ValueError(
            f"Invalid id {element_id!r}: it is reserved for the parts of an address"
        )
    if _BAND_ID.fullmatch(element_id):
        raise ValueError(
            f"Invalid id {element_id!r}: it is reserved for the bands of a timeline"
        )
    return element_id


def mint_id(prefix: str, taken: Container[str]) -> str:
    """Mint ``{prefix}_{N}`` with N the smallest positive integer not taken.

    This is the one id generator of the library. It holds no state: the same
    prefix and the same taken ids always give the same id, whatever was built
    before in the process.

    Args:
        prefix: The type prefix, optionally led by a role
            (``"clt"``, ``"region"``, ``"score:clt"``).
        taken: The ids already held in the scope the new id must be unique in.

    Returns:
        The minted id.

    Examples:
        >>> mint_id("clt", ())
        'clt_1'
        >>> mint_id("clt", {"clt_1", "clt_3"})
        'clt_2'
        >>> mint_id("score:clt", {"score:clt_1"})
        'score:clt_2'
    """
    number = 1
    while f"{prefix}_{number}" in taken:
        number += 1
    return f"{prefix}_{number}"


def timeline_id_prefix(unit: TimeUnit) -> str:
    """Return the id prefix of a timeline measured in *unit*.

    The prefix encodes the unit's discreteness and domain: ``clt``, ``dlt``,
    ``cpt``, ``dpt``, ``cgt``, ``dgt``.

    Examples:
        >>> timeline_id_prefix(TimeUnit.quarters)
        'clt'
        >>> timeline_id_prefix(TimeUnit.samples)
        'dpt'
    """
    return f"{'d' if unit.is_discrete else 'c'}{unit.domain.value[0]}t"


def _admit_id(
    element_id: str,
    *,
    minted: bool,
    prefix: str,
    taken: Mapping[str, str],
    element: str,
) -> str:
    """Return the id under which an element joins a scope.

    An id the scope does not hold is kept. A taken id that was minted is
    minted again in the scope; a taken id that was given raises.

    Args:
        element_id: The element's current id.
        minted: Whether the id was minted rather than given.
        prefix: The prefix to mint with.
        taken: Every id of the scope, mapped to a description of its holder.
        element: A description of the joining element, for the error.

    Raises:
        ValueError: If a given id is already taken.
    """
    if element_id not in taken:
        return element_id
    if minted:
        return mint_id(prefix, taken)
    raise ValueError(
        f"{element} cannot join: its id {element_id!r} is already held by "
        f"{taken[element_id]}"
    )


class _OpenObject:
    """An object with a dictionary form that keeps the keys it does not read.

    A TimeSkeleton object may carry keys its reader does not know. The
    reader keeps them on the object read from the JSON object they were
    found in (:meth:`_keep_unknown_keys`), and the writer writes them back
    after its own keys (:meth:`_dict_form`). A subclass lists its known keys
    in ``_dict_keys`` and the value each optional key has when absent in
    ``_dict_defaults``: the complete form writes every known key, the
    minimal form leaves out each key whose value is its default.
    """

    _dict_keys: ClassVar[tuple[str, ...]] = ()
    _dict_defaults: ClassVar[Mapping[str, Any]] = {}
    _unknown_keys: Mapping[str, Any] = MappingProxyType({})

    def _keep_unknown_keys(
        self, data: Mapping[str, Any], known: Container[str] | None = None
    ) -> None:
        """Keep a copy of every key of *data* outside *known* (``_dict_keys``)."""
        known = self._dict_keys if known is None else known
        unknown = {
            key: deepcopy(value) for key, value in data.items() if key not in known
        }
        object.__setattr__(self, "_unknown_keys", unknown)

    def _dict_form(
        self,
        fields: Mapping[str, Any],
        *,
        minimal: bool,
        defaults: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return *fields*, then the unknown keys kept.

        With *minimal*, a field whose value equals its default (in
        *defaults*, else ``_dict_defaults``) is left out.
        """
        defaults = self._dict_defaults if defaults is None else defaults
        written = {
            key: value
            for key, value in fields.items()
            if not (minimal and key in defaults and _same_json(value, defaults[key]))
        }
        written.update(deepcopy(dict(self._unknown_keys)))
        return written


def _same_json(value: Any, default: Any) -> bool:
    """Whether two JSON values are equal, a boolean never equal to a number."""
    if isinstance(value, bool) or isinstance(default, bool):
        return type(value) is type(default) and value == default
    return value == default


@dataclass(frozen=True)
class _IdentifiedElement(_OpenObject):
    """Identity shared by the frozen elements of a timeline manifest.

    A subclass names its kind in ``_id_prefix``. Constructed without an
    ``id``, an element mints ``{prefix}_1`` (its own scope is empty) and
    remembers that the id was minted, so that a collision when it joins a
    timeline re-mints it rather than raising. A given id is validated and
    kept verbatim. The minted flag is private and never serialized.
    """

    _id_prefix: ClassVar[str]

    id: str = field(default=None, kw_only=True)  # type: ignore[assignment]
    _minted: bool = field(default=False, kw_only=True, repr=False, compare=False)
    # A field, so that a copy made with ``dataclasses.replace`` keeps it.
    _unknown_keys: Mapping[str, Any] = field(
        default_factory=dict, kw_only=True, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.id is None:
            object.__setattr__(self, "id", mint_id(self._id_prefix, ()))
            object.__setattr__(self, "_minted", True)
        else:
            validate_id(self.id)


# endregion

# region ID Resolution Utilities


def resolve_ids(pattern: str, available_ids: list[str]) -> list[str]:
    """Resolve a partial string or regex pattern to matching IDs.

    Matching precedence:
    1. Exact match: Returns ``[pattern]`` immediately if found.
    2. Substring match: All IDs containing the pattern.
    3. Regex match: All IDs matching the compiled pattern via ``re.search``.

    Args:
        pattern: The search pattern (exact string, substring, or regex).
        available_ids: The IDs to search.

    Returns:
        List of matching IDs (may be empty).

    Examples:
        >>> resolve_ids("clt_1", ["clt_1", "clt_2", "dlt_1"])
        ['clt_1']

        >>> resolve_ids("clt", ["clt_1", "clt_2", "dlt_1"])
        ['clt_1', 'clt_2']

        >>> resolve_ids(r"^clt_\\d+$", ["clt_1", "clt_2", "dlt_1"])
        ['clt_1', 'clt_2']

        >>> resolve_ids("xyz", ["clt_1", "clt_2"])
        []
    """
    if not pattern or not available_ids:
        return []

    # 1. Exact match
    if pattern in available_ids:
        return [pattern]

    # 2. Substring match
    substring_matches = [id_ for id_ in available_ids if pattern in id_]
    if substring_matches:
        return substring_matches

    # 3. Regex match
    try:
        regex = re.compile(pattern)
        regex_matches = [id_ for id_ in available_ids if regex.search(id_)]
        return regex_matches
    except re.error:
        # Invalid regex, no matches
        return []


def resolve_id(
    pattern: str,
    available_ids: list[str],
    *,
    warn_multiple: bool = True,
) -> str:
    """Resolve a partial string or regex pattern to a single ID.

    When multiple IDs match, returns the first match and optionally
    warns (see ``warn_multiple``).

    Matching precedence:
    1. Exact match: Returns immediately.
    2. Substring match: If exactly one, returns it; if multiple, returns first.
    3. Regex match: Same logic.

    Args:
        pattern: The search pattern (exact string, substring, or regex).
        available_ids: The IDs to search.
        warn_multiple: If ``True`` (default), emit a warning when multiple IDs
            match and the first is returned.

    Returns:
        The matching ID.

    Raises:
        KeyError: If no ID matches the pattern.

    Examples:
        >>> resolve_id("clt_1", ["clt_1", "clt_2", "dlt_1"])
        'clt_1'

        >>> resolve_id("clt", ["clt_1", "clt_2", "dlt_1"])  # warns
        'clt_1'

        >>> resolve_id("xyz", ["clt_1", "clt_2"])
        Traceback (most recent call last):
            ...
        KeyError: "No ID matches pattern 'xyz'. Available: clt_1, clt_2"
    """
    import warnings

    matches = resolve_ids(pattern, available_ids)

    if not matches:
        available_str = ", ".join(available_ids[:10])
        if len(available_ids) > 10:
            available_str += f", ... ({len(available_ids)} total)"
        raise KeyError(f"No ID matches pattern {pattern!r}. Available: {available_str}")

    if len(matches) > 1 and warn_multiple:
        warnings.warn(
            f"Pattern {pattern!r} matches {len(matches)} IDs: {matches}. "
            f"Returning first match: {matches[0]!r}",
            UserWarning,
            stacklevel=2,
        )

    return matches[0]


# endregion
