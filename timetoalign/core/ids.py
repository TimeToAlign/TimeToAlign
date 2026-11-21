"""Identity management for TTA objects.

This module provides scoped IDs that preserve source provenance, the one
stateless generator of minted IDs, and utilities for resolving
partial/regex ID patterns.

Includes:
- ScopedId: An identifier with scope:local format.
- mint_ids: The one generator of minted ids, ``{prefix}_{N}`` with N above
  the highest held (e.g., ``clt_1``, ``region_2``, ``score:clt_1``,
  ``note_3``); ``mint_id`` mints one.
- validate_id: The rule every given id obeys.
- resolve_id: Resolve a partial string or regex to a single ID.
- resolve_ids: Resolve a pattern to all matching IDs.
"""

from __future__ import annotations

import re
from collections.abc import Container, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import timetoshare

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


# region Manifest element ids

_REFUSED_ID_CHARACTERS = re.compile(r"[\x00-\x1f]")


def validate_id(element_id: str) -> str:
    """Return *element_id* if it is a valid given id, else raise.

    An id is any non-empty string without a control character (U+0000 to
    U+001F, which an XML attribute cannot carry). Whitespace, slashes,
    colons, dots, URIs and unicode are permitted.

    Args:
        element_id: The id to check.

    Returns:
        The same id.

    Raises:
        ValueError: If the id is no string, is empty or contains a control
            character.

    Examples:
        >>> validate_id("Bass Kick")
        'Bass Kick'
        >>> validate_id("a/b")
        'a/b'
        >>> validate_id("a\\tb")
        Traceback (most recent call last):
        ...
        ValueError: Invalid id 'a\\tb': an id must be non-empty and contain no control character
    """
    if (
        not isinstance(element_id, str)
        or not element_id
        or (_REFUSED_ID_CHARACTERS.search(element_id))
    ):
        raise ValueError(
            f"Invalid id {element_id!r}: an id must be non-empty and contain no "
            "control character"
        )
    return element_id


def quote_id(element_id: str) -> str:
    """Return *element_id* in single quotes, as a message names an element.

    An id is quoted as it is, an apostrophe included, so that every message
    names an element alike (:func:`validate_id` alone shows the ``repr``,
    which makes a refused control character visible).

    Examples:
        >>> quote_id("Bach's")
        "'Bach's'"
    """
    return f"'{element_id}'"


#: The most digits of a minted number read back from a held id: the numbers
#: are compared as int64, and an id with more digits is no minted id.
_MINTED_DIGITS = 18


def _highest_minted_number(prefix: str, taken: pa.Array) -> int:
    """The highest N among the ids ``{prefix}_{N}`` in *taken*, 0 for none.

    Read column-wise: no id of *taken* becomes a Python object.
    """
    pattern = rf"^{re.escape(prefix)}_[1-9][0-9]{{0,{_MINTED_DIGITS - 1}}}$"
    minted = taken.filter(pc.match_substring_regex(taken, pattern=pattern))
    if not len(minted):
        return 0
    numbers = pc.utf8_slice_codeunits(minted, start=len(prefix) + 1)
    return pc.max(pc.cast(numbers, pa.int64())).as_py()


def id_column(taken: pa.Array | pa.ChunkedArray | Iterable[str]) -> pa.Array:
    """Read the ids held in a scope as one string array without nulls.

    This is how :func:`mint_ids` reads its *taken* ids; a caller minting
    several times against the same ids reads them once with it.

    Args:
        taken: The ids (a string array, chunked or not, or any iterable of
            strings; nulls are dropped).

    Returns:
        A string array of the ids, nulls dropped.

    Examples:
        >>> id_column(["note_1", None, "note_3"]).to_pylist()
        ['note_1', 'note_3']
    """
    if isinstance(taken, pa.ChunkedArray):
        taken = taken.combine_chunks()
    if not isinstance(taken, pa.Array):
        taken = pa.array(list(taken), pa.string())
    return pc.drop_null(pc.cast(taken, pa.string()))


def mint_ids(
    prefix: str,
    taken: pa.Array | pa.ChunkedArray | Iterable[str],
    count: int,
) -> pa.Array:
    """Mint *count* ids ``{prefix}_{N}``, the N above the highest held.

    This is the one id generator of the library, for timelines, maps,
    manifest elements, MatchClaims and events alike. N continues above the
    highest number held in the scope under that prefix, so an id is never
    given again: a number freed by a gap is not reissued, and the ids minted
    one after another grow with the scope. It holds no state: the same
    prefix and the same taken ids always give the same ids, whatever was
    built before in the process. The taken ids are read column-wise, so a
    scope of a million events costs no Python object per id.

    Args:
        prefix: The type prefix, optionally led by a role
            (``"clt"``, ``"claim"``, ``"score:clt"``).
        taken: The ids already held in the scope (a string array or any
            iterable of strings; nulls are ignored).
        count: How many ids to mint.

    Returns:
        A string array of *count* ids, ascending by number.

    Examples:
        >>> mint_ids("claim", pa.array(["claim_2", "x"]), 3).to_pylist()
        ['claim_3', 'claim_4', 'claim_5']
        >>> len(mint_ids("claim", pa.array([], pa.string()), 0))
        0
    """
    first = _highest_minted_number(prefix, id_column(taken)) + 1
    numbers = pa.array(np.arange(first, first + count, dtype=np.int64))
    return pc.binary_join_element_wise(f"{prefix}_", pc.cast(numbers, pa.string()), "")


def mint_id(prefix: str, taken: pa.Array | pa.ChunkedArray | Iterable[str]) -> str:
    """Mint one id ``{prefix}_{N}``, N above the highest held (:func:`mint_ids`).

    Args:
        prefix: The type prefix, optionally led by a role.
        taken: The ids already held in the scope the new id must be unique in.

    Returns:
        The minted id.

    Examples:
        >>> mint_id("clt", ())
        'clt_1'
        >>> mint_id("clt", {"clt_1", "clt_3"})
        'clt_4'
        >>> mint_id("score:clt", {"score:clt_1", "clt_7"})
        'score:clt_2'
    """
    if not taken:
        return f"{prefix}_1"
    return mint_ids(prefix, taken, 1)[0].as_py()


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
    scope: str = "",
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
        scope: A description of the scope (``"group 'g'"``), for the error,
            or empty for the manifest of a timeline.

    Raises:
        ValueError: If a given id is already taken.
    """
    if element_id not in taken:
        return element_id
    if minted:
        return mint_id(prefix, taken)
    raise ValueError(
        f"{element} cannot join{' ' + scope if scope else ''}: its id "
        f"{quote_id(element_id)} is already held by {taken[element_id]}"
    )


def _plan_admissions(
    entries: Sequence[tuple[str, bool, str, str]],
    taken: Mapping[str, str],
    *,
    scope: str = "",
    avoid: Iterable[str] = (),
) -> dict[int, str]:
    """Plan the ids under which several elements join a scope together.

    A given id is kept; it must be held neither in the scope nor by another
    joining element. A minted id held in the scope, given to a joining
    element or kept by a joining element before it is minted again. Nothing
    is changed: every check precedes the plan, which maps the index of each
    element to mint again to its new id.

    Args:
        entries: ``(id, minted, prefix, description)`` of each joining
            element, in the order they claim their ids.
        taken: Every id of the scope, mapped to a description of its holder.
        scope: A description of the scope, for the error.
        avoid: Further ids no id minted again may take: those of the other
            scopes the elements belong to already.

    Raises:
        ValueError: If a given id is already held, in the scope or by
            another joining element.
    """
    claimed = dict(taken)
    for element_id, minted, prefix, element in entries:
        if not minted:
            _admit_id(
                element_id,
                minted=False,
                prefix=prefix,
                taken=claimed,
                element=element,
                scope=scope,
            )
            claimed[element_id] = element
    occupied = set(claimed) | set(avoid) | {entry[0] for entry in entries}
    plan: dict[int, str] = {}
    for index, (element_id, minted, prefix, element) in enumerate(entries):
        if not minted:
            continue
        if element_id in claimed:
            element_id = plan[index] = mint_id(prefix, occupied)
            occupied.add(element_id)
        claimed[element_id] = element
    return plan


class _OpenObject:
    """An object with a dictionary form that keeps the keys it does not read.

    A TimeToShare object may carry keys its reader does not know. The
    reader keeps them on the object read from the JSON object they were
    found in (:meth:`_keep_unknown_keys`), and the writer writes them back
    after its own keys (:meth:`_dict_form`). A subclass lists its known keys
    in ``_dict_keys`` and names in ``_manifest_path`` where its JSON object
    lies in a manifest. The writer builds the complete form, every known key
    written; the minimal form is TimeToShare's (:func:`_minimal_at`).
    """

    _dict_keys: ClassVar[tuple[str, ...]] = ()
    _manifest_path: ClassVar[tuple[str, ...]] = ()
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
    ) -> dict[str, Any]:
        """Return *fields*, then the unknown keys kept.

        *fields* is the complete form. With *minimal*, the object is written
        in TimeToShare's minimal form at its ``_manifest_path``.
        """
        written = dict(fields)
        written.update(deepcopy(dict(self._unknown_keys)))
        return _minimal_at(written, self._manifest_path) if minimal else written


def _minimal_at(value: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    """Return *value*, a JSON object at *path* of a manifest, in its minimal form.

    The minimal form is the one :func:`timetoshare.minimal` writes: the
    object is placed as the first entry of each array *path* names, from a
    bare manifest down (``()`` is the manifest itself), and taken back out.
    A timeline holding it states no unit, so no own-axis value is respelled.
    """
    holder: Any = value
    for key in reversed(path):
        holder = {key: [holder]}
    result = timetoshare.minimal(holder)
    for key in path:
        result = result[key][0]
    return result


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

    def to_dict(self, *, minimal: bool = True) -> dict[str, Any]:
        """Return this element as its JSON object, coordinates as bare values.

        Args:
            minimal: Write the minimal form (the default): every key at its
                default value left out, in nested objects too. With False,
                the complete form: every key written, its default included.
        """
        return self._dict_form(self._dict_fields(), minimal=minimal)

    def _dict_fields(self) -> dict[str, Any]:
        """Return the element's keys in written order, in the complete form."""
        raise NotImplementedError


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
