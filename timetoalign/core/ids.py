"""Identity management for TTA objects.

This module mints the random ids of the library, checks given ids, and
resolves an id pattern against the ids held.

Includes:
- IdMinter: The minter of random ids, ``{prefix}_{suffix}`` with a suffix
  of base-36 characters drawn from a seeded counter (``note_3kq9x0m2ztb7a``).
- seed_from: The seed of a minter, computed from content.
- content_minter: The process's minter of a content seed;
  ``reset_content_minters`` forgets them all.
- mint_ids: The generator of numbered ids, ``{prefix}_{N}`` with N above
  the highest held (e.g., ``clt_1``, ``region_2``, ``score:clt_1``,
  ``note_3``); ``mint_id`` mints one.
- validate_id: The rule every given id obeys.
- resolve_id: Resolve an exact id or a regular expression to a single ID.
- resolve_ids: Resolve a pattern to all matching IDs.
"""

from __future__ import annotations

import hashlib
import json
import operator
import re
import secrets
import threading
from collections.abc import Container, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, TypeGuard

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import timetoshare

from .enums import TimeUnit

if TYPE_CHECKING:
    from pyarrow.lib import StringType


# region Manifest element ids

_REFUSED_ID_CHARACTERS = re.compile(r"[\x00-\x1f]")


def validate_id(element_id: object) -> str:
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


# endregion

# region The minter

#: The number of base-36 characters of a minted suffix: 36^13 exceeds 2^64,
#: so a minter of this length never repeats a suffix.
ID_LENGTH = 13

_MASK64 = (1 << 64) - 1
_MIX_FACTOR_1 = 0xBF58476D1CE4E5B9
_MIX_FACTOR_2 = 0x94D049BB133111EB
_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"
_BASE36_CODES = np.frombuffer(_BASE36.encode("ascii"), dtype=np.uint8)


def _mix64(value: int) -> int:
    """SplitMix64's finalizer: a bijection on the integers modulo 2^64.

    Examples:
        >>> hex(_mix64(0x9E3779B97F4A7C15))
        '0xe220a8397b1dcdaf'
    """
    value &= _MASK64
    value ^= value >> 30
    value = (value * _MIX_FACTOR_1) & _MASK64
    value ^= value >> 27
    value = (value * _MIX_FACTOR_2) & _MASK64
    return value ^ (value >> 31)


def _undo_xorshift(value: int, shift: int) -> int:
    """The *x* for which ``x ^ (x >> shift)`` is *value* (64 bits)."""
    result = value
    for _ in range(64 // shift):
        result = value ^ (result >> shift)
    return result


def _unmix64(value: int) -> int:
    """The inverse of :func:`_mix64`.

    Examples:
        >>> hex(_unmix64(0xE220A8397B1DCDAF))
        '0x9e3779b97f4a7c15'
    """
    value = _undo_xorshift(value & _MASK64, 31)
    value = (value * pow(_MIX_FACTOR_2, -1, 1 << 64)) & _MASK64
    value = _undo_xorshift(value, 27)
    value = (value * pow(_MIX_FACTOR_1, -1, 1 << 64)) & _MASK64
    return _undo_xorshift(value, 30)


def _mix64_array(
    values: np.ndarray[Any, np.dtype[np.uint64]],
) -> np.ndarray[Any, np.dtype[np.uint64]]:
    """:func:`_mix64` over a uint64 array; the products wrap modulo 2^64."""
    values = values ^ (values >> np.uint64(30))
    values = values * np.uint64(_MIX_FACTOR_1)
    values = values ^ (values >> np.uint64(27))
    values = values * np.uint64(_MIX_FACTOR_2)
    return values ^ (values >> np.uint64(31))


def _base36(value: int, length: int) -> str:
    """*value* in base 36, digits ``0-9a-z``, left-padded with ``0`` to *length*."""
    digits: list[str] = []
    for _ in range(length):
        value, digit = divmod(value, 36)
        digits.append(_BASE36[digit])
    return "".join(reversed(digits))


class IdMinter:
    """The minter of random ids: a seed, a length and a counter.

    The k-th id a minter mints (k = 1, 2, …), whatever its prefix, is
    ``prefix + "_" + base36(mix64(mix64(seed) + k) mod 36^length)``, all
    arithmetic modulo 2^64, ``mix64`` being SplitMix64's finalizer and the
    suffix left-padded with ``0`` to *length* characters. One seed gives the
    same ids in every process; the counter is shared by all prefixes. At the
    default length of 13 a minter never repeats a suffix; a shorter one
    guarantees nothing, and a repeat is refused where the id joins its scope
    like any other collision.

    Minting is thread-safe: each id draws its own step of the counter.

    Args:
        seed: The seed, taken modulo 2^64 (``-1`` is ``2**64 - 1``); ``None``
            draws 64 random bits.
        length: The number of suffix characters, 1 to 13.

    Raises:
        TypeError: If *seed* or *length* is no integer.
        ValueError: If *length* is outside 1 to 13.

    Examples:
        >>> minter = IdMinter(seed=0)
        >>> minter.mint("note")
        'note_1be6k8ij6dzad'
        >>> minter.mint("claim")
        'claim_3cccv6pc7fclm'
        >>> minter.mint_many("note", 2).to_pylist()
        ['note_0gloaudxbozjk', 'note_2sjbpc8k7u4gk']
        >>> IdMinter(seed=0, length=7).mint("note")
        'note_ij6dzad'
        >>> IdMinter(seed=-1).seed == 2**64 - 1
        True
    """

    def __init__(self, seed: int | None = None, *, length: int = ID_LENGTH) -> None:
        seed = secrets.randbits(64) if seed is None else operator.index(seed)
        length = operator.index(length)
        if not 1 <= length <= ID_LENGTH:
            raise ValueError(
                f"An id length is an integer from 1 to {ID_LENGTH}, not {length!r}"
            )
        self._seed = seed & _MASK64
        self._length = length
        self._start = _mix64(self._seed)
        self._counter = 0
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"IdMinter(seed={self._seed}, length={self._length})"

    @property
    def seed(self) -> int:
        """The seed, from 0 to 2^64 - 1."""
        return self._seed

    @property
    def length(self) -> int:
        """The number of suffix characters."""
        return self._length

    def _draw(self, count: int) -> int:
        """Reserve the next *count* steps of the counter; return the first."""
        with self._lock:
            first = self._counter + 1
            self._counter += count
        return first

    def mint(self, prefix: str) -> str:
        """Mint the next id under *prefix*.

        Args:
            prefix: The type prefix, optionally led by a role
                (``"note"``, ``"score:clt"``).

        Returns:
            The id ``{prefix}_{suffix}``.

        Examples:
            >>> IdMinter(seed=7).mint("score:clt")
            'score:clt_1hvif5xc2r1fv'
        """
        step = self._draw(1)
        suffix = _mix64((self._start + step) & _MASK64)
        if self._length < ID_LENGTH:
            suffix %= 36**self._length
        return f"{prefix}_{_base36(suffix, self._length)}"

    def mint_many(self, prefix: str, count: int) -> pa.Array[pa.Scalar[StringType]]:
        """Mint the next *count* ids under *prefix*, column-wise.

        The ids are those *count* calls of :meth:`mint` would give, computed
        without a Python object per id.

        Args:
            prefix: The type prefix, optionally led by a role.
            count: How many ids to mint, 0 or more.

        Returns:
            A string array of *count* ids, in minting order.

        Raises:
            ValueError: If *count* is negative, or the ids exceed the 2 GiB a
                string array holds.

        Examples:
            >>> IdMinter(seed=7, length=4).mint_many("event", 3).to_pylist()
            ['event_r1fv', 'event_a7jx', 'event_ial0']
        """
        if count < 0:
            raise ValueError(f"Cannot mint a negative number of ids: {count}")
        head = f"{prefix}_".encode()
        width = len(head) + self._length
        if count * width > np.iinfo(np.int32).max:
            raise ValueError(
                f"{count} ids of {width} bytes exceed the 2 GiB a string array holds"
            )
        first = self._draw(count)
        steps = np.arange(first, first + count, dtype=np.uint64)
        values = _mix64_array(np.uint64(self._start) + steps)
        if self._length < ID_LENGTH:
            values %= np.uint64(36**self._length)
        data = np.empty((count, width), dtype=np.uint8)
        data[:, : len(head)] = np.frombuffer(head, dtype=np.uint8)
        for column in range(width - 1, len(head) - 1, -1):
            data[:, column] = _BASE36_CODES[values % np.uint64(36)]
            values //= np.uint64(36)
        offsets = np.arange(0, (count + 1) * width, width, dtype=np.int32)
        return pa.StringArray.from_buffers(
            count, pa.py_buffer(offsets.tobytes()), pa.py_buffer(data.tobytes())
        )


def _encode_seed_part(part: object) -> bytes:
    """The bytes :func:`seed_from` hashes for *part*."""
    if isinstance(part, (bytes, bytearray)):
        return bytes(part)
    if isinstance(part, str):
        return part.encode("utf-8")
    try:
        return json.dumps(part, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except TypeError:
        raise TypeError(
            f"Cannot compute an id seed from a {type(part).__name__}: it has no JSON form"
        ) from None


def seed_from(*parts: object) -> int:
    """Compute a minter seed from content.

    The seed is the first 8 bytes, big-endian, of the SHA-256 digest over
    the parts, each part preceded by its length in 8 big-endian bytes and
    encoded first: bytes as they are, a ``str`` as UTF-8, anything else as
    JSON with sorted keys and no spaces. The length prefix keeps
    ``("ab", "c")`` and ``("a", "bc")`` apart.

    Args:
        *parts: The content (a loader's bytes, a class prefix and the
            values an object is built from).

    Returns:
        An integer from 0 to 2^64 - 1.

    Raises:
        TypeError: If a part is neither bytes nor ``str`` and JSON cannot
            encode it.

    Examples:
        >>> seed_from("clt", {"length": 16})
        8558461638158943518
        >>> seed_from() == int.from_bytes(hashlib.sha256().digest()[:8], "big")
        True
    """
    digest = hashlib.sha256()
    for part in parts:
        data = _encode_seed_part(part)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return int.from_bytes(digest.digest()[:8], "big")


_CONTENT_MINTERS: dict[tuple[int, int], IdMinter] = {}
_CONTENT_MINTERS_LOCK = threading.Lock()


def content_minter(*parts: object, length: int = ID_LENGTH) -> IdMinter:
    """Return the process's minter of a content seed.

    The seed is :func:`seed_from` of *parts*, the class's prefix first. The
    process keeps one minter per seed and length, so objects of equal
    content built one after another get distinct ids, in construction order,
    and a script gives the same ids on every run. A caller given a seed
    mints with a fresh ``IdMinter(seed)`` instead, which gives the same ids
    whatever ran before it.

    Args:
        *parts: The content the seed is computed from, the class's prefix
            first.
        length: The number of suffix characters, 1 to 13.

    Returns:
        The minter, shared by every caller of the same content and length.

    Examples:
        >>> reset_content_minters()
        >>> content_minter("region", 0, 4).mint("region")
        'region_16y3ae79lr7h9'
        >>> content_minter("region", 0, 4).mint("region")
        'region_35z1ve4l9h0nl'
        >>> content_minter("region", 0, 4).seed == seed_from("region", 0, 4)
        True
    """
    key = (seed_from(*parts), length)
    with _CONTENT_MINTERS_LOCK:
        minter = _CONTENT_MINTERS.get(key)
        if minter is None:
            minter = _CONTENT_MINTERS[key] = IdMinter(key[0], length=length)
    return minter


def reset_content_minters() -> None:
    """Forget every minter :func:`content_minter` holds.

    The next content minter of a seed starts its counter again, so a run
    after the reset mints the ids a fresh process would.
    """
    with _CONTENT_MINTERS_LOCK:
        _CONTENT_MINTERS.clear()


# endregion

# region Numbered ids and identified elements


#: The most digits of a minted number read back from a held id: the numbers
#: are compared as int64, and an id with more digits is no minted id.
_MINTED_DIGITS = 18


def _highest_minted_number(prefix: str, taken: pa.Array[pa.Scalar[StringType]]) -> int:
    """The highest N among the ids ``{prefix}_{N}`` in *taken*, 0 for none.

    Read column-wise: no id of *taken* becomes a Python object.
    """
    pattern = rf"^{re.escape(prefix)}_[1-9][0-9]{{0,{_MINTED_DIGITS - 1}}}$"
    minted = taken.filter(pc.match_substring_regex(taken, pattern=pattern))
    if not len(minted):
        return 0
    numbers = pc.utf8_slice_codeunits(minted, start=len(prefix) + 1)
    return pc.max(numbers.cast(pa.int64())).as_py()


def _is_arrow_column(value: object) -> TypeGuard[pa.Array[Any] | pa.ChunkedArray[Any]]:
    """Whether *value* is a PyArrow array, chunked or not, of any value type."""
    return isinstance(value, (pa.Array, pa.ChunkedArray))


def id_column(
    taken: pa.Array[Any] | pa.ChunkedArray[Any] | Iterable[str | None],
) -> pa.Array[pa.Scalar[StringType]]:
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
    if _is_arrow_column(taken):
        column = taken.combine_chunks() if isinstance(taken, pa.ChunkedArray) else taken
    else:
        column = pa.array(list(taken), pa.string())
    return pc.drop_null(column.cast(pa.string()))


def mint_ids(
    prefix: str,
    taken: pa.Array[Any] | pa.ChunkedArray[Any] | Iterable[str | None],
    count: int,
) -> pa.Array[pa.Scalar[StringType]]:
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
    return pc.utf8_replace_slice(
        numbers.cast(pa.string()), start=0, stop=0, replacement=f"{prefix}_"
    )


def mint_id(
    prefix: str, taken: pa.Array[Any] | pa.ChunkedArray[Any] | Iterable[str | None]
) -> str:
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
        default_factory=dict[str, Any], kw_only=True, repr=False, compare=False
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
    """Resolve an exact id or a regular expression to the matching IDs.

    Matching precedence:
    1. Exact match: Returns ``[pattern]`` if it is one of the IDs.
    2. Regular expression: Every ID the pattern matches in full
       (``re.fullmatch``), in the order given.

    A fragment of an id matches nothing: minted ids are random, so a
    substring would pick an unrelated one. Spell the rest of the id as a
    regular expression instead (``"score:.*"``).

    Args:
        pattern: An ID, or a regular expression matching whole IDs.
        available_ids: The IDs to search.

    Returns:
        List of matching IDs (may be empty, also for an invalid regular
        expression).

    Examples:
        >>> resolve_ids("clt_1", ["clt_1", "clt_2", "dlt_1"])
        ['clt_1']

        >>> resolve_ids(r"clt_\\d+", ["clt_1", "clt_2", "dlt_1", "score:clt_1"])
        ['clt_1', 'clt_2']

        >>> resolve_ids("clt", ["clt_1", "clt_2", "dlt_1"])
        []

        >>> resolve_ids("score:.*", ["score:clt_1", "perf:dlt_1"])
        ['score:clt_1']
    """
    if not pattern or not available_ids:
        return []

    if pattern in available_ids:
        return [pattern]

    try:
        regex = re.compile(pattern)
    except re.error:
        return []
    return [id_ for id_ in available_ids if regex.fullmatch(id_)]


def resolve_id(
    pattern: str,
    available_ids: list[str],
    *,
    warn_multiple: bool = True,
) -> str:
    """Resolve an exact id or a regular expression to a single ID.

    The candidates are those of :func:`resolve_ids`: an exact match, else
    every ID the pattern matches in full as a regular expression. When
    several IDs match, returns the first and optionally warns (see
    ``warn_multiple``).

    Args:
        pattern: An ID, or a regular expression matching whole IDs.
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

        >>> resolve_id("clt_.*", ["clt_1", "clt_2", "dlt_1"])  # warns
        'clt_1'

        >>> resolve_id("clt", ["clt_1", "clt_2"])
        Traceback (most recent call last):
            ...
        KeyError: "No ID matches pattern 'clt'. Available: clt_1, clt_2"
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
