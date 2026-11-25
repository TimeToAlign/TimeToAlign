"""Reading and writing TimeToShare manifests.

A TimeToShare manifest is one JSON object: the version marker
``timetoshare``, an array of root ``timelines`` and, optionally, the
manifest's ``id``, ``name`` and ``meta``, its ``groups``, its
``measure_maps``, its ``resources`` (the files it names, such as the
MatchClaim table beside it), its ``graphical``, and its inline JSON-LD
``@context`` with the manifest's ``statements`` and ``provenance`` (see
:mod:`~timetoalign.timelines.statements`).
:class:`~timetoalign.timelines.Timeline`,
:class:`~timetoalign.timelines.TimelineGroup` and
:class:`~timetoalign.alignment.AlignmentBundle` each write one with
``to_dict`` and read one with ``from_dict``. This module holds what they
share: the version marker, the reading of the manifest around its timelines
and groups, its resources (:class:`Resource`), the writing of that
envelope, of the measure maps its
timelines name and of the ``@context`` its statements need, and the codec of
a bare value on a timeline's own axis.

Every manifest read and every manifest written is validated whole by
TimeToShare's validator (:func:`timetoshare.validate`): the library reads
and writes exactly the manifests the format admits, and a refusal is the
validator's ``timetoshare.SchemaValidationError`` (a ``ValueError``) with
its ``code`` and message unchanged. Its warnings are not reported. A few
rules are the library's alone and checked by its readers of the elements: a
negative child offset, a region ending before it starts, the rows of a
measure map that do not follow one another, and maps the library cannot
evaluate.

A value on a timeline's own axis (``length``, ``offset``, region ``start``
and ``end``, the ``coordinate`` of a flow-control element and the ``by`` of
a destination) is written bare, represented as the axis's number type
selects: a JSON integer for ``int``, a JSON number for ``float``, a
``[numerator, denominator]`` pair for ``fraction``. Reading follows the
number type, never the lexical form: an integral JSON number on an ``int``
axis is that integer, an integer on a ``float`` axis that float, and a pair
with integral float members (``[1, 8.0]``) that Fraction. A value no reading
of the number type admits is refused.

A manifest file writes every ID, an element's own and every reference to
one, as ``#`` followed by the ID (``"id": "#clt_1"``); every object of the
library holds IDs bare (``clt_1``). The boundary is the whole manifest: a
dict carrying the ``timetoshare`` version marker, which ``to_dict`` returns
and ``from_dict`` takes, or a file, is in the file form; the dict of one
object inside it (a conversion map, a region, a measure map) is bare.
:meth:`Manifest.read` and :meth:`Manifest.write` convert, through
TimeToShare's :func:`timetoshare.with_bare_ids` and
:func:`timetoshare.with_file_ids`. Statements are the same in both forms: a
value naming an element is ``{"id": "#clt_1"}`` in memory too.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self
from urllib.parse import unquote, urlsplit

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import timetoshare
from timetoshare import FORMAT_CONTEXT, load_catalogue, loads
from timetoshare.serialization import Serialization
from timetoshare.tables import TableError, projection_problems

from timetoalign.core import Coordinate, Event, NumberType, Projection, TimeUnit
from timetoalign.core.fields import _null_column, derive_arrow_schema
from timetoalign.core.ids import _OpenObject, mint_id, quote_id, validate_id
from timetoalign.core.time import (
    COORDINATE_CELL_TYPE,
    PACKAGE_NUMBER_TYPE,
    RATIONAL_STRUCT_TYPE,
    build_number_struct_array,
    is_fraction_json,
    number_from_json,
    number_to_json,
    numbers_from_package,
    package_numbers,
)
from timetoalign.maps.base import MEASURE_MAP_ID_PREFIX
from timetoalign.storage.events import refuse_shared_projection_ids

from .statements import (
    STATE_KEYS,
    declared_prefixes,
    read_statement_state,
    renamed_state,
)

if TYPE_CHECKING:
    from .base import Timeline
    from .groups import _IdScope
    from .structure import MeasureMap

# region Bare values on a timeline's own axis


def axis_value_to_json(coordinate: Coordinate) -> int | float | list[int]:
    """Write a coordinate on its timeline's own axis as a bare value."""
    return number_to_json(coordinate.value)


def _json_kind(data: Any) -> str | None:
    """Name the number representation of *data*, or None if it is none."""
    if isinstance(data, bool):
        return None
    if isinstance(data, int):
        return "an int"
    if isinstance(data, float):
        return "a float"
    if is_fraction_json(data):
        return "a fraction"
    return None


_AXIS_KIND = {
    NumberType.int: "an int",
    NumberType.float: "a float",
    NumberType.fraction: "a fraction",
}


def _suggestion(data: Any, number_type: NumberType) -> str:
    """What to write instead of *data* on an axis of *number_type*.

    The rewritten value when the rewriting is exact, otherwise the axis's
    number type with its article.
    """
    value = Fraction(number_from_json(data))
    if number_type is NumberType.fraction:
        return json.dumps(number_to_json(value))
    if number_type is NumberType.float:
        written = float(value)
        return repr(written) if Fraction(written) == value else _AXIS_KIND[number_type]
    if value.denominator == 1:
        return str(int(value))
    return _AXIS_KIND[number_type]


def axis_value_from_json(
    data: Any,
    *,
    key: str,
    owner: str,
    unit: TimeUnit,
    number_type: NumberType,
) -> Coordinate:
    """Read a bare value on a timeline's own axis.

    The axis's number type decides how the value is read
    (:func:`~timetoalign.core.time.number_from_json`): an integral JSON
    number is read on an ``int`` axis, an integer on a ``float`` axis, a pair
    with integral float members on a ``fraction`` axis. A non-integral value
    on an ``int`` axis, a pair on an ``int`` or ``float`` axis and a bare
    number on a ``fraction`` axis are refused.

    Args:
        data: The JSON value.
        key: The key the value was found under, for the error.
        owner: The id of the object holding the key, for the error.
        unit: The axis's unit.
        number_type: The axis's number type.

    Returns:
        The coordinate on the axis.

    Raises:
        ValueError: If *data* is no number, or no reading of the axis's
            number type admits it, for example ``length 32 of
            'score' is an int on a fraction axis: write [32, 1]``; the
            rewritten value is named only when the rewriting is exact,
            otherwise the axis's number type (``write a float``).
    """
    kind = _json_kind(data)
    if kind is None:
        raise ValueError(f"{key} {json.dumps(data)} of '{owner}' is not a number")
    if isinstance(data, float) and not math.isfinite(data):
        raise ValueError(f"{key} {data!r} of '{owner}' is not finite")
    try:
        value = number_from_json(data, number_type)
    except ValueError as error:
        if "int64" in str(error):
            raise ValueError(f"{key} of '{owner}': {error}") from error
        raise ValueError(
            f"{key} {json.dumps(data)} of '{owner}' is {kind} on "
            f"{_AXIS_KIND[number_type]} axis: write {_suggestion(data, number_type)}"
        ) from error
    return Coordinate(value, unit, number_type=number_type)


def spell(value: Any) -> str:
    """Spell a JSON value as a message quotes it.

    A number as JSON writes it, an array as its items separated by ``, ``
    within brackets (``[32, 1]``), anything else as compact JSON: the
    spelling of the TimeToShare validators.
    """
    if isinstance(value, list):
        return "[" + ", ".join(spell(item) for item in value) + "]"
    if isinstance(value, bool) or not isinstance(value, int | float):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return repr(value)


# endregion


# region Objects


def require_object(
    data: Any, what: str, required: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Return *data* if it is a JSON object holding every *required* key.

    Raises:
        ValueError: If *data* is not an object or lacks a required key.
    """
    if not isinstance(data, dict):
        raise ValueError(f"{what} must be an object, got {type(data).__name__}")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{what} lacks the key(s) {missing}")
    return data


def read_name(data: Mapping[str, Any], what: str) -> str | None:
    """Read an optional ``name``: a string, or null (the default).

    Raises:
        ValueError: If the name is neither.
    """
    name = data.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError(f"{what} has a non-string name {name!r}")
    return name


def read_meta(data: Mapping[str, Any], what: str) -> dict[str, Any]:
    """Read an optional ``meta``: an object, ``{}`` by default.

    Raises:
        ValueError: If the meta is not an object.
    """
    meta = data.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError(f"{what} has a non-object meta")
    return copy.deepcopy(meta)


def read_id(value: Any, what: str) -> str:
    """Read a given id.

    Raises:
        ValueError: If the id is not a string or not a valid id.
    """
    if not isinstance(value, str):
        raise ValueError(f"{what} id must be a string, got {value!r}")
    return validate_id(value)


def _manifest_ids_of(timelines: Iterable[Timeline]) -> dict[str, str]:
    """Map every id of the manifests *timelines* belong to, to its holder's description.

    A timeline's manifest is its tree and, while the tree is held by groups
    or bundles, every tree those hold, with each group's and bundle's given
    id (``Timeline._manifest_ids``, which asks this for one timeline). Each
    tree and each group or bundle is read once however many of *timelines*
    it holds, so the roots of one bundle cost one pass over the bundle. An
    id held twice keeps its first holder: the trees in the order of
    *timelines*, then the groups and bundles in the order they are met.
    """
    ids: dict[str, str] = {}
    roots: set[int] = set()
    containers: dict[int, _IdScope] = {}
    for timeline in timelines:
        root = timeline._tree_root()
        if id(root) in roots:
            continue
        roots.add(id(root))
        for element_id, description in root._tree_ids().items():
            ids.setdefault(element_id, description)
        for container in root._scope_containers():
            containers.setdefault(id(container), container)
    for container in containers.values():
        for element_id, description in container._scope_ids().items():
            ids.setdefault(element_id, description)
    return ids


def _leave_out_unwritten(
    fields: dict[str, Any], roots: list[Timeline], written: set[str]
) -> dict[str, Any]:
    """Drop from the statements written the elements the write leaves out.

    A timeline or a group written on its own writes part of the manifests
    it belongs to: a value may name an element of those manifests
    (:func:`_manifest_ids_of` of the roots written) that this one does not
    hold. The pair naming it, its provenance entry and every ``basis`` or
    ``stated_by`` item naming it are left out. A reference naming no
    element of any of those manifests stays, for the check of the manifest
    written to refuse.

    Args:
        fields: The manifest written, IDs bare.
        roots: The roots written.
        written: Every ID the manifest written holds.
    """
    known = _manifest_ids_of(roots).keys()
    return timetoshare.with_references_renamed(fields, dict.fromkeys(known - written))


# endregion


# region Resources

_RESOURCE_SCHEMA = timetoshare.load_schema()["$defs"]["Resource"]["properties"]

INTEGRITY_ALGORITHMS: tuple[str, ...] = tuple(
    _RESOURCE_SCHEMA["integrity"]["properties"]["algorithm"]["enum"]
)
"""The hash functions a resource's ``integrity`` may name, as PREMIS names them."""

PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
"""The media type of the Parquet tables a package holds."""

TABLE_RESOURCE_TYPES: tuple[str, ...] = ("MatchClaim", "Projection", "Box")
"""The resource types a manifest names at most once: its package tables."""

_MEDIA_TYPE = re.compile(_RESOURCE_SCHEMA["media_type"]["pattern"])
_DIGEST = re.compile(r"[0-9a-f]+")


@dataclass(frozen=True)
class Integrity(_OpenObject):
    """The digest of a resource's raw bytes and the hash function computing it.

    Attributes:
        algorithm: The hash function as the PREMIS vocabulary of
            cryptographic hash functions names it, one of
            :data:`INTEGRITY_ALGORITHMS` (``"SHA-256"``).
        digest: The digest in lower-case hexadecimal.

    Raises:
        ValueError: If the algorithm is none of :data:`INTEGRITY_ALGORITHMS`
            or the digest is no lower-case hexadecimal string.

    Examples:
        >>> Integrity("SHA-256", "5f9a").to_dict()
        {'algorithm': 'SHA-256', 'digest': '5f9a'}
    """

    _dict_keys: ClassVar[tuple[str, ...]] = ("algorithm", "digest")

    algorithm: str
    digest: str
    _unknown_keys: Mapping[str, Any] = field(
        default_factory=dict, kw_only=True, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.algorithm not in INTEGRITY_ALGORITHMS:
            raise ValueError(
                f"{self.algorithm!r} is no hash function a resource's integrity "
                f"names: write one of {list(INTEGRITY_ALGORITHMS)}"
            )
        if not isinstance(self.digest, str) or not _DIGEST.fullmatch(self.digest):
            raise ValueError(
                f"the digest {self.digest!r} is not written in lower-case hexadecimal"
            )

    @classmethod
    def of_bytes(cls, data: bytes) -> Integrity:
        """The SHA-256 digest of *data*.

        Examples:
            >>> Integrity.of_bytes(b"").digest[:12]
            'e3b0c44298fc'
        """
        return cls("SHA-256", hashlib.sha256(data).hexdigest())

    @classmethod
    def from_dict(cls, data: Any, what: str = "a resource's integrity") -> Integrity:
        """Read an ``integrity`` object, keeping its unknown keys.

        Raises:
            ValueError: If *data* is no object holding ``algorithm`` and
                ``digest``, or either is malformed.
        """
        require_object(data, what, cls._dict_keys)
        integrity = cls(data["algorithm"], data["digest"])
        integrity._keep_unknown_keys(data)
        return integrity

    def to_dict(self) -> dict[str, Any]:
        """Write the ``integrity`` object."""
        return self._dict_form(
            {"algorithm": self.algorithm, "digest": self.digest}, minimal=False
        )


@dataclass(frozen=True)
class Resource(_OpenObject):
    """A file the manifest names: one entry of its ``resources``.

    A package is a manifest and the files it names. Each resource says what
    its rows are (``type``: ``"MatchClaim"`` for the MatchClaim table,
    ``"Projection"`` for the Projections table, or the type of the events
    an event table holds), for which timelines it holds data and where it
    lies, relative to the manifest, so that a bare file name lies beside
    it; optionally its media type, size in bytes and the digest of its
    bytes. IDs are bare, as the library holds them.

    Attributes:
        id: The resource's ID, held by no other element of the manifest
            (a new entry's is minted).
        type: What the file's rows are.
        timelines: The IDs of the timelines the file holds data for.
        url: Its location, relative to the manifest's own.
        media_type: Its media type, or None.
        size: Its size in bytes, or None.
        integrity: The digest of its bytes, or None.

    Raises:
        ValueError: If the ID is not a valid ID, the type is empty, a
            timeline ID is invalid or given twice, the URL is no string, the
            media type is malformed, or the size is no integer at least 0.

    Examples:
        >>> resource = Resource(
        ...     "resource_1", "MatchClaim", ("clt_1", "dpt_1"), "match_claims.parquet"
        ... )
        >>> resource.to_dict()
        {'id': 'resource_1', 'type': 'MatchClaim', 'timelines': ['clt_1', 'dpt_1'], 'url': 'match_claims.parquet'}
        >>> Resource.from_dict(resource.to_dict()) == resource
        True
    """

    _dict_keys: ClassVar[tuple[str, ...]] = (
        "id",
        "type",
        "timelines",
        "url",
        "media_type",
        "size",
        "integrity",
    )

    id: str
    type: str
    timelines: tuple[str, ...]
    url: str
    media_type: str | None = None
    size: int | None = None
    integrity: Integrity | None = None
    _unknown_keys: Mapping[str, Any] = field(
        default_factory=dict, kw_only=True, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        validate_id(self.id)
        what = f"resource {quote_id(self.id)}"
        if not isinstance(self.type, str) or not self.type:
            raise ValueError(f"{what} has no type")
        object.__setattr__(self, "timelines", tuple(self.timelines))
        for timeline_id in self.timelines:
            read_id(timeline_id, f"{what}: a timeline")
        if len(set(self.timelines)) != len(self.timelines):
            raise ValueError(f"{what} names a timeline twice: {list(self.timelines)}")
        if not isinstance(self.url, str):
            raise ValueError(f"{what} has a non-string url {self.url!r}")
        if self.media_type is not None and (
            not isinstance(self.media_type, str)
            or not _MEDIA_TYPE.fullmatch(self.media_type)
        ):
            raise ValueError(f"{what} has a malformed media type {self.media_type!r}")
        if self.size is not None and (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size < 0
        ):
            raise ValueError(
                f"{what} has the size {self.size!r}: a number of bytes, at least 0"
            )

    @classmethod
    def of_file(
        cls,
        path: str | PathLike[str],
        *,
        id: str,  # noqa: A002 - the key's name
        type: str,  # noqa: A002 - the key's name
        timelines: Iterable[str],
        url: str,
        media_type: str | None = PARQUET_MEDIA_TYPE,
    ) -> Resource:
        """Describe the file *path*: its size and SHA-256 digest from its bytes.

        Args:
            path: The file.
            id: The resource's ID.
            type: What its rows are.
            timelines: The timelines it holds data for.
            url: Its location relative to the manifest.
            media_type: Its media type, Parquet's by default.
        """
        data = Path(path).read_bytes()
        return cls(
            id,
            type,
            tuple(timelines),
            url,
            media_type,
            len(data),
            Integrity.of_bytes(data),
        )

    @classmethod
    def from_dict(cls, data: Any, what: str = "a resource") -> Resource:
        """Read one entry of ``resources``, IDs bare, keeping its unknown keys.

        Raises:
            ValueError: If *data* is no object holding ``id``, ``type``,
                ``timelines`` and ``url``, or an entry is malformed.
        """
        require_object(data, what, ("id", "type", "timelines", "url"))
        resource_id = read_id(data["id"], what)
        timelines = data["timelines"]
        if not isinstance(timelines, list):
            raise ValueError(
                f"resource {quote_id(resource_id)}'s 'timelines' must be an array"
            )
        integrity = data.get("integrity")
        resource = cls(
            resource_id,
            data["type"],
            tuple(timelines),
            data["url"],
            data.get("media_type"),
            data.get("size"),
            (
                None
                if integrity is None
                else Integrity.from_dict(
                    integrity, f"resource {quote_id(resource_id)}'s integrity"
                )
            ),
        )
        resource._keep_unknown_keys(data)
        return resource

    def to_dict(self) -> dict[str, Any]:
        """Write the entry, IDs bare: the optional keys only with a value.

        A resource has no defaults, so its minimal and complete forms are
        one.
        """
        fields: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "timelines": list(self.timelines),
            "url": self.url,
        }
        if self.media_type is not None:
            fields["media_type"] = self.media_type
        if self.size is not None:
            fields["size"] = self.size
        if self.integrity is not None:
            fields["integrity"] = self.integrity.to_dict()
        return self._dict_form(fields, minimal=False)

    def path_in(self, directory: str | PathLike[str]) -> Path:
        """The file's path for a manifest lying in *directory* (:func:`resource_path`)."""
        return resource_path(directory, self.url)


def resource_path(directory: str | PathLike[str], url: str) -> Path:
    """The path of the file *url* names for a manifest lying in *directory*.

    Raises:
        ValueError: If the URL names a scheme or a host, or no path: the
            library reads and writes the files of a package on disk,
            relative to its manifest.

    Examples:
        >>> resource_path("package", "tables/match%20claims.parquet").as_posix()
        'package/tables/match claims.parquet'
    """
    parts = urlsplit(url)
    if parts.scheme or parts.netloc or not parts.path:
        raise ValueError(
            f"a resource lies at {url!r}; the library reads and writes only a "
            f"file named relative to the manifest"
        )
    return Path(directory) / unquote(parts.path)


# endregion


# region The event tables

PROJECTIONS_FILE_NAME = "projections.parquet"
"""The file name of a package's Projections table beside its manifest."""

BOXES_FILE_NAME = "boxes.parquet"
"""The file name of a package's Box table beside its manifest."""

#: The columns an event table writes before the properties of its event type.
_EVENT_TABLE_HEADER = ("id", "timeline_id", "unit", "number_type")


def _package_type(type_: pa.DataType) -> pa.DataType:
    """*type_* with every number in it (a number struct or a coordinate cell)
    as the package number, at whatever depth."""
    if type_ in (RATIONAL_STRUCT_TYPE, COORDINATE_CELL_TYPE):
        return PACKAGE_NUMBER_TYPE
    if pa.types.is_struct(type_):
        return pa.struct(
            [pa.field(field.name, _package_type(field.type)) for field in type_]
        )
    return type_


def event_table_schema(event_type: type[Event]) -> pa.Schema:
    """The columns of the package table of *event_type*.

    ``id``, ``timeline_id``, ``unit`` and ``number_type`` (strings), then
    every property of the event type after ``id``, in field order, as the
    type derives it, each number in it written as the package number
    (:data:`~timetoalign.core.time.PACKAGE_NUMBER_TYPE`): the unit and the
    number type are stated once per row. For :class:`~timetoalign.core.Projection`
    it is TimeToShare's Projections table
    (:data:`timetoshare.tables.PROJECTION_SCHEMA`).

    Args:
        event_type: An event type (:class:`~timetoalign.core.Event` or a
            subclass).

    Returns:
        The schema.

    Examples:
        >>> from timetoalign.core import Box
        >>> event_table_schema(Box).names
        ['id', 'timeline_id', 'unit', 'number_type', 'start', 'duration', 'bounding_box', 'quadrilateral']
    """
    derived = derive_arrow_schema(event_type)
    return pa.schema(
        [
            *(pa.field(name, pa.string()) for name in _EVENT_TABLE_HEADER),
            *(
                pa.field(field.name, _package_type(field.type))
                for field in derived
                if field.name != "id"
            ),
        ]
    )


def _packaged(
    array: Any, type_: pa.DataType, number_type: NumberType, types: pa.Array
) -> pa.Array:
    """A memory column of an event table as its package column of *type_*.

    Each number (where *type_* is the package number) is written once by
    the row's number type (:func:`~timetoalign.core.time.package_numbers`),
    a struct member by member, by name, and anything else cast; a column
    the memory table lacks (``None``) or holds untyped is null throughout.
    """
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if array is None or pa.types.is_null(array.type):
        return _null_column(len(types), type_)
    if type_ == PACKAGE_NUMBER_TYPE:
        stored = build_number_struct_array(array, number_type=number_type)
        return package_numbers(stored, pc.if_else(stored.is_valid(), types, None))
    if not pa.types.is_struct(type_):
        return array.cast(type_)
    members = dict(
        zip((field.name for field in array.type), array.flatten(), strict=True)
    )
    return pa.StructArray.from_arrays(
        [_packaged(members[f.name], f.type, number_type, types) for f in type_],
        fields=list(type_),
        mask=array.is_null(),
    )


#: The columns of a timeline's events an event table holds without writing
#: them: it is the table of one event type, and an event's temporal type and
#: end follow from its ``start`` and ``duration``.
_IMPLIED_EVENT_COLUMNS = frozenset({"event_type", "temporal_type", "end"})


def _refuse_unwritten_columns(
    rows: pa.Table, schema: pa.Schema, event_type: type[Event], timeline_id: str
) -> None:
    """Refuse *rows* (one timeline's events of *event_type*) when one of them
    gives a value in a column the event table of *schema* does not have.

    Raises:
        ValueError: Naming the first such event, its timeline, the column
            and the value, and the members the event type holds.
    """
    kind = event_type.__name__
    for name in rows.column_names:
        if name in schema.names or name in _IMPLIED_EVENT_COLUMNS:
            continue
        column = rows.column(name)
        if column.null_count == len(column):
            continue
        at = pc.index(column.is_valid(), True).as_py()
        held = ", ".join(event_type.model_fields)
        raise ValueError(
            f"{kind} {quote_id(rows.column('id')[at].as_py())} of timeline "
            f"{quote_id(timeline_id)} gives {name} {column[at].as_py()!r}, which "
            f"the {kind} table has no column for: a {kind} holds {held}."
        )


def event_table(timelines: Iterable[Timeline], event_type: type[Event]) -> pa.Table:
    """The package table of *event_type*: every such event each timeline holds itself.

    One row per event, timeline by timeline in the order given and in each
    timeline's event order, in the columns of :func:`event_table_schema`:
    the event's ``id``, the ``timeline_id`` of the timeline holding it (a
    child's event is the child's), the ``unit`` and ``number_type`` of that
    timeline's axis, once for the row, then every property, each number
    written once, in the member the number type names (the package form,
    :func:`~timetoalign.core.time.package_numbers`), however deep it lies
    (a Box's shape). Nothing passes through a float that is not one.

    The table holds what the event type holds, and nothing is dropped: a
    row of a timeline's events giving a value in a column the table does
    not have (an event's ``name``, a further column such as ``staff``) is
    refused, naming the column, the event type and the timeline. The
    columns the table states by being of one type (``event_type``) or
    derives from ``start`` and ``duration`` (``temporal_type``, ``end``)
    are not written.

    Args:
        timelines: The timelines, each once.
        event_type: The event type whose rows are written (its class name is
            the rows' ``event_type``).

    Returns:
        The table, empty when no timeline holds such an event.

    Raises:
        ValueError: If a row gives a value in a column the table does not
            have.
    """
    schema = event_table_schema(event_type)
    parts = []
    for timeline in timelines:
        events = timeline.events.table
        if events.num_rows == 0 or "event_type" not in events.column_names:
            continue
        rows = events.filter(
            pc.equal(events.column("event_type"), pa.scalar(event_type.__name__))
        )
        n = rows.num_rows
        if n == 0:
            continue
        _refuse_unwritten_columns(rows, schema, event_type, timeline.id)
        number_type = NumberType(timeline.number_type)
        types = pa.array([number_type.name] * n, pa.string())
        header = {
            "timeline_id": pa.array([timeline.id] * n, pa.string()),
            "unit": pa.array([TimeUnit(timeline.unit).value] * n, pa.string()),
            "number_type": types,
        }

        def column(name: str, rows: pa.Table = rows) -> pa.ChunkedArray | None:
            """Return column *name* of the timeline's rows, ``None`` if they lack it."""
            return rows.column(name) if name in rows.column_names else None

        parts.append(
            pa.Table.from_arrays(
                [
                    (
                        header[field.name]
                        if field.name in header
                        else _packaged(
                            column(field.name), field.type, number_type, types
                        )
                    )
                    for field in schema
                ],
                schema=schema,
            )
        )
    if not parts:
        return schema.empty_table()
    return pa.concat_tables(parts)


def projection_table(timelines: Iterable[Timeline]) -> pa.Table:
    """The Projections table of *timelines*: every Projection each holds itself.

    The event table of :class:`~timetoalign.core.Projection`
    (:func:`event_table`), TimeToShare's Projections table form
    (:data:`timetoshare.tables.PROJECTION_SCHEMA`): ``id``, the
    ``timeline_id`` of the timeline holding it, the ``unit`` and
    ``number_type`` of that timeline's axis, ``start`` and ``duration`` each
    its number written once, and ``projection_of``.

    A Projection's ID is unique among all Projections of the package, so
    two timelines holding one Projection ID (each minted it before they
    joined one bundle) are refused, naming both, and the message says how
    to obtain a writable bundle: create the MatchClaims once the timelines
    share the bundle, so that their Projection IDs are minted unique. No ID
    is minted again, since a MatchClaim may name it.

    Args:
        timelines: The timelines, each once.

    Returns:
        The table, empty when no timeline holds a Projection.

    Raises:
        ValueError: If one Projection ID names Projections on two timelines.
    """
    table = event_table(timelines, Projection)
    refuse_shared_projection_ids(
        table.column("id"),
        table.column("timeline_id"),
        remedy=(
            "Create the MatchClaims once the timelines share the bundle (for "
            "instance with AlignmentBundle.create_match_claims), so that their "
            "Projection IDs are minted unique"
        ),
    )
    return table


def write_event_table(table: pa.Table, path: str | PathLike[str]) -> Path:
    """Write an event table to its Parquet file, replacing it whole.

    A write that fails leaves the file as it was and no temporary file.

    Args:
        table: The table, as :func:`event_table` builds it.
        path: The file.

    Returns:
        The path written.
    """
    target = Path(path)
    temporary = target.with_name(f".{target.name}.writing")
    try:
        pq.write_table(table, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _from_package(array: Any, number_types: pa.Array, where: str) -> pa.Array:
    """A package column of an event table in memory: every number in it
    (a struct of ``numerator``, ``denominator`` and ``value``) read by the
    row's number type, at whatever depth; anything else as it stands."""
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if not pa.types.is_struct(array.type):
        return array
    names = [field.name for field in array.type]
    if sorted(names) == sorted(PACKAGE_NUMBER_TYPE.names):
        return numbers_from_package(array, number_types, where=f"{where}'")
    members = [
        _from_package(member, number_types, f"{where}.{name}")
        for name, member in zip(names, array.flatten(), strict=True)
    ]
    return pa.StructArray.from_arrays(members, names=names, mask=array.is_null())


def _event_tables_by_timeline(table: pa.Table, what: str) -> dict[str, pa.Table]:
    """The rows of an event table in memory, by the ID of the timeline holding
    them, in row order: ``timeline_id`` dropped, every other column kept and
    each number read by its row's number type (*what* names the table in a
    message)."""
    number_types = table.column("number_type").combine_chunks()
    memory = pa.table(
        {
            name: _from_package(table.column(name), number_types, f"{what} '{name}")
            for name in table.column_names
        }
    )
    timeline_ids = memory.column("timeline_id")
    rows = memory.drop_columns(["timeline_id"])
    return {
        timeline_id: rows.filter(pc.equal(timeline_ids, timeline_id))
        for timeline_id in pc.unique(timeline_ids).to_pylist()
    }


def read_event_table(path: str | PathLike[str]) -> dict[str, pa.Table]:
    """Read an event table: the rows of each timeline, in memory, in row order.

    The inverse of :func:`event_table`: per timeline ID, the table of its
    rows (``id``, ``unit``, ``number_type`` and the properties), each number
    read in the number type its row states
    (:func:`~timetoalign.core.time.numbers_from_package`), both sides filled
    as memory holds them, however deep it lies; a further column is kept as
    it stands. An event type's ``from_row(row, unit=row["unit"],
    number_type=row["number_type"])`` reads a row.

    Args:
        path: The file.

    Returns:
        The rows by the ID of the timeline holding them.

    Raises:
        ValueError: If a number does not hold its value as its row's number
            type says, naming the file, the column and the row.
    """
    return _event_tables_by_timeline(
        pq.read_table(os.fspath(path) if isinstance(path, PathLike) else path),
        Path(path).name,
    )


def read_projection_table(
    path: str | PathLike[str], manifest: Mapping[str, Any]
) -> dict[str, list[Projection]]:
    """Read a Projections table: the Projections of each timeline, in row order.

    The table is checked by TimeToShare's rules against the manifest naming
    it (:func:`timetoshare.tables.projection_problems`): every column of the
    table type, every timeline one of the manifest's, every cell on its
    timeline's own axis, every ID given once in the table (a Projection's
    ID is unique in the package, so two timelines never both hold
    ``projection_1``). Each ``start`` and ``duration`` is read in the
    number type its row states, as :func:`read_event_table` reads an event
    table, and a number not standing as that type says is refused.

    Args:
        path: The file.
        manifest: The manifest naming it, as the file holds it.

    Returns:
        The Projections by the ID of the timeline holding them.

    Raises:
        timetoshare.tables.TableError: A ``ValueError``, the first problem
            of the table.
        ValueError: If a ``start`` or ``duration`` does not hold its number
            as its row's number type says.
    """
    table = pq.read_table(os.fspath(path) if isinstance(path, PathLike) else path)
    problems = projection_problems(table, manifest)
    if problems:
        raise TableError(problems[0])
    return {
        timeline_id: [
            Projection.from_row(
                {
                    **row,
                    "start": _coordinate_cell(row["start"], row),
                    "duration": _coordinate_cell(row["duration"], row),
                }
            )
            for row in rows.to_pylist()
        ]
        for timeline_id, rows in _event_tables_by_timeline(
            table, "Projections table"
        ).items()
    }


def _coordinate_cell(
    number: dict[str, Any] | None, row: Mapping[str, Any]
) -> dict[str, Any] | None:
    """A number of an event row with its row's axis: a coordinate cell."""
    if number is None:
        return None
    return {**number, "unit": row["unit"], "number_type": row["number_type"]}


# endregion


# region The manifest


@dataclass(eq=False)
class Manifest(_OpenObject):
    """What a manifest holds beside its timelines and groups.

    The manifest's ``id`` (absent when it has none), ``name`` and ``meta``,
    its statements and provenance with the prefixes it declares, the members
    of its ``graphical`` other than ``order``
    (``styling``) and the band orders of its timelines (``band_orders``),
    both kept as given and never interpreted (the format's validator checks
    them as it checks the whole manifest read and written), whether it holds
    ``graphical`` at all, the measure maps no timeline names, and every key
    the library does not read.

    Attributes:
        id: The manifest id, or None when it has none (an anonymous bundle,
            or one named by ``name`` alone).
        name: The manifest name, or None.
        meta: The manifest's free metadata.
        statement_state: The manifest's ``statements`` and ``provenance``,
            each only when not empty, as the file writes them.
        prefixes: The prefixes the manifest's ``@context`` declares that the
            catalogue lacks or gives another namespace, used by its
            statements or not.
        styling: ``graphical`` without ``order``: ``styles``,
            ``rules``, ``timestamps`` and any other key, kept as given;
            a named timestamp's ``display`` is written in canonical order,
            its default stated in the complete form only.
        band_orders: The entries of ``graphical.order`` keyed by a
            timeline's id, each that timeline's band order, kept as given.
        has_graphical: Whether the manifest holds a ``graphical`` object,
            which is then written back even with every member at its
            default.
        measure_maps: The entries of ``measure_maps`` no timeline names,
            written back with the measure maps the timelines name.
        resources: The files the manifest names (``resources``), each one
            :class:`Resource`.
    """

    _dict_keys: ClassVar[tuple[str, ...]] = (
        "@context",
        "timetoshare",
        "id",
        "name",
        "meta",
        "timelines",
        "groups",
        "measure_maps",
        "resources",
        "flow_maps",
        "virtual_timelines",
        *STATE_KEYS,
        "graphical",
    )

    id: str | None = None
    name: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    statement_state: dict[str, Any] = field(default_factory=dict)
    prefixes: dict[str, str] = field(default_factory=dict)
    styling: dict[str, Any] = field(default_factory=dict)
    band_orders: dict[str, Any] = field(default_factory=dict)
    has_graphical: bool = False
    measure_maps: list[MeasureMap] = field(default_factory=list)
    resources: list[Resource] = field(default_factory=list)

    @classmethod
    def read(cls, data: Any) -> tuple[Manifest, ManifestContent]:
        """Read a manifest: its envelope, its timelines and its raw groups.

        *data* is the manifest as a file holds it, every ID written ``#``
        followed by the ID. It is validated first, whole, by the format's
        validator (:func:`timetoshare.validate`), so that the library reads
        exactly the manifests the format admits; its warnings (unknown keys
        and the like) are not reported. It is then read with bare IDs
        (:func:`timetoshare.with_bare_ids`), as every object of the library
        holds them. The measure maps are read next, then every root timeline
        is rebuilt, its conversion maps reading the measure maps they name,
        and every element of the manifest is indexed by id (:func:`unique_ids`).
        Each FlowMap joins the root timeline it names and each virtual
        timeline the root of the tree it comes from
        (:func:`~timetoalign.timelines.flow.flowmap.place_on_roots`), so that
        a timeline carries them wherever it goes.
        Groups are left as JSON for the class reading them.
        ``graphical.order`` is split by its keys: a group's display order
        goes to the content, a timeline's band order to the envelope. The
        measure maps no timeline names are kept on the envelope, and so are
        its resources. The manifest, every timeline and every region keep
        their statements and provenance as written; every timeline and
        region with the prefixes of the manifest's ``@context`` it uses that
        the catalogue lacks or gives another namespace, the manifest with
        every such prefix its ``@context`` declares, used or not (the
        context its MatchClaims' actors are read under).

        Returns:
            The envelope, and the manifest's content.

        Raises:
            timetoshare.SchemaValidationError: The validator's refusal, a
                ``ValueError`` carrying the validator's ``code`` and message
                unchanged: ``#/graphical/timestamps/0/timeline: 'clt_9' names
                no timeline of this manifest``.
            timetoshare.UnsupportedVersionError: A ``ValueError``, if *data*
                names another version than the library's.
            ValueError: If the manifest is valid but breaks a rule of the
                library alone: a negative child offset, a region ending
                before it starts, a measure map or conversion map the library
                cannot read (see the element readers), virtual timelines
                naming one another in a circle.
        """
        from .flow.flowmap import FlowMap, VirtualTimeline, place_on_roots

        timetoshare.validate(data)
        bare = timetoshare.with_bare_ids(data)
        own = {
            key: value
            for key, value in data.get("@context", {}).items()
            if key not in FORMAT_CONTEXT
        }
        data = bare
        manifest_id = data.get("id")
        state = read_statement_state(data)
        manifest = cls(
            id=manifest_id,
            name=data.get("name"),
            meta=copy.deepcopy(data.get("meta", {})),
            statement_state=state,
            prefixes=declared_prefixes(own),
        )
        manifest._keep_unknown_keys(data)

        holders: list[tuple[str, str]] = []
        if manifest_id is not None:
            holders.append((manifest_id, "the manifest"))
        read_maps = _read_measure_maps(data.get("measure_maps", []))
        holders.extend(
            (measure_map.id, f"measure map {quote_id(measure_map.id)}")
            for measure_map in read_maps
        )
        measure_maps = {measure_map.id: measure_map for measure_map in read_maps}
        roots, timelines = _read_roots(data["timelines"], measure_maps)
        if own:
            for timeline in timelines.values():
                timeline._take_prefixes(own)
        place_on_roots(
            [FlowMap.from_dict(item) for item in data.get("flow_maps", [])],
            [
                VirtualTimeline.from_dict(item)
                for item in data.get("virtual_timelines", [])
            ],
            timelines,
        )
        holders.extend(
            (element_id, description)
            for root in roots
            for element_id, description in root._tree_ids().items()
        )
        manifest.resources = [
            Resource.from_dict(item, f"resource {index}")
            for index, item in enumerate(data.get("resources", []))
        ]
        holders.extend(
            (resource.id, f"resource {quote_id(resource.id)}")
            for resource in manifest.resources
        )
        groups = data.get("groups", [])
        group_ids = [group["id"] for group in groups]
        holders.extend(
            (group_id, f"group {quote_id(group_id)}") for group_id in group_ids
        )
        ids = unique_ids(holders)
        named = {
            id(measure_map)
            for timeline in timelines.values()
            for cmap in timeline._all_conversion_maps()
            for measure_map in cmap._measure_maps()
        }
        manifest.measure_maps = [
            measure_map
            for measure_map in measure_maps.values()
            if id(measure_map) not in named
        ]

        order: dict[str, Any] = {}
        if "graphical" in data:
            graphical = data["graphical"]
            manifest.has_graphical = True
            order = graphical.get("order", {})
            manifest.band_orders = {
                key: copy.deepcopy(entry)
                for key, entry in order.items()
                if key not in group_ids
            }
            order = {key: entry for key, entry in order.items() if key in group_ids}
            # The styling is kept in TimeToShare's minimal form, so that the
            # complete and the minimal form read to the same styling.
            styling = {key: value for key, value in graphical.items() if key != "order"}
            manifest.styling = timetoshare.minimal({"graphical": styling})["graphical"]

        return manifest, ManifestContent(roots, timelines, groups, order, ids)

    def write(
        self,
        roots: list[Timeline],
        *,
        groups: list[dict[str, Any]] = (),  # type: ignore[assignment]
        order: dict[str, list[str]] | None = None,
        minimal: bool = True,
        events: bool = False,
        external_references: bool = False,
        table_prefixes: Collection[str] = (),
    ) -> dict[str, Any]:
        """Write the manifest around root timelines and written groups.

        The roots are written sorted by id, each with its
        :attr:`~timetoalign.timelines.Timeline.root_offset` as its
        ``offset`` when it has one. ``measure_maps`` holds every measure map
        the timelines name and every one the manifest kept, sorted by id
        (see :class:`MeasureMapEntries`). ``flow_maps`` and
        ``virtual_timelines`` hold the FlowMaps and virtual timelines the
        timelines written hold, each sorted by id, and are written only when
        there is one. ``graphical.order`` holds the
        groups' member orders and, unchanged, the band order of every
        written timeline that has one; the band order of a timeline not
        written, removed or given another id, is left out, so that every
        key names a timeline or group of the manifest. ``graphical`` is
        written when there is an order to keep, styling, or a ``graphical``
        read with the manifest; ``id`` only when the manifest has one.
        ``resources`` holds the manifest's resources, each naming only the
        timelines written among its own. The
        manifest's, every timeline's and every region's statements and
        provenance are written where not empty; a value naming an element
        of the manifests the roots belong to that this one leaves out is
        left out with its provenance (:func:`_leave_out_unwritten`). The
        form is TimeToShare's: the
        manifest is built whole and put in its form by
        :func:`timetoshare.minimal` or :func:`timetoshare.complete`, and
        every ID is written as a file writes it, ``#`` followed by the ID
        (:func:`timetoshare.with_file_ids`). Every manifest written carries
        its inline JSON-LD ``@context``: the format's terms, then, sorted by
        name, the prefixes its statements use (:func:`timetoshare.with_context`),
        the *table_prefixes* and those the provenance of its FlowMaps uses,
        each standing for the namespace the holders using it declare, else
        the catalogue's (:meth:`_prefix_table`; a FlowMap's are the
        manifest's, as a table's are); a prefix none of them uses is not
        written, declared or not. The manifest
        written is validated whole by the format's validator
        (:func:`timetoshare.validate`), as a manifest read is, so that a
        manifest the format refuses is never emitted; its warnings are not
        reported. Writing a file, :meth:`ManifestFileMixin.to_file`
        validates the manifest as the file holds it instead.

        Args:
            roots: The root timelines.
            groups: The written groups, each in the complete form.
            order: Every group's member order, keyed by group id.
            minimal: Write the minimal form (the default): every key at its
                default value left out. With False, the complete form.
            events: Write each timeline's event rows.
            external_references: Write each timeline's external reference
                table.
            table_prefixes: The prefixes a table written beside the manifest
                uses (the actors of a bundle's MatchClaim table), read under
                the manifest's ``@context``, so a reload must find them
                there; each stands for the namespace the manifest declares
                (:attr:`prefixes`), else the catalogue's, and one neither
                declares is not written.

        Raises:
            ValueError: If an id is held twice among the manifest, its
                timelines' trees and its groups (:func:`unique_ids`), or a
                measure map cannot be written (two different ones carry one
                given id, or one's given id is held by another element), two
                holders written use one prefix for two namespaces (``the
                prefix 'ex' stands for 'http://a.example/' on timeline
                'clt_1' and for 'http://b.example/' on timeline 'clt_2'``).
            timetoshare.SchemaValidationError: The validator's refusal of the
                manifest written, a ``ValueError`` carrying its ``code`` and
                message unchanged: a named timestamp kept with the styling
                beyond the length its timeline was given since, for instance
                (``#/graphical/timestamps/0/coordinate: coordinate [11, 1] of
                'timestamp_1' is beyond the length [10, 1] of its timeline
                'clt_1'``).
        """
        trees = [
            timeline
            for root in roots
            for timeline in (root, *(child for _, child in root._iter_descendants()))
        ]
        holders: list[tuple[str, str]] = []
        if self.id is not None:
            holders.append((self.id, "the manifest"))
        holders.extend(
            (element_id, description)
            for root in roots
            for element_id, description in root._tree_ids().items()
        )
        holders.extend(
            (group["id"], f"group {quote_id(group['id'])}") for group in groups
        )
        holders.extend(
            (resource.id, f"resource {quote_id(resource.id)}")
            for resource in self.resources
        )
        entries = MeasureMapEntries(
            unique_ids(holders),
            [
                *self.measure_maps,
                *(
                    measure_map
                    for timeline in trees
                    for cmap in timeline._all_conversion_maps()
                    for measure_map in cmap._measure_maps()
                ),
            ],
        )
        timelines = [
            root._to_timeline_object(
                events=events,
                external_references=external_references,
                offset=root._root_offset,
                entries=entries,
            )
            for root in sorted(roots, key=lambda root: root.id)
        ]
        for measure_map in self.measure_maps:
            entries.id_of(measure_map)
        fields: dict[str, Any] = {"timetoshare": timetoshare.SCHEMA_VERSION}
        if self.id is not None:
            fields["id"] = self.id
        fields.update(
            name=self.name,
            meta=copy.deepcopy(self.meta),
            timelines=timelines,
            groups=list(groups),
            measure_maps=entries.write(),
        )
        for key, held in (
            ("flow_maps", "_flow_maps"),
            ("virtual_timelines", "_virtual_timelines"),
        ):
            elements = [
                element
                for timeline in trees
                for element in getattr(timeline, held).values()
            ]
            if elements:
                fields[key] = [
                    element.to_dict(minimal=False)
                    for element in sorted(elements, key=lambda element: element.id)
                ]
        fields.update(copy.deepcopy(self.statement_state))
        fields = _leave_out_unwritten(
            fields,
            roots,
            {*entries.taken, *(entry["id"] for entry in fields["measure_maps"])},
        )
        # A FlowMap's provenance is read under the manifest's context, as a
        # table beside the manifest is.
        table_prefixes = sorted(
            {
                *table_prefixes,
                *timetoshare.used_prefixes({"flow_maps": fields.get("flow_maps", [])}),
            }
        )
        prefixes = self._prefix_table(fields, trees, table_prefixes)
        written = _timeline_ids(fields["timelines"])
        fields["resources"] = [
            {
                **resource.to_dict(),
                "timelines": [tid for tid in resource.timelines if tid in written],
            }
            for resource in self.resources
        ]
        band_orders = {
            key: copy.deepcopy(entry)
            for key, entry in self.band_orders.items()
            if key in written
        }
        order = dict(sorted({**band_orders, **(order or {})}.items()))
        if order or self.styling or self.has_graphical:
            fields["graphical"] = {"order": order, **copy.deepcopy(self.styling)}
        built = self._dict_form(fields, minimal=False)
        result = timetoshare.minimal(built) if minimal else timetoshare.complete(built)
        result = timetoshare.with_context(timetoshare.with_file_ids(result), prefixes)
        declared = {
            **{k: v for k, v in result["@context"].items() if k not in FORMAT_CONTEXT},
            **{p: prefixes[p] for p in table_prefixes if p in prefixes},
        }
        result["@context"] = {**FORMAT_CONTEXT, **dict(sorted(declared.items()))}
        if not _VALIDATED_BY_THE_FILE.get():
            timetoshare.validate(result)
        return result

    def _prefix_table(
        self,
        fields: Mapping[str, Any],
        trees: list[Timeline],
        table_prefixes: Collection[str] = (),
    ) -> dict[str, str]:
        """Return the namespace of every prefix the statements written use.

        The catalogue's prefixes, then holder by holder in written order
        (the manifest; each timeline object, depth first, followed by its
        regions): a prefix a holder uses stands for the namespace the holder
        declares, else the catalogue's. A prefix a table beside the manifest
        uses (*table_prefixes*) is used by the manifest: it stands for the
        manifest's namespace (:attr:`prefixes`), else the catalogue's,
        wherever it is used.

        Raises:
            ValueError: If two holders use one prefix for two namespaces, the
                manifest counting as a holder of each prefix its tables use.
        """
        catalogue = load_catalogue().prefixes
        held: dict[str, Any] = {}
        for timeline in trees:
            held[timeline.id] = timeline
            held.update(timeline._regions)
        table = dict(catalogue)
        fixed: dict[str, tuple[str, str]] = {}
        for prefix in table_prefixes:
            iri = self.prefixes.get(prefix, catalogue.get(prefix))
            if iri is not None:
                fixed[prefix] = (iri, "the manifest")
                table[prefix] = iri
        for _path, element, kind in timetoshare.subjects(fields):
            state = {key: element[key] for key in STATE_KEYS if key in element}
            if not state:
                continue
            if kind == "manifest":
                declared, holder = self.prefixes, "the manifest"
            else:
                element_holder = held[element["id"]]
                declared = element_holder._own_prefixes()
                holder = element_holder._statement_holder_name()
            for prefix in timetoshare.used_prefixes(state):
                iri = declared.get(prefix, catalogue.get(prefix))
                if iri is None:
                    continue
                first = fixed.setdefault(prefix, (iri, holder))
                if first[0] != iri:
                    raise ValueError(
                        f"the prefix '{prefix}' stands for '{first[0]}' on {first[1]} "
                        f"and for '{iri}' on {holder}"
                    )
                table[prefix] = iri
        return table

    def _rename_statement_references(self, renamed: Mapping[str, str | None]) -> None:
        """Rename or drop the elements the manifest's statements name."""
        self.statement_state = renamed_state(self.statement_state, renamed)


class MeasureMapEntries:
    """The measure maps a manifest being written holds, each under one id.

    Every id of the manifest is unique: :attr:`taken` holds every id the
    manifest written holds besides its measure maps, mapped to its holder,
    and grows with every id minted while it is written.
    A given measure map id is kept: two different measure maps carrying
    one, or one carrying an id another element holds, are refused. A minted
    id is minted again around every id the manifest holds, the given ones
    taken first. Equal measure maps of equal name and meta share one entry,
    so two timelines holding the same measure map, or equal ones, name one
    entry.

    While a timeline's conversion maps are written, :attr:`own` is that
    timeline's measure map, which a measure-derived map names when it
    derives from it.

    Args:
        taken: Every id of the manifest but its measure maps', mapped to its
            holder.
        measure_maps: The measure maps the manifest names, whose given ids
            are taken before any is minted.

    Raises:
        ValueError: If a given id is refused.
    """

    def __init__(
        self, taken: dict[str, str], measure_maps: list[MeasureMap] = ()  # type: ignore[assignment]
    ) -> None:
        self.own: MeasureMap | None = None
        self.taken = taken
        self._entries: dict[str, MeasureMap] = {}
        # By object identity: the id each measure map met is written under,
        # with the map kept alive so that its identity is not reused.
        self._written: dict[int, tuple[MeasureMap, str]] = {}
        for measure_map in measure_maps:
            if not measure_map._id_minted:
                self._admit_given(measure_map)

    def id_of(self, measure_map: MeasureMap) -> str:
        """Return the id under which *measure_map* is written, adding it.

        Raises:
            ValueError: If its given id is refused.
        """
        known = self._written.get(id(measure_map))
        if known is not None:
            return known[1]
        if not measure_map._id_minted:
            return self._admit_given(measure_map)
        for entry_id, entry in self._entries.items():
            if _same_entry(entry, measure_map):
                self._written[id(measure_map)] = (measure_map, entry_id)
                return entry_id
        entry_id = mint_id(MEASURE_MAP_ID_PREFIX, {*self._entries, *self.taken})
        self._entries[entry_id] = measure_map
        self._written[id(measure_map)] = (measure_map, entry_id)
        return entry_id

    def id_of_rows(self, rows: list[dict[str, Any]]) -> str:
        """Return the id of the measure map compressed *rows* describe, adding it."""
        from .structure import MeasureMap

        return self.id_of(MeasureMap.from_rows(rows))

    def write(self) -> list[dict[str, Any]]:
        """Return the ``measure_maps`` entries, sorted by id, in the complete form."""
        return [
            {**self._entries[entry_id].to_dict(minimal=False), "id": entry_id}
            for entry_id in sorted(self._entries)
        ]

    def _admit_given(self, measure_map: MeasureMap) -> str:
        """Add a measure map under its given id, and return the id.

        Raises:
            ValueError: If another element of the manifest, or another
                measure map, carries the id.
        """
        entry_id = measure_map.id
        if entry_id in self.taken:
            raise ValueError(
                f"measure map {quote_id(entry_id)} cannot be written: its id is already "
                f"held by {self.taken[entry_id]}"
            )
        held = self._entries.get(entry_id)
        if held is not None and not _same_entry(held, measure_map):
            raise ValueError(
                f"two different measure maps carry the id {quote_id(entry_id)}"
            )
        self._entries.setdefault(entry_id, measure_map)
        self._written[id(measure_map)] = (measure_map, entry_id)
        return entry_id


def _same_entry(left: MeasureMap, right: MeasureMap) -> bool:
    """Whether two measure maps write the same entry but for its id."""
    return (
        left == right
        and left.name == right.name
        and left.meta == right.meta
        and dict(left._unknown_keys) == dict(right._unknown_keys)
        and dict(left._unknown_row_keys) == dict(right._unknown_row_keys)
    )


@dataclass
class ManifestContent:
    """The timelines and groups a manifest holds.

    Attributes:
        roots: The root timelines, rebuilt, in the manifest's order.
        timelines: Every timeline of the roots' trees, by id.
        groups: The manifest's groups, as JSON.
        order: The display orders of ``graphical.order``: member orders
            keyed by group id.
        ids: Every id of the manifest, mapped to a description of its
            holder.
    """

    roots: list[Timeline]
    timelines: dict[str, Timeline]
    groups: list[Any]
    order: dict[str, Any]
    ids: dict[str, str]

    def candidates(self) -> list[tuple[str, str | None]]:
        """Every timeline of the manifest as ``(id, name)``, for an error."""
        return [(timeline.id, timeline._name) for timeline in self.timelines.values()]


def _timeline_ids(timelines: list[dict[str, Any]]) -> set[str]:
    """The ids of written timeline objects and of all their children."""
    ids: set[str] = set()
    pending = list(timelines)
    while pending:
        timeline = pending.pop()
        ids.add(timeline["id"])
        pending.extend(timeline.get("children", []))
    return ids


def _read_measure_maps(items: list[Any]) -> list[MeasureMap]:
    """Read the ``measure_maps`` entries, in order.

    Raises:
        ValueError: If an entry is malformed.
    """
    from .structure import MeasureMap

    return [MeasureMap.from_dict(item) for item in items]


def _read_roots(
    items: list[dict[str, Any]], measure_maps: Mapping[str, MeasureMap]
) -> tuple[list[Timeline], dict[str, Timeline]]:
    """Rebuild the root timelines and index every timeline of their trees by id.

    *items* are the root timeline objects of a manifest the format's
    validator admitted.
    """
    from .base import Timeline

    roots: list[Timeline] = []
    index: dict[str, Timeline] = {}
    for item in items:
        root = Timeline._from_timeline_object(item, measure_maps=measure_maps)
        roots.append(root)
        for timeline in [
            root,
            *(descendant for _, descendant in root._iter_descendants()),
        ]:
            index.setdefault(timeline.id, timeline)
    return roots, index


def unique_ids(holders: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Map each id to its holder, refusing an id held twice.

    Args:
        holders: ``(id, description)`` of every element of a manifest.

    Raises:
        ValueError: ``the id 'intro' is held twice in the manifest: by
            region 'intro' of timeline 'score' and by timeline 'intro'``.
    """
    ids: dict[str, str] = {}
    for element_id, holder in holders:
        if element_id in ids:
            raise ValueError(
                f"the id {quote_id(element_id)} is held twice in the manifest: by "
                f"{ids[element_id]} and by {holder}"
            )
        ids[element_id] = holder
    return ids


def select(
    candidates: list[Any],
    *,
    id: str | None,
    name: str | None,
    what: str,
    names: list[tuple[str, str | None]],
) -> Any:
    """Return the one candidate whose id and name match the given ones.

    Args:
        candidates: Objects with ``id`` and ``_name``-or-``name``.
        id: The id to match, or None to match any.
        name: The name to match, or None to match any.
        what: What is selected (``"timeline"``, ``"group"``), for the error.
        names: Every candidate as ``(id, name)``, for the error.

    Raises:
        ValueError: If no candidate or several match; the message names the
            manifest's candidates.
    """
    matches = [
        candidate
        for candidate in candidates
        if (id is None or candidate.id == id)
        and (name is None or _name_of(candidate) == name)
    ]
    if len(matches) == 1:
        return matches[0]
    wanted = ", ".join(
        f"{key} {quote_id(value) if key == 'id' else repr(value)}"
        for key, value in (("id", id), ("name", name))
        if value is not None
    )
    found = "no" if not matches else "several"
    raise ValueError(
        f"the manifest holds {found} {what} with {wanted}; its {what}s: {listed(names)}"
    )


def matches_any(candidates: list[Any], *, id: str | None, name: str | None) -> bool:
    """Whether the given id or the given name names one of the candidates.

    Each given argument is matched on its own, so an argument naming
    something the manifest holds asks for a selection even when the other
    names nothing.
    """
    return any(
        (id is not None and candidate.id == id)
        or (name is not None and _name_of(candidate) == name)
        for candidate in candidates
    )


def listed(names: list[tuple[str, str | None]]) -> str:
    """Spell candidates given as ``(id, name)`` for an error message."""
    return (
        ", ".join(f"{quote_id(cid)} (name {cname!r})" for cid, cname in names) or "none"
    )


def _name_of(candidate: Any) -> str | None:
    """Return a candidate's own name, never a derived display name."""
    return getattr(candidate, "_name", candidate.name)


# endregion


# region Manifest files


@dataclass(frozen=True)
class FileForm(timetoshare.FileForm):
    """How a manifest file is written: TimeToShare's file form, and the timeline tables.

    The form, the styling and the serialization are ``timetoshare.FileForm``'s;
    the two tables are the library's own.

    Attributes:
        events: Whether any timeline object holds its event rows
            (``events``).
        external_references: Whether any timeline object holds its
            external reference table (``external_references``).
    """

    events: bool = False
    external_references: bool = False

    @classmethod
    def of(cls, text: str, data: Any) -> FileForm:
        """Infer the form, styling, serialization and table keys of a manifest file.

        The form, the styling and the serialization are the ones
        ``timetoshare.file_form`` reads off *text*; ``events`` and
        ``external_references`` are each written when a timeline object of
        the manifest, at any depth, holds the key.

        Args:
            text: The file's text.
            data: The manifest *text* holds.
        """
        keys = {key for timeline in _timeline_objects(data) for key in timeline}
        return cls(
            **asdict(timetoshare.file_form(text)),
            events="events" in keys,
            external_references="external_references" in keys,
        )


_NEW_FILE = FileForm(minimal=True, styling=True, serialization="yaml")
"""The form of an object not read from a file: minimal, styled, canonical YAML."""

_VALIDATED_BY_THE_FILE: ContextVar[bool] = ContextVar(
    "_VALIDATED_BY_THE_FILE", default=False
)
"""Set while :meth:`ManifestFileMixin._file_text` writes a manifest it validates itself."""


@contextmanager
def _validated_by_the_file() -> Iterator[None]:
    """Leave the validation of the manifests written to the file being written."""
    token = _VALIDATED_BY_THE_FILE.set(True)
    try:
        yield
    finally:
        _VALIDATED_BY_THE_FILE.reset(token)


def _timeline_objects(data: Any) -> Iterator[dict[str, Any]]:
    """Yield every timeline object of a manifest, the children included."""
    pending = list(data.get("timelines") or ())
    while pending:
        timeline = pending.pop()
        if isinstance(timeline, dict):
            yield timeline
            pending.extend(timeline.get("children") or ())


class ManifestFileMixin:
    """Read a manifest file and write it back in the form it came in.

    Shared by :class:`~timetoalign.timelines.Timeline`,
    :class:`~timetoalign.timelines.TimelineGroup` and
    :class:`~timetoalign.alignment.AlignmentBundle`, each of which reads a
    manifest with ``from_dict`` and writes one with ``to_dict``. An object
    read from a file keeps the file's :class:`FileForm`, which
    :meth:`to_file` follows unless told otherwise: a file in canonical YAML,
    pretty JSON or RFC 8785 JSON keeps its bytes; any other text is written
    back in the canonical form of the serialization it was read as. A file
    read, changed and written back changes only where the object changed.

    A manifest and the files it names make a package. Reading a file, the
    object reads the tables its manifest names beside it
    (:meth:`_read_tables`); writing one, it writes its tables beside the
    manifest first and names them (:meth:`_write_tables`). A timeline and a
    group hold no package table; the bundle holds its MatchClaim table and
    the Projections of its timelines.
    """

    _file_form: FileForm | None = None
    """The form of the file the object was read from, or None."""

    from_dict: ClassVar[Callable[..., Any]]
    """The classmethod reading the object from a manifest, which each class
    using the mixin defines with its own selection keywords (``timeline_id``
    and ``name`` for a timeline, ``group_id`` and ``name`` for a group);
    :meth:`from_file` calls it with the manifest and those keywords."""

    def _read_tables(self, directory: Path, data: Mapping[str, Any]) -> None:
        """Read the tables the manifest names, lying in *directory*.

        Args:
            directory: The manifest's directory.
            data: The manifest as the file holds it.
        """

    def _write_tables(self, directory: Path) -> bool:
        """Write the object's tables into *directory* and name them as resources.

        Args:
            directory: The manifest's directory.

        Returns:
            Whether a table was written (none here), so that the manifest
            naming it is written anew.
        """
        return False

    def to_dict(
        self,
        *,
        minimal: bool = True,
        events: bool = False,
        external_references: bool = False,
    ) -> dict[str, Any]:
        """Write the object's manifest; each class using the mixin defines it.

        Args:
            minimal: Write the minimal form (True) or the complete form
                (False).
            events: Write every timeline's event rows.
            external_references: Write every timeline's external references.
        """
        raise NotImplementedError

    @classmethod
    def from_file(cls, path: str | PathLike[str], **selection: str | None) -> Self:
        """Read a manifest file in any serialization, as ``from_dict`` reads a manifest.

        The text is read with ``timetoshare.loads``: canonical YAML, pretty
        JSON, RFC 8785 JSON or any other JSON or YAML text. The object read
        records the file's form, styling and serialization, as
        ``timetoshare.file_form`` reads them, and whether its timeline
        objects hold ``events`` and ``external_references``, which
        :meth:`to_file` writes back. An
        :class:`~timetoalign.alignment.AlignmentBundle` reads the tables its
        manifest names beside it, where they lie there.

        Args:
            path: The manifest file, UTF-8.
            **selection: The keyword arguments of ``from_dict`` that select
                one timeline or group of the manifest (``timeline_id`` and
                ``name`` for a timeline, ``group_id`` and ``name`` for a
                group).

        Returns:
            The object read.

        Raises:
            ValueError: If the text cannot be read (with the line and column
                of the fault), or ``from_dict`` refuses the manifest.
        """
        with Path(path).open(encoding="utf-8", newline="") as file:
            text = file.read()
        data = loads(text)
        result = cls.from_dict(data, **selection)
        result._read_tables(Path(path).parent, data)
        result._file_form = FileForm.of(text, data)
        return result

    def to_file(
        self,
        path: str | PathLike[str],
        *,
        minimal: bool | None = None,
        styling: bool | None = None,
        serialization: Serialization | None = None,
        events: bool | None = None,
        external_references: bool | None = None,
    ) -> Path:
        """Write the object's manifest to a file.

        The manifest is ``to_dict``'s complete form, written with
        ``timetoshare.dumps_file`` in the form, styling and serialization
        chosen: every object's keys sorted, its arrays in canonical order, so
        that equal manifests give equal text and a change shows in a diff as the
        lines it touches. Each argument left None takes the value the file
        the object was read from had (see :meth:`from_file`); for an object
        not read from a file, the minimal form, the styling included, in
        canonical YAML, without event rows or external references. An
        :class:`~timetoalign.alignment.AlignmentBundle` holding MatchClaims
        or Projections writes its tables beside the manifest and names them
        in its ``resources``: its manifest is written (validated and
        serialized) once before any table, and once more naming the tables
        written. The manifest validated is the one the file holds: without
        its styling where ``styling=False`` leaves the styling out, so a
        manifest whose styling the format refuses (a named timestamp beyond
        the length its timeline was given since) is written without it.

        Args:
            path: The file to write, UTF-8 with line feeds; it is
                overwritten.
            minimal: Write the minimal form (True) or the complete form
                (False).
            styling: Write the ``graphical`` object, when the manifest has
                one (True), or leave it out (False).
            serialization: ``"yaml"`` (canonical YAML), ``"json"`` (pretty
                JSON) or ``"jcs"`` (RFC 8785 JSON).
            events: Write every timeline's event rows (``events``).
            external_references: Write every timeline's external reference
                table (``external_references``).

        Returns:
            The path written.

        Raises:
            ValueError: If ``to_dict`` cannot write the manifest (the
                format's validator refusing it among others), the
                serialization is unknown or refuses a value of the manifest,
                or a table beside it is refused. Each of these refusals
                comes before any file is written and leaves the object's
                ``resources`` unchanged.

        Examples:
            >>> import tempfile
            >>> from fractions import Fraction
            >>> from pathlib import Path
            >>> from timetoalign.timelines import ContinuousLogicalTimeline, Timeline
            >>> score = ContinuousLogicalTimeline(length=Fraction(32), uid="piece")
            >>> path = score.to_file(Path(tempfile.mkdtemp()) / "piece.yaml")
            >>> text = path.read_text(encoding="utf-8")
            >>> text.startswith('"@context":')
            True
            >>> print(text[text.index("\\ntimelines:") + 1 :], end="")
            timelines:
              -
                id: "#piece"
                length: [32, 1]
                unit: quarters
            timetoshare: "0.2.0"
            >>> Timeline.from_file(path).length
            Coordinate(Fraction(32, 1), quarters)
        """
        given = {
            "minimal": minimal,
            "styling": styling,
            "serialization": serialization,
            "events": events,
            "external_references": external_references,
        }
        form = replace(
            self._file_form or _NEW_FILE,
            **{key: value for key, value in given.items() if value is not None},
        )
        path = Path(path)
        # The manifest is written (hence validated) and serialized before any
        # file is: a manifest the format refuses, or a serialization refusing
        # the form or a value of it, leaves no table behind. A second manifest
        # differs from the first in its resources alone, so the serialization
        # refuses nothing in it that it accepted in the first.
        text = self._file_text(form)
        if self._write_tables(path.parent):
            # The resources now name the tables written.
            text = self._file_text(form)
        path.write_text(text, encoding="utf-8", newline="")
        return path

    def _file_text(self, form: FileForm) -> str:
        """The text of the manifest file *form* writes, validated as written.

        ``to_dict`` validates the whole manifest; here the form the file
        holds is validated instead (``timetoshare.dumps_file`` leaves the
        ``graphical`` object out where ``form.styling`` is False).
        """
        with _validated_by_the_file():
            manifest = self.to_dict(
                minimal=False,
                events=form.events,
                external_references=form.external_references,
            )
        if not form.styling:
            manifest = {
                key: value for key, value in manifest.items() if key != "graphical"
            }
        timetoshare.validate(manifest)
        return timetoshare.dumps_file(manifest, form)


# endregion
