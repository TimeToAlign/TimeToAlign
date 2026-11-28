"""Loader: Base classes for loading music representations into TimeToAlign!

This module provides the loader hierarchy for TimeToAlign!:

- **Loader (ABC)**: Returns EventStore (events only)
- **ManifestLoader (ABC)**: Returns ManifestData (dimensions/metadata only)
- **AlignmentLoader (ABC)**: Returns AlignmentStore (events + C-maps)

Design principles:
- Multi-source: One Loader can aggregate multiple files
- Metadata in table: Deterministic, stored in PyArrow schema metadata
- Polymorphic returns: Different loader categories return different types
- Subclassable: Domain-specific loaders extend these ABCs
- Sources anywhere: a source is a filesystem path or a member of a ZIP
  archive (``zipfile.Path``), read in place; a parser that needs a file on
  disk gets the member through
  :func:`~timetoalign.loader.materialize`
"""

from __future__ import annotations

import functools
import html
import importlib
import importlib.metadata
import inspect
import json
import logging
import re
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Concatenate,
    Generic,
    ParamSpec,
    TypeVar,
)
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from typing_extensions import Self

from timetoalign.core import (
    IntervalPolicy,
    NumberType,
    TimeUnit,
)
from timetoalign.core.fields import (
    TIMETOALIGN_METADATA_KEY,
    _compared,
    _filled,
    _if_else,
    _joined_strings,
    field_metadata,
)
from timetoalign.core.ids import (
    CMAP_PREFIX,
    ID_MINTER,
    IdMinter,
    minting_from,
    seed_from,
)
from timetoalign.core.time import struct_to_coordinate
from timetoalign.display.html import affordance_html, code
from timetoalign.storage import EventData, EventRegistry, Parse

from ._sources import (
    Source,
    SourceLike,
    as_source,
    describe_source,
    member_uri,
    source_digest,
    source_record,
)

#: The column of a loader's events that says where in its source file each
#: event was read (the loader's rule, in its docstring); a timeline holds it
#: in the event's registry row. Its text is :func:`locate`'s.
LOCATOR = "locator"

#: What follows a locator's pointer where the source states a key of its
#: own for the event: ``row:3 key:n12``.
KEY_MARK = " key:"

#: The column naming the parse (one per source file read) each event
#: of a loader comes from (:attr:`Loader.parses`).
PARSE_ID = "parse_id"

#: The key of a Parquet file's schema metadata under which a loader saving
#: its events (``to_parquet``) keeps the parses they name: their rows
#: of the Parse table, as a JSON array.
PARSES_KEY = b"timetoalign.parses"

# Event loaders return source metadata and a vectorized field dictionary. Other
# payload families return their payload directly and use ``_accept_source``.
LoadSourceResult = tuple[
    dict[str, Any], dict[str, np.ndarray | pa.Array | pa.ChunkedArray]
]

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.storage.store import AlignmentStore, EventStore
    from timetoalign.timelines.base import Timeline
    from timetoalign.timelines.groups import TimelineGroup

module_logger = logging.getLogger(__name__)

# Type variable for polymorphic loader payloads.
PayloadT = TypeVar("PayloadT")

_EventDataT = TypeVar("_EventDataT", bound=EventData)


def _source_part(source: Source) -> bytes | str:
    """What a source contributes to a loader's seed: its digest, else its name."""
    return source_digest(source) if source.exists() else describe_source(source)


def _seed_part(part: Source | bytes | str) -> bytes | str:
    """What one thing a loader read contributes to its seed."""
    return part if isinstance(part, (bytes, str)) else _source_part(part)


def _canonical(value: object) -> object:
    """*value* as JSON can hold it, for a seed: enum members by name, the rest as is.

    Mappings and sequences are converted member by member; a ``Fraction``
    becomes ``[numerator, denominator]``, a path its string.
    """
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical(member) for key, member in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(member) for member in value]
    return value


def write_events(data: EventData, path: Path | str) -> None:
    """Write *data*'s table to the Parquet file *path*, with the parses it carries.

    The parses (:attr:`EventData.parses
    <timetoalign.storage.EventData.parses>`) go into the file's schema
    metadata under :data:`PARSES_KEY`; :func:`read_parses` gives
    them back to the table read from it.
    """
    table = data.table
    if data.parses:
        rows = [parse.to_row() for parse in data.parses]
        table = table.replace_schema_metadata(
            {
                **(table.schema.metadata or {}),
                PARSES_KEY: json.dumps(rows, sort_keys=True).encode(),
            }
        )
    pq.write_table(table, path)


def read_parses(data: _EventDataT) -> _EventDataT:
    """*data*, read from a file :func:`write_events` wrote, carrying its parses."""
    saved = (data.table.schema.metadata or {}).get(PARSES_KEY)
    if saved is None:
        return data
    return data.with_parses(*map(Parse.from_row, json.loads(saved)))


def _key_text(key: object) -> str | None:
    """*key* as a locator writes it: verbatim, a number in its decimal digits.

    ``None``, an empty string and a missing number (NaN) state no key.
    """
    if key is None or (isinstance(key, float) and key != key):
        return None
    text = str(key)
    return text or None


def locate(pointer: str, key: object = None) -> str:
    """The locator of an event: where its source states it, then its own key.

    Every loader writes its locators in this one form. The *pointer* is
    the event's structural place in the source file, unique within one
    parse (``row:<i>``, ``track:<i> message:<j>``, a JSON Pointer, an
    XPath, ``hole:<k>``); where the source states a key of its own for the
    event (an ``id`` column or member, an ``xml:id``, a measure number),
    :data:`KEY_MARK` and the key follow, verbatim and last: everything after
    the first ``" key:"`` is the key. The pointer keeps the locator unique
    where a key repeats.

    Args:
        pointer: The event's place in the file, by the loader's rule.
        key: The key the source states for the event; ``None``, an empty
            string or NaN for none.

    Examples:
        >>> locate("row:3")
        'row:3'
        >>> locate("row:3", "n12")
        'row:3 key:n12'
        >>> locate("/mei/music/measure[2]", 12)
        '/mei/music/measure[2] key:12'
    """
    text = _key_text(key)
    return pointer if text is None else f"{pointer}{KEY_MARK}{text}"


def locators(
    pointers: pa.Array | list[str], keys: pa.Array | list[object] | None = None
) -> pa.Array:
    """The locators of many events, row for row :func:`locate`'s text.

    Args:
        pointers: Each event's place in the file.
        keys: Each event's own key in the source, null (or an empty
            string) where it states none; ``None`` where it states none at
            all. Numbers are written in their decimal digits; a list is
            read as :func:`locate` reads one key (NaN states none).

    Returns:
        A string array as long as *pointers*.
    """
    placed = (
        pointers if isinstance(pointers, pa.Array) else pa.array(pointers, pa.string())
    )
    if keys is None:
        return placed
    stated = (
        pc.cast(keys, pa.string())
        if isinstance(keys, pa.Array)
        else pa.array([_key_text(key) for key in keys], pa.string())
    )
    # A key is stated where it is valid and not empty: every other string
    # sorts after the empty one.
    given = _filled(_compared("greater", stated, ""), pa.scalar(False))
    return _if_else(given, _joined_strings(placed, stated, KEY_MARK), placed)


def row_locators(
    count: int, start: int = 0, keys: pa.Array | list[object] | None = None
) -> pa.Array:
    """The locators ``row:<i>`` of *count* data rows of a table, from row *start* on.

    A data row is counted from 0, the header line not counted. Where *keys*
    states a row's own key (a table's ``id`` column), ``key:<key>``
    follows (:func:`locators`).
    """
    numbers = np.arange(start, start + count).astype(str)
    pointers = pa.array(np.char.add("row:", numbers).astype(object), pa.string())
    return locators(pointers, keys)


@functools.cache
def _distribution_version(name: str) -> str:
    """The installed version of the package *name*, as its metadata or module states it."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = getattr(importlib.import_module(name), "__version__", None)
        return "(version unknown)" if version is None else str(version)


def _source_text(source: Source) -> str:
    """A source file as a parse names it: its path, a member's ``jar:`` URI."""
    if isinstance(source, Path):
        return source.as_posix()
    return member_uri(source)


def _source_of_text(text: str, root: Path | None) -> Source:
    """The source a parse names (:func:`_source_text`), relative ones under *root*."""
    if text.startswith("jar:"):
        archive, _, member = text.removeprefix("jar:").partition("!/")
        return zipfile.Path(url2pathname(urlparse(archive).path), at=unquote(member))
    path = Path(text)
    return path if root is None or path.is_absolute() else root / path


def _walk(result: object) -> list[Any]:
    """The timelines *result* holds (a timeline, or a collection of them), descendants after each."""
    from timetoalign.timelines.base import Timeline

    found: list[Timeline] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for member in item.values():
                visit(member)
        elif isinstance(item, (list, tuple)):
            for member in item:
                visit(member)
        elif isinstance(item, Timeline):
            found.append(item)
            for uid in item.list_children():
                visit(item.get_child(uid))

    visit(result)
    return found


def _rows_read_again(built: object) -> list[tuple[str, EventData, Parse]]:
    """What :meth:`Loader.update` hands the registry, per timeline *built* and parse.

    For each timeline *built* (and each descendant) and each parse its
    located rows name: the timeline's ID, its events of that parse
    with their locators (a column ``locator``), local to the timeline, and
    the parse. Rows that name no locator are left out.
    """
    pieces: list[tuple[str, EventData, Parse]] = []
    for timeline in _walk(built):
        own = timeline.registry.filter(within=timeline.id)
        rows = own.table
        rows = rows.filter(rows.column(LOCATOR).is_valid())
        if not rows.num_rows:
            continue
        parses = {parse.id: parse for parse in own.parses}
        events = timeline.events
        table = events.table
        for parse_id in pc.unique(rows.column(PARSE_ID)).to_pylist():
            mine = rows.filter(pc.equal(rows.column(PARSE_ID), parse_id))
            ids = mine.column("id")
            picked = table.filter(pc.is_in(table.column("id"), value_set=ids))
            at = pc.index_in(picked.column("id"), value_set=ids)
            picked = picked.append_column(
                LOCATOR, mine.column(LOCATOR).combine_chunks().take(at)
            )
            pieces.append(
                (
                    timeline.id,
                    type(events)(picked, events.unit, events.number_type),
                    parses[parse_id],
                )
            )
    return pieces


_LoaderT = TypeVar("_LoaderT", bound="Loader[Any]")
_Params = ParamSpec("_Params")
_Result = TypeVar("_Result")


def _minting_from_the_loader(
    method: Callable[Concatenate[_LoaderT, _Params], _Result],
) -> Callable[Concatenate[_LoaderT, _Params], _Result]:
    """Run a door of a loader with the loader's minter as the call's minter.

    Every element the method builds without an ID of its own (a group, a
    bundle, a map, flow control, events) takes it from the loader's
    :class:`~timetoalign.core.ids.IdMinter` of the latest ``load`` call, in
    build order (:func:`~timetoalign.core.ids.minting_from`), never from a
    minter the process keeps: what a fresh loader builds from unchanged
    sources does not depend on anything built before it in the process.
    """

    loads = method.__name__ == "load"

    @functools.wraps(method)
    def door(self: _LoaderT, *args: _Params.args, **kwargs: _Params.kwargs) -> _Result:
        outermost_load = loads and not getattr(self, "_loading", False)
        if outermost_load:
            self._loading = True
        try:
            with minting_from(lambda: self._id_minter):
                result = method(self, *args, **kwargs)
        finally:
            if outermost_load:
                self._loading = False
        if outermost_load:
            self._load_calls.append((args, dict(kwargs)))
        return result

    return door


def _mint_doors_from_the_loader(cls: type) -> None:
    """Wrap every door *cls* defines: ``load``, ``load_*`` and ``create_*`` methods.

    See :func:`_minting_from_the_loader`. The outermost ``load`` call is
    recorded (its arguments, for :meth:`Loader.update`).
    """
    for name, member in list(vars(cls).items()):
        door = name == "load" or name.startswith(("load_", "create_"))
        if door and inspect.isfunction(member):
            setattr(cls, name, _minting_from_the_loader(member))


class Loader(ABC, Generic[PayloadT]):
    """Shared lifecycle for all loader families.

    The root owns source tracking, the ``load`` template, convenience
    construction, and timeline-creation surfaces. Payload-specific state and
    APIs belong to subclasses such as :class:`EventLoader`,
    :class:`ManifestLoader`, and :class:`AlignmentLoader`.

    Every ``load``, ``load_*`` and ``create_*`` method of every loader
    mints from the loader's minter (see :meth:`load`): the groups, bundles,
    maps and flow control it builds without an ID of their own included.
    """

    #: Whether one loader reads one source only. ``load`` then refuses a second
    #: source, in the same call or a later one, until ``clear()``.
    _single_source: ClassVar[bool] = False

    #: The constructor arguments that are enums, by name: a parse
    #: records their members' names (:meth:`from_registry`).
    _enum_parameters: ClassVar[dict[str, type[Enum]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _mint_doors_from_the_loader(cls)

    def __init__(self, *, seed: int | None = None) -> None:
        """Initialize shared loader state.

        Args:
            seed: The seed of the loader's ID minter; without one, the
                minter is seeded from the sources (see :meth:`load`).
        """
        self._sources: list[Source] = []
        self._source_metadata: list[dict[str, Any]] = []
        self._reset_minter(seed)
        self._reset_parses()

    @abstractmethod
    def _load_source(self, source: Source) -> PayloadT:
        """Parse one source and return the family-specific payload.

        *source* is a ``Path`` or a member of a ZIP archive (``zipfile.Path``).
        A parser taking a file object reads it in place
        (``timetoalign.loader._sources.readable`` or ``source.open()``); one
        that needs a file on disk reads the copy
        :func:`~timetoalign.loader.materialize` makes, through
        ``timetoalign.loader._sources.on_disk``.
        """
        ...

    def _unpack_source_result(
        self, result: PayloadT
    ) -> tuple[dict[str, Any], PayloadT]:
        """Return metadata and payload from a source parser result."""
        return {}, result

    def _accept_source(
        self, path: Source, source_meta: dict[str, Any], payload: PayloadT
    ) -> None:
        """Record a parsed payload.

        Generic payload loaders override this hook to retain their parsed
        object. The default records source lifecycle information only.
        """
        self._sources.append(path)
        self._source_metadata.append(source_meta)

    def _refuse_further_source(self, *sources: object) -> None:
        """Refuse *sources* when a single-source loader cannot take them.

        A loader whose ``_single_source`` is set reads one source; it requires
        that source, so it refuses a call with none, more than one in one
        call, and any while it holds one. Every door into such a loader's
        parse calls this before parsing anything.

        Raises:
            ValueError: If the loader reads one source and *sources* are none
                or more than one, or it already holds one.
        """
        if not self._single_source:
            return
        name = self.__class__.__name__
        if not sources:
            raise ValueError(f"{name} reads one source; got none.")
        held = self._held_source()
        if held is not None:
            raise ValueError(
                f"{name} reads one source and already holds {held}; use a new "
                f"loader for {', '.join(map(describe_source, sources))}, or call clear() "
                "first."
            )
        if len(sources) > 1:
            raise ValueError(
                f"{name} reads one source; got {len(sources)}: "
                f"{', '.join(map(describe_source, sources))}."
            )

    def _held_source(self) -> str | None:
        """Name the source this loader holds, or ``None`` when it holds none."""
        return describe_source(self._sources[0]) if self._sources else None

    def load(self, *sources: SourceLike) -> Self:
        """Load and accept one or more sources through the shared template.

        Each source is a filesystem path (``Path`` or ``str``) or a member of
        a ZIP archive (``zipfile.Path``), which is read without extracting
        the archive. Its entry in ``metadata["sources"]`` records ``path``
        (for a member: the archive's absolute path, and ``member``, the
        member's path inside it) and ``loaded_at``.

        Every ID the loader mints (events, timelines, maps, MatchClaims,
        Projections) comes from its :class:`~timetoalign.core.ids.IdMinter`,
        in the order the loader builds them. Every ``load`` call seeds a
        new one: the first call after the loader is made or cleared from
        the loader's ``seed`` when one was given, else from
        :func:`~timetoalign.core.ids.seed_from` of the loader's class name,
        its configuration (the constructor arguments that shape the
        events) and the SHA-256 digests of the call's sources, in order;
        every later call from ``seed_from`` of the previous call's seed and
        its own sources' digests, so later sources count too. A fresh
        loader loading unchanged sources in the same calls mints the same
        IDs, in any process; IDs the source states are kept as they are.

        Raises:
            FileNotFoundError: If a ``zipfile.Path`` names a member its
                archive does not hold.
            ValueError: If the loader reads one source and is given none or
                more than one, or any while it holds one (after ``clear()``
                it reads one again). Nothing is parsed then.
        """
        self._refuse_further_source(*sources)
        held_sources = [as_source(source) for source in sources]
        self._start_minter(*held_sources)
        for held in held_sources:
            source_meta, payload = self._unpack_source_result(self._load_source(held))
            source_meta.update(source_record(held))
            source_meta["loaded_at"] = datetime.now(timezone.utc).isoformat()
            self._accept_source(held, source_meta, payload)
        return self

    def _reset_minter(self, seed: int | None = None) -> None:
        """Forget every seed: the next ID comes from a minter seeded afresh.

        With *seed*, the minter of the first ``load`` call (or of a loader
        that loads nothing) is ``IdMinter(seed)``.
        """
        self._seed = seed
        self._seed_calls: list[tuple[Source | bytes | str, ...]] = []
        self._call_seeds: list[int] = []
        self._minter: IdMinter | None = None
        self._minter_calls = 0

    def _start_minter(self, *parts: Source | bytes | str) -> None:
        """Seed the loader's ID minter for one ``load`` call reading *parts*.

        Every door into a loader's parse calls this once, before minting
        anything. The seed (see :meth:`load`) is computed when the first ID
        is minted (:attr:`_id_minter`), so a loader that mints nothing
        reads nothing twice: a source stands for the SHA-256 digest of its
        bytes (a folder for its files', a source that does not exist for
        its name, so that its parser reports it), bytes and text for
        themselves. A loader reading in-memory content passes a canonical
        serialization of it.
        """
        self._seed_calls.append(parts)

    def _seed_config(self) -> dict[str, object]:
        """The configuration that shapes the events a loader builds, for its seed.

        A loader taking constructor arguments that change what it reads
        extends this with them (through ``super()``), so two loaders of one
        class configured differently mint different IDs from one source.
        Values are enum members, numbers, strings, paths, fractions and
        mappings or sequences of them.
        """
        return {}

    def _seed_of_call(self, calls: int) -> int:
        """The seed of the minter after *calls* ``load`` calls (see :meth:`load`)."""
        while len(self._call_seeds) < calls:
            parts = tuple(
                _seed_part(part) for part in self._seed_calls[len(self._call_seeds)]
            )
            if self._call_seeds:
                seed = seed_from(self._call_seeds[-1], *parts)
            else:
                seed = self._first_seed(*parts)
            self._call_seeds.append(seed)
        return self._call_seeds[calls - 1] if calls else self._first_seed()

    def _first_seed(self, *parts: bytes | str) -> int:
        """The seed of the first ``load`` call reading *parts*: ``seed``, else from the content."""
        if self._seed is not None:
            return self._seed
        return seed_from(type(self).__name__, _canonical(self._seed_config()), *parts)

    @property
    def _id_minter(self) -> IdMinter:
        """The loader's ID minter for its latest ``load`` call (see :meth:`load`)."""
        calls = len(self._seed_calls)
        if self._minter is None or self._minter_calls != calls:
            self._minter = IdMinter(self._seed_of_call(calls))
            self._minter_calls = calls
        return self._minter

    @classmethod
    def from_file(cls, *paths: SourceLike, **kwargs: Any) -> Self:
        """Instantiate the loader, load *paths*, and return it.

        A keyword the loader's :meth:`load` declares keyword-only (what to
        read of a source, as ``performance`` of
        :class:`~timetoalign.loader.alignment.MpmLoader`) goes to
        :meth:`load`; every other keyword to the constructor.

        Args:
            *paths: Filesystem paths or members of ZIP archives
                (``zipfile.Path``), as :meth:`load` takes them.
            **kwargs: The keyword-only arguments of :meth:`load`, and the
                loader's constructor arguments, ``seed`` among them (the
                seed of the loader's ID minter).

        Examples:
            >>> from zipfile import Path as ZipPath
            >>> member = ZipPath("corpus.zip", at="corpus/measures/piece.measures.tsv")
            >>> loader = Ms3Loader.from_file(member)
        """
        load_keywords = {
            name: kwargs.pop(name) for name in cls._load_keywords() if name in kwargs
        }
        loader = cls(**kwargs)
        loader.load(*paths, **load_keywords)
        return loader

    @classmethod
    def _load_keywords(cls) -> tuple[str, ...]:
        """The names of the keyword-only parameters of the loader's :meth:`load`."""
        return tuple(
            name
            for name, parameter in inspect.signature(cls.load).parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        )

    @property
    def sources(self) -> list[Source]:
        """The loaded sources: ``Path`` objects and ZIP members (``zipfile.Path``)."""
        return list(self._sources)

    @property
    def metadata(self) -> dict[str, Any]:
        """Aggregated source metadata."""
        return {
            "loader_class": self.__class__.__name__,
            "source_count": len(self._sources),
            "sources": self._source_metadata,
        }

    def _read_back(
        self,
        path: Path | str,
        seed: int | None,
        parses: Iterable[Parse] = (),
    ) -> None:
        """Set the shared state of a loader made from its own saved events at *path*.

        The loader holds no sources, and the *parses* its saved events
        name. The IDs it mints later come from ``IdMinter(seed)``, else
        from a minter seeded with :func:`~timetoalign.core.ids.seed_from` of
        the class name and the digest of *path* (a folder's files).
        """
        self._sources = []
        self._source_metadata = []
        if seed is None:
            seed = seed_from(type(self).__name__, _source_part(Path(path)))
        self._reset_minter(seed)
        self._reset_parses()
        for parse in parses:
            self._parses.setdefault(parse.source or parse.id, parse)

    def clear(self) -> None:
        """Clear shared source state; the next ``load`` seeds as the first did."""
        self._sources.clear()
        self._source_metadata.clear()
        self._reset_minter(self._seed)
        self._reset_parses()

    # region Parses

    def _reset_parses(self) -> None:
        """Forget the parses and the ``load`` calls recorded."""
        self._parses: dict[str, Parse] = {}
        self._load_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._loading = False

    @property
    def parses(self) -> list[Parse]:
        """One :class:`~timetoalign.storage.Parse` per source file read, in order.

        Each records the file and its SHA-256 digest, the parser (the
        loader's class name) and its version (this library's and the
        backend's, ``timetoalign 1.2.0; mido 1.3.2``), the parameters it
        ran with (the configuration that shapes its events, as the seed of
        its IDs reads it), the ID minter's algorithm and length, and the
        loader's ``seed`` where one was given. Every event the loader
        reads from a file names its parse (``parse_id``) and its
        place in the file (``locator``); a timeline the loader creates
        holds both in its registry rows.
        """
        return list(self._parses.values())

    def _backends(self) -> tuple[str, ...]:
        """The packages, by distribution name, that parse the loader's sources.

        Their versions follow this library's in a parse's
        ``parser_version``. A loader parsing with a package extends this.
        """
        return ()

    def _parser_version(self) -> str:
        """``timetoalign <version>`` and each backend's ``<name> <version>``, ``; ``-joined."""
        from timetoalign import __version__

        return "; ".join(
            [f"timetoalign {__version__}"]
            + [f"{name} {_distribution_version(name)}" for name in self._backends()]
        )

    def _parse_of(self, source: Source) -> Parse:
        """The parse of this loader's run over the file *source*, made once.

        A file read twice (two ``load`` calls) has one parse: its ID
        is seeded from the file, its digest, the parser and its version,
        the parameters, the minter and the seed, never from the date.
        """
        text = _source_text(source)
        held = self._parses.get(text)
        if held is not None:
            return held
        fields: dict[str, Any] = {
            "parser": type(self).__name__,
            "parser_version": self._parser_version(),
            "parameters": _canonical(self._seed_config()),
            "minter": ID_MINTER,
            "id_length": self._id_minter.length,
            "seed": self._seed,
        }
        parse = Parse.of_source(source, source=text, **fields)
        self._parses[text] = parse
        return parse

    def _stamps(
        self, source: Source, locators: pa.Array | list[str | None]
    ) -> dict[str, pa.Array]:
        """The registry columns of rows read from *source*: ``locator``, ``parse_id``.

        Args:
            source: The file the rows were read from.
            locators: Each row's place in the file, by the loader's rule.
        """
        locator = (
            locators
            if isinstance(locators, pa.Array)
            else pa.array(locators, pa.string())
        )
        parse = self._parse_of(source)
        return {
            LOCATOR: locator,
            PARSE_ID: pa.array(
                np.full(len(locator), parse.id, dtype=object), pa.string()
            ),
        }

    def _adopt_parses(self, loader: Loader[Any]) -> None:
        """Hold the parses of *loader*, one this loader composes for a file.

        Events this loader takes from it keep their ``parse_id`` and
        ``locator``; what this loader creates holds those parses too.
        """
        for text, parse in loader._parses.items():
            self._parses.setdefault(text, parse)

    def _stamp(
        self, source: Source, pointer: str | None, key: object = None
    ) -> dict[str, str | None]:
        """The registry keys of one event read from *source* at *pointer*.

        The locator is :func:`locate`'s of *pointer* and *key*, the key the
        source states for the event (none by default).
        """
        return {
            LOCATOR: None if pointer is None else locate(pointer, key),
            PARSE_ID: self._parse_of(source).id,
        }

    def _stamp_rows(
        self, rows: list[dict[str, Any]], source: Source, locators: list[str | None]
    ) -> list[dict[str, Any]]:
        """*rows* (event dictionaries read from *source*) with their registry keys.

        Each row takes its locator and the parse of *source*, in place.
        """
        parse_id = self._parse_of(source).id
        for row, locator in zip(rows, locators, strict=True):
            row[LOCATOR] = locator
            row[PARSE_ID] = parse_id
        return rows

    def _carrying(self, data: _EventDataT) -> _EventDataT:
        """*data* carrying the parses of this loader that its rows name.

        A timeline made from *data* (by the timeline factory, as
        :meth:`create_timeline` makes it) holds them with the rows
        (:attr:`EventData.parses <timetoalign.storage.EventData.parses>`).
        """
        table = data.table
        if PARSE_ID not in table.column_names or not self._parses:
            return data
        named = set(pc.unique(table.column(PARSE_ID)).to_pylist())
        carried = [e for e in self._parses.values() if e.id in named]
        return data.with_parses(*carried) if carried else data

    @classmethod
    def _config_of(cls, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """The constructor arguments a parse's *parameters* record.

        Each parameter the constructor takes (by name, or through its
        ``**kwargs``), an enum by its member's name (:attr:`_enum_parameters`).
        """
        signature = inspect.signature(cls.__init__).parameters
        takes_any = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.values()
        )
        config: dict[str, Any] = {}
        for name, value in parameters.items():
            if name not in signature and not takes_any:
                continue
            enum = cls._enum_parameters.get(name)
            config[name] = (
                enum[value] if enum is not None and value is not None else value
            )
        return config

    @classmethod
    def from_registry(
        cls,
        registry: EventRegistry,
        *sources: SourceLike,
        root: Path | str | None = None,
        **kwargs: Any,
    ) -> Self:
        """A loader set up from *registry*'s parses, to refresh it with :meth:`update`.

        The loader runs with the parameters (and the seed) the registry's
        parses of this loader's class recorded, and reads their source
        files, unless *sources* names what to read (a loader reading a
        folder, an archive or a project is given that). Nothing is read
        until :meth:`update`.

        Args:
            registry: The registry to refresh, a timeline's, a group's or a
                bundle's.
            *sources: What the loader reads, as :meth:`load` takes it; by
                default the source files of the parses, in order.
            root: The folder a relative source file of a parse lies
                in; by default the current one.
            **kwargs: The keyword-only arguments of :meth:`load`, and
                constructor arguments, which must leave the parameters as
                the parses recorded them.

        Returns:
            The loader, its ``load`` call recorded but not run.

        Raises:
            ValueError: If the parses of this class record two sets of
                parameters or seeds, the loader set up runs with other
                parameters than they record, or neither *sources* nor an
                parse names a source file.
        """
        own = [e for e in registry.parses if e.parser == cls.__name__]
        recorded = {json.dumps(e.parameters, sort_keys=True) for e in own}
        if len(recorded) > 1:
            raise ValueError(
                f"the registry's parses of {cls.__name__} ran with "
                f"{len(recorded)} sets of parameters; set up one loader per set"
            )
        seeds = {e.seed for e in own}
        if len(seeds) > 1:
            raise ValueError(
                f"the registry's parses of {cls.__name__} name {len(seeds)} seeds: "
                f"{sorted(map(str, seeds))}"
            )
        load_keywords = {
            name: kwargs.pop(name) for name in cls._load_keywords() if name in kwargs
        }
        parameters = own[0].parameters if own else None
        config = {**cls._config_of(parameters or {}), **kwargs}
        seed = own[0].seed if own else None
        if seed is not None and "seed" not in config:
            config["seed"] = int(seed)
        loader = cls(**config)
        if parameters is not None:
            ran = _canonical(loader._seed_config())
            assert isinstance(ran, dict)
            differing = sorted(
                name
                for name in {*ran, *parameters}
                if ran.get(name) != parameters.get(name)
            )
            if differing:
                raise ValueError(
                    f"{cls.__name__} as set up runs with other parameters than the "
                    f"registry's parses record: {differing}"
                )
        if not sources:
            named = list(dict.fromkeys(e.source for e in own if e.source is not None))
            if not named:
                raise ValueError(
                    f"the registry names no source file {cls.__name__} read; "
                    "name what to read"
                )
            folder = None if root is None else Path(root)
            sources = tuple(_source_of_text(text, folder) for text in named)
        loader._load_calls = [(tuple(sources), load_keywords)]
        return loader

    def update(self, registry: EventRegistry) -> list[str]:
        """Re-read the loader's sources and update *registry* with what they hold now.

        The loader is cleared and its ``load`` calls are run again (those
        :meth:`from_registry` set up, or the loader's own); each timeline
        :meth:`create_timelines` builds from them (and each descendant) then
        updates the registry's timeline that holds the rows its locators
        name (else the one of its ID), one parse at a time
        (:meth:`EventRegistry.update <timetoalign.storage.EventRegistry.update>`):
        a row found again by its locator keeps its ID and takes the new
        content, position and parse; a new row is added with the ID
        it was given (minted from the content, as at every ``load``); a
        row of the source no longer found is kept and marked vanished in
        the new parse. Nothing is deleted. Rows that name no locator
        (a child's place, events made in code) are left as they are.

        Args:
            registry: The registry to update: a timeline's, a group's or a
                bundle's (:attr:`Timeline.registry
                <timetoalign.timelines.Timeline.registry>`).

        Returns:
            The IDs of the registry's timelines updated, in order.

        Raises:
            RuntimeError: If the loader has no ``load`` call to run again
                (it read nothing, or in-memory content only).
            ValueError: If a timeline built holds rows that match no
                timeline of *registry*, by locator or by ID; nothing is
                updated then.
        """
        calls = list(self._load_calls)
        if not calls:
            raise RuntimeError(
                f"{type(self).__name__} has no load() call to run again: it read no "
                "source file (set it up with from_registry, or load first)"
            )
        self.clear()
        for args, keywords in calls:
            self.load(*args, **keywords)
        planned: list[tuple[str, EventData, Parse]] = []
        unmatched: list[str] = []
        for timeline_id, data, parse in _rows_read_again(self.create_timelines()):
            target = registry.update_target(data, parse, timeline_id=timeline_id)
            if target is None:
                unmatched.append(f"{timeline_id} ({parse.source})")
            else:
                planned.append((target, data, parse))
        if unmatched:
            raise ValueError(
                "rows read again match no timeline of the registry, by locator or by "
                f"ID: those of {unmatched}; nothing is updated"
            )
        for target, data, parse in planned:
            # Through the view of the target alone, so that each update goes to
            # the timeline chosen before any of them changed the registry.
            registry.filter(within=target).update(data, parse, timeline_id=target)
        return list(dict.fromkeys(target for target, _, _ in planned))

    # endregion

    def create_timeline(self, uid: str | None = None, **kwargs: Any) -> "Timeline":
        """Create a timeline from the loaded payload.

        Concrete loader families implement payload-specific construction.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_timeline()."
        )

    def create_timelines(self, id_pattern: str | None = None) -> "list[Timeline]":
        """Create all timelines, optionally filtered by an implementation."""
        return [self.create_timeline()]

    def create_group(self, **kwargs: Any) -> "TimelineGroup | None":
        """Create a timeline group when supported."""
        return None

    def create_bundle(self, **kwargs: Any) -> "AlignmentBundle":
        """Create an alignment bundle when supported."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_bundle()."
        )

    def __repr__(self) -> str:
        """Return a concise shared-lifecycle representation."""
        return f"{self.__class__.__name__}(sources={len(self._sources)})"


_mint_doors_from_the_loader(Loader)


class EventLoader(Loader[LoadSourceResult]):
    """Abstract base class for event-producing loaders.

    Loader provides a unified interface for loading one or more source files
    and aggregating their events into an EventData. Metadata about the sources
    is stored in the PyArrow table's schema metadata for determinism.

    Subclasses must implement:
    - _load_source(): Parse a single source file into event rows
    - _default_unit: The default time unit for this loader type

    Attributes:
        events: The EventData containing all loaded events.
        sources: List of loaded source file paths.
        unit: The time unit for coordinates.
        number_type: The number type for coordinates.

    Examples:
        >>> # Subclass implementation
        >>> class MidiLoader(Loader):
        ...     _default_unit = TimeUnit.ticks
        ...     _event_data_class = EventData
        ...
        ...     def _load_source(self, path):
        ...         # Parse MIDI file, return (metadata_dict, event_rows)
        ...         return {"format": "midi"}, [{"id": "n1", ...}]
        >>>
        >>> loader = MidiLoader()
        >>> loader.load("piece.mid")
        >>> print(loader.event_summary())
    """

    # Class-level configuration (subclasses override)
    _default_unit: ClassVar[TimeUnit] = TimeUnit.seconds
    # A class-level default; a loader may choose its EventData per instance.
    _event_data_class: type[EventData] = EventData

    # Canonical ordering for RawField fields in assembled events tables.
    # Fields matching these names appear first (in this order), followed by
    # non-canonical RawFields (in field_specs order), then property fields.
    _CANONICAL_FIELD_ORDER: ClassVar[list[str]] = ["start", "end", "duration"]

    _enum_parameters: ClassVar[dict[str, type[Enum]]] = {
        "unit": TimeUnit,
        "number_type": NumberType,
        "interval_policy": IntervalPolicy,
    }

    def __init__(
        self,
        unit: TimeUnit | None = None,
        number_type: NumberType | None = None,
        interval_policy: IntervalPolicy | str = IntervalPolicy.warn,
        *,
        seed: int | None = None,
    ) -> None:
        """Initialize the Loader.

        Args:
            unit: The time unit for coordinates. Defaults to class's _default_unit.
            number_type: The number type for coordinates. Defaults to the
                one the unit itself uses.
            interval_policy: How to handle end/duration inconsistencies in
                interval events.  Accepts an `IntervalPolicy` enum member or
                a string (``"warn"``, ``"prefer_end"``, ``"prefer_duration"``,
                ``"strict"``).  Default: ``"warn"`` -- prefer ``end``, warn
                on discrepancies.
            seed: The seed of the loader's ID minter (see ``Loader.load``).
        """
        super().__init__(seed=seed)
        self._unit = unit or self._default_unit
        self._number_type = self._unit.resolve_number_type(number_type)
        self._interval_policy = IntervalPolicy(interval_policy)
        self._events: EventData = self._event_data_class.empty(
            self._unit, self._number_type
        )

    def _seed_config(self) -> dict[str, object]:
        """The unit, number type and interval policy the events are built in, for the seed."""
        return {
            **super()._seed_config(),
            "unit": self._unit,
            "number_type": self._number_type,
            "interval_policy": self._interval_policy,
        }

    # region Abstract Methods

    @abstractmethod
    def _load_source(self, source: Source) -> LoadSourceResult:
        """Load a single source file.

        Subclasses implement this to parse their specific format.

        Event loaders return ``(metadata_dict, field_dict)`` where
        ``field_dict`` contains numpy or PyArrow arrays for each field. Other
        loader families may return their payload directly.

        Args:
            source: The source file: a path or a ZIP archive member.

        Returns:
            A source result. Event loaders return a tuple of file metadata and
            vectorized field arrays; payload loaders return their payload.

        Raises:
            FileNotFoundError: If the source file doesn't exist.
            ValueError: If the source file is invalid.
        """
        ...

    # endregion

    # region Properties

    @property
    def events(self) -> EventData:
        """The EventData containing all loaded events."""
        return self._events

    @property
    def unit(self) -> TimeUnit:
        """The time unit for coordinates."""
        return self._unit

    @property
    def number_type(self) -> NumberType:
        """The number type for coordinates."""
        return self._number_type

    @property
    def metadata(self) -> dict[str, Any]:
        """Aggregated metadata from all sources.

        Returns:
            Dict with loader info and per-source metadata.
        """
        return {
            "loader_class": self.__class__.__name__,
            "unit": str(self._unit),
            "number_type": str(self._number_type),
            "source_count": len(self._sources),
            "sources": self._source_metadata,
        }

    @property
    def store(self) -> "EventStore":
        """Return an EventStore wrapping the loader's events.

        Subclasses may override to return specialized stores
        (e.g., ScoreStore, MidiStore). Default implementation
        wraps self.events in a SingleStore.

        Returns:
            An EventStore providing uniform access to loaded data.
        """
        from timetoalign.storage.store import SingleStore

        return SingleStore(self._events, name="events")

    def create_timeline(
        self,
        uid: str | None = None,
        store_filters: dict[str, dict[str, Any]] | None = None,
        include_stores: list[str] | None = None,
        exclude_stores: list[str] | None = None,
        flatten: bool = False,
    ) -> "Timeline":
        """Create a Timeline from the loaded events.

        Builds the timeline from ``self.store``; every timeline ID it mints
        comes from the loader's minter.

        Args:
            uid: Unique ID for the parent timeline. Auto-generated if None.
            store_filters: Per-data filter kwargs to apply before timeline
                creation. Example: {"notes": {"event_type": "Note"}}.
            include_stores: Only include these data (default: all non-empty).
            exclude_stores: Exclude these data from the timeline.
            flatten: If True, merge all events into a single parent timeline.

        Returns:
            A Timeline containing the loaded events.

        Examples:
            >>> loader = Ms3Loader()
            >>> loader.load("notes.tsv")
            >>> timeline = loader.create_timeline(uid="my_score")
        """
        from timetoalign.timelines.factory import create_timeline_from_bundle

        return create_timeline_from_bundle(
            self.store,
            uid=uid,
            store_filters=store_filters,
            include_stores=include_stores,
            exclude_stores=exclude_stores,
            flatten=flatten,
            minter=self._id_minter,
        )

    def create_timelines(
        self,
        id_pattern: str | None = None,
    ) -> "list[Timeline]":
        """Create all timelines, optionally filtered by regex pattern.

        The default implementation returns a single-element list with
        ``create_timeline()``. Subclasses with multi-timeline output
        (e.g., ``TiliaJsonLoader``, ``MatchfileLoader``) override this.

        Args:
            id_pattern: Optional regex pattern to filter timeline IDs.

        Returns:
            List of `Timeline` objects.
        """
        return [self.create_timeline()]

    def create_group(self, **kwargs: Any) -> "TimelineGroup | None":
        """Create a `TimelineGroup`. Override in subclasses that support groups.

        Returns:
            A `TimelineGroup`, or ``None`` if the loader does not produce groups.
        """
        return None

    def create_bundle(self, **kwargs: Any) -> "AlignmentBundle":
        """Create an `AlignmentBundle`. Override in subclasses that support bundles.

        Raises:
            NotImplementedError: If the concrete loader does not produce bundles.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_bundle()."
        )

    # endregion

    # region Item Access

    def __getitem__(self, key: str) -> "EventData":
        """Access a named EventData table from the store.

        Delegates to ``self.store[key]``.

        Args:
            key: The name of the EventData table.

        Returns:
            The `EventData` for the given key.

        Raises:
            KeyError: If the key is not found in the store.
        """
        return self.store[key]

    # endregion

    # region Events Assembly

    def get_events(self, properties: bool | str | tuple[str, ...] = True) -> EventData:
        """Return an EventData assembled from loaded data.

        This is the primary access method for the loader-first pipeline.
        It assembles EventData from the loaded data, with control over
        which property fields are included.

        Args:
            properties: Controls which non-field-spec data fields to include.
                - ``True``: Include all property fields (default).
                - ``False``: Only include semantic-field fields (start, end,
                  duration, pitch, etc.) and core fields (id, name,
                  event_type, temporal_type).
                - Single string: shorthand for a one-element tuple — include
                  only the named property field.
                - Tuple of strings: Include only the named property fields.

        Returns:
            An ``EventData`` containing the assembled events.
        """
        if isinstance(properties, str):
            properties = (properties,)
        return self._assemble_events_table(properties)

    def _assemble_events_table(
        self, properties: bool | str | tuple[str, ...] = True
    ) -> EventData:
        """Assemble an EventData from internal state, with field ordering.

        Field ordering follows the canonical field order:
        1. Core fields (id, name, event_type, temporal_type)
        2. Canonical fields (start, end, duration, pitch -- in order, if present)
        3. Non-canonical fields (in field_specs order)
        4. Property fields (controlled by the *properties* parameter)

        The default implementation returns ``self._events`` unmodified
        when ``properties=True``.  Subclasses with ``field_specs`` can
        override to produce filtered/reordered tables.

        Args:
            properties: Field filter (see ``get_events``).

        Returns:
            An ``EventData`` with the requested fields.
        """
        if isinstance(properties, str):
            properties = (properties,)
        if properties is True:
            return self._events

        # Determine which fields to keep
        events = self._events
        if len(events) == 0:
            return events

        table = events.table
        all_names = set(table.column_names)

        # Core fields always included
        core_names = {"id", "name", "event_type", "temporal_type"}

        # Canonical-spec fields always included (when present)
        spec_names = set()
        for name in self._CANONICAL_FIELD_ORDER:
            if name in all_names:
                spec_names.add(name)

        # Also include any fields with timetoalign metadata (semantic fields)
        for i in range(len(table.schema)):
            pa_field = table.schema.field(i)
            if pa_field.metadata and TIMETOALIGN_METADATA_KEY in pa_field.metadata:
                spec_names.add(pa_field.name)

        keep = core_names | spec_names

        if properties is False:
            # Only fields + core
            pass
        elif isinstance(properties, tuple):
            # Add requested property fields
            for name in properties:
                if name in all_names:
                    keep.add(name)

        # Build ordered field list preserving table order
        ordered = [c for c in table.column_names if c in keep]
        if set(ordered) == all_names:
            return events

        filtered_table = table.select(ordered)
        return EventData(filtered_table, events.unit, events.number_type).with_parses(
            *events.parses
        )

    # endregion

    # region Loading

    def _unpack_source_result(
        self, result: LoadSourceResult
    ) -> tuple[dict[str, Any], dict[str, np.ndarray | pa.Array | pa.ChunkedArray]]:
        """Split an event parser result into metadata and field arrays."""
        return result

    def _accept_source(
        self,
        path: Source,
        source_meta: dict[str, Any],
        payload: dict[str, np.ndarray | pa.Array | pa.ChunkedArray],
    ) -> None:
        """Accept one loaded source using the event payload implementation."""
        if not isinstance(payload, dict):
            raise TypeError(
                f"{self.__class__.__name__} returned an unsupported payload "
                f"type: {type(payload).__name__}"
            )

        self._sources.append(path)
        self._source_metadata.append(source_meta)
        if not payload:
            return

        # Every row names the parse of its file and, by the loader's
        # rule, its place in it (a ``locator`` column of the payload).
        rows = len(next(iter(payload.values())))
        locators = payload.pop(LOCATOR, None)
        if locators is None:
            locators = pa.nulls(rows, pa.string())
        elif isinstance(locators, pa.ChunkedArray):
            locators = locators.combine_chunks()
        elif not isinstance(locators, pa.Array):
            locators = pa.array(locators, pa.string())
        payload = {**payload, **self._stamps(path, locators)}

        extra_fields = getattr(self, "_extra_schema_fields", None)
        new_data = self._event_data_class.from_arrays(
            payload,
            self._unit,
            self._number_type,
            validate=getattr(self, "_validate_vectorized", True),
            extra_fields=extra_fields,
            interval_policy=self._interval_policy,
            minter=self._id_minter,
        ).with_parses(self._parse_of(path))
        if len(self._events) == 0:
            self._events = new_data
        else:
            self._events = self._events.concat(new_data)

    def clear(self) -> None:
        """Clear all loaded sources and events."""
        super().clear()
        self._events = self._event_data_class.empty(self._unit, self._number_type)

    # endregion

    # region Stats (delegated to EventData)

    def event_summary(self) -> dict[str, Any]:
        """Get a summary of loaded events.

        Returns:
            Dict with event counts, types, coordinate range, etc.
        """
        summary = self._events.summary()
        summary["sources"] = [describe_source(s) for s in self._sources]
        return summary

    def count_events_by_type(self) -> dict[str, int]:
        """Count events grouped by event_type.

        Returns:
            Dict mapping event type names to counts.
        """
        return self._events.count_by("event_type")

    def count_events_by_temporal_type(self) -> dict[str, int]:
        """Count events grouped by temporal_type (instant/interval).

        Returns:
            Dict mapping "instant"/"interval" to counts.
        """
        return self._events.count_by("temporal_type")

    # endregion

    # region Magic Methods

    def __len__(self) -> int:
        """Return the total number of events across all store tables."""
        return self.store.event_count

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"sources={len(self._sources)}, "
            f"events={self.store.event_count}, "
            f"unit={self._unit})"
        )

    # endregion

    # region C-Map Creation

    def create_cmap(
        self,
        source_field: str,
        target_field: str,
        *,
        map_type: type | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a ConversionMap from two coordinate fields.

        This method creates a C-Map from loaded coordinate data, enabling
        conversion between different coordinate systems (e.g., seconds to pixels).

        Both fields must contain coordinate data (either core coordinates like
        'start'/'end', or CoordinateField extra fields).

        Args:
            source_field: Name of the source coordinate field.
            target_field: Name of the target coordinate field.
            map_type: The map class to create. Defaults to TableMap.
                Supported: TableMap, LinearMap, ScalarMap.
            **kwargs: Additional arguments passed to the map constructor.
                For TableMap: kind, extrapolate
                For LinearMap: (computed automatically from data)
                Without ``uid``, the map's ID is minted by the loader's
                minter.

        Returns:
            A ConversionMap instance.

        Raises:
            ValueError: If fields don't exist or aren't coordinate fields.
            ValueError: If insufficient data points for the map type.

        Examples:
            >>> # Load data with dual coordinates
            >>> loader.load("data.tsv")

            >>> # Create TableMap (default) from start -> x_pixels
            >>> cmap = loader.create_cmap("start", "x_pixels")

            >>> # Create LinearMap (fits y = ax + b to data)
            >>> cmap = loader.create_cmap("start", "x_pixels", map_type=LinearMap)

            >>> # TableMap with custom interpolation
            >>> cmap = loader.create_cmap("start", "x_pixels", kind="cubic")
        """
        from timetoalign.maps import LinearMap, ScalarMap, TableMap

        # Default to TableMap
        if map_type is None:
            map_type = TableMap
        if map_type not in (TableMap, LinearMap, ScalarMap):
            raise ValueError(
                f"Unsupported map_type: {map_type}. "
                f"Supported: TableMap, LinearMap, ScalarMap"
            )

        # Extract coordinate values from fields
        source_values = self._extract_coordinate_values(source_field)
        target_values = self._extract_coordinate_values(target_field)

        if len(source_values) != len(target_values):
            raise ValueError(
                f"Field length mismatch: {source_field} has {len(source_values)} "
                f"values, {target_field} has {len(target_values)}"
            )

        if len(source_values) < 2:
            raise ValueError(
                f"Need at least 2 data points to create a C-Map, got {len(source_values)}"
            )

        # Get units from field metadata
        source_unit = self._get_field_unit(source_field)
        target_unit = self._get_field_unit(target_field)
        if kwargs.get("uid") is None:
            kwargs["uid"] = self._id_minter.mint(CMAP_PREFIX)

        # Create the appropriate map type
        if map_type is TableMap:
            return TableMap(
                x_values=source_values,
                y_values=target_values,
                source_unit=source_unit,
                target_unit=target_unit,
                **kwargs,
            )
        elif map_type is LinearMap:
            # Fit linear regression: y = ax + b
            x = np.array(source_values, dtype=np.float64)
            y = np.array(target_values, dtype=np.float64)
            # Use numpy's polyfit for simple linear regression
            coeffs = np.polyfit(x, y, 1)
            scalar, offset = coeffs[0], coeffs[1]
            return LinearMap(
                scalar=float(scalar),
                offset=float(offset),
                source_unit=source_unit,
                target_unit=target_unit,
                **kwargs,
            )
        else:
            # Fit pure scaling: y = ax (no offset)
            x = np.array(source_values, dtype=np.float64)
            y = np.array(target_values, dtype=np.float64)
            # Least squares fit through origin: scalar = sum(x*y) / sum(x*x)
            scalar = np.sum(x * y) / np.sum(x * x)
            return ScalarMap(
                scalar=float(scalar),
                source_unit=source_unit,
                target_unit=target_unit,
                **kwargs,
            )

    def _extract_coordinate_values(
        self, field_name: str
    ) -> list[int | float | Fraction]:
        """Extract a coordinate field's values in their own representation.

        A C-Map built from an exact column deserves exact anchors: a tick
        grid read off a score should convert back to the ratios it came
        from, not to whatever the float projection rounds to. So this reads
        each cell in the representation the field declares rather than
        reaching straight for the float side.

        Args:
            field_name: The field name to extract.

        Returns:
            List of coordinate values.

        Raises:
            ValueError: If field doesn't exist or isn't a coordinate field.
        """
        table = self._events.table

        if field_name not in table.column_names:
            raise ValueError(
                f"Field '{field_name}' not found. " f"Available: {table.column_names}"
            )

        arr = table.column(field_name)

        # Check if it's a coordinate struct (has 'value' field)
        if pa.types.is_struct(arr.type):
            inner_names = [f.name for f in arr.type]
            if "value" in inner_names:
                number_type = self._get_field_number_type(field_name)
                decoded: list[int | float | Fraction] = []
                for cell in arr.to_pylist():
                    if cell is None:
                        decoded.append(0.0)
                        continue
                    try:
                        decoded.append(struct_to_coordinate(cell, number_type))
                    except (TypeError, ValueError, ZeroDivisionError):
                        decoded.append(float(cell.get("value") or 0.0))
                return decoded
            else:
                raise ValueError(
                    f"Field '{field_name}' is a struct but doesn't have a 'value' "
                    f"field. Inner fields: {inner_names}"
                )
        elif pa.types.is_floating(arr.type) or pa.types.is_integer(arr.type):
            # Plain numeric field
            return [float(v) for v in arr.to_pylist() if v is not None]
        else:
            raise ValueError(
                f"Field '{field_name}' is not a coordinate field. " f"Type: {arr.type}"
            )

    def _get_field_unit(self, field_name: str) -> str | None:
        """Get the unit metadata from a coordinate field.

        Args:
            field_name: The field name.

        Returns:
            The unit string, or None if not specified.
        """
        table = self._events.table
        schema = table.schema

        field_idx = schema.get_field_index(field_name)
        if field_idx < 0:
            return None

        unit = field_metadata(schema.field(field_idx)).get("unit")
        if unit:
            return unit

        # For core coordinate fields (start, end, duration), use loader's unit
        if field_name in ("start", "end", "duration"):
            return str(self._unit.value) if self._unit else None

        return None

    def _get_field_number_type(self, field_name: str) -> NumberType:
        """Return the representation a coordinate field declares.

        Falls back to the loader's own, which is what an unannotated core
        coordinate column is written in.
        """
        schema = self._events.table.schema
        field_idx = schema.get_field_index(field_name)
        if field_idx >= 0:
            declared = field_metadata(schema.field(field_idx)).get("number_type")
            if declared:
                return NumberType(declared)
        return self._number_type

    # endregion

    # region HTML Representation

    def _repr_html_(self) -> str:
        """Rich HTML affordance card for Jupyter notebooks.

        Built from the overridable :meth:`_repr_rows` /
        :meth:`_repr_count_row` / :meth:`_repr_affordances` hooks so the
        whole loader hierarchy shares one rendering path (no hand-rolled
        ``<table>`` markup per subclass).
        """
        return affordance_html(
            type(self).__name__,
            self._repr_rows(),
            affordances=self._repr_affordances(),
        )

    def _repr_count_row(self) -> tuple[str, str]:
        """Return the payload-count row shown in :meth:`_repr_html_`.

        Default is ``("Events", str(len(self)))`` — ``len(self)`` (NOT
        ``self.store.event_count``) so whole-directory / single-file
        loaders that keep their payload in claims or timelines report the
        meaningful count via their ``__len__`` override.  Loaders whose
        payload is better named otherwise (e.g. alignment loaders counting
        claims) override this hook rather than appending a second row.
        """
        return ("Events", str(len(self)))

    def _repr_rows(self) -> list[tuple[str, str]]:
        """Return the ``(label, value)`` rows for :meth:`_repr_html_`.

        Default rows: Sources (count + first up-to-5 names), the
        :meth:`_repr_count_row` payload-count row, and Unit.  Subclasses
        extend by calling ``super()._repr_rows()`` and adding their own.
        """
        n_sources = len(self._sources)
        names = ", ".join(code(s.name) for s in self._sources[:5])
        if n_sources > 5:
            names += f", … (+{n_sources - 5} more)"
        sources_value = f"{n_sources} file(s)" + (f": {names}" if names else "")
        return [
            ("Sources", sources_value),
            self._repr_count_row(),
            ("Unit", html.escape(str(self._unit.value))),
        ]

    def _repr_affordances(self) -> list[str]:
        """Return the invocation snippets for the ``Try`` row.

        Detects the create methods this loader overrides plus the
        universal ``get_events(...)`` accessor and, when the class defines
        a ``diagram`` method, ``diagram()``.
        """
        snippets = ["create_timeline()"]
        if self.create_group.__func__ is not EventLoader.create_group:
            snippets.append("create_group()")
        if self.create_bundle.__func__ is not EventLoader.create_bundle:
            snippets.append("create_bundle()")
        snippets.append("get_events(...)")
        if hasattr(type(self), "diagram"):
            snippets.append("diagram()")
        return snippets

    # endregion

    # region Serialization

    def to_parquet(self, path: Path | str) -> None:
        """Save the loaded events to a Parquet file.

        The metadata (including source info) is preserved in the file, and
        the parses the events name (:attr:`parses`), which a
        loader read back from it (:meth:`from_parquet`) holds again.

        Args:
            path: Path to write the Parquet file.
        """
        write_events(self._carrying(self.events), path)

    @classmethod
    def from_parquet(cls, path: Path | str, *, seed: int | None = None) -> Self:
        """Load a Loader from a Parquet file.

        Note: This creates a new Loader with the EventData loaded,
        but source paths may not be accessible for re-loading. The events
        keep their IDs; the loader's minter, for the timeline IDs it mints
        later, is ``IdMinter(seed)``, else seeded from the class name and
        the file's bytes.

        Args:
            path: Path to the Parquet file.
            seed: The seed of the loader's ID minter.

        Returns:
            A new Loader with events loaded from the file.
        """
        events = read_parses(cls._event_data_class.from_parquet(path))
        loader = cls.__new__(cls)
        loader._unit = events.unit
        loader._number_type = events.number_type
        loader._events = events
        loader._read_back(path, seed, events.parses)
        return loader

    # endregion


# region ManifestData


@dataclass
class ManifestData:
    """Container for manifest/metadata from ManifestLoader.

    ManifestData holds structural information about a source without events.
    This includes dimensions, sample rates, page counts, etc.

    Attributes:
        dimensions: Dict of named dimensions (e.g., {"width": 1920, "height": 1080}).
        metadata: Additional metadata about the source.
        source_path: The source file: a path or a ZIP archive member.
        source_type: Type of source (e.g., "audio", "image", "pdf").

    Examples:
        >>> # Audio manifest
        >>> manifest = ManifestData(
        ...     dimensions={"duration_samples": 44100 * 60, "duration_seconds": 60.0},
        ...     metadata={"sample_rate": 44100, "channels": 2},
        ...     source_type="audio",
        ... )

        >>> # PDF manifest
        >>> manifest = ManifestData(
        ...     dimensions={"pages": 10, "width": 612, "height": 792},
        ...     metadata={"title": "Score", "dpi": 72},
        ...     source_type="pdf",
        ... )
    """

    dimensions: dict[str, int | float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Source | None = None
    source_type: str = "unknown"

    def __repr__(self) -> str:
        dims_str = ", ".join(f"{k}={v}" for k, v in self.dimensions.items())
        return f"ManifestData({self.source_type}, {dims_str})"


# endregion


# region ManifestLoader


class ManifestLoader(Loader[ManifestData]):
    """Abstract base class for loaders that return ManifestData.

    ManifestLoader is for sources that provide dimensional/structural
    information without timeline events:
    - Audio files -> duration, sample rate, channels
    - PDF files -> pages, dimensions
    - Images -> width, height, DPI

    Subclasses must implement:
    - _load_source(): Parse a single source into ManifestData

    The loaded ManifestData can be used to create empty timelines with
    correct dimensions, or to configure subsequent event loading.

    Examples:
        >>> class AudioManifestLoader(ManifestLoader):
        ...     def _load_source(self, source: Path) -> ManifestData:
        ...         # Read audio header
        ...         return ManifestData(
        ...             dimensions={"duration_samples": n_samples},
        ...             metadata={"sample_rate": sr},
        ...             source_type="audio",
        ...         )
    """

    def __init__(self, *, seed: int | None = None) -> None:
        """Initialize the ManifestLoader.

        Args:
            seed: The seed of the loader's ID minter (see ``Loader.load``).
        """
        super().__init__(seed=seed)
        self._manifests: list[ManifestData] = []
        self._logger = module_logger.getChild(self.__class__.__name__)

    # region Abstract Methods

    @abstractmethod
    def _load_source(self, source: Source) -> ManifestData:
        """Load a single source file into ManifestData.

        Subclasses implement this to parse their specific format.

        Args:
            source: The source file: a path or a ZIP archive member.

        Returns:
            ManifestData containing dimensions and metadata.

        Raises:
            FileNotFoundError: If the source file doesn't exist.
            ValueError: If the source file is invalid.
        """
        ...

    # endregion

    def _accept_source(
        self,
        path: Source,
        source_meta: dict[str, Any],
        payload: ManifestData,
    ) -> None:
        """Record one manifest payload from the root lifecycle."""
        payload.source_path = path
        self._sources.append(path)
        self._source_metadata.append(source_meta)
        self._manifests.append(payload)
        self._logger.debug(f"Loaded manifest from {describe_source(path)}")

    # region Properties

    @property
    def manifests(self) -> list[ManifestData]:
        """List of loaded ManifestData objects."""
        return list(self._manifests)

    @property
    def manifest(self) -> ManifestData | None:
        """The first (or only) manifest, for convenience."""
        return self._manifests[0] if self._manifests else None

    # endregion

    def clear(self) -> None:
        """Clear all loaded sources and manifests."""
        super().clear()
        self._manifests.clear()

    # region Timeline Creation

    def create_timeline(
        self,
        uid: str | None = None,
        **kwargs: Any,
    ) -> "Timeline":
        """Create a Timeline from the loaded manifest.

        Subclasses override this to create the appropriate timeline type
        (e.g., ``DiscretePhysicalTimeline`` for audio, ``DiscreteGraphicalTimeline``
        for images).

        Args:
            uid: Unique identifier for the timeline. Auto-generated if None.
            **kwargs: Additional arguments for subclass implementations.

        Returns:
            A Timeline representing the loaded source.

        Raises:
            NotImplementedError: If the subclass does not provide an implementation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_timeline(). "
            "Subclasses should override this method."
        )

    def create_timelines(
        self,
        id_pattern: str | None = None,
    ) -> "list[Timeline]":
        """Create all timelines, optionally filtered by regex pattern.

        Default implementation returns a single-element list.

        Args:
            id_pattern: Optional regex pattern to filter timeline IDs.

        Returns:
            List of `Timeline` objects.
        """
        return [self.create_timeline()]

    # endregion

    # region HTML Representation

    def _repr_html_(self) -> str:
        """Rich HTML affordance card for Jupyter notebooks."""
        return affordance_html(
            type(self).__name__,
            self._repr_rows(),
            affordances=self._repr_affordances(),
        )

    def _repr_rows(self) -> list[tuple[str, str]]:
        """Return the ``(label, value)`` rows for :meth:`_repr_html_`.

        Sources, the manifest count, then one row per dimension of the
        first manifest (the manifest's structural shape).
        """
        n_sources = len(self._sources)
        names = ", ".join(code(s.name) for s in self._sources[:5])
        if n_sources > 5:
            names += f", … (+{n_sources - 5} more)"
        sources_value = f"{n_sources} file(s)" + (f": {names}" if names else "")
        rows: list[tuple[str, str]] = [
            ("Sources", sources_value),
            ("Manifests", str(len(self))),
        ]
        if self._manifests:
            for k, v in self._manifests[0].dimensions.items():
                rows.append((str(k), html.escape(str(v))))
        return rows

    def _repr_affordances(self) -> list[str]:
        """Return the invocation snippets for the ``Try`` row."""
        return ["create_timeline()", "create_timelines()"]

    # endregion

    # region Magic Methods

    def __len__(self) -> int:
        """Return the number of loaded manifests."""
        return len(self._manifests)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(sources={len(self._sources)})"

    # endregion


# endregion


# region AlignmentLoader


class AlignmentLoader(Loader["AlignmentStore"]):
    """Abstract base class for loaders that return AlignmentStore.

    AlignmentLoader is for formats that encode aligned multimodal data:
    - TiLiA JSON -> hierarchical annotations with alignments
    - Match files -> score-to-performance alignment
    - MPM, parangonada, Listen Here!, MAPS, and PerformancePrecision exports

    Subclasses must implement:
    - _load_source(): Parse a single source into AlignmentStore

    AlignmentStore contains:
    - EventData tables for each domain/layer
    - ConversionMaps between coordinate systems

    MatchClaims are made by the loader itself (``create_bundle`` and the
    ``create_*`` methods), never held in the store.

    Examples:
        >>> loader = MatchfileLoader()
        >>> loader.load("performance.match")
        >>> bundle = loader.create_bundle()
    """

    def __init__(self, *, seed: int | None = None) -> None:
        """Initialize the AlignmentLoader.

        Args:
            seed: The seed of the loader's ID minter (see ``Loader.load``).
        """
        super().__init__(seed=seed)
        self._store: AlignmentStore | None = None
        self._logger = module_logger.getChild(self.__class__.__name__)

    # region Abstract Methods

    @abstractmethod
    def _load_source(self, source: Source) -> "AlignmentStore":
        """Load a single source file into AlignmentStore.

        Subclasses implement this to parse their specific format.

        Args:
            source: The source file: a path or a ZIP archive member.

        Returns:
            AlignmentStore containing events and C-maps.

        Raises:
            FileNotFoundError: If the source file doesn't exist.
            ValueError: If the source file is invalid.
        """
        ...

    # endregion

    def _accept_source(
        self,
        path: Source,
        source_meta: dict[str, Any],
        payload: "AlignmentStore",
    ) -> None:
        """Store or merge one alignment payload from the root lifecycle."""
        self._sources.append(path)
        self._source_metadata.append(source_meta)
        if self._store is None:
            self._store = payload
        else:
            self._store.extend(payload)
        self._logger.debug(f"Loaded alignment from {describe_source(path)}")

    # region Properties

    @property
    def store(self) -> "AlignmentStore | None":
        """The AlignmentStore containing all loaded data."""
        return self._store

    @property
    def metadata(self) -> dict[str, Any]:
        """Aggregated metadata from all sources."""
        return {
            "loader_class": self.__class__.__name__,
            "source_count": len(self._sources),
            "sources": self._source_metadata,
        }

    # endregion

    def clear(self) -> None:
        """Clear all loaded sources and store."""
        super().clear()
        self._store = None

    # region Timeline & Bundle Creation

    def create_timeline(self, uid: str | None = None, **kwargs: Any) -> "Timeline":
        """Create a single Timeline by uid.

        Subclasses override this to assemble timelines from loaded data.

        Args:
            uid: Timeline identifier. Interpretation is subclass-specific.
            **kwargs: Additional arguments for subclass implementations.

        Returns:
            A Timeline.

        Raises:
            NotImplementedError: If the subclass does not provide an implementation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_timeline(). "
            "Subclasses should override this method."
        )

    def create_timelines(
        self,
        id_pattern: str | None = None,
    ) -> "list[Timeline]":
        """Create all timelines, optionally filtered by regex pattern.

        Args:
            id_pattern: Optional regex pattern to filter timeline IDs.

        Returns:
            List of `Timeline` objects.

        Raises:
            NotImplementedError: If the subclass does not provide an implementation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_timelines(). "
            "Subclasses should override this method."
        )

    @staticmethod
    def _filter_timelines_by_id_pattern(
        timelines: list["Timeline"], id_pattern: str | None
    ) -> list["Timeline"]:
        """Filter timelines by an optional regular expression on their IDs."""
        return [
            tl for tl in timelines if id_pattern is None or re.search(id_pattern, tl.id)
        ]

    def create_group(self, **kwargs: Any) -> "TimelineGroup | None":
        """Create a `TimelineGroup`. Override in subclasses that support groups.

        Returns:
            A `TimelineGroup`, or ``None`` if not supported.
        """
        return None

    def create_bundle(self, **kwargs: Any) -> "AlignmentBundle":
        """Create an `AlignmentBundle`. Override in subclasses that support bundles.

        Raises:
            NotImplementedError: If the concrete loader does not produce bundles.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_bundle()."
        )

    # endregion

    # region HTML Representation

    def _repr_html_(self) -> str:
        """Rich HTML affordance card for Jupyter notebooks.

        Built from the overridable :meth:`_repr_rows` /
        :meth:`_repr_count_row` / :meth:`_repr_affordances` hooks so every
        alignment loader shares one rendering path.
        """
        return affordance_html(
            type(self).__name__,
            self._repr_rows(),
            affordances=self._repr_affordances(),
        )

    def _repr_count_row(self) -> tuple[str, str]:
        """Return the payload-count row shown in :meth:`_repr_html_`.

        Default ``("Events", str(len(self)))``.  Whole-directory /
        single-file alignment loaders whose payload is claims override
        this to ``("Claims", …)`` rather than appending a second row, so
        the count row never doubles up.
        """
        return ("Events", str(len(self)))

    def _repr_rows(self) -> list[tuple[str, str]]:
        """Return the ``(label, value)`` rows for :meth:`_repr_html_`.

        Default rows: Sources (count + first up-to-5 names) and the
        :meth:`_repr_count_row` payload-count row.  Subclasses extend by
        calling ``super()._repr_rows()`` and adding their own.
        """
        n_sources = len(self._sources)
        names = ", ".join(code(s.name) for s in self._sources[:5])
        if n_sources > 5:
            names += f", … (+{n_sources - 5} more)"
        sources_value = f"{n_sources} file(s)" + (f": {names}" if names else "")
        return [
            ("Sources", sources_value),
            self._repr_count_row(),
        ]

    def _repr_affordances(self) -> list[str]:
        """Return the invocation snippets for the ``Try`` row."""
        snippets = ["create_timeline()", "create_timelines()"]
        if self.create_group.__func__ is not AlignmentLoader.create_group:
            snippets.append("create_group()")
        if self.create_bundle.__func__ is not AlignmentLoader.create_bundle:
            snippets.append("create_bundle()")
        if hasattr(type(self), "diagram"):
            snippets.append("diagram()")
        return snippets

    # endregion

    # region Magic Methods

    def __len__(self) -> int:
        """Return total event count across all stores."""
        if self._store is None:
            return 0
        return self._store.event_count

    def __repr__(self) -> str:
        """Return string representation."""
        n_events = len(self) if self._store else 0
        return (
            f"{self.__class__.__name__}("
            f"sources={len(self._sources)}, "
            f"events={n_events})"
        )

    # endregion


# endregion
