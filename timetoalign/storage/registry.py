"""The event registry: one row per event, its position on its root's axis.

Every timeline belongs to one *family*: a root timeline and every
descendant of it, all of one unit and one number type. A family's events
are held by its :class:`EventRegistry`, which a timeline of the family
reads through and every change of its rows goes through. The registry
holds two things for every event:

- a **registry row** in the columns of :data:`REGISTRY_SCHEMA`: the event's
  ID, its kind (its event type), the timeline holding it, its ``start`` and
  ``duration`` on the axis of the family's **root**, the axis's unit and
  number type, its name, and the :class:`Extraction` that made it, with
  the parser's ``locator`` of the event in its source and, after an
  update, the extraction that no longer found it (``vanished_in``);
- its **content**, the event's own columns (a note's pitch, a Box's shape),
  in the timeline's own coordinates: :attr:`Timeline.events
  <timetoalign.timelines.Timeline.events>` is an :class:`EventData` of
  them, positions local to the timeline.

Both coordinates are held: a root position is a local one shifted by the
offsets of the timeline's ancestors, in the family's number type
(:func:`~timetoalign.core.time.combine_number_columns`, column-wise, the
innermost offset first), and a local one read from a package is a root
position less those offsets. On an ``int`` or ``fraction`` axis the two
are exact and agree in both directions. On a ``float`` axis IEEE binary64
does not always give back what it shifted (``0.1 + 0.2 - 0.2``): built in
code, both are held as computed; read from a package, the local position
is what binary64 gives, and a source's own position stays in its own
columns.

A timeline or object built in code holds its own private registry; a
child joining a parent brings its family's rows into the parent's family,
shifted once onto the new root. A family whose rows were read from a
package (``add_rows(..., registered=True)``) never joins a parent: its
positions are its root's, and re-rooting it would give every row new
positions. A registry seen from a timeline, a group or a bundle
(:attr:`Timeline.registry <timetoalign.timelines.Timeline.registry>`) is a
view: a filter over one or more families, read when asked. A view shows
events only: the rows recording where a child lies (:data:`SEGMENT_KIND`)
are the family's own bookkeeping.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import quote

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from timetoalign.core import NumberType, TimeUnit, resolve_coordinate_spec
from timetoalign.core.fields import _filled, _if_else, _null_column, _struct_array
from timetoalign.core.ids import ID_LENGTH, ID_MINTER, IdMinter, quote_id, seed_from
from timetoalign.core.time import (
    RATIONAL_STRUCT_TYPE,
    combine_number_columns,
    compare_number_column,
)

from .events import (
    TIMELINE_ID,
    EventData,
    _column_refusal,
    _extreme_coordinate,
    joined_tables,
)
from .schema import make_base_schema

if TYPE_CHECKING:
    from os import PathLike

__all__ = [
    "EXTRACTION_SCHEMA",
    "NOT_CONTENT_COLUMNS",
    "REGISTRY_SCHEMA",
    "SEGMENT_KIND",
    "EventRegistry",
    "Extraction",
    "RegistryFamily",
    "RegistryHost",
    "WrittenEvents",
    "extraction_table",
    "member_uri",
]


SEGMENT_KIND = "Segment"
"""The kind of the row recording a child's place on its parent.

A child timeline's ``Segment`` row (its ID the child's) is held in memory
among its parent's rows, for the timelines' own use: it names no
extraction, is never written to a package and no view of the registry
(:attr:`EventRegistry.table`) shows it.
"""

_UNTYPED_KIND = "Event"
"""The kind of an event stating no event type: the ``Event`` base's."""

_BOX_KIND = "Box"
"""The kind of a Box, an area of the image a timeline runs along."""

REGISTRY_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("location", pa.string()),
        pa.field("timeline_id", pa.string()),
        pa.field("unit", pa.string()),
        pa.field("number_type", pa.string()),
        pa.field("start", RATIONAL_STRUCT_TYPE),
        pa.field("duration", RATIONAL_STRUCT_TYPE),
        pa.field("name", pa.string()),
        pa.field("extraction_id", pa.string()),
        pa.field("locator", pa.string()),
        pa.field("vanished_in", pa.string()),
    ]
)
"""The registry's columns in memory, in the order of TimeToShare's Registry table.

``start`` and ``duration`` are storage cells (both sides of each number,
:data:`~timetoalign.core.time.RATIONAL_STRUCT_TYPE`), on the root's axis;
the row's ``number_type`` says which side is authoritative.
"""

_REGISTRY_COLUMNS = tuple(REGISTRY_SCHEMA.names)

#: The columns of an event row that the registry holds and no event table
#: does: given in an EventData, they are moved into the registry row.
_RESERVED_COLUMNS = ("locator", "extraction_id")

NOT_CONTENT_COLUMNS = frozenset(
    {"start", "end", "duration", "instant", "temporal_type", "event_type", "name"}
)
"""The columns of an event that are its registry row's, never its content.

Its position, its type and its name: an event's content
(:meth:`EventRegistry.content`, a pre-parsed table) is every other column
but ``id``, which keys it.
"""

EXTRACTION_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("source", pa.string()),
        pa.field(
            "source_integrity",
            pa.struct(
                [pa.field("algorithm", pa.string()), pa.field("digest", pa.string())]
            ),
        ),
        pa.field("parser", pa.string(), nullable=False),
        pa.field("parser_version", pa.string(), nullable=False),
        pa.field("parameters", pa.string()),
        pa.field("minter", pa.string(), nullable=False),
        pa.field("id_length", pa.int64(), nullable=False),
        pa.field("seed", pa.string()),
        pa.field("date", pa.string(), nullable=False),
    ]
)
"""The columns of the Extraction table, one row per :class:`Extraction`."""

_EXTRACTION_PREFIX = "extraction"


def _now() -> str:
    """The current date-time in UTC, ISO 8601 to the second (``...Z``)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extraction_id(*parts: object) -> str:
    """The ID of an extraction of the content *parts*: the same in every run.

    Minted by a minter of its own seeded from the content, never by a
    call's minter, so an extraction recorded on the way takes no ID from
    the events it records.
    """
    return IdMinter(seed_from(_EXTRACTION_PREFIX, *parts)).mint(_EXTRACTION_PREFIX)


def member_uri(member: zipfile.Path) -> str:
    """The ``jar:`` URI of a member of a ZIP archive opened from its path.

    ``jar:<archive file URI>!/<member>``, the archive's absolute path and
    the member's POSIX path (without a closing ``/``) both percent-encoded:
    for instance ``jar:file:///data/melod%20y.zip!/works/1.ttl``.

    Args:
        member: A member of a ZIP archive.

    Returns:
        The member's URI.

    Raises:
        ValueError: If the archive was opened from a file object: it has no
            path, so the member has no URI.
    """
    filename = member.root.filename
    if filename is None:
        raise ValueError(
            f"The ZIP archive of member '{member.at}' was opened from a file object: "
            "it has no path, so the member has no URI; open the archive from its path."
        )
    return f"jar:{Path(filename).absolute().as_uri()}!/{quote(member.at.rstrip('/'))}"


def _canonical_json(value: Mapping[str, Any]) -> str:
    """*value* as canonical JSON text: sorted keys, no spaces (RFC 8785's form)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# region Extraction


@dataclass(frozen=True, kw_only=True)
class Extraction:
    """One run of a parser over one source file: what made a set of events.

    A row of the Extraction table (CRMdig's D10 Software Execution): the
    source file and its digest, the parser and its version, the parameters
    it ran with, and the ID minter's algorithm, length and seed, so that a
    run again over the same file mints the same IDs. Events made without a
    source file (in code, by a script) name the tool's extraction
    (:meth:`of_tool`). Extractions sharing an ID hash alike.

    Attributes:
        id: The extraction's ID (``extraction_…``).
        source: The source file, a path or URI reference; None for events
            made without one.
        source_integrity: ``{"algorithm": "SHA-256", "digest": <hex>}`` of
            the source file; None exactly where *source* is.
        parser: The parser's name.
        parser_version: Its version.
        parameters: The parameters it ran with, or None.
        minter: The algorithm of the ID minter
            (:data:`~timetoalign.core.ids.ID_MINTER`).
        id_length: The length of the minted IDs' suffixes.
        seed: The seed given to the minter, if one was, as text.
        date: When it ran, ISO 8601 in UTC (``2026-11-27T21:30:00Z``).

    Raises:
        ValueError: If *source* and *source_integrity* are not both given
            or both None.
    """

    id: str
    source: str | None = None
    source_integrity: Mapping[str, str] | None = None
    parser: str
    parser_version: str
    parameters: Mapping[str, Any] | None = None
    minter: str = ID_MINTER
    id_length: int = ID_LENGTH
    seed: str | None = None
    date: str = field(default_factory=_now)

    @classmethod
    def of_tool(
        cls,
        parser: str = "timetoalign",
        parser_version: str | None = None,
        *,
        scope: str | None = None,
        id: str | None = None,
        date: str | None = None,
    ) -> Extraction:
        """The extraction of events made without a source file.

        A registry makes one of its own (for its root timeline) when events
        made in code join it.

        Args:
            parser: The tool's name; ``"timetoalign"`` by default.
            parser_version: Its version; this library's by default.
            scope: What the extraction is one of (a root timeline's ID); a
                minted ID is seeded from it, the parser and its version.
            id: The extraction's ID; minted when None.
            date: When it ran; now when None.
        """
        if parser_version is None:
            from timetoalign import __version__

            parser_version = __version__
        if id is None:
            id = _extraction_id(parser, parser_version, scope)
        return cls(
            id=id,
            parser=parser,
            parser_version=parser_version,
            date=_now() if date is None else date,
        )

    @classmethod
    def of_source(
        cls,
        path: str | PathLike[str] | zipfile.Path,
        *,
        source: str | None = None,
        parser: str,
        parser_version: str,
        parameters: Mapping[str, Any] | None = None,
        minter: str = ID_MINTER,
        id_length: int = ID_LENGTH,
        seed: int | str | None = None,
        id: str | None = None,
        date: str | None = None,
    ) -> Extraction:
        """The extraction of a parser's run over the file *path*.

        The file's SHA-256 digest is read from its bytes. The ID, when
        minted, is seeded from the digest, the parser and its version, the
        parameters, the minter's algorithm and length and the seed, never
        from where the file lies: a run over the same bytes with the same
        parser and parameters has the same ID on every machine, in every
        run and every process.

        Args:
            path: The source file, read here: a path, or a member of a ZIP
                archive read in place (a ``zipfile.Path`` of an archive
                opened from its path).
            source: The reference recorded for it (a path relative to a
                package's root, a URI); when None, *path* as given (POSIX
                separators), a member's ``jar:`` URI
                (``jar:file:///data/set.zip!/notes/1.csv``, the archive's
                absolute path and the member's path percent-encoded).
            parser: The parser's name.
            parser_version: Its version.
            parameters: The parameters it ran with.
            minter: The algorithm of the ID minter.
            id_length: The length of the IDs it minted.
            seed: The seed given to the minter, if one was.
            id: The extraction's ID; minted when None.
            date: When it ran; now when None.

        Raises:
            OSError: If the file cannot be read.
            ValueError: If *path* is a member of an archive opened from a
                file object and *source* is None: it has no URI.
        """
        if isinstance(path, zipfile.Path):
            source = member_uri(path) if source is None else source
            opened = path.open("rb")
        else:
            source = Path(path).as_posix() if source is None else source
            opened = open(path, "rb")
        digest = hashlib.sha256()
        with opened as source_file:
            for block in iter(lambda: source_file.read(1 << 20), b""):
                digest.update(block)
        seed_text = None if seed is None else str(seed)
        integrity = {"algorithm": "SHA-256", "digest": digest.hexdigest()}
        if id is None:
            id = _extraction_id(
                integrity["digest"],
                parser,
                parser_version,
                None if parameters is None else _canonical_json(parameters),
                minter,
                id_length,
                seed_text,
            )
        return cls(
            id=id,
            source=source,
            source_integrity=integrity,
            parser=parser,
            parser_version=parser_version,
            parameters=None if parameters is None else dict(parameters),
            minter=minter,
            id_length=id_length,
            seed=seed_text,
            date=_now() if date is None else date,
        )

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Extraction:
        """The extraction of a row of the Extraction table (:meth:`to_row`)."""
        parameters = row.get("parameters")
        integrity = row.get("source_integrity")
        return cls(
            id=row["id"],
            source=row.get("source"),
            source_integrity=None if integrity is None else dict(integrity),
            parser=row["parser"],
            parser_version=row["parser_version"],
            parameters=None if parameters is None else json.loads(parameters),
            minter=row["minter"],
            id_length=int(row["id_length"]),
            seed=row.get("seed"),
            date=row["date"],
        )

    def __post_init__(self) -> None:
        if (self.source is None) != (self.source_integrity is None):
            raise ValueError(
                f"extraction {quote_id(self.id)} names a source file and its "
                "digest together or neither"
            )

    def __hash__(self) -> int:
        return hash(self.id)

    def to_row(self) -> dict[str, Any]:
        """The extraction as a row of the Extraction table.

        ``parameters`` is canonical JSON text (sorted keys, no spaces).
        """
        return {
            "id": self.id,
            "source": self.source,
            "source_integrity": (
                None if self.source_integrity is None else dict(self.source_integrity)
            ),
            "parser": self.parser,
            "parser_version": self.parser_version,
            "parameters": (
                None if self.parameters is None else _canonical_json(self.parameters)
            ),
            "minter": self.minter,
            "id_length": self.id_length,
            "seed": self.seed,
            "date": self.date,
        }


def _without_date(extraction: Extraction) -> dict[str, Any]:
    """*extraction*'s row but its date: what two runs of one extraction share."""
    row = extraction.to_row()
    del row["date"]
    return row


def extraction_table(extractions: Iterable[Extraction]) -> pa.Table:
    """The Extraction table of *extractions*, one row each, in order."""
    return pa.Table.from_pylist(
        [extraction.to_row() for extraction in extractions], schema=EXTRACTION_SCHEMA
    )


# endregion


# region Columns


def _ids_in(table: pa.Table) -> pa.ChunkedArray:
    """The event IDs *table* holds, as strings, nulls left out."""
    if "id" not in table.column_names:
        return pa.chunked_array([], pa.string())
    column = table.column("id")
    if column.null_count == len(column):
        return pa.chunked_array([], pa.string())
    if column.type != pa.string():
        column = pc.cast(column, pa.string())
    return pc.drop_null(column)


def _strings(table: pa.Table, name: str) -> pa.Array:
    """The column *name* of *table* as one string array, nulls where it is absent."""
    if name not in table.column_names:
        return pa.nulls(table.num_rows, pa.string())
    column = table.column(name).combine_chunks()
    if isinstance(column, pa.DictionaryArray):
        column = column.dictionary_decode()
    return column if column.type == pa.string() else pc.cast(column, pa.string())


def number_cells(column: pa.Array | pa.ChunkedArray) -> pa.StructArray:
    """The storage cells of a number *column*, one contiguous struct array.

    A dictionary-encoded column is decoded; the cells are of
    :data:`~timetoalign.core.time.RATIONAL_STRUCT_TYPE`.
    """
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if isinstance(column, pa.DictionaryArray):
        column = column.dictionary_decode()
    cells = _struct_array(column)
    if cells.type != RATIONAL_STRUCT_TYPE:
        cells = _struct_array(cells.cast(RATIONAL_STRUCT_TYPE))
    return cells


def _cells(table: pa.Table, name: str) -> pa.StructArray:
    """The number cells of the column *name* of *table*, nulls where it is absent."""
    if name not in table.column_names:
        return pa.nulls(table.num_rows, RATIONAL_STRUCT_TYPE)
    return number_cells(table.column(name))


def _shifted(
    cells: pa.StructArray,
    offsets: Sequence[int | float | Fraction],
    number_type: NumberType,
    *,
    subtract: bool = False,
) -> pa.StructArray:
    """*cells* moved by *offsets*, the innermost first, column-wise.

    Adding, each offset is added in turn (a child's own first, the
    outermost last), as a family's rows are shifted onto its root step by
    step; subtracting runs the other way. Null cells stay null.
    """
    steps = reversed(offsets) if subtract else offsets
    for offset in steps:
        cells = combine_number_columns(
            cells, offset, "subtract" if subtract else "add", number_type
        )
    return cells


def _compacted(table: pa.Table) -> pa.Table:
    """*table* with its trailing small chunks joined: O(log n) chunks in all.

    Rows added one call at a time each add a chunk; each chunk is joined
    with the one before it as long as it is at least half as long, so a
    timeline's rows stay in few chunks and each row is copied O(log n)
    times over all additions.
    """
    batches = table.to_batches()
    if len(batches) < 2 or batches[-1].num_rows * 2 < batches[-2].num_rows:
        return table
    while len(batches) > 1 and batches[-1].num_rows * 2 >= batches[-2].num_rows:
        joined = pa.Table.from_batches(batches[-2:], schema=table.schema)
        batches[-2:] = joined.combine_chunks().to_batches()
    return pa.Table.from_batches(batches, schema=table.schema)


def _first_listed(candidates: pa.ChunkedArray, ids: pa.Array) -> str:
    """The first of *ids*, in their order, that *candidates* lists."""
    at = pc.index(pc.is_in(ids, value_set=candidates), True).as_py()
    return str(ids[at].as_py())


def _collision(event_id: str, holder: str, joining: str) -> str:
    """The refusal of the event ID *event_id*, held by *holder*, joining *joining*."""
    held = (
        f"Event id {quote_id(event_id)} is already held by an event of timeline "
        f"{quote_id(holder)}"
    )
    if holder == joining:
        return held
    return (
        f"{held}, so it cannot join timeline {quote_id(joining)}: event ids are "
        "unique among a root timeline and its descendants"
    )


def _repeated(ids: pa.Array, timeline_id: str) -> str | None:
    """The refusal of an ID given twice in *ids*, joining *timeline_id*, or None."""
    counts = pc.value_counts(ids)
    repeated = counts.filter(pc.greater(counts.field("counts"), pa.scalar(1)))
    if not len(repeated):
        return None
    return (
        f"Event id {quote_id(str(repeated.field('values')[0].as_py()))} is "
        f"given to {repeated.field('counts')[0].as_py()} events joining "
        f"timeline {quote_id(timeline_id)}; event ids are unique among a root "
        "timeline and its descendants"
    )


def _flat(array: pa.Array | pa.ChunkedArray) -> pa.Array:
    """*array* as one array, its chunks joined."""
    return array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array


def _flags(mask: object) -> np.ndarray:
    """The boolean *mask* (an array, chunked or not) as NumPy flags, null False."""
    if isinstance(mask, pa.ChunkedArray):
        mask = mask.combine_chunks()
    if not isinstance(mask, pa.Array):
        raise TypeError(f"a mask is an array of booleans, not {type(mask).__name__}")
    return np.asarray(mask.fill_null(False).to_numpy(zero_copy_only=False), dtype=bool)


def _repeated_string(value: str, n: int) -> pa.Array:
    """The string *value* on *n* rows, one buffer for all (never a Python list)."""
    return pa.repeat(pa.scalar(value, pa.string()), n)


def _runs(column: pa.Array | pa.ChunkedArray) -> list[tuple[Any, np.ndarray]]:
    """The rows of *column* per value: each value with its row indices.

    The values in the order they first appear; one stable sort of the
    column, so the indices of each value ascend and its rows keep their
    order. Never a scan per value.
    """
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if not len(column):
        return []
    order = pc.sort_indices(column).to_numpy()
    counted = pc.value_counts(column.take(order))
    counts = counted.field("counts").to_numpy()
    starts = np.cumsum(counts) - counts
    runs = [
        (value, order[start:][:count])
        for value, count, start in zip(
            counted.field("values").to_pylist(), counts, starts, strict=True
        )
    ]
    return sorted(runs, key=lambda run: int(run[1][0]))


def _inverse(positions: Sequence[np.ndarray]) -> pa.Array:
    """The indices that put rows taken at *positions*, in turn, back in order."""
    return pc.sort_indices(pa.array(np.concatenate(positions)))


def _own(table: pa.Table) -> pa.Table:
    """*table* without the columns the registry rows hold for it."""
    return table.drop_columns(
        [name for name in _RESERVED_COLUMNS if name in table.column_names]
    )


def _refuse_repeated_locators(rows: pa.Table, held: pa.Table | None = None) -> None:
    """Refuse a locator naming two events of one extraction, among *rows* and *held*.

    A locator names one event of its source: the registry rows *rows*
    joining may not repeat an ``(extraction_id, locator)`` pair, among
    themselves or with the rows *held* already. Rows without a locator
    are not compared.

    Raises:
        ValueError: Naming the locator, the number of events and the
            extraction.
    """
    named = pc.and_(
        rows.column("locator").is_valid(), rows.column("extraction_id").is_valid()
    )
    if not pc.any(named).as_py():
        return
    pairs = rows.select(["extraction_id", "locator"]).filter(named)
    if held is not None and held.num_rows:
        near = pc.and_(
            pc.is_in(
                held.column("extraction_id"), value_set=pairs.column("extraction_id")
            ),
            pc.is_in(held.column("locator"), value_set=pairs.column("locator")),
        )
        pairs = pa.concat_tables(
            [
                pairs,
                held.select(["extraction_id", "locator"]).filter(
                    _filled(near, pa.scalar(False))
                ),
            ]
        )
    counted = pairs.group_by(["extraction_id", "locator"], use_threads=False).aggregate(
        [("locator", "count")]
    )
    repeated = counted.filter(pc.greater(counted.column("locator_count"), pa.scalar(1)))
    if repeated.num_rows:
        first = repeated.slice(0, 1).to_pylist()[0]
        raise ValueError(
            f"locator {quote_id(str(first['locator']))} is given to "
            f"{first['locator_count']} events of extraction "
            f"{quote_id(str(first['extraction_id']))}; a locator names one event "
            "of its source"
        )


# endregion


class RegistryHost(Protocol):
    """What a family's registry asks of a timeline whose rows it takes in.

    A timeline holding its events in a registry
    (:class:`~timetoalign.timelines.Timeline`) is its host: rows read from a
    package (:meth:`EventRegistry.add_rows`) are held within its length and
    its Boxes checked against their shapes, as the rows it adds itself are.
    """

    def _ensure_capacity(
        self, required_end: Any, allow_expansion: bool = False
    ) -> None:
        """Hold the timeline long enough to reach *required_end*, or refuse."""
        ...

    def _refuse_boxes_off_their_shapes(
        self, added: EventData, given_ids: Sequence[str | None]
    ) -> None:
        """Refuse a Box of *added* that contradicts its shape."""
        ...


class _Family:
    """The events of one family of timelines (a root and its descendants).

    Per timeline, in the order the timelines joined (the root first): its
    events as an :class:`EventData` in its own coordinates (``parts``),
    and row for row its registry rows on the root's axis (``rows``); rows
    read from a package wait as registry rows past the part's end until
    the family's events are first read (:meth:`_materialise`), and are
    then read in their order. The offsets leading from each timeline to
    the root (``paths``, innermost first) shift the one into the other.

    Every change checks first and changes after: a refusal leaves the
    family as it was.
    """

    def __init__(
        self, timeline_id: str, data: EventData, host: RegistryHost | None = None
    ) -> None:
        self.root_id = timeline_id
        self.unit: TimeUnit = data.unit
        self.number_type: NumberType = data.number_type
        empty = data._derived(data.table.slice(0, 0))
        self.parts: dict[str, EventData] = {}
        self.rows: dict[str, pa.Table] = {}
        self.paths: dict[str, tuple[int | float | Fraction, ...]] = {timeline_id: ()}
        self.extractions: dict[str, Extraction] = {}
        self.tool: Extraction | None = None
        self.contents: dict[str, list[pa.Table | Callable[[], pa.Table]]] = {}
        self.registered = False
        # Per timeline: the events that copy another event of the family (a
        # child cut from a region), by ID, to the ID of the event copied.
        self.copies: dict[str, dict[str, str]] = {}
        # The timelines holding their events here, by ID, where one was given.
        self.hosts: dict[str, RegistryHost] = (
            {} if host is None else {timeline_id: host}
        )
        # The family this one joined, once it joined a parent's.
        self.merged_into: _Family | None = None
        self.version = 0
        self._table: pa.Table | None = None
        self._events_table: pa.Table | None = None
        self._set_part(timeline_id, empty)
        self.rows[timeline_id] = REGISTRY_SCHEMA.empty_table()
        self.replace(timeline_id, data)

    # region Reading

    def current(self) -> _Family:
        """This family, or the family it joined (and so on)."""
        family = self
        while family.merged_into is not None:
            family = family.merged_into
        return family

    def table(self) -> pa.Table:
        """Every registry row of the family, the timelines in joining order."""
        if self._table is None:
            self._table = pa.concat_tables(list(self.rows.values()))
        return self._table

    def events_table(self) -> pa.Table:
        """The registry rows of the family's events: :meth:`table` but ``Segment`` rows."""
        if self._events_table is None:
            table = self.table()
            self._events_table = table.filter(
                pc.not_equal(table.column("kind"), pa.scalar(SEGMENT_KIND))
            )
        return self._events_table

    def others(self, timeline_id: str) -> pa.Table:
        """The registry rows of every timeline of the family but *timeline_id*."""
        return pa.concat_tables(
            [rows for held, rows in self.rows.items() if held != timeline_id]
            or [REGISTRY_SCHEMA.empty_table()]
        )

    def events_of(self, timeline_id: str) -> EventData:
        """The events of *timeline_id*, in its own coordinates."""
        held = self.parts.get(timeline_id)
        if held is None:
            raise KeyError(
                f"No timeline {quote_id(timeline_id)} in this family. "
                f"Available timelines: {list(self.parts)}"
            )
        if self.rows[timeline_id].num_rows > len(held):
            self._materialise()
            held = self.parts[timeline_id]
        return held

    def holder_of(self, event_id: str, skip: str | None = None) -> str | None:
        """The timeline holding the event *event_id* (*skip* left out), or None."""
        wanted = pa.array([event_id], pa.string())
        for timeline_id, rows in self.rows.items():
            if timeline_id == skip or not rows.num_rows:
                continue
            if pc.any(pc.is_in(rows.column("id"), value_set=wanted)).as_py():
                return timeline_id
        return None

    def origin_of(self, event_id: str) -> str | None:
        for copies in self.copies.values():
            origin = copies.get(event_id)
            if origin is not None:
                return origin
        return None

    def held_ids(self, skip: str | None = None) -> pa.ChunkedArray | None:
        """Every event ID the family holds, but those of the timeline *skip*."""
        chunks = [
            chunk
            for timeline_id, rows in self.rows.items()
            if timeline_id != skip
            for chunk in rows.column("id").chunks
            if len(chunk)
        ]
        return pa.chunked_array(chunks, pa.string()) if chunks else None

    def bound_on_root(self, bound: Any) -> int | float | Fraction | None:
        """The query bound *bound* as a position on the root's axis, or None.

        A bare number lies on the family's axis; a :class:`Coordinate` in
        the family's unit too, in another unit it names a position of no
        row of the family (None); an :class:`IdCoordinate` naming a timeline
        of the family is moved onto the root by that timeline's offsets, the
        innermost first (exact on an exact axis, one double addition per
        level where the axis or the bound is a float), and one naming a
        timeline the family does not hold names a position of none (None).
        """
        resolved = resolve_coordinate_spec(bound)
        if resolved.unit is not None and resolved.unit != self.unit:
            return None
        value = resolved.value
        if resolved.timeline_id is None:
            return value
        path = self.paths.get(resolved.timeline_id)
        if path is None:
            return None
        for offset in path:
            value = value + offset
        return value

    # endregion

    # region Checking

    def refuse_held_ids(
        self,
        timeline_id: str,
        ids: pa.Table | pa.Array | pa.ChunkedArray,
        *,
        own: bool = False,
    ) -> None:
        """Refuse event IDs joining *timeline_id* that the family holds, or repeats."""
        if isinstance(ids, pa.Table):
            column = _ids_in(ids)
        else:
            column = pa.chunked_array([ids]) if isinstance(ids, pa.Array) else ids
            if column.type != pa.string():
                column = pc.cast(column, pa.string())
        joining = pc.drop_null(column).combine_chunks()
        if not len(joining):
            return
        repeated = _repeated(joining, timeline_id)
        if repeated is not None:
            raise ValueError(repeated)
        skip = timeline_id if own else None
        held = self.held_ids(skip)
        if held is None:
            return
        hits = held.filter(pc.is_in(held, value_set=joining))
        if len(hits):
            event_id = _first_listed(hits, joining)
            holder = self.holder_of(event_id, skip)
            assert holder is not None
            raise ValueError(_collision(event_id, holder, timeline_id))

    def refuse_joining(self, other: _Family) -> None:
        """Refuse *other* joining this family as a child's, changing nothing."""
        if other.registered:
            raise ValueError(
                f"timeline {quote_id(other.root_id)} holds events read from a "
                "package, positioned on its own axis: it cannot become a child "
                "(that would move every position); relate the two through a "
                "group or MatchClaims"
            )
        if (other.unit, other.number_type) != (self.unit, self.number_type):
            raise ValueError(
                f"timeline {quote_id(other.root_id)} lies on {other.unit} in "
                f"{other.number_type.name}, but the family it joins on "
                f"{self.unit} in {self.number_type.name}"
            )
        shared = [tid for tid in other.parts if tid in self.parts]
        if shared:
            raise ValueError(
                f"timeline {quote_id(shared[0])} is held in this family already"
            )
        held = self.held_ids()
        joining = other.held_ids()
        if held is not None and joining is not None:
            hits = held.filter(pc.is_in(held, value_set=joining))
            if len(hits):
                event_id = _first_listed(hits, joining.combine_chunks())
                holder, joiner = self.holder_of(event_id), other.holder_of(event_id)
                assert holder is not None and joiner is not None
                raise ValueError(_collision(event_id, holder, joiner))
        self.refuse_extractions(
            extraction
            for extraction in other.extractions.values()
            if not (extraction is other.tool and self.tool is not None)
        )
        _refuse_repeated_locators(other.table(), self.table())

    def on_axis(self, timeline_id: str, data: EventData) -> EventData:
        """*data* as a table of *timeline_id*: on the family's axis, standing alone.

        A table spanning timelines whose every row lies on the family's
        axis stands alone (its key column, ``timeline_id``, dropped); rows
        are never cast to make them fit.

        Raises:
            ValueError: If a row lies on another unit or number type.
        """
        axes = (
            {(data.unit, data.number_type)}
            if data._axes is None
            else set(data._axes.values())
        )
        if axes != {(self.unit, self.number_type)}:
            spelled = " and ".join(
                sorted(f"{unit} in {number_type.name}" for unit, number_type in axes)
            )
            raise ValueError(
                f"the rows of timeline {quote_id(timeline_id)} lie on its axis, "
                f"{self.unit} in {self.number_type.name}; these lie on {spelled}"
            )
        if data._axes is None:
            return data
        alone = data._derived(data.table.drop_columns([TIMELINE_ID]))
        alone._axes = None
        alone._timeline_id = None
        return alone

    def refuse_extractions(self, extractions: Iterable[Extraction]) -> None:
        """Refuse an extraction held under its ID with other content (but its date)."""
        for extraction in extractions:
            held = self.extractions.get(extraction.id)
            if held is not None and _without_date(held) != _without_date(extraction):
                raise ValueError(
                    f"extraction {quote_id(extraction.id)} is held already, of "
                    "other content"
                )

    # endregion

    # region Changing

    def _changed(self) -> None:
        self.version += 1
        self._table = None
        self._events_table = None

    def _set_part(self, timeline_id: str, part: EventData) -> None:
        """Make *part* the events of *timeline_id*: a table of that timeline's."""
        held = part._derived(part.table)
        held._timeline_id = timeline_id
        held._extractions = ()
        self.parts[timeline_id] = held

    def _record(self, extractions: Iterable[Extraction]) -> None:
        """Hold *extractions*, refused already where they conflict; one held stays."""
        for extraction in extractions:
            self.extractions.setdefault(extraction.id, extraction)

    def registry_rows(
        self,
        timeline_id: str,
        data: EventData,
        extraction: Extraction | None,
        kept: pa.Table | None = None,
    ) -> tuple[pa.Table, list[Extraction], Extraction | None]:
        """The registry rows of *data*'s rows, joining *timeline_id*; nothing changed.

        Positions shifted onto the root; the reserved columns ``locator``
        and ``extraction_id`` read where *data* holds them; a row an ID of
        *kept* (rows it replaces) names keeps that row's location,
        extraction, locator and vanishing, unless *data* states them; the
        other event rows name *extraction*, else the family's tool
        extraction. A ``Segment`` row names none. An event without an
        event type is of kind ``Event``.

        Returns:
            The rows; the extractions to hold with them (*extraction*, and
            those *data* carries that the rows name); the family's tool
            extraction where the rows name it and the family has none yet.

        Raises:
            ValueError: If a row names an extraction that is neither held,
                nor *extraction*, nor one *data* carries.
        """
        table = data.table
        n = table.num_rows
        path = self.paths[timeline_id]
        ids = _strings(table, "id")
        kinds = _filled(_strings(table, "event_type"), pa.scalar(_UNTYPED_KIND))
        segment = pc.equal(kinds, pa.scalar(SEGMENT_KIND))
        stated = _strings(table, "extraction_id")
        locator = _strings(table, "locator")
        location = pa.nulls(n, pa.string())
        vanished = pa.nulls(n, pa.string())
        named = {str(name) for name in pc.unique(pc.drop_null(stated)).to_pylist()}
        known = {*self.extractions, *(item.id for item in data.extractions)}
        if extraction is not None:
            known.add(extraction.id)
        unknown = sorted(named - known)
        if unknown:
            raise ValueError(
                f"the rows name the extraction {quote_id(unknown[0])}, which this "
                "family does not hold: give it with the rows "
                "(EventData.with_extractions)"
            )
        recorded = [] if extraction is None else [extraction]
        recorded.extend(item for item in data.extractions if item.id in named)
        if kept is not None and kept.num_rows:
            at = pc.index_in(ids, value_set=kept.column("id").combine_chunks())

            def carried(name: str, own: pa.Array) -> pa.Array:
                old = kept.column(name).combine_chunks().take(at)
                return pc.coalesce(own, old)

            stated = carried("extraction_id", stated)
            locator = carried("locator", locator)
            location = carried("location", location)
            vanished = carried("vanished_in", vanished)
        tool: Extraction | None = None
        unnamed = pc.and_(stated.is_null(), pc.invert(segment))
        if pc.any(unnamed).as_py():
            fallback = extraction
            if fallback is None:
                fallback = self.tool
                if fallback is None:
                    fallback = tool = Extraction.of_tool(scope=self.root_id)
                    recorded.append(tool)
            stated = _filled(stated, pa.scalar(fallback.id, pa.string()))
        stated = _if_else(segment, pa.nulls(n, pa.string()), stated)
        rows = pa.Table.from_arrays(
            [
                ids,
                kinds,
                location,
                _repeated_string(timeline_id, n),
                _repeated_string(self.unit.value, n),
                _repeated_string(self.number_type.name, n),
                _shifted(_cells(table, "start"), path, self.number_type),
                _cells(table, "duration"),
                _strings(table, "name"),
                stated,
                locator,
                vanished,
            ],
            schema=REGISTRY_SCHEMA,
        )
        return rows, recorded, tool

    def _commit_extractions(
        self, recorded: Iterable[Extraction], tool: Extraction | None
    ) -> None:
        self._record(recorded)
        if tool is not None:
            self.tool = tool

    def add(
        self,
        timeline_id: str,
        data: EventData,
        *,
        extraction: Extraction | None = None,
        checked: bool = False,
    ) -> None:
        """Add *data*'s rows, in *timeline_id*'s coordinates, after its rows."""
        if not len(data):
            return
        data = self.on_axis(timeline_id, data)
        part = self.events_of(timeline_id)
        if not checked:
            self.refuse_held_ids(timeline_id, data.table)
        rows, recorded, tool = self.registry_rows(timeline_id, data, extraction)
        self.refuse_extractions(recorded)
        _refuse_repeated_locators(rows, self.table())
        joined = joined_tables(
            [part.table, _own(data.table)], _refusal_for(timeline_id)
        )
        self._commit_extractions(recorded, tool)
        self._set_part(timeline_id, part._derived(_compacted(joined)))
        self.rows[timeline_id] = _compacted(
            pa.concat_tables([self.rows[timeline_id], rows])
        )
        self._changed()

    def replace(
        self,
        timeline_id: str,
        data: EventData,
        *,
        extraction: Extraction | None = None,
    ) -> None:
        """Make *data*'s rows those of *timeline_id*, its class the timeline's.

        A row keeping the ID of a row it replaces keeps that row's
        extraction, locator and location.
        """
        data = self.on_axis(timeline_id, data)
        # Rows waiting to be read are read first: *data* replaces them too.
        self.events_of(timeline_id)
        held = self.rows[timeline_id]
        given = _ids_in(data.table)
        unchanged = held.column("id").equals(given) if held.num_rows else not len(given)
        if not unchanged:
            self.refuse_held_ids(timeline_id, given, own=True)
        rows, recorded, tool = self.registry_rows(
            timeline_id, data, extraction, kept=held
        )
        self.refuse_extractions(recorded)
        _refuse_repeated_locators(rows, self.others(timeline_id))
        self._commit_extractions(recorded, tool)
        self._set_part(timeline_id, data._derived(_own(data.table)))
        self.rows[timeline_id] = rows
        copies = self.copies.get(timeline_id)
        if copies and not unchanged:
            kept = set(
                pc.filter(
                    given, pc.is_in(given, value_set=pa.array(list(copies)))
                ).to_pylist()
            )
            self.copies[timeline_id] = {
                copy: origin for copy, origin in copies.items() if copy in kept
            }
        self._changed()

    def record_copies(
        self, timeline_id: str, copies: Sequence[str], origins: Sequence[str]
    ) -> None:
        self.copies.setdefault(timeline_id, {}).update(
            zip(copies, origins, strict=True)
        )

    def attach(
        self, other: _Family, parent_id: str, offset: int | float | Fraction
    ) -> None:
        """Take *other*, a child's family joining under *parent_id* at *offset*.

        Its rows are shifted onto this family's root once (column-wise); its
        events stay as they are. *other* forwards to this family from now
        on. The caller refuses first (:meth:`refuse_joining`).
        """
        steps = (offset, *self.paths[parent_id])
        tool = self.tool
        for timeline_id in other.parts:
            # A row read lazily is placed with the rest.
            rows = other.rows[timeline_id]
            moved = rows.set_column(
                rows.schema.get_field_index("start"),
                REGISTRY_SCHEMA.field("start"),
                _shifted(_cells(rows, "start"), steps, self.number_type),
            )
            if other.tool is not None and tool is not None and other.tool != tool:
                index = moved.schema.get_field_index("extraction_id")
                column = moved.column(index)
                moved = moved.set_column(
                    index,
                    REGISTRY_SCHEMA.field("extraction_id"),
                    pc.if_else(
                        _filled(
                            pc.equal(column, pa.scalar(other.tool.id)), pa.scalar(False)
                        ),
                        pa.scalar(tool.id, pa.string()),
                        column,
                    ),
                )
            self.parts[timeline_id] = other.parts[timeline_id]
            self.rows[timeline_id] = moved
            self.paths[timeline_id] = (*other.paths[timeline_id], *steps)
        self._record(
            extraction
            for extraction in other.extractions.values()
            if not (extraction is other.tool and tool is not None)
        )
        if tool is None and other.tool is not None:
            self.tool = other.tool
        for kind, contents in other.contents.items():
            self.contents.setdefault(kind, []).extend(contents)
        for timeline_id, copies in other.copies.items():
            self.copies.setdefault(timeline_id, {}).update(copies)
        self.hosts.update(other.hosts)
        other.merged_into = self
        other.parts, other.rows, other.paths, other.hosts = {}, {}, {}, {}
        other._changed()
        self._changed()

    def refuse_rename(
        self, old_id: str, new_id: str, *, parent: str | None = None
    ) -> None:
        if old_id == new_id:
            return
        if old_id not in self.parts:
            raise KeyError(
                f"No timeline {quote_id(old_id)} in this family. "
                f"Available timelines: {list(self.parts)}"
            )
        if new_id in self.parts:
            raise ValueError(
                f"timeline {quote_id(new_id)} is held in this family already"
            )
        if parent is not None:
            self.refuse_held_ids(parent, pa.array([new_id], pa.string()))

    def rename(self, old_id: str, new_id: str) -> None:
        self.refuse_rename(old_id, new_id)
        if old_id == new_id:
            return

        def renamed(mapping: dict[str, Any]) -> dict[str, Any]:
            return {
                (new_id if key == old_id else key): value
                for key, value in mapping.items()
            }

        self.parts = renamed(self.parts)
        self._set_part(new_id, self.parts[new_id])
        self.paths = renamed(self.paths)
        self.copies = renamed(self.copies)
        self.hosts = renamed(self.hosts)
        rows = renamed(self.rows)
        moved = rows[new_id]
        index = moved.schema.get_field_index("timeline_id")
        rows[new_id] = moved.set_column(
            index,
            REGISTRY_SCHEMA.field("timeline_id"),
            _repeated_string(new_id, moved.num_rows),
        )
        self.rows = rows
        if self.root_id == old_id:
            self.root_id = new_id
        self._changed()

    # endregion

    # region Rows read from a package

    def state(self) -> tuple[Any, ...]:
        """What :meth:`restore` puts back: every holding of the family, copied."""
        return (
            dict(self.parts),
            dict(self.rows),
            dict(self.paths),
            dict(self.extractions),
            self.tool,
            {kind: list(entries) for kind, entries in self.contents.items()},
            self.registered,
            {timeline_id: dict(copies) for timeline_id, copies in self.copies.items()},
        )

    def restore(self, state: tuple[Any, ...]) -> None:
        """Put back the holdings :meth:`state` copied."""
        (
            self.parts,
            self.rows,
            self.paths,
            self.extractions,
            self.tool,
            self.contents,
            self.registered,
            self.copies,
        ) = state
        self._changed()

    def not_stated_already(self, rows: pa.Table) -> pa.Table:
        """*rows* (registry rows) but those the family holds already.

        A row whose ID the family holds on the same timeline, as the same
        kind at the same root position (``start`` and ``duration`` compared
        exactly, cell for cell), is the event a manifest's own event rows
        state: the held one is kept and the row passed over. A row whose ID
        the family holds otherwise is kept here, for :meth:`add_rows` to
        refuse.
        """
        held = self.table()
        if not held.num_rows or not rows.num_rows:
            return rows
        held_ids = held.column("id").combine_chunks()
        hits = _filled(
            pc.is_in(rows.column("id"), value_set=held_ids), pa.scalar(False)
        )
        if not pc.any(hits).as_py():
            return rows
        at = pc.index_in(rows.column("id"), value_set=held_ids)
        same = _flags(hits)
        for name in ("timeline_id", "kind"):
            same &= _flags(pc.equal(rows.column(name), held.column(name).take(at)))
        for name in ("start", "duration"):
            mine = rows.column(name).combine_chunks()
            theirs = held.column(name).combine_chunks().take(at)
            both_null = _flags(pc.and_(mine.is_null(), theirs.is_null()))
            equal = _flags(
                pc.and_(
                    pc.equal(
                        pc.struct_field(mine, "numerator"),
                        pc.struct_field(theirs, "numerator"),
                    ),
                    pc.equal(
                        pc.struct_field(mine, "denominator"),
                        pc.struct_field(theirs, "denominator"),
                    ),
                )
            )
            same &= both_null | equal
        return rows.filter(pa.array(~same))

    def refuse_rows(
        self, rows: pa.Table, extractions: Sequence[Extraction]
    ) -> pa.Array | None:
        """Refuse registry *rows* the family cannot hold; nothing changes.

        Returns:
            The timeline of each row, row for row, or None for no rows.

        Raises:
            KeyError: If a row names a timeline the family does not hold.
            ValueError: As :meth:`add_rows` refuses.
        """
        self.refuse_extractions(extractions)
        if not rows.num_rows:
            return None
        timelines = rows.column("timeline_id").combine_chunks()
        if timelines.null_count:
            raise ValueError(
                "a registry row of an event names no timeline; an event lies on "
                "one timeline"
            )
        unknown = pc.filter(
            timelines,
            pc.invert(pc.is_in(timelines, value_set=pa.array(list(self.parts)))),
        )
        if len(unknown):
            raise KeyError(
                f"A registry row names the timeline {quote_id(unknown[0].as_py())}, "
                f"which this family does not hold. Available timelines: "
                f"{list(self.parts)}"
            )
        off = pc.invert(
            _filled(
                pc.and_(
                    pc.equal(rows.column("unit"), pa.scalar(self.unit.value)),
                    pc.equal(
                        rows.column("number_type"), pa.scalar(self.number_type.name)
                    ),
                ),
                pa.scalar(False),
            )
        )
        if pc.any(off).as_py():
            row = rows.slice(pc.index(off, True).as_py(), 1).to_pylist()[0]
            raise ValueError(
                f"event {quote_id(row['id'])} lies on {row['unit']} in "
                f"{row['number_type']}, but the root of its timeline "
                f"{quote_id(row['timeline_id'])} on {self.unit.value} in "
                f"{self.number_type.name}"
            )
        known = {*self.extractions, *(extraction.id for extraction in extractions)}
        named = pc.unique(_strings(rows, "extraction_id").fill_null("")).to_pylist()
        missing = [name for name in named if name not in known]
        if missing:
            raise ValueError(
                f"a registry row names the extraction {quote_id(str(missing[0]))}, which "
                "is given with none of the extractions"
            )
        repeated = _repeated(rows.column("id").combine_chunks(), self.root_id)
        if repeated is not None:
            raise ValueError(repeated)
        held = self.held_ids()
        if held is not None:
            joining = rows.column("id").combine_chunks()
            hits = held.filter(pc.is_in(held, value_set=joining))
            if len(hits):
                event_id = _first_listed(hits, joining)
                holder = self.holder_of(event_id)
                assert holder is not None
                at = pc.index(joining, pa.scalar(event_id)).as_py()
                raise ValueError(
                    _collision(event_id, holder, str(timelines[at].as_py()))
                )
        _refuse_repeated_locators(rows, self.table())
        return timelines

    def fit_lengths(self, rows: pa.Table, timelines: pa.Array) -> None:
        """Hold each host timeline of *rows* long enough for its rows.

        The largest end of each timeline's rows (``start + duration``, the
        ``start`` of an instant), decided exactly on an exact axis, made
        local by the offsets of the timeline's ancestors, is held by the
        timeline as the end of rows it adds (:meth:`RegistryHost._ensure_capacity`).
        A timeline without a host is not held.

        Raises:
            ValueError: If a row ends past the length of a locked timeline;
                the timelines before it are grown already.
        """
        number_type = self.number_type
        for timeline_id, at in _runs(timelines):
            host = self.hosts.get(str(timeline_id))
            if host is None:
                continue
            part = rows.take(pa.array(at, pa.int64()))
            start = part.column("start").combine_chunks()
            duration = part.column("duration").combine_chunks()
            ends = combine_number_columns(start, duration, "add", number_type)
            ends = pc.if_else(duration.is_valid(), ends, start)
            values = pc.struct_field(ends, "value")
            if values.null_count == len(values):
                continue
            end = _extreme_coordinate(
                pa.chunked_array([ends]),
                pa.chunked_array([values]),
                pc.max(values),
                number_type,
                max,
            )
            for offset in self.paths[str(timeline_id)]:
                end = end - offset
            host._ensure_capacity(end)

    def add_rows(
        self,
        rows: pa.Table,
        contents: Mapping[str, pa.Table | Callable[[], pa.Table]],
        extractions: Iterable[Extraction],
        *,
        registered: bool,
    ) -> None:
        """Hold registry rows on the root's axis, their contents read lazily.

        Rows the family holds already are passed over
        (:meth:`not_stated_already`); the others are checked
        (:meth:`refuse_rows`), held within the lengths of their timelines
        (:meth:`fit_lengths`) and taken in; a timeline taking a Box checks
        its Boxes against their shapes, every row read with its content,
        and a refusal there puts the family back as it was.
        """
        rows = self.not_stated_already(_as_registry(rows))
        extractions = list(extractions)
        timelines = self.refuse_rows(rows, extractions)
        if timelines is None:
            self._record(extractions)
            self._changed()
            return
        self.fit_lengths(rows, timelines)
        before = self.state()
        # Nothing is refused past here but a Box, which puts the family back.
        self._record(extractions)
        for kind, content in contents.items():
            self.contents.setdefault(kind, []).append(content)
        for timeline_id, at in _runs(timelines):
            # The rows past the part's end are read when the family's events
            # are first read.
            self.rows[timeline_id] = pa.concat_tables(
                [self.rows[timeline_id], rows.take(at)]
            )
        if registered:
            self.registered = True
        self._changed()
        is_box = _filled(
            pc.equal(rows.column("kind"), pa.scalar(_BOX_KIND)), pa.scalar(False)
        )
        try:
            for timeline_id in pc.unique(pc.filter(timelines, is_box)).to_pylist():
                host = self.hosts.get(str(timeline_id))
                if host is None:
                    continue
                # A Box read passes the check every Box added passes: its
                # shape read with it, never trusted over its position.
                events = self.events_of(str(timeline_id))
                host._refuse_boxes_off_their_shapes(
                    events, events.table.column("id").to_pylist()
                )
        except BaseException:
            self.restore(before)
            raise

    def content_of(self, kind: str) -> pa.Table | None:
        """The registered content of *kind*, every table of it read and joined once."""
        entries = self.contents.get(kind)
        if not entries:
            return None
        tables = [entry() if callable(entry) else entry for entry in entries]
        joined = (
            tables[0] if len(tables) == 1 else joined_tables(tables, _column_refusal)
        )
        self.contents[kind] = [joined]
        return joined

    def _materialise(self) -> None:
        """Read every waiting registry row of the family into its timeline's events.

        Each kind's content is looked up once for the whole family (one
        hash of its IDs); each timeline's rows are built column-wise,
        positions made local, and keep the registry's order.
        """
        waiting = [
            (timeline_id, self.rows[timeline_id].slice(len(part)))
            for timeline_id, part in self.parts.items()
            if self.rows[timeline_id].num_rows > len(part)
        ]
        if not waiting:
            return
        number_type = self.number_type
        rows = pa.concat_tables([table for _, table in waiting])
        start = pa.concat_arrays(
            [
                _shifted(
                    _cells(table, "start"),
                    self.paths[timeline_id],
                    number_type,
                    subtract=True,
                )
                for timeline_id, table in waiting
            ]
        )
        duration = _cells(rows, "duration")
        kinds = rows.column("kind").combine_chunks()
        base = make_base_schema(self.unit, number_type=number_type)
        columns: dict[str, pa.Array] = {
            "id": rows.column("id").combine_chunks(),
            "name": _strings(rows, "name"),
            "temporal_type": pc.if_else(
                duration.is_valid(), pa.scalar("interval"), pa.scalar("instant")
            ),
            "event_type": kinds,
            "start": start,
            "end": combine_number_columns(start, duration, "add", number_type),
            "duration": duration,
        }
        events = pa.Table.from_arrays(
            list(columns.values()),
            schema=pa.schema([base.field(name) for name in columns]),
        )
        by_kind: list[tuple[pa.Table, np.ndarray]] = []
        for kind, at in _runs(kinds):
            table = events.take(at)
            content = self.content_of(str(kind))
            if content is not None and "id" in content.column_names:
                found = pc.index_in(
                    table.column("id"), value_set=content.column("id").combine_chunks()
                )
                for field_ in content.schema:
                    if field_.name == "id" or field_.name in columns:
                        continue
                    table = table.append_column(
                        field_.with_nullable(True),
                        content.column(field_.name).combine_chunks().take(found),
                    )
            by_kind.append((table, at))
        first = 0
        for timeline_id, table in waiting:
            last = first + table.num_rows
            pieces: list[pa.Table] = []
            where: list[np.ndarray] = []
            for kind_table, at in by_kind:
                low, high = np.searchsorted(at, [first, last])
                if high > low:
                    pieces.append(kind_table.slice(int(low), int(high - low)))
                    where.append(at[low:high] - first)
            part = self.parts[timeline_id]
            joined = joined_tables([part.table, *pieces], _refusal_for(timeline_id))
            order = pa.concat_arrays(
                [
                    pa.array(np.arange(len(part)), pa.int64()),
                    pc.add(
                        pc.cast(_inverse(where), pa.int64()),
                        pa.scalar(len(part), pa.int64()),
                    ),
                ]
            )
            self._set_part(timeline_id, part._derived(joined.take(order)))
            first = last

    def update(self, timeline_id: str, data: EventData, extraction: Extraction) -> None:
        """Update *timeline_id*'s rows from *data*, a parser's new run.

        Rows are matched by ``locator`` among the rows of extractions of the
        same source as *extraction*: a matched row keeps its ID and takes
        the new row's content and *extraction*; a new row matching none is
        added; an old row of that source no longer found is kept and names
        *extraction* in ``vanished_in``. Nothing is deleted.
        """
        if "locator" not in data.table.column_names:
            raise ValueError(
                "an update matches rows by their locator: the rows give none "
                "(a column 'locator')"
            )
        data = self.on_axis(timeline_id, data)
        part = self.events_of(timeline_id)
        held = self.rows[timeline_id]
        # The events and the registry rows of a timeline lie row for row.
        assert (
            part.table.column("id")
            .combine_chunks()
            .equals(held.column("id").combine_chunks())
        ), timeline_id
        new_locators = _strings(data.table, "locator")
        _refuse_repeated_locators(
            pa.table(
                {
                    "extraction_id": _repeated_string(extraction.id, len(new_locators)),
                    "locator": new_locators,
                }
            )
        )
        self.refuse_extractions([extraction])
        same_source = [
            held_id
            for held_id, held_extraction in self.extractions.items()
            if held_extraction.source == extraction.source
        ]
        of_source = _filled(
            pc.is_in(
                held.column("extraction_id").combine_chunks(),
                value_set=pa.array(same_source, pa.string()),
            ),
            pa.scalar(False),
        )
        old_locators = pc.if_else(
            of_source,
            held.column("locator").combine_chunks(),
            pa.scalar(None, pa.string()),
        )
        # For each new row the old row it updates, for each old row the new.
        match = pc.index_in(new_locators, value_set=old_locators, skip_nulls=True)
        found = pc.index_in(old_locators, value_set=new_locators, skip_nulls=True)
        matched_new = match.is_valid()
        # The new rows: a matched one takes its old row's ID.
        old_ids = held.column("id").combine_chunks()
        ids = pc.if_else(matched_new, old_ids.take(match), _strings(data.table, "id"))
        new = data.table.set_column(
            data.table.schema.get_field_index("id"),
            data.table.schema.field("id"),
            ids,
        )
        if "extraction_id" in new.column_names:
            new = new.drop_columns(["extraction_id"])
        n_old = part.table.num_rows
        # Old rows in their places (an updated one by its new row), then the
        # new rows matching none.
        order = pc.if_else(
            found.is_valid(),
            pc.add(pc.cast(found, pa.int64()), pa.scalar(n_old, pa.int64())),
            pa.array(np.arange(n_old), pa.int64()),
        )
        added = pc.add(
            pc.indices_nonzero(pc.invert(matched_new)), pa.scalar(n_old, pa.uint64())
        )
        order = pa.concat_arrays([order, pc.cast(added, pa.int64())])
        joined = joined_tables([part.table, _own(new)], _refusal_for(timeline_id)).take(
            order
        )
        stated, _, _ = self.registry_rows(timeline_id, data._derived(new), extraction)
        registry = pa.concat_tables([held, stated]).take(order)
        # Old rows of the source not found vanished in this run; a found one
        # is present again.
        vanished = pc.if_else(
            pc.and_(of_source, pc.invert(found.is_valid())),
            pa.scalar(extraction.id, pa.string()),
            pc.if_else(
                found.is_valid(),
                pa.scalar(None, pa.string()),
                held.column("vanished_in").combine_chunks(),
            ),
        )
        vanished = pa.concat_arrays(
            [vanished, pa.nulls(len(order) - n_old, pa.string())]
        )
        registry = registry.set_column(
            registry.schema.get_field_index("vanished_in"),
            REGISTRY_SCHEMA.field("vanished_in"),
            vanished,
        )
        self.refuse_held_ids(timeline_id, registry.column("id"), own=True)
        _refuse_repeated_locators(registry, self.others(timeline_id))
        self._record([extraction])
        self._set_part(timeline_id, part._derived(joined))
        self.rows[timeline_id] = registry
        self._changed()

    # endregion


def _refusal_for(timeline_id: str) -> Callable[[str, pa.DataType, pa.DataType], str]:
    """The refusal of a column of *timeline_id* holding two types."""

    def refusal(column: str, held: pa.DataType, joining: pa.DataType) -> str:
        return (
            f"the column {column!r} of timeline {quote_id(timeline_id)} would "
            f"hold values of type {held} and of type {joining}"
        )

    return refusal


def _as_registry(rows: pa.Table) -> pa.Table:
    """*rows* in the registry's columns and types, a column they lack null.

    Raises:
        ValueError: If *rows* holds a column the registry has none of.
    """
    extra = [name for name in rows.column_names if name not in _REGISTRY_COLUMNS]
    if extra:
        raise ValueError(
            f"registry rows hold the column {extra[0]!r}, which the registry has "
            f"none of. Available columns: {list(_REGISTRY_COLUMNS)}"
        )
    columns = []
    for field_ in REGISTRY_SCHEMA:
        if field_.name not in rows.column_names:
            columns.append(_null_column(rows.num_rows, field_.type))
            continue
        column = rows.column(field_.name).combine_chunks()
        if isinstance(column, pa.DictionaryArray):
            column = column.dictionary_decode()
        if column.type != field_.type:
            column = column.cast(field_.type)
        columns.append(column)
    table = pa.Table.from_arrays(columns, schema=REGISTRY_SCHEMA)
    for name in ("id", "kind"):
        if table.column(name).null_count:
            raise ValueError(f"a registry row's {name!r} is never null")
    return table


# region Views


@dataclass(frozen=True)
class RegistryFamily:
    """One family of timelines a registry reads: its root and its axis.

    Attributes:
        root_id: The ID of the family's root timeline.
        unit: The unit of the family's axis, every timeline's.
        number_type: Its number type, every timeline's.
        timelines: The IDs of its timelines, in the order they joined (the
            root first).
    """

    root_id: str
    unit: TimeUnit
    number_type: NumberType
    timelines: tuple[str, ...]


@dataclass(frozen=True)
class WrittenEvents:
    """What a writer stores of a registry's events (:meth:`EventRegistry.written`).

    Attributes:
        rows: The registry rows (:data:`REGISTRY_SCHEMA`), positions on the
            axis of each family's root as storage cells, the families in
            order, each family's rows in its order (its timelines in
            joining order); no row records a child's place, and every row
            names its extraction.
        contents: The content of each timeline's events of each kind, keyed
            ``(timeline_id, kind)``, the families and their timelines in
            order and the kinds of a timeline sorted: ``id`` first, then
            every column but :data:`NOT_CONTENT_COLUMNS` in which one of
            these events holds a value, row for row the timeline's events of
            that kind in their order.
        extractions: The extractions the rows name (``extraction_id`` and
            ``vanished_in``), sorted by ID.
        families: The families the rows are of, in order.
    """

    rows: pa.Table
    contents: Mapping[tuple[str, str], pa.Table]
    extractions: tuple[Extraction, ...]
    families: tuple[RegistryFamily, ...]

    def family_of(self, timeline_id: str) -> RegistryFamily:
        """The family holding the timeline *timeline_id*.

        Raises:
            KeyError: If no family holds it.
        """
        for family in self.families:
            if timeline_id in family.timelines:
                return family
        raise KeyError(
            f"No family holds the timeline {quote_id(timeline_id)}. Available "
            f"timelines: {[tid for family in self.families for tid in family.timelines]}"
        )


def _places(
    ids: pa.Array | pa.ChunkedArray, kinds: pa.Array | pa.ChunkedArray, family: _Family
) -> pa.Array:
    """Row for row, whether a row records a child's place in *family*: a
    ``Segment`` row whose ID is a timeline of the family. A ``Segment`` row
    of any other ID is an event like every other."""
    return _flat(
        _filled(
            pc.and_(
                pc.equal(kinds, pa.scalar(SEGMENT_KIND)),
                pc.is_in(ids, value_set=pa.array(list(family.paths), pa.string())),
            ),
            pa.scalar(False),
        )
    )


@dataclass(frozen=True)
class _Criteria:
    """One filter of a view: every condition given must hold."""

    within: frozenset[str] | None = None
    kinds: frozenset[str] | None = None
    extractions: frozenset[str] | None = None
    vanished: bool | None = None
    min_coord: Any = None
    max_coord: Any = None

    def mask(self, table: pa.Table, family: _Family) -> pa.Array:
        """Which rows of *table*, *family*'s, these criteria keep."""
        mask = pa.array(np.ones(table.num_rows, dtype=bool))
        for name, wanted in (
            ("timeline_id", self.within),
            ("kind", self.kinds),
            ("extraction_id", self.extractions),
        ):
            if wanted is not None:
                mask = pc.and_(
                    mask,
                    _filled(
                        pc.is_in(
                            table.column(name), value_set=pa.array(sorted(wanted))
                        ),
                        pa.scalar(False),
                    ),
                )
        if self.vanished is not None:
            valid = table.column("vanished_in").is_valid()
            mask = pc.and_(mask, valid if self.vanished else pc.invert(valid))
        for bound, comparison in (
            (self.min_coord, "greater_equal"),
            (self.max_coord, "less"),
        ):
            if bound is None:
                continue
            value = family.bound_on_root(bound)
            if value is None:
                # A position of another unit, or of a timeline the family
                # does not hold: no row of the family lies there.
                return pa.array(np.zeros(table.num_rows, dtype=bool))
            start = table.column("start").combine_chunks()
            mask = pc.and_(
                mask,
                _filled(
                    compare_number_column(start, value, comparison, family.number_type),
                    pa.scalar(False),
                ),
            )
        return mask


@dataclass(frozen=True)
class _Source:
    """One family a view reads, through its filters (all of them must hold)."""

    family: _Family
    criteria: tuple[_Criteria, ...] = ()


def _frozen(values: str | Iterable[str] | None) -> frozenset[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return frozenset({values})
    return frozenset(values)


class EventRegistry:
    """The registry rows of the events of one or more families, as a view.

    A view is a filter over the families it reads, read when asked: it
    follows the rows as they change. :attr:`table` holds one row per event
    in the columns of :data:`REGISTRY_SCHEMA`, positions on each family's
    root axis; :meth:`content` the content of the events of one kind. The
    rows recording where a child lies (:data:`SEGMENT_KIND`) are no events
    and no view shows them.

    Rows come in through the doors of package readers and loaders, on a
    view too: :meth:`add_rows` (rows read from a package), :meth:`update`
    (a parser's new run, :meth:`update_target` the timeline it updates);
    writers read through :meth:`written`. A timeline adds its own rows
    through :meth:`Timeline.add_events
    <timetoalign.timelines.Timeline.add_events>`.

    Args:
        timeline_id: The ID of the root timeline of a new family.
        data: The root's events, on its axis; their class is the class of
            the family's events. The extractions it carries
            (:attr:`EventData.extractions`) are held with them.
        host: The timeline holding its events here
            (:class:`RegistryHost`): rows read for it are held within its
            length and its Boxes checked. None for a registry standing
            alone.

    Raises:
        ValueError: If *data* holds one event ID twice, or names an
            extraction it does not carry.

    Examples:
        >>> from fractions import Fraction
        >>> from timetoalign.timelines import ContinuousLogicalTimeline
        >>> root = ContinuousLogicalTimeline(length=8, unit="quarters", uid="clt_root")
        >>> child = ContinuousLogicalTimeline(length=4, unit="quarters", uid="clt_a")
        >>> child.add_events([{"id": "a1", "event_type": "Beat", "start": Fraction(1, 2)}])
        ['a1']
        >>> root.add_child(child, offset=2)
        >>> root.registry.filter(kinds="Beat").table.column("start")[0].as_py()["value"]
        2.5
        >>> child.events.table.column("start")[0].as_py()["value"]
        0.5
    """

    @classmethod
    def _view(cls, sources: Iterable[_Source]) -> EventRegistry:
        view = cls.__new__(cls)
        view._sources = tuple(sources)
        view._cache = None
        view._sources_cache = None
        return view

    @classmethod
    def concat(cls, *registries: EventRegistry) -> EventRegistry:
        """One view of the rows of *registries*, each row once.

        A family read by several of them keeps the rows any of them keeps,
        in the family's order; families in the order they first appear.
        """
        return cls._view(
            source for registry in registries for source in registry._sources
        )

    def __init__(
        self, timeline_id: str, data: EventData, *, host: RegistryHost | None = None
    ) -> None:
        self._sources: tuple[_Source, ...] = (
            _Source(_Family(timeline_id, data, host)),
        )
        self._cache: tuple[tuple[int, ...], pa.Table] | None = None
        self._sources_cache: tuple[tuple[int, ...], pa.Table] | None = None

    def __len__(self) -> int:
        """The number of events of this view."""
        return self.table.num_rows

    def __repr__(self) -> str:
        families = self._families()
        return (
            f"EventRegistry(rows={len(self)}, families="
            f"{[family.root_id for family, _ in families]})"
        )

    # region Reading

    @property
    def table(self) -> pa.Table:
        """The rows of this view's events in the registry's columns, families in order."""
        families = self._families()
        key = tuple(
            value for family, _ in families for value in (id(family), family.version)
        )
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        tables: list[pa.Table] = []
        for family, filters in families:
            table = family.events_table()
            if any(not criteria for criteria in filters):
                tables.append(table)
                continue
            mask: pa.Array | None = None
            for criteria in filters:
                kept = pa.array(np.ones(table.num_rows, dtype=bool))
                for criterion in criteria:
                    kept = pc.and_(kept, criterion.mask(table, family))
                mask = kept if mask is None else pc.or_(mask, kept)
            assert mask is not None
            tables.append(table.filter(mask))
        joined = pa.concat_tables(tables) if tables else REGISTRY_SCHEMA.empty_table()
        self._cache = (key, joined)
        return joined

    @property
    def extractions(self) -> list[Extraction]:
        """The extractions the rows of this view name, in the families' order."""
        named = set(
            pc.unique(self.table.column("extraction_id").combine_chunks()).to_pylist()
        )
        found: dict[str, Extraction] = {}
        for family, _ in self._families():
            for extraction_id, extraction in family.extractions.items():
                if extraction_id in named:
                    found.setdefault(extraction_id, extraction)
        return list(found.values())

    @property
    def _family(self) -> _Family:
        """The one family of a family's registry.

        Raises:
            TypeError: If this registry is a view reading through filters
                or several families.
        """
        if len(self._sources) != 1 or self._sources[0].criteria:
            raise TypeError(
                "rows join the registry of one family (a timeline's), not a view "
                "of it"
            )
        return self._sources[0].family.current()

    def _families(self) -> list[tuple[_Family, list[tuple[_Criteria, ...]]]]:
        """Each family read, once, with the filters of every source reading it."""
        families: dict[int, tuple[_Family, list[tuple[_Criteria, ...]]]] = {}
        for source in self._sources:
            family = source.family.current()
            families.setdefault(id(family), (family, []))[1].append(source.criteria)
        return list(families.values())

    def filter(
        self,
        *,
        within: str | Iterable[str] | None = None,
        kinds: str | Iterable[str] | None = None,
        extractions: str | Iterable[str] | None = None,
        vanished: bool | None = None,
        min_coord: Any = None,
        max_coord: Any = None,
    ) -> EventRegistry:
        """A view of the rows of this one meeting every condition given.

        Args:
            within: Keep the rows of these timelines (IDs).
            kinds: Keep the rows of these kinds (event types).
            extractions: Keep the rows made by these extractions (IDs).
            vanished: True keeps the rows an update no longer found, False
                the others; None keeps both.
            min_coord: Keep the rows starting at or after this position
                (inclusive), compared as :meth:`EventData.filter` compares a
                bound: exactly where it is exact and the axis ``int`` or
                ``fraction``, as a key where it is a float. A bare number
                lies on each family's root axis; a ``Coordinate`` keeps rows
                of the families in its unit only (no row of a family on
                another unit lies at it); an ``IdCoordinate`` lies on the
                timeline it names, moved onto that timeline's root by the
                offsets of its ancestors, and keeps rows of that family
                only.
            max_coord: Keep the rows starting before this position
                (exclusive), read as *min_coord*.

        Raises:
            ValueError: If a bound is NaN, when the rows are read.
        """
        criterion = _Criteria(
            within=_frozen(within),
            kinds=_frozen(kinds),
            extractions=_frozen(extractions),
            vanished=vanished,
            min_coord=min_coord,
            max_coord=max_coord,
        )
        return self._view(
            _Source(source.family, (*source.criteria, criterion))
            for source in self._sources
        )

    def counts(self, by: str = "kind") -> pa.Table:
        """The number of rows per value of the column *by*, sorted by value.

        Returns:
            A table of two columns, *by* and ``count``, one row per value
            (null last); the total is ``len(registry)``.

        Raises:
            KeyError: If *by* is no registry column.
        """
        if by not in _REGISTRY_COLUMNS or by in ("start", "duration"):
            raise KeyError(
                f"No registry column {by!r} to count by. Available columns: "
                f"{[name for name in _REGISTRY_COLUMNS if name not in ('start', 'duration')]}"
            )
        counted = (
            self.table.select([by])
            .group_by(by, use_threads=False)
            .aggregate([(by, "count", pc.CountOptions(mode="all"))])
        )
        counted = pa.Table.from_arrays(
            [counted.column(by), counted.column(f"{by}_count")], names=[by, "count"]
        )
        return counted.sort_by([(by, "ascending")])

    def content(self, kind: str, ids: Iterable[str] | None = None) -> pa.Table:
        """The content of this view's events of *kind*, keyed by ``id``.

        The columns of the events' own (a pre-parsed table's): ``id`` first,
        then every column an event of *kind* holds a value in, without its
        position, type, name or registry columns. One row per event of
        *kind* this view holds (of *ids* only, if given), in the view's
        order.
        """
        selected = self.table.filter(
            _filled(
                pc.equal(self.table.column("kind"), pa.scalar(kind)), pa.scalar(False)
            )
        )
        if ids is not None:
            selected = selected.filter(
                pc.is_in(
                    selected.column("id"), value_set=pa.array(list(ids), pa.string())
                )
            )
        if not selected.num_rows:
            return pa.table({"id": pa.array([], pa.string())})
        holders = {
            timeline_id: family
            for family, _ in self._families()
            for timeline_id in family.parts
        }
        selected_ids = selected.column("id").combine_chunks()
        tables: list[pa.Table] = []
        where: list[np.ndarray] = []
        for timeline_id, at in _runs(selected.column("timeline_id")):
            events = holders[str(timeline_id)].events_of(str(timeline_id)).table
            found = pc.index_in(
                selected_ids.take(at), value_set=events.column("id").combine_chunks()
            )
            tables.append(events.take(found))
            where.append(at)
        joined = joined_tables(tables, _column_refusal).take(_inverse(where))
        names = [
            name
            for name in joined.column_names
            if name == "id"
            or (
                name not in NOT_CONTENT_COLUMNS
                and joined.column(name).null_count < joined.num_rows
            )
        ]
        return joined.select(names)

    def written(self, kinds: str | Iterable[str] | None = None) -> WrittenEvents:
        """What a writer stores of the events of the families this view reads.

        Every family this view reads, whole (a package holds a family whole:
        a child's events lie on its root's axis), its rows and contents read
        when asked. The rows recording where a child lies (a ``Segment`` row
        named by a timeline of the family) are left out; a user's
        ``Segment`` row, an event like any other, is kept, naming the
        family's extraction of events made in code
        (:meth:`Extraction.of_tool`) where it names none.

        Args:
            kinds: Keep the rows and contents of these kinds only; every
                kind when None.

        Returns:
            The rows, the contents per timeline and kind, the extractions
            the rows name and the families
            (:class:`WrittenEvents`).
        """
        wanted = _frozen(kinds)
        row_parts: list[pa.Table] = []
        contents: dict[tuple[str, str], pa.Table] = {}
        found: dict[str, Extraction] = {}
        families: list[RegistryFamily] = []
        for family, _ in self._families():
            families.append(
                RegistryFamily(
                    root_id=family.root_id,
                    unit=family.unit,
                    number_type=family.number_type,
                    timelines=tuple(family.parts),
                )
            )
            held = family.table()
            if held.num_rows:
                kept = pc.invert(
                    _places(held.column("id"), held.column("kind"), family)
                )
                if wanted is not None:
                    kept = pc.and_(
                        kept,
                        pc.is_in(
                            held.column("kind"), value_set=pa.array(sorted(wanted))
                        ),
                    )
                rows = held.filter(kept)
                unnamed = rows.column("extraction_id").is_null()
                if pc.any(unnamed).as_py():
                    tool = family.tool or Extraction.of_tool(scope=family.root_id)
                    found.setdefault(tool.id, tool)
                    at = rows.schema.get_field_index("extraction_id")
                    rows = rows.set_column(
                        at,
                        rows.schema.field(at),
                        _filled(rows.column("extraction_id"), pa.scalar(tool.id)),
                    )
                row_parts.append(rows)
            for timeline_id in family.parts:
                table = family.events_of(timeline_id).table
                if not table.num_rows:
                    continue
                kinds_of = _filled(
                    _strings(table, "event_type"), pa.scalar(_UNTYPED_KIND)
                )
                own = pc.invert(_places(_strings(table, "id"), kinds_of, family))
                for kind in sorted(
                    str(value) for value in pc.unique(kinds_of.filter(own)).to_pylist()
                ):
                    if wanted is not None and kind not in wanted:
                        continue
                    rows = table.filter(
                        pc.and_(own, pc.equal(kinds_of, pa.scalar(kind)))
                    )
                    n = rows.num_rows
                    contents[(timeline_id, kind)] = rows.select(
                        [
                            name
                            for name in rows.column_names
                            if name == "id"
                            or (
                                name not in NOT_CONTENT_COLUMNS
                                and rows.column(name).null_count < n
                            )
                        ]
                    )
            for extraction_id, extraction in family.extractions.items():
                found.setdefault(extraction_id, extraction)
        table = (
            pa.concat_tables(row_parts) if row_parts else REGISTRY_SCHEMA.empty_table()
        )
        named = {
            str(value)
            for column in ("extraction_id", "vanished_in")
            for value in pc.unique(table.column(column)).to_pylist()
            if value is not None
        }
        return WrittenEvents(
            rows=table,
            contents=contents,
            extractions=tuple(found[key] for key in sorted(named) if key in found),
            families=tuple(families),
        )

    # endregion

    # region Taking rows in

    def add(
        self,
        timeline_id: str,
        data: EventData,
        *,
        extraction: Extraction | None = None,
    ) -> None:
        """Add *data*'s rows to the timeline *timeline_id*, after its rows.

        *data* is on the timeline's axis, in its own coordinates; the
        registry rows are shifted onto the root's, column-wise. The columns
        ``locator`` and ``extraction_id``, where *data* holds them, are moved
        into the registry rows. A row naming no extraction names
        *extraction*, else the family's tool extraction
        (:meth:`Extraction.of_tool`, one per family). The extractions *data*
        carries (:attr:`EventData.extractions`) that the rows name are held
        with them. Taken by the registry of one family (a timeline's
        ``_registry``), not by a view.

        Raises:
            KeyError: If the family holds no timeline *timeline_id*.
            TypeError: If this registry is a view.
            ValueError: If *data* lies on another axis than the family's, an
                event ID of it is held in the family or twice in *data*, a
                row names an extraction neither held, nor *extraction*, nor
                carried by *data*, a locator names two events of one
                extraction, or a column of it holds values of a type the
                timeline's column does not read; nothing changes then.
        """
        self._family.add(timeline_id, data, extraction=extraction)

    def add_rows(
        self,
        rows: pa.Table,
        contents: Mapping[str, pa.Table | Callable[[], pa.Table]] | None = None,
        extractions: Iterable[Extraction] = (),
        *,
        registered: bool = True,
    ) -> None:
        """Hold registry rows already on the root's axis, as read from a package.

        Each row joins the family of the timeline it names, among the
        families this registry (a view too: a timeline's, a group's, a
        bundle's) reads. A row the family holds already, on the same
        timeline, of the same kind, at the same root position (``start`` and
        ``duration`` compared cell for cell), is passed over: it is the
        event a manifest states inline. Each timeline taking rows is held
        long enough for them, as for the rows it adds itself (an unlocked
        timeline grows to the last end, made local by its ancestors'
        offsets; a locked one refuses). The rows' content is read lazily:
        the family's events are built from its rows and *contents* (per
        kind, a table keyed by ``id`` or a callable giving one) when a
        timeline's events are first read, positions made local (the root
        position less the offsets of the timeline's ancestors), each
        timeline's in the order of its rows; a timeline taking a Box reads
        its events at once and checks every Box against its shape, as a Box
        added is checked.

        Args:
            rows: Registry rows (:data:`REGISTRY_SCHEMA`'s columns; a column
                absent is null), ``start`` and ``duration`` as storage cells.
            contents: The content of each kind, keyed by ``id``; every
                family taking rows reads it (each its own events' rows).
            extractions: The extractions the rows name; each family taking
                rows holds them all.
            registered: The rows are a package's: a family taking them never
                joins a parent afterwards.

        Raises:
            KeyError: If a row names a timeline no family of this registry
                holds.
            ValueError: If a row lies on another axis than its family's,
                names an extraction not given or held, holds an ID held in
                the family or twice, a locator naming two events of one
                extraction, ends past the length of a locked timeline, or a
                Box contradicts its shape. Nothing changes then, but the
                lengths of timelines grown before a refusal.
        """
        rows = _as_registry(rows)
        families = [family for family, _ in self._families()]
        extractions = list(extractions)
        if len(families) == 1:
            families[0].add_rows(
                rows, contents or {}, extractions, registered=registered
            )
            return
        timelines = rows.column("timeline_id").combine_chunks()
        if timelines.null_count:
            raise ValueError(
                "a registry row of an event names no timeline; an event lies on "
                "one timeline"
            )
        holder: dict[str, _Family] = {
            timeline_id: family for family in families for timeline_id in family.parts
        }
        unknown = pc.filter(
            timelines,
            pc.invert(
                pc.is_in(timelines, value_set=pa.array(list(holder), pa.string()))
            ),
        )
        if len(unknown):
            raise KeyError(
                f"A registry row names the timeline {quote_id(unknown[0].as_py())}, "
                f"which no family of this registry holds. Available timelines: "
                f"{list(holder)}"
            )
        by_family: dict[int, tuple[_Family, list[np.ndarray]]] = {}
        for timeline_id, at in _runs(timelines):
            family = holder[str(timeline_id)]
            by_family.setdefault(id(family), (family, []))[1].append(at)
        states: list[tuple[_Family, tuple[Any, ...]]] = []
        try:
            for family, positions in by_family.values():
                states.append((family, family.state()))
                family.add_rows(
                    rows.take(pa.array(np.sort(np.concatenate(positions)), pa.int64())),
                    contents or {},
                    extractions,
                    registered=registered,
                )
        except BaseException:
            for family, state in reversed(states):
                family.restore(state)
            raise

    def update_target(
        self,
        data: EventData,
        extraction: Extraction,
        *,
        timeline_id: str | None = None,
    ) -> str | None:
        """The timeline :meth:`update` would update from *data*, or None.

        The timeline of this registry holding most of the rows that *data*'s
        locators name among the rows made by extractions of the same source
        file as *extraction* (ties: the timeline met first); where they name
        none, the timeline *timeline_id* (else the one *data* is a table of)
        if a family of this registry holds it. Read column-wise: one table
        of each held extraction's ID and source, and set lookups.

        Raises:
            ValueError: If *data* gives no locators (a column ``locator``).
        """
        if "locator" not in data.table.column_names:
            raise ValueError(
                "an update matches rows by their locator: the rows give none "
                "(a column 'locator')"
            )
        held = self.table
        sources = self._extraction_sources()
        same_source = sources.column("id").filter(
            _filled(
                pc.equal(
                    sources.column("source"), pa.scalar(extraction.source, pa.string())
                ),
                pa.scalar(False),
            )
            if extraction.source is not None
            else sources.column("source").is_null()
        )
        locators = pc.drop_null(_strings(data.table, "locator"))
        candidates = held.filter(
            _filled(
                pc.and_(
                    pc.is_in(held.column("extraction_id"), value_set=same_source),
                    pc.is_in(held.column("locator"), value_set=locators),
                ),
                pa.scalar(False),
            )
        )
        if candidates.num_rows:
            counted = (
                candidates.select(["timeline_id"])
                .group_by("timeline_id", use_threads=False)
                .aggregate([("timeline_id", "count")])
                .sort_by([("timeline_id_count", "descending")])
            )
            return str(counted.column("timeline_id")[0].as_py())
        fallback = data._timeline_id if timeline_id is None else timeline_id
        if fallback is not None and any(
            fallback in family.parts for family, _ in self._families()
        ):
            return fallback
        return None

    def update(
        self,
        data: EventData,
        extraction: Extraction,
        *,
        timeline_id: str | None = None,
    ) -> str:
        """Update the rows of a timeline of this registry from *data*, a parser's new run.

        *data* holds the rows of one extraction, local to the timeline they
        were read for, each with its ``locator`` (a column). The timeline
        updated is :meth:`update_target`'s. Its rows are matched by the full
        ``locator`` among the rows made by extractions of the same source as
        *extraction*: a matched row keeps its ID and takes the new row's
        content, position and *extraction*; a new row matching none is
        added; an old row of that source no longer found is kept, naming
        *extraction* in ``vanished_in``. Nothing is deleted. A view (a
        timeline's, a group's, a bundle's) updates the family holding the
        timeline.

        Args:
            data: The rows read again, with their locators.
            extraction: The run that read them.
            timeline_id: The timeline updated where no locator matches.

        Returns:
            The ID of the timeline updated.

        Raises:
            ValueError: If *data* gives no locators or one locator twice,
                matches no timeline of this registry (by locator or by ID),
                lies on another axis, or an added row's ID is held; nothing
                changes then.
        """
        target = self.update_target(data, extraction, timeline_id=timeline_id)
        if target is None:
            named = data._timeline_id if timeline_id is None else timeline_id
            raise ValueError(
                f"the rows of extraction {quote_id(extraction.id)} "
                f"({extraction.source}) match no timeline of the registry, by "
                f"locator or by ID ({named}); nothing is updated"
            )
        family = next(
            family for family, _ in self._families() if target in family.parts
        )
        family.update(target, data, extraction)
        return target

    def _extraction_sources(self) -> pa.Table:
        """Every extraction the families of this registry hold: ``id`` and ``source``.

        Built once per state of the families (each family's version and
        number of extractions), then reused.
        """
        families = [family for family, _ in self._families()]
        key = tuple(
            value
            for family in families
            for value in (id(family), family.version, len(family.extractions))
        )
        cached = self._sources_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        ids: list[str] = []
        sources: list[str | None] = []
        for family in families:
            for extraction_id, extraction in family.extractions.items():
                ids.append(extraction_id)
                sources.append(extraction.source)
        table = pa.table(
            {"id": pa.array(ids, pa.string()), "source": pa.array(sources, pa.string())}
        )
        self._sources_cache = (key, table)
        return table

    # endregion


# endregion
