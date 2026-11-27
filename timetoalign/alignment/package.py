"""An AlignmentBundle as a dictionary, as a manifest with its files, and in a package.

The bundle's manifest dictionary (:meth:`AlignmentBundle.to_dict`,
:meth:`AlignmentBundle.from_dict`), the tables written beside it, the merge
of several bundles into one (:meth:`AlignmentBundle.from_bundles`) and the
resources such a merge names; and :class:`Package`, a folder of manifests
sharing their tables, which ``package.yaml`` lists.
"""

from __future__ import annotations

import copy
import hashlib
import os
import posixpath
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote, urlsplit

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import timetoshare
from timetoshare import loads
from timetoshare.tables import (
    PACKAGE_FILE,
    PROVENANCE_METADATA_KEY,
    TYPE_METADATA_KEY,
    package_problems,
    read_table,
    table_files,
)

from timetoalign.alignment.claim_store import _refuse_shared_claim_ids
from timetoalign.alignment.claims import MatchClaimField
from timetoalign.core.fields import _filled, _null_column
from timetoalign.core.ids import (
    IdMinter,
    _refuse_held_ids,
    content_minter,
    quote_id,
    seed_from,
    table_seed_part,
)
from timetoalign.timelines import TimelineGroup
from timetoalign.timelines.groups import _held_ids
from timetoalign.timelines.manifest import (
    _EVENT_ROW_COLUMNS,
    _NEW_FILE,
    PARQUET_MEDIA_TYPE,
    Integrity,
    Manifest,
    Resource,
    _joined_schema,
    _projection_tables,
    _timeline_ids,
    _validated_by_the_file,
    _write_package_file,
    event_file_name,
    read_event_table,
    resource_path,
    write_event_table,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from timetoalign.alignment.bundle import AlignmentBundle


def _merged_resources(
    bundles: Sequence["AlignmentBundle"], merged: "AlignmentBundle"
) -> list[Resource]:
    """The resources a merge of *bundles* into *merged* names, every id kept.

    Every source's resources are carried as they stand, in source order,
    with one exception: the merge holds one table of each type (its
    MatchClaim table holds every source's MatchClaims, the Projections,
    Boxes and other events lie on the timelines they travel with), and the
    library writes a
    bundle's rows of a table type into one table, at the first entry of
    that type (:func:`_write_tables`). So the first entry of a type, in
    source order, names the merged table, and every later entry of that
    type, of the same source or of a later one, is not carried.
    :meth:`AlignmentBundle.to_file` writes the merged table at that entry's
    URL under its ID. A table a source carries as it stands (it holds no
    events) is carried like any other resource.

    Raises:
        ValueError: If the id of a resource carried is held in the merge
            already, by a resource of another source, an element of a
            source's timeline trees, a source's group or *merged* itself,
            naming the id and both holders, before anything is merged.
    """
    carried: list[tuple[AlignmentBundle, Resource]] = []
    table_types: set[str] = set()
    for bundle in bundles:
        for resource in bundle.resources:
            if resource.is_table and resource.id not in bundle._carried_tables:
                if resource.type in table_types:
                    continue
                table_types.add(resource.type)
            carried.append((bundle, resource))
    if not carried:
        return []
    taken = _held_ids(
        member for bundle in bundles for member in bundle._scope_members()
    )
    for bundle in bundles:
        for group_id in bundle.groups:
            taken.setdefault(group_id, f"group {quote_id(group_id)}")
    taken.setdefault(merged.id, f"bundle {quote_id(merged.id)}")
    _refuse_held_ids(
        [
            (
                resource.id,
                f"resource {quote_id(resource.id)} of bundle {quote_id(bundle.id)}",
            )
            for bundle, resource in carried
        ],
        taken,
        scope=f"bundle {quote_id(merged.id)}",
    )
    return [resource for _, resource in carried]


def _from_dict(cls: type[AlignmentBundle], data: dict[str, Any]) -> "AlignmentBundle":
    """Implement :meth:`AlignmentBundle.from_dict`."""
    manifest, content = Manifest.read(data)
    bundle = cls(
        id=manifest.id or "",
        name=manifest.name,
        meta=manifest.meta,
        styling=manifest.styling,
        resources=manifest.resources,
    )
    bundle._unknown_keys = manifest._unknown_keys
    bundle._has_graphical = manifest.has_graphical
    bundle._band_orders = manifest.band_orders
    bundle._measure_maps = manifest.measure_maps
    bundle._set_statement_state(manifest.statement_state, manifest.prefixes)
    for group_data in content.groups:
        group = TimelineGroup._from_group_object(
            group_data, content.timelines, content.order.get(group_data["id"])
        )
        bundle.add_group(group)
    # Manifest.read found every id of the manifest unique, the groups'
    # included, and a group read holds its given id: admitting a root
    # (``add_timeline``) would rename nothing and refuse nothing, at the
    # cost of a pass over the bundle's trees per root.
    for root in content.roots:
        if bundle.timelines.get(root.id) is not root:
            bundle._register(root)
    return bundle


def _from_bundles(
    cls: type[AlignmentBundle],
    bundles: list["AlignmentBundle"],
    *,
    id: str = "",
    name: str | None = None,
    seed: int | None = None,
) -> "AlignmentBundle":
    """Implement :meth:`AlignmentBundle.from_bundles`."""
    _refuse_shared_claim_ids(bundles)
    merged = cls(id=id, name=name, seed=seed)
    merged.resources = _merged_resources(bundles, merged)
    merged._carried_tables = {
        resource_id: path
        for bundle in bundles
        for resource_id, path in bundle._carried_tables.items()
    }
    for bundle in bundles:
        for group in bundle.groups.values():
            rebuilt = TimelineGroup(
                id=group.id,
                name=group.name,
                meta=group.meta,
            )
            # A member keeps its own lock: merging locks nothing.
            for member in group:
                rebuilt._admit_member(member)
            merged._take_group(rebuilt)
        grouped = bundle.timeline_to_groups
        for timeline_id, timeline in bundle.timelines.items():
            if timeline_id in grouped:
                continue
            if merged.timelines.get(timeline.id) is timeline:
                continue
            merged.add_timeline(timeline)
        merged._take_claims_of(bundle)
    return merged


_FILTERED: ContextVar[bool] = ContextVar("_FILTERED", default=False)
"""Set while :meth:`Package.bundle` reads a manifest of its package: every
table is then read filtered, single files as folders are."""


def _holds_event_columns(kind: str, names: Sequence[str]) -> bool:
    """Whether a file of the type *kind* whose columns are *names* holds rows
    a bundle reads: MatchClaims or Projections, the format's table types,
    whatever the file holds (their checks refuse a malformed one), or
    events, the file holding the columns every event table holds (``id``,
    ``timeline_id``, ``unit``, ``number_type``, ``start``)."""
    return kind in ("MatchClaim", "Projection") or set(_EVENT_ROW_COLUMNS) <= set(names)


def _holds_rows(resource: Resource, path: Path) -> bool:
    """Whether the table of *resource* lying at *path* holds rows a bundle
    reads: a MatchClaim or Projections table, or one whose every file holds
    an event table's columns (:func:`_holds_event_columns`). A table of
    other rows, and a folder holding no Parquet file, which holds no rows,
    is carried as it stands."""
    if resource.type in ("MatchClaim", "Projection"):
        return True
    files = table_files(path) if resource.names_folder else [path]
    return bool(files) and all(
        _holds_event_columns(resource.type, pq.read_schema(file).names)
        for file in files
    )


def _resource_table(resource: Resource, path: Path) -> pa.Table:
    """The rows the table resource lying at *path* names for its manifest.

    A file of a lone manifest is read whole: every row of it is the
    manifest's, and a row on a timeline the manifest lacks is refused by the
    reading of its type. A folder (:attr:`Resource.names_folder`), and in a
    package (:meth:`Package.bundle`) a file as well, is read through
    ``timetoshare.tables.read_table``, keeping the rows of the timelines the
    resource lists (a MatchClaim's either of whose timelines it lists): the
    files of a package may hold the rows of its other manifests.

    Raises:
        timetoshare.tables.TableError: If the files state another type than
            the resource's, or cannot be read as one table.
    """
    if resource.names_folder or _FILTERED.get():
        return read_table(path, timelines=resource.timelines, type=resource.type)
    return pq.read_table(path)


def _read_tables(
    self: AlignmentBundle, directory: Path, data: Mapping[str, Any]
) -> None:
    """Implement :meth:`AlignmentBundle._read_tables`."""
    tables: list[tuple[str, pa.Table]] = []
    carried: dict[str, Path] = {}
    for resource in self.resources:
        if not resource.is_table:
            continue
        path = resource.path_in(directory)
        if not (path.is_dir() if resource.names_folder else path.is_file()):
            continue
        if not _holds_rows(resource, path):
            carried[resource.id] = Path(os.path.abspath(path))
            continue
        tables.append((resource.type, _resource_table(resource, path)))
    self._carried_tables = carried
    # Every resource of a type is read, in the order the manifest lists
    # them: the Projections first, then every other event type, then the
    # MatchClaims, whose side B may name a Projection. Each timeline's rows
    # are added as a table.
    for table_type, table in tables:
        if table_type == "Projection":
            for timeline_id, rows in _projection_tables(table, data).items():
                timeline = self._require_known_timeline(timeline_id)
                timeline._add_event_table(rows, table_type)
    for table_type, table in tables:
        if table_type in ("Projection", "MatchClaim"):
            continue
        for timeline_id, rows in read_event_table(table).items():
            self._require_known_timeline(timeline_id)._add_event_table(rows, table_type)
    for table_type, table in tables:
        if table_type == "MatchClaim":
            self.add_match_claim_field(MatchClaimField.from_package_table(table))


def _write_tables(
    self: AlignmentBundle, directory: Path, *, events: bool = False
) -> bool:
    """Implement :meth:`AlignmentBundle._write_tables`."""
    claims = self.match_claim_table
    event_tables = self._event_tables(events=events)
    kinds = {*event_tables, *(("MatchClaim",) if len(claims) else ())}
    # A table carried as it stands (no events, no MatchClaims) names no table.
    tabled = [
        resource
        for resource in self.resources
        if resource.is_table and resource.id not in self._carried_tables
    ]
    for resource in tabled:
        if resource.names_folder and resource.type in kinds:
            raise NotImplementedError(
                f"The resource {quote_id(resource.id)} names the folder "
                f"{resource.url!r} for its {resource.type} rows: "
                f"AlignmentBundle.to_file writes a manifest's tables as single "
                f"files beside it and does not write into a folder, whose files "
                f"other manifests may share; write the bundle into its package "
                f"with Package.write_bundle."
            )
    if not kinds:
        return False
    # The first entry of each table type names its table; a later entry of
    # a type written lists timelines whose rows that table now holds.
    named: dict[str, Resource] = {}
    for resource in tabled:
        named.setdefault(resource.type, resource)
    taken = {
        **self._scope_ids(),
        **{r.id: f"resource {quote_id(r.id)}" for r in self.resources},
    }
    written: dict[str, Resource] = {}

    def place(table_type: str, default: str) -> tuple[str, Path]:
        """Return the URL and resolved path a table is written to, creating its folder.

        A held entry's URL is kept; otherwise *default* is used.
        """
        held = named.get(table_type)
        url = held.url if held is not None else default
        path = resource_path(directory, url)
        path.parent.mkdir(parents=True, exist_ok=True)
        return url, path

    # Every table to write: its type, URL, path, timelines and writer.
    tables: list[tuple[str, str, Path, list[str], Callable[[Path], object]]] = []
    # Every place is resolved (a URL refused) before the first file is written.
    if len(claims):
        url, path = place("MatchClaim", MatchClaimField.file_name)
        tables.append(
            ("MatchClaim", url, path, _claim_timelines(claims.table), claims.to_parquet)
        )
    for kind, table in event_tables.items():
        url, path = place(kind, quote(event_file_name(kind)))
        tables.append(
            (
                kind,
                url,
                path,
                _event_timelines(table),
                lambda path, table=table: write_event_table(table, path),
            )
        )
    # An entry of a type not written stays where its file lies; a folder
    # of such a type, and a table carried, stay as they are.
    kept = {id(r) for r in self.resources if r.id in self._carried_tables} | {
        id(r)
        for r in tabled
        if r.type not in kinds
        and (r.names_folder or resource_path(directory, r.url).is_file())
    }
    # Every entry's id, refused where the manifest holds it, before the
    # first file is written. A held entry's id is reused, and its
    # timeline order kept when it names the same timelines; otherwise a
    # new entry mints its id.
    entries: list[tuple[str, str, Path, list[str], str]] = []
    for table_type, url, path, timelines, _ in tables:
        held = named.get(table_type)
        if held is not None and set(held.timelines) == set(timelines):
            timelines = list(held.timelines)
        if held is None:
            resource_id = Resource._minted_id(table_type, timelines, url)
            taken = _refuse_held_ids(
                [(resource_id, f"resource {quote_id(resource_id)}")], taken
            )
        else:
            resource_id = held.id
        entries.append((table_type, url, path, timelines, resource_id))
    for (_, _, _, _, write), (table_type, url, path, timelines, resource_id) in zip(
        tables, entries, strict=True
    ):
        write(path)
        written[table_type] = Resource.of_file(
            path, id=resource_id, type=table_type, timelines=timelines, url=url
        )
    resources: list[Resource] = []
    for r in self.resources:
        if not r.is_table or id(r) in kept:
            resources.append(r)
        elif r is named.get(r.type) and r.type in written:
            resources.append(written.pop(r.type))
    self.resources = resources + list(written.values())
    return True


def _event_timelines(table: pa.Table) -> list[str]:
    """The timelines an event table's rows lie on, sorted."""
    named = pc.unique(table.column("timeline_id")).to_pylist()
    return sorted(value for value in named if isinstance(value, str))


def _claim_timelines(table: pa.Table) -> list[str]:
    """The timelines a MatchClaim table's rows name on either side, sorted."""
    named = pc.unique(
        pa.concat_arrays(
            [
                table.column("timeline_a_id").combine_chunks(),
                table.column("timeline_b_id").combine_chunks(),
            ]
        )
    )
    return sorted(value for value in named.to_pylist() if isinstance(value, str))


def _to_dict(
    self: AlignmentBundle,
    *,
    minimal: bool = True,
    events: bool = False,
    external_references: bool = False,
    resources: Sequence[Resource] | None = None,
) -> dict[str, Any]:
    """Implement :meth:`AlignmentBundle.to_dict`; the manifest names
    *resources* in place of the bundle's where they are given."""
    groups = [self.groups[group_id] for group_id in sorted(self.groups)]
    manifest = Manifest(
        id=self.id if self._id_given else None,
        name=self.name,
        meta=self.meta,
        statement_state=self._statement_state,
        prefixes=self._prefixes,
        styling=self.styling,
        band_orders=self._band_orders,
        has_graphical=self._has_graphical,
        measure_maps=self._measure_maps,
        resources=list(self.resources if resources is None else resources),
    )
    manifest._unknown_keys = self._unknown_keys
    return manifest.write(
        self._roots(),
        groups=[group._to_group_object() for group in groups],
        order={group.id: group.timeline_ids for group in groups},
        minimal=minimal,
        events=events,
        external_references=external_references,
        table_prefixes=self.match_claim_table._used_prefixes(),
    )


# region Packages

#: The folder of a package's tables, one folder per type inside it.
_TABLES_FOLDER = "tables"

#: The prefix of the names of the files the package writer mints.
_PART_PREFIX = "part"

#: The columns naming a row's timelines in a MatchClaim table.
_CLAIM_TIMELINES = ("timeline_a_id", "timeline_b_id")

#: The sorting columns of a written MatchClaim file.
_CLAIM_ORDER = ("timeline_a_id", "timeline_b_id", "id")


def _sorted_by(kind: str) -> tuple[str, ...]:
    """The columns a package file of *kind* is sorted by and declares."""
    return _CLAIM_ORDER if kind == "MatchClaim" else ("timeline_id",)


def _claim_file_table(claims: MatchClaimField) -> pa.Table:
    """A bundle's MatchClaim table as a package file holds it.

    The package form (:attr:`MatchClaimField.package_table`) with every
    row's provenance in its own cell, a shared provenance written into each
    row, since a file's shared provenance is lost when the files of a folder
    are read as one table; sorted by ``timeline_a_id``, ``timeline_b_id``
    and ``id``; its metadata naming its type alone.
    """
    table = claims.package_table
    if claims.provenance is not None:
        at = table.schema.get_field_index("provenance")
        rows = pa.Table.from_struct_array(claims._rows_with_provenance())
        table = table.set_column(at, table.schema.field(at), rows.column("provenance"))
    table = table.replace_schema_metadata({TYPE_METADATA_KEY: "MatchClaim"})
    return table.take(
        pc.sort_indices(table, sort_keys=[(name, "ascending") for name in _CLAIM_ORDER])
    )


def _manifest_timelines(manifest: Mapping[str, Any]) -> frozenset[str]:
    """Every timeline ID of a manifest in the file form, at any depth, its
    virtual timelines' included."""
    bare = timetoshare.with_bare_ids(manifest)
    return frozenset(
        {
            *_timeline_ids(list(bare.get("timelines") or ())),
            *(entry["id"] for entry in bare.get("virtual_timelines") or ()),
        }
    )


def _rows_of(table: pa.Table, columns: Sequence[str], timelines: Any) -> pa.Array:
    """Whether each row of *table* names one of *timelines* in any of *columns*."""
    listed = pa.array(sorted(timelines), pa.string())
    mask = pa.repeat(pa.scalar(False), table.num_rows)
    for name in columns:
        if name in table.column_names:
            held = pc.is_in(table.column(name), value_set=listed)
            mask = pc.or_(mask, _filled(held, pa.scalar(False)))
    return mask


@dataclass(frozen=True)
class _Copy:
    """A table a bundle carries, as the package holds it.

    Attributes:
        type: The resource's type, the type of the package's entry.
        source: Where the bundle read it, a file or a folder.
        target: Its path from the package's root (a folder's without the
            closing ``/``).
        folder: Whether it is a folder.
        copied: Whether the write copies it there; otherwise it lies there
            already (the bundle was read from it, or equal content lies at
            its name).
        entry: The ``tables`` entry ``package.yaml`` lists for it.
    """

    type: str
    source: Path
    target: str
    folder: bool
    copied: bool
    entry: dict[str, Any]


@dataclass(frozen=True)
class _Written:
    """A bundle as :meth:`Package.write_bundle` prepares it: what
    :meth:`Package.to_file` writes for it.

    Attributes:
        removed: The timelines whose rows the write removes from the files
            on disk: the manifest's and the one it replaces; none for a
            package not written yet.
        carried: Each table the bundle carries, by its resource ID.
    """

    url: str
    bundle: AlignmentBundle
    text: str
    media_type: str
    tables: dict[str, pa.Table]
    resources: list[Resource]
    timelines: frozenset[str]
    removed: frozenset[str]
    carried: dict[str, _Copy]

    @cached_property
    def id_pairs(self) -> dict[bool, list[tuple[str, str]]]:
        """The IDs of the events (``False``) and of the MatchClaims
        (``True``), each with its timeline (:func:`_id_pairs`), converted
        once."""
        return {claims: _id_pairs(self, claims) for claims in (False, True)}


def _collision(
    what: str, held_id: str, timeline: str, url: str, holder: str, place: str
) -> ValueError:
    """The refusal of the *what* *held_id* of *timeline* of the manifest at
    *url*, whose ID the *what* of timeline *holder* in *place* holds."""
    return ValueError(
        f"The {what} {quote_id(held_id)} of timeline {quote_id(timeline)} of the "
        f"manifest {url!r} cannot join the package: its ID is held by the {what} "
        f"of timeline {quote_id(holder)} in {place}; nothing is renamed."
    )


def _id_columns(written: _Written, claims: bool) -> tuple[pa.Array, pa.Array]:
    """The IDs of *written*'s MatchClaims (with *claims*) or of its events,
    each with the timeline it lies on (a MatchClaim's ``timeline_a_id``)."""
    column = "timeline_a_id" if claims else "timeline_id"
    tables = [
        table.select(["id", column])
        for kind, table in written.tables.items()
        if (kind == "MatchClaim") == claims
    ]
    if not tables:
        return pa.array([], pa.string()), pa.array([], pa.string())
    joined = pa.concat_tables(tables)
    return (
        joined.column("id").combine_chunks(),
        joined.column(column).combine_chunks(),
    )


def _id_pairs(written: _Written, claims: bool) -> list[tuple[str, str]]:
    """Each ID of *written*'s MatchClaims (with *claims*) or of its events
    with the timeline it lies on, in table order (no ID is null)."""
    ids, timelines = _id_columns(written, claims)
    return [
        (held_id, timeline_id)
        for held_id, timeline_id in zip(
            ids.to_pylist(), timelines.to_pylist(), strict=True
        )
        if isinstance(held_id, str) and isinstance(timeline_id, str)
    ]


@dataclass(frozen=True)
class _Ids:
    """The IDs a package holds on disk besides a bundle being written: its
    rows' (``id`` and the timeline each lies on) and each holder's place."""

    ids: pa.Array
    timelines: pa.Array
    places: list[str]
    place: pa.Array

    @classmethod
    def of(cls, parts: list[tuple[pa.Table, str]], column: str) -> _Ids:
        """The IDs of *parts*, each a table of ``id`` and *column* with the
        place its rows lie in."""
        ids = [table.column("id").combine_chunks() for table, _ in parts]
        timelines = [table.column(column).combine_chunks() for table, _ in parts]
        place = [
            pa.repeat(pa.scalar(index, pa.int64()), table.num_rows)
            for index, (table, _) in enumerate(parts)
        ]
        return cls(
            pa.concat_arrays(ids) if ids else pa.array([], pa.string()),
            pa.concat_arrays(timelines) if timelines else pa.array([], pa.string()),
            [where for _, where in parts],
            pa.concat_arrays(place) if place else pa.array([], pa.int64()),
        )

    def refuse(self, ids: pa.Array, timelines: pa.Array, what: str, url: str) -> None:
        """Refuse *ids* (each on the timeline of *timelines*, of the manifest
        at *url*) where one is held here or two of them are one.

        Raises:
            ValueError: Naming the first such ID and both holders.
        """
        order = pc.sort_indices(ids).to_numpy()
        ordered = ids.take(order)
        if len(ordered) > 1:
            twice = pc.equal(ordered.slice(1), ordered.slice(0, len(ordered) - 1))
            if pc.any(twice).as_py():
                at = pc.index(twice, True).as_py()
                first, again = sorted((int(order[at]), int(order[at + 1])))
                raise _collision(
                    what,
                    ids[again].as_py(),
                    timelines[again].as_py(),
                    url,
                    timelines[first].as_py(),
                    f"the same manifest {url!r}",
                )
        if not len(self.ids):
            return
        held = pc.is_in(ids, value_set=self.ids)
        if not pc.any(held).as_py():
            return
        at = pc.index(held, True).as_py()
        found = pc.index(self.ids, ids[at]).as_py()
        raise _collision(
            what,
            ids[at].as_py(),
            timelines[at].as_py(),
            url,
            self.timelines[found].as_py(),
            self.places[self.place[found].as_py()],
        )


class _Staged:
    """The IDs the bundles a package stages hold, kept as each is staged:
    staging a bundle checks its own IDs alone, against these, never the
    staged bundles' tables again.

    Attributes:
        owners: The manifest URL owning each timeline ID staged.
        ids: Of the events (``False``) and of the MatchClaims (``True``),
            the timeline and the manifest URL of each ID staged.
    """

    def __init__(self) -> None:
        self.owners: dict[str, str] = {}
        self.ids: dict[bool, dict[str, tuple[str, str]]] = {False: {}, True: {}}
        self._staged: dict[str, _Written] = {}

    def refuse(self, written: _Written) -> None:
        """Refuse *written* where a timeline ID or an ID of its rows is held
        by a bundle staged at another URL.

        Raises:
            ValueError: Naming the first such ID and both holders.
        """
        _refuse_held_ids(
            [
                (
                    timeline_id,
                    f"timeline {quote_id(timeline_id)} of the manifest {written.url!r}",
                )
                for timeline_id in sorted(written.timelines)
            ],
            {
                timeline_id: f"a timeline of the manifest {self.owners[timeline_id]!r}"
                for timeline_id in written.timelines
                if self.owners.get(timeline_id, written.url) != written.url
            },
            scope="the package",
        )
        for claims in (False, True):
            held = self.ids[claims]
            pairs = written.id_pairs[claims]
            hits = held.keys() & {held_id for held_id, _ in pairs}
            if not hits:
                continue
            for held_id, timeline_id in pairs:
                if held_id in hits and held[held_id][1] != written.url:
                    holder, url = held[held_id]
                    raise _collision(
                        "MatchClaim" if claims else "event",
                        held_id,
                        timeline_id,
                        written.url,
                        holder,
                        f"the manifest {url!r}",
                    )

    def add(self, written: _Written) -> _Written | None:
        """Stage *written*, in place of the bundle staged at its URL, which is
        returned (None where none was)."""
        previous = self.remove(written.url)
        self._staged[written.url] = written
        self.owners.update(dict.fromkeys(written.timelines, written.url))
        for claims in (False, True):
            self.ids[claims].update(
                (held_id, (timeline_id, written.url))
                for held_id, timeline_id in written.id_pairs[claims]
            )
        return previous

    def remove(self, url: str) -> _Written | None:
        """Unstage the bundle staged at *url*, returned (None where none was)."""
        previous = self._staged.pop(url, None)
        if previous is None:
            return None
        for timeline_id in previous.timelines:
            if self.owners.get(timeline_id) == url:
                del self.owners[timeline_id]
        for claims in (False, True):
            held = self.ids[claims]
            for held_id, _ in previous.id_pairs[claims]:
                if held.get(held_id, ("", ""))[1] == url:
                    del held[held_id]
        return previous


class _Undo:
    """What writing a package changes on disk, restored where the write fails.

    Every file is recorded before it is first changed: the bytes of one
    lying there, else that it is new; every folder created is recorded.
    """

    def __init__(self) -> None:
        self._kept: dict[Path, bytes] = {}
        self._new: list[Path] = []
        self._folders: list[Path] = []

    def keep(self, path: Path) -> None:
        """Record *path* before it is written, replaced or deleted."""
        if path in self._kept or path in self._new:
            return
        if path.exists():
            self._kept[path] = path.read_bytes()
        else:
            self._new.append(path)

    def folder(self, path: Path) -> None:
        """Create the folder *path*, recording every folder created."""
        missing: list[Path] = []
        at = path
        while not at.exists():
            missing.append(at)
            at = at.parent
        path.mkdir(parents=True, exist_ok=True)
        self._folders.extend(reversed(missing))

    def restore(self) -> None:
        """Restore every file recorded: a new one removed, another written
        again with its bytes; then every folder created and left empty
        removed."""
        for path in reversed(self._new):
            path.unlink(missing_ok=True)
        for path, data in self._kept.items():
            temporary = path.with_name(f".{path.name}.restoring")
            temporary.write_bytes(data)
            os.replace(temporary, path)
        for folder in reversed(self._folders):
            try:
                folder.rmdir()
            except OSError:
                pass


class Package:
    """A TimeToShare package: a folder of manifests sharing their tables.

    ``package.yaml`` at the folder's root lists the package's manifests and
    its tables, each a file or a folder of Parquet files (``tables/<Type>/``
    as this class writes them) holding the rows of one type, of every
    manifest that names it. A manifest names each file or folder it reads
    in its ``resources``, with the timelines whose rows it keeps; nothing in
    a file names a manifest. An event ID is held once among all events of
    the package, every type; a MatchClaim ID once among its MatchClaims; a
    timeline ID by one manifest.

    :meth:`from_file` opens a package; :meth:`bundle` reads one of its
    manifests as an :class:`~timetoalign.alignment.AlignmentBundle`, every
    table filtered by the timelines its resources list.
    :meth:`write_bundle` writes a bundle into the package as one of its
    manifests, and :meth:`to_file` writes a new package where it is told.

    Args:
        id: The package's IRI where it is published (its repository or
            DOI), or None.
        name: Its human-readable name, or None.

    Examples:
        >>> import tempfile
        >>> from fractions import Fraction
        >>> from pathlib import Path
        >>> from timetoalign.alignment import AlignmentBundle, Package
        >>> from timetoalign.timelines import ContinuousLogicalTimeline
        >>> score = ContinuousLogicalTimeline(length=Fraction(4), uid="score")
        >>> _ = score.add_events([{"id": "n1", "event_type": "Note", "start": 0}])
        >>> bundle = AlignmentBundle(id="piece")
        >>> _ = bundle.add_timeline(score)
        >>> package = Package(name="Pieces")
        >>> package.write_bundle(bundle, "piece/manifest.yaml")
        >>> path = package.to_file(Path(tempfile.mkdtemp()) / "package.yaml")
        >>> package.manifests, [table["url"] for table in package.tables]
        (['piece/manifest.yaml'], ['tables/Note/'])
        >>> read = Package.from_file(path).bundle("piece/manifest.yaml")
        >>> read.get_timeline("score").events.table.column("id").to_pylist()
        ['n1']
    """

    def __init__(
        self,
        id: str | None = None,  # noqa: A002
        name: str | None = None,
    ) -> None:
        document: dict[str, Any] = {
            "timetoshare": timetoshare.SCHEMA_VERSION,
            "manifests": [],
            "tables": [],
        }
        if id is not None:
            document["id"] = id
        if name is not None:
            document["name"] = name
        timetoshare.validate_package(document)
        self._document = document
        self._path: Path | None = None
        self._pending: dict[str, _Written] = {}
        self._staged = _Staged()
        self._on_disk: dict[str, frozenset[str]] = {}

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> Package:
        """Open the package whose ``package.yaml`` lies at *path*.

        The file is read in any serialization of the format and validated
        (:func:`timetoshare.validate_package`); its manifests and tables
        are read when asked for (:meth:`bundle`). A file a write cut short
        left hidden in a folder of ``tables/`` (``.part_….parquet``, which
        no reader reads) is removed.

        Raises:
            FileNotFoundError: If no file lies at *path*.
            timetoshare.SchemaValidationError: If the file is no valid
                ``package.yaml``.
        """
        path = Path(path)
        data = loads(path.read_text(encoding="utf-8"))
        timetoshare.validate_package(data)
        tables = path.parent / _TABLES_FOLDER
        if tables.is_dir():
            for folder in tables.iterdir():
                if folder.is_dir():
                    for stray in folder.glob(f".{_PART_PREFIX}_*.parquet"):
                        stray.unlink()
        package = cls.__new__(cls)
        package._document = data
        package._path = path
        package._pending = {}
        package._staged = _Staged()
        package._on_disk = {}
        return package

    def __repr__(self) -> str:
        return (
            f"Package(name={self.name!r}, manifests={len(self.manifests)}, "
            f"tables={len(self.tables)})"
        )

    @property
    def id(self) -> str | None:
        """The package's IRI, or None."""
        return self._document.get("id")

    @property
    def name(self) -> str | None:
        """The package's name, or None."""
        return self._document.get("name")

    @property
    def path(self) -> Path | None:
        """Where the package's ``package.yaml`` lies, or None for a package
        not written yet."""
        return self._path

    @property
    def manifests(self) -> list[str]:
        """The URLs of the package's manifests, relative to its root, in the
        order ``package.yaml`` lists them."""
        return [entry["url"] for entry in self._document["manifests"]]

    @property
    def tables(self) -> list[dict[str, Any]]:
        """The package's tables as ``package.yaml`` lists them: each a
        ``type`` and a ``url`` (a folder's ending in ``/``), with the members
        given (``media_type``, a file's ``size`` and ``integrity``)."""
        return [dict(entry) for entry in self._document["tables"]]

    def bundle(self, url: str) -> AlignmentBundle:
        """The manifest at *url* as an :class:`~timetoalign.alignment.AlignmentBundle`.

        Read as :meth:`AlignmentBundle.from_file
        <timetoalign.alignment.AlignmentBundle.from_file>` reads it, but
        every table its resources name, a single file as a folder, is read
        filtered: its rows of the timelines the resource lists (a
        MatchClaim's either of whose timelines it lists), the other
        manifests' passed over. A manifest read alone instead refuses a row
        of a timeline it lacks in a single file.

        Args:
            url: The manifest's URL as :attr:`manifests` lists it.

        Raises:
            KeyError: If the package lists no manifest at *url*, naming
                those it lists.
            ValueError: If the manifest is not written yet, or as
                ``AlignmentBundle.from_file`` refuses it.
        """
        from timetoalign.alignment.bundle import AlignmentBundle

        if url not in self.manifests:
            raise KeyError(
                f"The package lists no manifest {url!r}. Available manifests: "
                f"{self.manifests}"
            )
        if self._path is None or url in self._pending:
            raise ValueError(
                f"The manifest {url!r} is not written yet: write the package "
                f"(to_file) before reading it."
            )
        token = _FILTERED.set(True)
        try:
            return AlignmentBundle.from_file(resource_path(self._path.parent, url))
        finally:
            _FILTERED.reset(token)

    def write_bundle(self, bundle: AlignmentBundle, url: str) -> None:
        """Write *bundle* into the package as its manifest at *url*.

        The manifest lies at *url* under the package's root, its resources
        naming the folders ``tables/<Type>/`` (relative to it, without size
        or digest, since other manifests' rows share them), one per type it
        has rows of, with its timelines holding them (a MatchClaim's both);
        a resource that is no table (a recording) is kept. A table the
        bundle carries, read as a table of other rows (a Parquet file
        without an event table's columns, a folder holding no Parquet file),
        is copied into the package with its bytes, at ``tables/<Type>/``
        under its own name (a name holding other content: one seeded from
        the table's content), listed in ``package.yaml``, and its resource
        names the copy; one lying in the package already stays where it
        lies, and equal content lying at its name is that copy. A table
        resource of a type the bundle holds no rows of and does not carry
        is refused: the writer has no file to copy. Its rows of every
        event type (:func:`~timetoalign.timelines.manifest.event_table`,
        Projections and Boxes among them) and its MatchClaims (each row with
        its own provenance) go into the package's files: each write adds one
        new file per type, ``tables/<Type>/part_<13 base-36
        characters>.parquet``, holding the rows of every bundle it writes
        (:meth:`to_file` writes every bundle staged at once), sorted by
        ``timeline_id`` with each timeline's rows in their order (the
        MatchClaims by ``timeline_a_id``, ``timeline_b_id`` and ``id``),
        named from its content through the library's ID minter, so that a
        rebuild names its files again. A manifest the package holds at *url*
        already is replaced: the rows of its timelines and of the bundle's
        leave the files holding them, each such file rewritten whole (one
        left without a row removed), the other manifests' rows kept, sorted
        as they are declared; its MatchClaims must be kept by the bundle, as
        :meth:`MatchClaimField.to_parquet
        <timetoalign.alignment.MatchClaimField.to_parquet>` requires, so a
        bundle holding none while the package holds MatchClaims of that
        manifest is refused.

        Before anything is written, every ID is checked in memory: each
        event ID among every event of the package, every type; each
        MatchClaim ID among its MatchClaims; each timeline ID among the
        timelines of its other manifests. A collision is refused, naming
        the ID and both holders; nothing is renamed. A new package (not
        written yet) stages the bundle, which :meth:`to_file` writes: the
        bundle's own IDs are checked against an index of the IDs staged, so
        staging N bundles costs in the number of their rows. A package lying
        on disk (:meth:`from_file`, or written by :meth:`to_file`) is
        written at once, ``package.yaml`` with it, and each call reads the
        ``id`` and timeline columns of every file of the package, rewrites
        every file holding a row of the manifest replaced, and checks the
        package whole (:func:`timetoshare.tables.package_problems`): its
        cost grows with the package, so a package of many manifests is
        built new, staging every bundle and writing them with one
        :meth:`to_file`. Once written, the bundle's :attr:`resources
        <timetoalign.alignment.AlignmentBundle.resources>` are the
        manifest's, and the tables it carries the package's copies.

        A refusal leaves the package as it was, on disk and in memory: what
        can be checked in memory is checked before any file is written (the
        IDs, the bundle's resources, an opened package's format version);
        the new files are written under names readers skip and renamed into
        place once the old rows are removed; and where the write or the
        package check after it fails, every file rewritten gets its bytes
        back, every new file is removed. This holds for an exception, not
        for a process killed between the removal of the old rows and the
        renaming of the new files: the package then lacks the rows of the
        manifests written, until they are written again (:meth:`from_file`
        removes the new files left hidden).

        Args:
            bundle: The bundle.
            url: The manifest's location relative to the package's root,
                such as ``bach/manifest.yaml``.

        Raises:
            ValueError: If *url* leaves the package's root or names no file
                in it, an ID collides, the bundle cannot be written as a
                manifest, names a table of a type it holds no rows of that
                it does not carry or whose file no longer lies where it was
                read (the writer has nothing to copy), or its
                MatchClaims do not keep the ones the package holds for it,
                or an opened package is of another format version, before
                anything is written; if the package check refuses the
                package written, with everything restored.
        """
        url = self._manifest_url(url)
        written = self._prepared(bundle, url)
        entries = self._document["manifests"]
        listed = url in self.manifests
        previous = self._staged.add(written)
        self._pending[url] = written
        if not listed:
            entries.append({"url": url, "media_type": written.media_type})
        if self._path is None:
            return
        try:
            self._write()
        except BaseException:
            self._staged.remove(url)
            del self._pending[url]
            if previous is not None:
                self._staged.add(previous)
                self._pending[url] = previous
            if not listed:
                entries.pop()
            raise

    def to_file(self, path: str | os.PathLike[str]) -> Path:
        """Write the package: every bundle staged, then ``package.yaml``.

        ``package.yaml`` is written in canonical YAML, every manifest listed
        explicitly and one table entry per type folder. Every bundle staged
        is written in one go: one new file per type, holding the rows of
        them all (:meth:`write_bundle`). Once written, the package is
        checked whole (:func:`timetoshare.tables.package_problems`), and its
        first problem is raised.

        A refusal leaves the package as it was: on disk, every file
        rewritten gets its bytes back and every file and folder created is
        removed, ``package.yaml`` included; in memory, a new package stays
        unwritten with its bundles staged.

        Args:
            path: The ``package.yaml`` of a new package, in a folder holding
                none yet; a package opened or written before is written where
                it lies.

        Returns:
            The path written.

        Raises:
            ValueError: If *path* is not named ``package.yaml``, a new package
                would be written over a file lying there, a package lying
                elsewhere would be written at *path*, an opened package is
                of another format version, or the package check finds a
                problem.
        """
        path = Path(path)
        if self._path is None:
            if path.name != PACKAGE_FILE:
                raise ValueError(
                    f"A package's file is named {PACKAGE_FILE!r}, not {path.name!r}."
                )
            if path.exists():
                raise ValueError(
                    f"'{path}' exists already: a new package is written into a "
                    f"folder holding none; open that one with Package.from_file."
                )
            self._path = path
            try:
                self._write()
            except BaseException:
                self._path = None
                raise
            return path
        if os.path.abspath(path) != os.path.abspath(self._path):
            raise ValueError(
                f"The package lies at '{self._path}' and is written there, not at "
                f"'{path}': its manifests and tables lie beside it."
            )
        self._write()
        return self._path

    # Preparing a bundle.

    def _manifest_url(self, url: str) -> str:
        """*url* as the package lists a manifest at it: a relative path under
        the package's root, normalised.

        Raises:
            ValueError: If it names a scheme, a host, a query, a fragment or
                a folder, leaves the root, or names ``package.yaml`` or the
                tables' folder.
        """
        parts = urlsplit(url)
        path = posixpath.normpath(parts.path) if parts.path else ""
        if (
            parts.scheme
            or parts.netloc
            or parts.query
            or parts.fragment
            or not path
            or parts.path.endswith("/")
            or path.startswith(("/", "../"))
            or path in (".", "..", PACKAGE_FILE)
            or path.split("/")[0] == _TABLES_FOLDER
        ):
            raise ValueError(
                f"A manifest of a package lies at a file path relative to its "
                f"root, outside {_TABLES_FOLDER}/, not at {url!r}."
            )
        return path

    def _prepared(self, bundle: AlignmentBundle, url: str) -> _Written:
        """What writing *bundle* at *url* writes, every ID checked."""
        tables: dict[str, pa.Table] = dict(bundle._event_tables(events=True))
        claims = bundle.match_claim_table
        if len(claims):
            tables["MatchClaim"] = _claim_file_table(claims)
        folder = posixpath.dirname(url) or "."
        carried = self._carried(bundle, set(tables), url)
        resources: list[Resource] = []
        for resource in bundle.resources:
            placed = carried.get(resource.id)
            if placed is not None:
                relative = posixpath.relpath(placed.target, folder)
                location = quote(relative + ("/" if placed.folder else ""), safe="/.")
                resources.append(replace(resource, url=location))
            elif not resource.is_table:
                resources.append(resource)
        held: dict[str, str] = {}
        for resource in bundle.resources:
            if resource.is_table and resource.id not in carried:
                held.setdefault(resource.type, resource.id)
        taken = {
            **bundle._scope_ids(),
            **{r.id: f"resource {quote_id(r.id)}" for r in resources},
        }
        for kind, table in tables.items():
            relative = posixpath.relpath(f"{_TABLES_FOLDER}/{kind}", folder) + "/"
            location = quote(relative, safe="/.")
            timelines = (
                _claim_timelines(table)
                if kind == "MatchClaim"
                else _event_timelines(table)
            )
            resource_id = held.get(kind)
            if resource_id is None:
                resource_id = Resource._minted_id(kind, timelines, location)
            taken = _refuse_held_ids(
                [(resource_id, f"resource {quote_id(resource_id)}")], taken
            )
            resources.append(
                Resource(
                    resource_id, kind, tuple(timelines), location, PARQUET_MEDIA_TYPE
                )
            )
        form = replace(bundle._file_form or _NEW_FILE, events=False)
        with _validated_by_the_file():
            manifest = _to_dict(bundle, minimal=False, resources=resources)
        if not form.styling:
            manifest = {
                key: value for key, value in manifest.items() if key != "graphical"
            }
        timetoshare.validate(manifest)
        text = timetoshare.dumps_file(manifest, form)
        timelines = _manifest_timelines(manifest)
        before = self._disk_manifest_timelines(url)
        written = _Written(
            url=url,
            bundle=bundle,
            text=text,
            media_type=(
                "application/yaml"
                if form.serialization == "yaml"
                else "application/json"
            ),
            tables=tables,
            resources=resources,
            timelines=timelines,
            removed=timelines | before if self._path is not None else frozenset(),
            carried=carried,
        )
        self._refuse_held(written)
        if before:
            # Every MatchClaim of the manifest replaced is kept, even by a
            # bundle holding none.
            self._refuse_unless_succeeding(claims, before)
        return written

    def _carried(
        self, bundle: AlignmentBundle, written: set[str], url: str
    ) -> dict[str, _Copy]:
        """Where the package holds each table *bundle* carries, by resource ID.

        Every table resource of a type the bundle writes no rows of
        (*written*), and every table the bundle carries
        (``_carried_tables``), is carried: placed by :meth:`_placed`, beside
        the copies of the bundles staged at other URLs.

        Raises:
            ValueError: If such a resource is not carried by the bundle (it
                was never read from a file) or its file or folder no longer
                lies where the bundle read it, naming it and why.
        """
        reserved = {
            placed.target: placed
            for staged in self._pending.values()
            if staged.url != url
            for placed in staged.carried.values()
        }
        found: dict[str, _Copy] = {}
        for resource in bundle.resources:
            if not resource.is_table:
                continue
            source = bundle._carried_tables.get(resource.id)
            what = (
                f"The resource {quote_id(resource.id)} of the bundle "
                f"{quote_id(bundle.id)} names the table {resource.url!r} of the "
                f"type {resource.type}"
            )
            if source is None:
                if resource.type in written:
                    continue
                raise ValueError(
                    f"{what}, of which the bundle holds no rows, and the bundle "
                    f"was not read from it (built in code, or its file missing "
                    f"when it was read): the package writer copies such a table "
                    f"into the package from the file the bundle was read with and "
                    f"has none to copy; nothing is written."
                )
            if not (source.is_dir() if resource.names_folder else source.is_file()):
                raise ValueError(
                    f"{what}, which the bundle carries from '{source}', where no "
                    f"{'folder' if resource.names_folder else 'file'} lies any "
                    f"more: the package writer has nothing to copy into the "
                    f"package; nothing is written."
                )
            placed = self._placed(resource, source, reserved)
            reserved[placed.target] = placed
            found[resource.id] = placed
        return found

    def _placed(
        self, resource: Resource, source: Path, reserved: Mapping[str, _Copy]
    ) -> _Copy:
        """Where the package holds the table of *resource* the bundle carries
        from *source*: where it lies, if the package lists it there; else at
        ``tables/<Type>/`` under its own name, or, where that name holds
        other content on disk or in *reserved* (the copies staged), under a
        name seeded from its content; equal content lying at a name is that
        copy."""
        folder = resource.names_folder
        entry: dict[str, Any] = {"type": resource.type}
        for key in ("media_type", "size"):
            if getattr(resource, key) is not None:
                entry[key] = getattr(resource, key)
        if resource.integrity is not None:
            entry["integrity"] = resource.integrity.to_dict()

        def placed(target: str, copied: bool) -> _Copy:
            url = quote(target + ("/" if folder else ""), safe="/.")
            return _Copy(
                resource.type, source, target, folder, copied, {**entry, "url": url}
            )

        root = None if self._path is None else self._path.parent
        if root is not None:
            try:
                inside = source.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                inside = None
            if inside is not None and any(
                table["type"] == resource.type
                and posixpath.normpath(unquote(urlsplit(table["url"]).path)) == inside
                for table in self._document["tables"]
            ):
                return placed(inside, copied=False)
        name = source.name
        stem = name if folder else Path(name).stem
        suffix = "" if folder else Path(name).suffix
        digest: str | None = None
        minter: IdMinter | None = None
        while True:
            target = f"{_TABLES_FOLDER}/{resource.type}/{name}"
            staged = reserved.get(target)
            lying = None if root is None else root / target
            if staged is None and (lying is None or not lying.exists()):
                return placed(target, copied=True)
            if digest is None:
                digest = _content_digest(source)
            if staged is not None:
                if _content_digest(staged.source) == digest:
                    return staged
            elif lying is not None and _content_digest(lying) == digest:
                return placed(target, copied=False)
            if minter is None:
                minter = IdMinter(seed_from(resource.type, digest))
            name = f"{minter.mint(stem)}{suffix}"

    def _disk_manifest_timelines(self, url: str) -> frozenset[str]:
        """The timelines of the manifest lying at *url* in the package on disk,
        none where none lies there."""
        if self._path is None or url not in self.manifests:
            return frozenset()
        held = self._on_disk.get(url)
        if held is None:
            path = resource_path(self._path.parent, url)
            held = (
                _manifest_timelines(loads(path.read_text(encoding="utf-8")))
                if path.is_file()
                else frozenset()
            )
            self._on_disk[url] = held
        return held

    def _files(self) -> Iterator[tuple[str, Path, str]]:
        """Each file of the package's tables on disk, once: its type, path and
        path from the package's root."""
        if self._path is None:
            return
        root = self._path.parent
        seen: set[Path] = set()
        for entry in self._document["tables"]:
            target = resource_path(root, entry["url"])
            if not target.exists():
                continue
            for file in table_files(target):
                resolved = file.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield entry["type"], file, file.relative_to(root).as_posix()

    def _refuse_held(self, written: _Written) -> None:
        """Refuse *written* where one of its IDs is held in the package by
        another manifest, before anything is written: on disk, every file's
        rows but those of the timelines the write removes; staged, the
        index of the bundles staged (:class:`_Staged`).

        Raises:
            ValueError: Naming the ID and both holders.
        """
        owners: dict[str, str] = {}
        if self._path is not None:
            for url in self.manifests:
                if url == written.url or url in self._pending:
                    continue
                for timeline_id in self._disk_manifest_timelines(url):
                    owners.setdefault(
                        timeline_id, f"a timeline of the manifest {url!r}"
                    )
        _refuse_held_ids(
            [
                (
                    timeline_id,
                    f"timeline {quote_id(timeline_id)} of the manifest {written.url!r}",
                )
                for timeline_id in sorted(written.timelines)
            ],
            owners,
            scope="the package",
        )
        # A package not written yet holds no file: its bundles' IDs are
        # checked against the index of those staged alone.
        removed = (
            frozenset().union(
                written.removed, *(w.removed for w in self._pending.values())
            )
            if self._path is not None
            else frozenset()
        )
        for claims in (False, True):
            column = "timeline_a_id" if claims else "timeline_id"
            parts: list[tuple[pa.Table, str]] = []
            for kind, file, name in self._files():
                if (kind == "MatchClaim") != claims:
                    continue
                schema = pq.read_schema(file)
                columns = [
                    c
                    for c in ("id", *(_CLAIM_TIMELINES if claims else (column,)))
                    if c in schema.names
                ]
                if "id" not in columns or column not in columns:
                    continue
                rows = pq.read_table(file, columns=columns)
                keep = pc.invert(
                    _rows_of(rows, _CLAIM_TIMELINES if claims else (column,), removed)
                )
                parts.append((rows.filter(keep).select(["id", column]), f"'{name}'"))
            ids, timelines = _id_columns(written, claims)
            _Ids.of(parts, column).refuse(
                ids, timelines, "MatchClaim" if claims else "event", written.url
            )
        self._staged.refuse(written)

    def _refuse_unless_succeeding(
        self, claims: MatchClaimField, before: frozenset[str]
    ) -> None:
        """Refuse *claims* unless they keep the MatchClaims the package holds
        for the manifest they replace (of its timelines *before*).

        Raises:
            ValueError: As :meth:`MatchClaimField.to_parquet` refuses a table
                not succeeding the one it is written over.
        """
        if self._path is None:
            return
        root = self._path.parent
        parts = [
            read_table(
                resource_path(root, entry["url"]), timelines=before, type="MatchClaim"
            )
            for entry in self._document["tables"]
            if entry["type"] == "MatchClaim"
            and resource_path(root, entry["url"]).exists()
        ]
        parts = [part for part in parts if part.num_rows]
        if not parts:
            return
        old = pa.concat_tables(parts, promote_options="default")
        claims._refuse_unless_succeeding(MatchClaimField.from_package_table(old))

    # Writing.

    def _write(self) -> None:
        """Write every bundle staged, then ``package.yaml``, then check the
        package whole; restore the package where anything fails.

        In order: the format version of an opened package is checked; the
        new files (one per type, :func:`_joined_tables`) are written under
        names starting with ``.``, which readers skip; the rows of the
        timelines replaced leave the files holding them; the new files are
        renamed into place; the tables the bundles carry are copied in and
        listed; the manifests and ``package.yaml`` are written; the package
        is checked. Where a step raises, every file changed
        gets its bytes back, every file and folder created is removed, and
        the package's document is as it was; the bundles stay staged.

        Raises:
            ValueError: If an opened package is of another format version
                (before anything is written), or the package check finds a
                problem, its first.
        """
        assert self._path is not None
        version = self._document.get("timetoshare")
        if version != timetoshare.SCHEMA_VERSION:
            raise ValueError(
                f"The package at '{self._path}' is of the format version "
                f"{version!r}, and the library writes {timetoshare.SCHEMA_VERSION!r}: "
                f"a package holds files of one version; nothing is written."
            )
        root = self._path.parent
        pending = list(self._pending.values())
        removed = frozenset().union(*(written.removed for written in pending))
        document = copy.deepcopy(self._document)
        entries = document["tables"]
        undo = _Undo()
        try:
            placed: list[tuple[Path, Path]] = []
            for kind, table in _joined_tables(pending).items():
                folder_url = f"{_TABLES_FOLDER}/{quote(kind)}/"
                folder = resource_path(root, folder_url)
                undo.folder(folder)
                final = folder / _file_name(kind, table, folder)
                hidden = folder / f".{final.name}"
                undo.keep(hidden)
                undo.keep(final)
                _write_package_file(table, hidden, _sorted_by(kind))
                placed.append((hidden, final))
                if not any(
                    entry["type"] == kind and entry["url"] == folder_url
                    for entry in entries
                ):
                    entries.append(
                        {
                            "type": kind,
                            "url": folder_url,
                            "media_type": PARQUET_MEDIA_TYPE,
                        }
                    )
            if removed:
                self._remove_rows(removed, undo)
            for hidden, final in placed:
                os.replace(hidden, final)
            copied: set[str] = set()
            for written in pending:
                for carried in written.carried.values():
                    if carried.target in copied:
                        continue
                    copied.add(carried.target)
                    if carried.copied:
                        _copy_into(carried, root, undo)
                    if not any(
                        entry["type"] == carried.type
                        and entry["url"] == carried.entry["url"]
                        for entry in entries
                    ):
                        entries.append(dict(carried.entry))
            by_url = {entry["url"]: entry for entry in document["manifests"]}
            for written in pending:
                target = resource_path(root, written.url)
                undo.folder(target.parent)
                undo.keep(target)
                target.write_text(written.text, encoding="utf-8", newline="")
                entry = by_url[written.url]
                if {"size", "integrity"} & entry.keys():
                    # A manifest rewritten states its new bytes.
                    data = target.read_bytes()
                    entry["size"] = len(data)
                    entry["integrity"] = Integrity.of_bytes(data).to_dict()
            undo.folder(root)
            undo.keep(self._path)
            self._path.write_text(
                timetoshare.dumps(document), encoding="utf-8", newline=""
            )
            problems = package_problems(self._path)
            if problems:
                problem = problems[0]
                row = "" if problem.row is None else f", row {problem.row}"
                raise ValueError(
                    f"The package check refuses the package written at "
                    f"'{self._path}': {problem.code} in '{problem.file}'{row}: "
                    f"{problem.message}; the package is restored as it was."
                )
        except BaseException:
            undo.restore()
            raise
        self._document = document
        for written in pending:
            self._on_disk[written.url] = written.timelines
            written.bundle.resources = list(written.resources)
            written.bundle._carried_tables = {
                resource_id: root / carried.target
                for resource_id, carried in written.carried.items()
            }
        self._pending.clear()
        self._staged = _Staged()

    def _remove_rows(self, removed: frozenset[str], undo: _Undo) -> None:
        """Remove the rows of the timelines *removed* from every file of the
        package holding one, each such file recorded in *undo*, rewritten
        whole and a file left without a row deleted; every other row is
        kept, stable-sorted by the columns the file declares
        (:func:`_sorted_by`: each timeline's rows keep their order). A
        MatchClaim file stating a provenance its rows share is rewritten
        with that provenance in each row kept, as the package writer writes
        provenance. A table of other rows (:func:`_holds_event_columns`),
        which a manifest carries, is left as it lies."""
        for kind, file, _ in list(self._files()):
            columns = _CLAIM_TIMELINES if kind == "MatchClaim" else ("timeline_id",)
            names = pq.read_schema(file).names
            present = [name for name in columns if name in names]
            if not present or not _holds_event_columns(kind, names):
                # A table of other rows is carried, never rewritten.
                continue
            mask = _rows_of(pq.read_table(file, columns=present), present, removed)
            if not pc.any(mask).as_py():
                continue
            undo.keep(file)
            shared = PROVENANCE_METADATA_KEY.encode() in (
                pq.read_schema(file).metadata or {}
            )
            table = (
                read_table(file, type=kind)
                if kind == "MatchClaim" and shared
                else pq.read_table(file)
            )
            kept = table.filter(pc.invert(mask))
            if kept.num_rows == 0:
                file.unlink()
                continue
            metadata = dict(table.schema.metadata or {})
            metadata.setdefault(TYPE_METADATA_KEY.encode(), kind.encode())
            _write_package_file(
                _sorted(kept.replace_schema_metadata(metadata), kind),
                file,
                _sorted_by(kind),
            )


def _content_digest(path: Path) -> str:
    """The SHA-256 digest of a file's bytes, or of a folder's files: each
    file under it, in sorted order, its path from the folder and its
    bytes."""
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        name = file.relative_to(path).as_posix().encode("utf-8")
        data = file.read_bytes()
        for part in (name, data):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def _copy_into(carried: _Copy, root: Path, undo: _Undo) -> None:
    """Copy the table *carried* into the package lying at *root*, every file
    and folder created recorded in *undo*.

    Raises:
        ValueError: If a file or folder lies at its place already (one
            written there since the bundle was staged).
    """
    target = root / carried.target
    if target.exists():
        raise ValueError(
            f"'{target}' lies in the package already: the {carried.type} table "
            f"'{carried.source}' carried by a manifest cannot be copied there; "
            f"nothing is written."
        )
    if not carried.folder:
        undo.folder(target.parent)
        undo.keep(target)
        shutil.copyfile(carried.source, target)
        return
    undo.folder(target)
    for item in sorted(carried.source.rglob("*")):
        place = target / item.relative_to(carried.source)
        if item.is_dir():
            undo.folder(place)
        else:
            undo.folder(place.parent)
            undo.keep(place)
            shutil.copyfile(item, place)


def _sorted(table: pa.Table, kind: str) -> pa.Table:
    """*table* stable-sorted by the columns a package file of *kind* declares
    (:func:`_sorted_by`), so each timeline's rows keep their order."""
    return table.take(
        pc.sort_indices(
            table, sort_keys=[(name, "ascending") for name in _sorted_by(kind)]
        )
    )


def _joined_tables(pending: Sequence[_Written]) -> dict[str, pa.Table]:
    """The table of each type a package writes for the bundles *pending*: the
    rows of every bundle holding the type, in one table, sorted
    (:func:`_sorted`). The timelines of two bundles are distinct, so each
    timeline's rows keep the order its bundle's table gives them; a further
    column joins the columns of the type in the order first met.

    Raises:
        ValueError: If two bundles hold a column of one name in two types,
            naming the column and both manifests.
    """
    by_kind: dict[str, list[tuple[str, pa.Table]]] = {}
    for written in pending:
        for kind, table in written.tables.items():
            by_kind.setdefault(kind, []).append((written.url, table))
    joined: dict[str, pa.Table] = {}
    for kind, parts in by_kind.items():
        schema = _joined_schema(parts[0][1].schema, parts, kind, holder="manifest")
        table = pa.concat_tables(
            [
                pa.Table.from_arrays(
                    [
                        (
                            part.column(field.name)
                            if field.name in part.column_names
                            else _null_column(part.num_rows, field.type)
                        )
                        for field in schema
                    ],
                    schema=schema,
                )
                for _, part in parts
            ]
        )
        joined[kind] = _sorted(table, kind)
    return joined


def _file_name(kind: str, table: pa.Table, folder: Path) -> str:
    """The name of a new file of *table* in *folder*: ``part_`` and 13 base-36
    characters minted from the table's content, drawn again where a file of
    that name, or of that name being written, lies there."""
    minter = content_minter(_PART_PREFIX, kind, table_seed_part(table))
    while True:
        name = f"{minter.mint(_PART_PREFIX)}.parquet"
        if not (folder / name).exists() and not (folder / f".{name}").exists():
            return name


# endregion
