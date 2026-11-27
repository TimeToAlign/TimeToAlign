"""An AlignmentBundle as a dictionary, as a manifest with its files, and in a package.

The bundle's manifest dictionary (:meth:`AlignmentBundle.to_dict`,
:meth:`AlignmentBundle.from_dict`), the tables written beside it, the merge
of several bundles into one (:meth:`AlignmentBundle.from_bundles`) and the
resources such a merge names; and :class:`Package`, a folder of manifests
sharing their tables, which ``package.yaml`` lists.
"""

from __future__ import annotations

import os
import posixpath
from collections.abc import Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlsplit

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
from timetoalign.core.fields import _filled
from timetoalign.core.ids import (
    _refuse_held_ids,
    content_minter,
    quote_id,
    table_seed_part,
)
from timetoalign.timelines import TimelineGroup
from timetoalign.timelines.groups import _held_ids
from timetoalign.timelines.manifest import (
    _NEW_FILE,
    PARQUET_MEDIA_TYPE,
    Integrity,
    Manifest,
    Resource,
    _timeline_ids,
    _validated_by_the_file,
    _write_package_file,
    event_file_name,
    read_event_table,
    read_projection_table,
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
    URL under its ID.

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
            if resource.is_table:
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


def _resource_table(resource: Resource, directory: Path) -> pa.Table | None:
    """The rows a table resource names for its manifest, or None where
    nothing lies at its URL.

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
    path = resource.path_in(directory)
    if not (path.is_dir() if resource.names_folder else path.is_file()):
        return None
    if resource.names_folder or _FILTERED.get():
        return read_table(path, timelines=resource.timelines, type=resource.type)
    return pq.read_table(path)


def _read_tables(
    self: AlignmentBundle, directory: Path, data: Mapping[str, Any]
) -> None:
    """Implement :meth:`AlignmentBundle._read_tables`."""
    tables = [
        (resource.type, table)
        for resource in self.resources
        if resource.is_table
        and (table := _resource_table(resource, directory)) is not None
    ]
    # Every resource of a type is read, in the order the manifest lists
    # them: the Projections first, then every other event type, then the
    # MatchClaims, whose side B may name a Projection.
    for table_type, table in tables:
        if table_type != "Projection":
            continue
        for timeline_id, projections in read_projection_table(table, data).items():
            self._require_known_timeline(timeline_id).add_projections(projections)
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
    for resource in self.resources:
        if resource.is_table and resource.names_folder:
            raise NotImplementedError(
                f"The resource {quote_id(resource.id)} names the folder "
                f"{resource.url!r} for its {resource.type} rows: "
                f"AlignmentBundle.to_file writes a manifest's tables as single "
                f"files beside it and does not write into a folder, whose files "
                f"other manifests may share; write the bundle into its package "
                f"with Package.write_bundle."
            )
    claims = self.match_claim_table
    event_tables = self._event_tables(events=events)
    if len(claims) == 0 and not event_tables:
        return False
    # The first entry of each table type names its table; a later entry of
    # a type written lists timelines whose rows that table now holds.
    named: dict[str, Resource] = {}
    for resource in self.resources:
        if resource.is_table:
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
    written_types = {table_type for table_type, *_ in tables}
    # An entry of a type not written stays where its file lies.
    kept = {
        id(r)
        for r in self.resources
        if r.is_table
        and r.type not in written_types
        and resource_path(directory, r.url).is_file()
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
class _Written:
    """A bundle as :meth:`Package.write_bundle` prepares it: what
    :meth:`Package.to_file` writes for it."""

    url: str
    bundle: AlignmentBundle
    text: str
    media_type: str
    tables: dict[str, pa.Table]
    resources: list[Resource]
    timelines: frozenset[str]
    removed: frozenset[str]


@dataclass(frozen=True)
class _Ids:
    """The IDs a package holds besides a bundle being written: its rows'
    (``id`` and the timeline each lies on) and each holder's description."""

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
                raise ValueError(
                    f"The {what} {quote_id(ids[again].as_py())} of timeline "
                    f"{quote_id(timelines[again].as_py())} cannot join the "
                    f"package: its ID is held by the {what} of timeline "
                    f"{quote_id(timelines[first].as_py())} of the same manifest "
                    f"{url!r}; nothing is renamed."
                )
        held = pc.is_in(ids, value_set=self.ids)
        if not pc.any(held).as_py():
            return
        at = pc.index(held, True).as_py()
        found = pc.index(self.ids, ids[at]).as_py()
        raise ValueError(
            f"The {what} {quote_id(ids[at].as_py())} of timeline "
            f"{quote_id(timelines[at].as_py())} of the manifest {url!r} cannot "
            f"join the package: its ID is held by the {what} of timeline "
            f"{quote_id(self.timelines[found].as_py())} in "
            f"{self.places[self.place[found].as_py()]}; nothing is renamed."
        )


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
        self, id: str | None = None, name: str | None = None
    ) -> None:  # noqa: A002
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
        self._on_disk: dict[str, frozenset[str]] = {}

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> Package:
        """Open the package whose ``package.yaml`` lies at *path*.

        The file is read in any serialization of the format and validated
        (:func:`timetoshare.validate_package`); its manifests and tables
        are read when asked for (:meth:`bundle`).

        Raises:
            FileNotFoundError: If no file lies at *path*.
            timetoshare.SchemaValidationError: If the file is no valid
                ``package.yaml``.
        """
        path = Path(path)
        data = loads(path.read_text(encoding="utf-8"))
        timetoshare.validate_package(data)
        package = cls.__new__(cls)
        package._document = data
        package._path = path
        package._pending = {}
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
        a resource that is no table (a recording) is kept. Its rows of every
        event type (:func:`~timetoalign.timelines.manifest.event_table`,
        Projections and Boxes among them) and its MatchClaims (each row with
        its own provenance) go into one new file per type,
        ``tables/<Type>/part_<13 base-36 characters>.parquet``, named from
        its content through the library's ID minter, so that a rebuild names
        its files again. A manifest the package holds at *url* already is
        replaced: the rows of its timelines and of the bundle's leave the
        files holding them, each such file rewritten whole (one left
        without a row removed), the other manifests' rows kept, before the
        new files are written; its MatchClaims must be kept by the bundle,
        as :meth:`MatchClaimField.to_parquet
        <timetoalign.alignment.MatchClaimField.to_parquet>` requires.

        Before anything is written, every ID is checked in memory: each
        event ID among every event of the package, every type; each
        MatchClaim ID among its MatchClaims; each timeline ID among the
        timelines of its other manifests. A collision is refused, naming
        the ID and both holders; nothing is renamed. A package lying on
        disk (:meth:`from_file`, or written by :meth:`to_file`) is written
        at once, ``package.yaml`` with it; a new one is written by
        :meth:`to_file`. Once written, the bundle's :attr:`resources
        <timetoalign.alignment.AlignmentBundle.resources>` are the
        manifest's.

        Args:
            bundle: The bundle.
            url: The manifest's location relative to the package's root,
                such as ``bach/manifest.yaml``.

        Raises:
            ValueError: If *url* leaves the package's root or names no file
                in it, an ID collides, the bundle cannot be written as a
                manifest, or its MatchClaims do not keep the ones the
                package holds for it, before anything is written; if the
                package check refuses the package written
                (:meth:`to_file`).
        """
        url = self._manifest_url(url)
        written = self._prepared(bundle, url)
        self._pending[url] = written
        entries = self._document["manifests"]
        if url not in self.manifests:
            entries.append({"url": url, "media_type": written.media_type})
        if self._path is not None:
            self._write()

    def to_file(self, path: str | os.PathLike[str]) -> Path:
        """Write the package: every bundle written into it, then ``package.yaml``.

        ``package.yaml`` is written in canonical YAML, every manifest listed
        explicitly and one table entry per type folder. Once written, the
        package is checked whole (:func:`timetoshare.tables.package_problems`),
        and its first problem is raised.

        Args:
            path: The ``package.yaml`` of a new package, in a folder holding
                none yet; a package opened or written before is written where
                it lies.

        Returns:
            The path written.

        Raises:
            ValueError: If *path* is not named ``package.yaml``, a new package
                would be written over a file lying there, a package lying
                elsewhere would be written at *path*, or the package check
                finds a problem.
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
        elif os.path.abspath(path) != os.path.abspath(self._path):
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
        resources = [resource for resource in bundle.resources if not resource.is_table]
        held: dict[str, str] = {}
        for resource in bundle.resources:
            if resource.is_table:
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
            removed=timelines | before,
        )
        self._refuse_held(written)
        if before and len(claims):
            self._refuse_unless_succeeding(claims, before)
        return written

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
        another manifest, before anything is written.

        Raises:
            ValueError: Naming the ID and both holders.
        """
        others = [w for url, w in self._pending.items() if url != written.url]
        owners: dict[str, str] = {}
        for url in self.manifests:
            if url == written.url or url in self._pending:
                continue
            for timeline_id in self._disk_manifest_timelines(url):
                owners.setdefault(timeline_id, f"a timeline of the manifest {url!r}")
        for other in others:
            for timeline_id in other.timelines:
                owners.setdefault(
                    timeline_id, f"a timeline of the manifest {other.url!r}"
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
        removed = frozenset().union(written.removed, *(w.removed for w in others))
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
            for other in others:
                for kind, table in other.tables.items():
                    if (kind == "MatchClaim") == claims:
                        parts.append(
                            (
                                table.select(["id", column]),
                                f"the manifest {other.url!r}",
                            )
                        )
            held = _Ids.of(parts, column)
            new = [
                table.select(["id", column])
                for kind, table in written.tables.items()
                if (kind == "MatchClaim") == claims
            ]
            if not new:
                continue
            joined = pa.concat_tables(new)
            held.refuse(
                joined.column("id").combine_chunks(),
                joined.column(column).combine_chunks(),
                "MatchClaim" if claims else "event",
                written.url,
            )

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
        """Write every bundle prepared, then ``package.yaml``, then check the
        package whole.

        Raises:
            ValueError: If the package check finds a problem, its first.
        """
        assert self._path is not None
        root = self._path.parent
        pending = list(self._pending.values())
        removed = frozenset().union(*(written.removed for written in pending))
        if removed:
            self._remove_rows(removed)
        entries = self._document["tables"]
        for written in pending:
            for kind, table in written.tables.items():
                folder_url = f"{_TABLES_FOLDER}/{quote(kind)}/"
                folder = resource_path(root, folder_url)
                folder.mkdir(parents=True, exist_ok=True)
                _write_package_file(
                    table, folder / _file_name(kind, table, folder), _sorted_by(kind)
                )
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
            target = resource_path(root, written.url)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(written.text, encoding="utf-8", newline="")
            for entry in self._document["manifests"]:
                if entry["url"] == written.url and {"size", "integrity"} & entry.keys():
                    # A manifest rewritten states its new bytes.
                    data = target.read_bytes()
                    entry["size"] = len(data)
                    entry["integrity"] = Integrity.of_bytes(data).to_dict()
            self._on_disk[written.url] = written.timelines
            written.bundle.resources = list(written.resources)
        self._pending.clear()
        self._path.write_text(
            timetoshare.dumps(self._document), encoding="utf-8", newline=""
        )
        problems = package_problems(self._path)
        if problems:
            problem = problems[0]
            row = "" if problem.row is None else f", row {problem.row}"
            raise ValueError(
                f"The package check refuses the package written at '{self._path}': "
                f"{problem.code} in '{problem.file}'{row}: {problem.message}"
            )

    def _remove_rows(self, removed: frozenset[str]) -> None:
        """Remove the rows of the timelines *removed* from every file of the
        package holding one, each such file rewritten whole and a file left
        without a row deleted; every other row is kept as it stands. A
        MatchClaim file stating a provenance its rows share is rewritten with
        that provenance in each row kept, as the package writer writes
        provenance."""
        for kind, file, _ in list(self._files()):
            columns = _CLAIM_TIMELINES if kind == "MatchClaim" else ("timeline_id",)
            names = pq.read_schema(file).names
            present = [name for name in columns if name in names]
            if not present:
                continue
            mask = _rows_of(pq.read_table(file, columns=present), present, removed)
            if not pc.any(mask).as_py():
                continue
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
                kept.replace_schema_metadata(metadata), file, _sorted_by(kind)
            )


def _file_name(kind: str, table: pa.Table, folder: Path) -> str:
    """The name of a new file of *table* in *folder*: ``part_`` and 13 base-36
    characters minted from the table's content, drawn again where a file of
    that name lies there."""
    minter = content_minter(_PART_PREFIX, kind, table_seed_part(table))
    while True:
        name = f"{minter.mint(_PART_PREFIX)}.parquet"
        if not (folder / name).exists():
            return name


# endregion
