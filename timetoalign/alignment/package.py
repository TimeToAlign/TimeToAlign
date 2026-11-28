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
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote, urlsplit
from urllib.request import url2pathname

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import timetoshare
from timetoshare import loads
from timetoshare.tables import (
    ELEMENT_KINDS,
    EXTRACTION_SCHEMA,
    NON_EVENT_KINDS,
    PACKAGE_FILE,
    PROVENANCE_METADATA_KEY,
    REGISTRY_SCHEMA,
    TYPE_METADATA_KEY,
    check_extractions,
    check_projections,
    check_registry,
    package_problems,
    read_table,
)
from timetoshare.tables import resolve as timetoshare_resolve
from timetoshare.tables import (
    table_files,
    write_non_event_rows,
    write_registry,
)
from timetoshare.tables.package import non_event_rows_of, table_non_event_rows
from timetoshare.validation.diagnostics import manifest_ids

from timetoalign.alignment.claim_store import _refuse_shared_claim_ids
from timetoalign.alignment.claims import MatchClaimField
from timetoalign.core import NumberType, TimeUnit
from timetoalign.core.fields import _filled
from timetoalign.core.ids import (
    IdMinter,
    _refuse_held_ids,
    content_minter,
    quote_id,
    seed_from,
    table_seed_part,
)
from timetoalign.storage.registry import Extraction, extraction_table
from timetoalign.timelines import TimelineGroup
from timetoalign.timelines.manifest import (
    _NEW_FILE,
    ALWAYS_WRITTEN_KINDS,
    EXTRACTIONS_FILE_NAME,
    PARQUET_MEDIA_TYPE,
    REGISTRY_FILE_NAME,
    EventTables,
    Integrity,
    Manifest,
    Resource,
    _timeline_ids,
    _validated_by_the_file,
    _write_package_file,
    content_in_memory,
    distinct_strings,
    event_tables,
    joined_parts,
    non_event_table,
    pre_parsed_file_name,
    pre_parsed_schema,
    registry_rows_in_memory,
    resource_path,
)

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle


def _merged_resources(
    bundles: Sequence["AlignmentBundle"], merged: "AlignmentBundle"
) -> list[Resource]:
    """The resources a merge of *bundles* into *merged* names, every id kept.

    Every source's resources are carried as they stand, in source order,
    with one exception: the merge holds one table of each type (its
    MatchClaim table holds every source's MatchClaims, its registry the
    events that travel with the timelines), and the library writes a
    bundle's rows of a table type into one table, at the first entry of
    that type (:func:`_write_tables`). So the first entry of a type, in
    source order, names the merged table, and every later entry of that
    type, of the same source or of a later one, is not carried.
    :meth:`AlignmentBundle.to_file` writes the merged table at that entry's
    URL under its ID. A table a source carries as it stands (of a type the
    bundle does not read) is carried like any other resource.

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
    taken: dict[str, str] = {}
    for bundle in bundles:
        # Every ID of the bundle's trees and groups; the bundle's own ID is
        # not carried into the merge.
        itself = f"bundle {quote_id(bundle.id)}"
        for held_id, holder in bundle._scope_ids().items():
            if held_id != bundle.id or holder != itself:
                taken.setdefault(held_id, holder)
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
    merged._shared_tables = frozenset().union(
        *(bundle._shared_tables for bundle in bundles)
    ) & {resource.id for resource in merged.resources}
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


#: The table types a bundle reads into its timelines and writes from them;
#: a table resource of any other type is carried as it stands.
_READ_TYPES = ("Registry", "MatchClaim")


def _registry_folder(url: str) -> ValueError:
    """The refusal of a registry whose URL *url* names a folder."""
    return ValueError(
        f"The registry {url!r} names a folder: a registry is one Parquet file, "
        f"which holds every row."
    )


def _package_root(manifest: Path) -> tuple[Path, bool]:
    """The root of the package the manifest file *manifest* lies in, and
    whether a ``package.yaml`` lists it: the nearest folder above it whose
    ``package.yaml`` lists it, else its own folder (a manifest alone)."""
    resolved = manifest.resolve()
    for folder in resolved.parents:
        listing = folder / PACKAGE_FILE
        if not listing.is_file():
            continue
        try:
            document = loads(listing.read_text(encoding="utf-8"))
        except ValueError:
            continue
        for entry in document.get("manifests") or ():
            url = entry.get("url") if isinstance(entry, Mapping) else None
            if not isinstance(url, str):
                continue
            try:
                listed = resource_path(folder, url)
            except ValueError:
                continue
            if listed.resolve() == resolved:
                return folder, True
    return resolved.parent, False


def _location_path(root: Path, location: str) -> Path | None:
    """The path a registry row's *location* names in the package lying at
    *root*, or None for an absolute URI (a resource outside the package)."""
    if urlsplit(location).scheme:
        return None
    return resource_path(root, location)


def _reference(path: Path, root: Path) -> str:
    """*path* as a source a package lying at *root* names: its path from
    *root* (a URI reference resolved against it) where it lies under it,
    else its ``file:`` URI."""
    absolute = path.resolve()
    try:
        inside = absolute.relative_to(root.resolve())
    except ValueError:
        return absolute.as_uri()
    return quote(inside.as_posix(), safe="/")


def _file_path(text: str) -> Path | None:
    """The local path a source *text* names (a path, or a ``file:`` URI),
    None for a URI of another scheme."""
    parts = urlsplit(text)
    if parts.scheme and len(parts.scheme) > 1:
        if parts.scheme != "file" or parts.netloc not in ("", "localhost"):
            return None
        return Path(url2pathname(parts.path))
    return Path(text)


def _source_written(source: str | None, root: Path) -> str | None:
    """An extraction's *source* as a package lying at *root* (a manifest
    alone: its folder) writes it, so that the package names no path of the
    machine that wrote it where its source lies in it.

    A file lying under *root* is written as its path from *root*, a URI
    reference resolved against it; any other file as its ``file:`` URI; a
    member of a ZIP archive (``jar:<archive URI>!/<member>``) keeps its
    member, its archive written by the same rule; a URI of any other scheme
    as it stands. A source as the library holds it is a path (absolute, or
    relative to the current folder) or a ``jar:`` URI
    (:meth:`Extraction.of_source <timetoalign.storage.Extraction.of_source>`).

    Examples:
        >>> from pathlib import Path
        >>> root = Path("/data/package")
        >>> _source_written("/data/package/scores/a b.mid", root)
        'scores/a%20b.mid'
        >>> _source_written("/elsewhere/a.mid", root)
        'file:///elsewhere/a.mid'
        >>> _source_written("jar:file:///data/package/MTD.zip!/MTD/x.mid", root)
        'jar:MTD.zip!/MTD/x.mid'
    """
    if source is None:
        return None
    if source.startswith("jar:"):
        archive, mark, member = source.removeprefix("jar:").partition("!/")
        path = _file_path(archive)
        if path is None or not mark:
            return source
        return f"jar:{_reference(path, root)}{mark}{member}"
    path = _file_path(source)
    return source if path is None else _reference(path, root)


def _source_read(source: str | None, root: Path) -> str | None:
    """An extraction's *source* read from a package lying at *root*, as the
    library holds a source (the inverse of :func:`_source_written`): a
    reference resolved against *root*, a ``file:`` URI as its path, an
    archive's member with its archive's ``file:`` URI; a URI of any other
    scheme as it stands.

    Examples:
        >>> from pathlib import Path
        >>> root = Path("/data/package")
        >>> _source_read("scores/a%20b.mid", root)
        '/data/package/scores/a b.mid'
        >>> _source_read("file:///elsewhere/a.mid", root)
        '/elsewhere/a.mid'
        >>> _source_read("jar:MTD.zip!/MTD/x.mid", root)
        'jar:file:///data/package/MTD.zip!/MTD/x.mid'
    """
    if source is None:
        return None
    if source.startswith("jar:"):
        archive, mark, member = source.removeprefix("jar:").partition("!/")
        if not mark or urlsplit(archive).scheme:
            return source
        return f"jar:{(root / unquote(archive)).as_uri()}{mark}{member}"
    parts = urlsplit(source)
    if parts.scheme and len(parts.scheme) > 1:
        path = _file_path(source)
        return source if path is None else path.as_posix()
    return (root / unquote(source)).as_posix()


def _extraction_written(extraction: Extraction, root: Path) -> Extraction:
    """*extraction* with its source as a package lying at *root* writes it."""
    source = _source_written(extraction.source, root)
    return (
        extraction
        if source == extraction.source
        else replace(extraction, source=source)
    )


def _extraction_read(extraction: Extraction, root: Path) -> Extraction:
    """*extraction* read from a package lying at *root*, its source as the
    library holds it."""
    source = _source_read(extraction.source, root)
    return (
        extraction
        if source == extraction.source
        else replace(extraction, source=source)
    )


def _read_extractions(
    registries: Sequence[Path], root: Path, named: Sequence[str], what: str
) -> dict[str, Extraction]:
    """The extractions *named*, read from the Extraction tables the
    non-event rows of the registries lying at *registries* (every one the
    manifest names) locate, each by ID.

    Raises:
        ValueError: If an extraction is located by no non-event row, or its
            table holds no row of it.
        timetoshare.tables.TableError: If an Extraction table breaks a rule.
    """
    if not named:
        return {}
    located = pa.concat_tables(
        [read_table(path, ids=named, type="Registry") for path in registries],
        promote_options="default",
    )
    located = located.filter(
        _filled(
            pc.equal(located.column("kind"), pa.scalar("Extraction")), pa.scalar(False)
        )
    )
    found: dict[str, Extraction] = {}
    for location in distinct_strings(located.column("location")):
        path = _location_path(root, location)
        if path is None or not path.exists():
            continue
        ids = located.filter(pc.equal(located.column("location"), pa.scalar(location)))
        table = read_table(
            path, ids=distinct_strings(ids.column("id")), type="Extraction"
        )
        check_extractions(table)
        for row in table.to_pylist():
            found.setdefault(
                row["id"], _extraction_read(Extraction.from_row(row), root)
            )
    missing = [extraction_id for extraction_id in named if extraction_id not in found]
    if missing:
        raise ValueError(
            f"The events of {what} name the extraction {quote_id(missing[0])}, which "
            f"no Extraction table its non-event rows locate holds: an event names the "
            f"extraction that made it."
        )
    return found


def _content_reader(
    kind: str,
    entries: list[tuple[Path, pa.Array]],
    unit: Any,
    number_type: Any,
    manifest: Mapping[str, Any] | None,
) -> Callable[[], pa.Table]:
    """The reader of the content of one family's events of *kind*: each
    pre-parsed table of *entries* read by the IDs given, when first asked, a
    Projection table checked (against *manifest*, where given)."""

    def read() -> pa.Table:
        parts = []
        for path, ids in entries:
            table = read_table(path, ids=ids, type=kind)
            if kind == "Projection":
                check_projections(table, manifest)
            parts.append(
                content_in_memory(
                    table, unit, number_type, f"{kind} table '{path.name}'"
                )
            )
        read = (
            parts[0]
            if len(parts) == 1
            else pa.concat_tables(parts, promote_options="default")
        )
        # A column the family's events of the kind hold no value in is
        # another family's (the table spans them); a declared one is kept.
        declared = set(pre_parsed_schema(kind).names)
        return read.select(
            [
                name
                for name in read.column_names
                if name in declared or read.column(name).null_count < read.num_rows
            ]
        )

    return read


def _install_registry(
    bundle: AlignmentBundle,
    resource: Resource,
    path: Path,
    root: Path,
    manifest: Mapping[str, Any],
    filtered: bool,
    registries: Sequence[Path],
) -> bool:
    """Install the event rows of the registry *resource* names (lying at
    *path*) in the families of the bundle's timelines (``_read_tables``),
    their extractions located by the non-event rows of *registries*.

    Each family takes its rows through its registry
    (:meth:`EventRegistry.add_rows <timetoalign.storage.EventRegistry.add_rows>`),
    which passes over a row the manifest states already, holds each
    timeline long enough for its rows and checks every Box against its
    shape; its content is read from the pre-parsed tables the rows locate
    when the family's events are first read.

    Returns:
        Whether a row is of a kind other than Projection and Box.
    """
    what = f"the registry '{path.name}'"
    rows = read_table(
        path, timelines=resource.timelines if filtered else None, type="Registry"
    )
    events = rows.filter(pc.invert(_is_non_event(rows)))
    check_registry(events, manifest)
    if not events.num_rows:
        return False
    memory = registry_rows_in_memory(events, what)
    named = sorted(
        {
            *distinct_strings(memory.column("extraction_id")),
            *distinct_strings(memory.column("vanished_in")),
        }
    )
    extractions = _read_extractions(registries, root, named, what)
    timelines = memory.column("timeline_id").combine_chunks()
    held = pc.unique(timelines)
    roots = [bundle._root_of(str(timeline_id)) for timeline_id in held.to_pylist()]
    root_ids = pa.array([family.id for family in roots], pa.string())
    family_of = root_ids.take(pc.index_in(timelines, value_set=held))
    for family in {family.id: family for family in roots}.values():
        own = memory.filter(pc.equal(family_of, pa.scalar(family.id)))
        located: dict[str, list[tuple[Path, pa.Array]]] = {}
        stored = own.filter(own.column("location").is_valid())
        for (kind, location), where in _pairs(stored):
            target = _location_path(root, location)
            if target is None:
                # An absolute URI names content outside the package.
                continue
            if not target.exists():
                first = stored.column("id")[int(where[0])].as_py()
                raise ValueError(
                    f"The {kind} {quote_id(first)} of {what} names the location "
                    f"{location!r}, where no table lies in the package."
                )
            ids = stored.column("id").combine_chunks().take(pa.array(where, pa.int64()))
            located.setdefault(kind, []).append((target, ids))
        contents = {
            kind: _content_reader(
                kind,
                entries,
                family.unit,
                family.number_type,
                # In a package a Projection may stand for an event of
                # another manifest's timeline: the package check's to see.
                None if filtered else manifest,
            )
            for kind, entries in located.items()
        }
        own_named = {
            *distinct_strings(own.column("extraction_id")),
            *distinct_strings(own.column("vanished_in")),
        }
        family.registry.add_rows(
            own,
            contents,
            [extractions[key] for key in sorted(own_named)],
            registered=True,
        )
    kinds = set(distinct_strings(memory.column("kind")))
    return bool(kinds - set(ALWAYS_WRITTEN_KINDS))


def _pairs(rows: pa.Table) -> list[tuple[tuple[str, str], np.ndarray]]:
    """The positions of *rows* by their ``kind`` and ``location``, each pair once."""
    if not rows.num_rows:
        return []
    keys = pa.Table.from_arrays(
        [
            rows.column("kind"),
            rows.column("location"),
            pa.array(np.arange(rows.num_rows), pa.int64()),
        ],
        names=["kind", "location", "at"],
    )
    grouped = keys.group_by(["kind", "location"], use_threads=False).aggregate(
        [("at", "list")]
    )
    return [
        ((str(kind), str(location)), np.asarray(at, dtype=np.int64))
        for kind, location, at in zip(
            grouped.column("kind").to_pylist(),
            grouped.column("location").to_pylist(),
            grouped.column("at_list").to_pylist(),
            strict=True,
        )
    ]


def _read_tables(self: AlignmentBundle, path: Path, data: Mapping[str, Any]) -> bool:
    """Implement :meth:`AlignmentBundle._read_tables`."""
    directory = path.parent
    root, listed = _package_root(path)
    carried: dict[str, Path] = {}
    read: list[tuple[Resource, Path]] = []
    for resource in self.resources:
        if not resource.is_table:
            continue
        if resource.type == "Registry" and resource.names_folder:
            raise _registry_folder(resource.url)
        target = resource.path_in(directory)
        if not (target.is_dir() if resource.names_folder else target.is_file()):
            continue
        if resource.type not in _READ_TYPES:
            carried[resource.id] = Path(os.path.abspath(target))
            continue
        read.append((resource, target))
    self._carried_tables = carried
    # A package's manifest shares its registry and MatchClaim table with the
    # package's other manifests.
    self._shared_tables = frozenset(
        resource.id
        for resource in self.resources
        if listed and resource.is_table and resource.type in _READ_TYPES
    )
    others = False
    registries = [target for resource, target in read if resource.type == "Registry"]
    for resource, target in read:
        if resource.type == "Registry":
            others |= _install_registry(
                self, resource, target, root, data, listed, registries
            )
    # The MatchClaims last: a side may name a Projection.
    for resource, target in read:
        if resource.type == "MatchClaim":
            table = (
                read_table(target, timelines=resource.timelines, type="MatchClaim")
                if listed or resource.names_folder
                else pq.read_table(target)
            )
            self.add_match_claim_field(MatchClaimField.from_package_table(table))
    return others


def _held_entry(tabled: Sequence[Resource], table_type: str) -> Resource | None:
    """The first entry of *table_type* among *tabled*, the one naming its table."""
    return next((resource for resource in tabled if resource.type == table_type), None)


def _non_event_rows_of(
    manifest: Mapping[str, Any],
    location: str,
    claims: tuple[pa.Array, str] | None,
    extractions: tuple[pa.Array, str] | None,
) -> pa.Table:
    """The non-event rows of a manifest written (in the file form) at
    *location*, of its MatchClaims and of its extractions (their IDs and
    their table's URL), in the Registry's columns, sorted by ``location``,
    then ``id``."""
    parts = [non_event_rows_of(timetoshare.with_bare_ids(manifest), location)]
    for given, kind in ((claims, "MatchClaim"), (extractions, "Extraction")):
        if given is not None:
            parts.append(table_non_event_rows(given[0], kind, given[1]))
    joined = pa.concat_tables(parts).sort_by(
        [("location", "ascending"), ("id", "ascending")]
    )
    return non_event_table(joined)


def _kept_extractions(tables: EventTables, target: Path, root: Path) -> pa.Table:
    """The Extraction table of *tables* to write at *target*, for a manifest
    alone in the folder *root*: each source written as that folder holds it
    (:func:`_source_written`); an extraction the table lying there holds
    under its ID, of the same content but its date, is that extraction,
    written with the date it was first written with; every other as the
    bundle holds it."""
    held: dict[str, Extraction] = {}
    if target.is_file():
        for row in read_table(target, type="Extraction").to_pylist():
            held.setdefault(row["id"], Extraction.from_row(row))
    rows = []
    for extraction in tables.extractions:
        extraction = _extraction_written(extraction, root)
        kept = held.get(extraction.id)
        same = kept is not None and _undated(kept) == _undated(extraction)
        rows.append(kept if kept is not None and same else extraction)
    table = extraction_table(rows).cast(EXTRACTION_SCHEMA)
    return table.replace_schema_metadata({TYPE_METADATA_KEY: "Extraction"})


def _stale_pre_parsed(registry: Path, root: Path, written: set[Path]) -> list[Path]:
    """The pre-parsed files the registry lying at *registry* (a manifest's
    alone in the folder *root*) locates that the write replacing it does not
    write (*written*): the files of the kinds the new registry no longer
    holds, which nothing would name once it is written. A file outside the
    folder is never one of them."""
    if not registry.is_file():
        return []
    names = pq.read_schema(registry).names
    if "location" not in names or "kind" not in names:
        return []
    rows = pq.read_table(registry, columns=["kind", "location"])
    rows = rows.filter(pc.invert(_is_non_event(rows)))
    stale: list[Path] = []
    for location in distinct_strings(rows.column("location")):
        target = _location_path(root, location)
        if target is None or target in written or not target.is_file():
            continue
        if not target.resolve().is_relative_to(root.resolve()):
            continue
        stale.append(target)
    return stale


def _write_tables(self: AlignmentBundle, path: Path, *, events: bool = False) -> bool:
    """Implement :meth:`AlignmentBundle._write_tables`."""
    directory = path.parent
    tables = event_tables(self.registry, events=events)
    claims = self.match_claim_table
    with_events = tables.rows.num_rows > 0
    kinds = {
        *(("Registry",) if with_events else ()),
        *(("MatchClaim",) if len(claims) else ()),
    }
    # A table carried as it stands names no table written.
    tabled = [
        resource
        for resource in self.resources
        if resource.is_table and resource.id not in self._carried_tables
    ]
    for resource in tabled:
        if resource.type not in kinds:
            continue
        if resource.type == "Registry" and resource.names_folder:
            raise _registry_folder(resource.url)
        if resource.names_folder or resource.id in self._shared_tables:
            held = "folder" if resource.names_folder else "package table"
            raise NotImplementedError(
                f"The resource {quote_id(resource.id)} names the {held} "
                f"{resource.url!r} for its {resource.type} rows: "
                f"AlignmentBundle.to_file writes a manifest's tables as single "
                f"files beside it and does not write into a package's tables, "
                f"which other manifests may share; write the bundle into its "
                f"package with Package.write_bundle."
            )
    if not kinds:
        return False
    taken = {
        **self._scope_ids(),
        **{r.id: f"resource {quote_id(r.id)}" for r in self.resources},
    }
    planned: dict[str, Resource] = {}
    for kind, default, timelines in (
        ("Registry", REGISTRY_FILE_NAME, tables.timelines),
        ("MatchClaim", MatchClaimField.file_name, _claim_timelines(claims.table)),
    ):
        if kind not in kinds:
            continue
        held = _held_entry(tabled, kind)
        url = held.url if held is not None else quote(default)
        if held is not None and set(held.timelines) == set(timelines):
            timelines = list(held.timelines)
        if held is None:
            resource_id = Resource._minted_id(kind, timelines, url)
            taken = _refuse_held_ids(
                [(resource_id, f"resource {quote_id(resource_id)}")], taken
            )
        else:
            resource_id = held.id
        planned[kind] = Resource(
            resource_id, kind, tuple(timelines), url, PARQUET_MEDIA_TYPE
        )
    # An entry of a type not written stays where its file lies; a folder of
    # such a type, and a table carried, stay as they are.
    kept = {id(r) for r in self.resources if r.id in self._carried_tables} | {
        id(r)
        for r in tabled
        if r.type not in kinds
        and (r.names_folder or resource_path(directory, r.url).is_file())
    }

    def resources_with(written: Mapping[str, Resource]) -> list[Resource]:
        """The bundle's resources, each entry written in place of the first
        entry of its type, the new ones last."""
        pending = dict(written)
        resources: list[Resource] = []
        for r in self.resources:
            if not r.is_table or id(r) in kept:
                resources.append(r)
            elif r is _held_entry(tabled, r.type) and r.type in pending:
                resources.append(pending.pop(r.type))
        return resources + list(pending.values())

    files: list[tuple[Path, pa.Table, tuple[str, ...]]] = []
    if with_events:
        urls = {kind: quote(pre_parsed_file_name(kind)) for kind in tables.contents}
        rows = tables.located(urls)
        extraction_url = quote(EXTRACTIONS_FILE_NAME)
        with _validated_by_the_file():
            manifest = _to_dict(self, minimal=False, resources=resources_with(planned))
        claim_ids = (
            (claims.table.column("id").combine_chunks(), planned["MatchClaim"].url)
            if "MatchClaim" in planned
            else None
        )
        extraction_ids = pa.array(
            [extraction.id for extraction in tables.extractions], pa.string()
        )
        non_events = _non_event_rows_of(
            manifest,
            quote(path.name),
            claim_ids,
            (extraction_ids, extraction_url) if len(extraction_ids) else None,
        )
        registry = pa.concat_tables([non_events, rows]).replace_schema_metadata(
            {TYPE_METADATA_KEY: "Registry"}
        )
        files.append((resource_path(directory, planned["Registry"].url), registry, ()))
        if len(extraction_ids):
            target = resource_path(directory, extraction_url)
            files.append((target, _kept_extractions(tables, target, directory), ()))
        for kind, content in tables.contents.items():
            files.append((resource_path(directory, urls[kind]), content, ()))
    stale = (
        _stale_pre_parsed(
            resource_path(directory, planned["Registry"].url),
            directory,
            {target for target, _, _ in files},
        )
        if with_events
        else []
    )
    undo = _Undo()
    try:
        for target in stale:
            undo.keep(target)
            target.unlink()
        hidden: list[tuple[Path, Path]] = []
        for target, table, sorted_by in files:
            undo.folder(target.parent)
            veiled = target.with_name(f".{target.name}")
            undo.keep(veiled)
            undo.keep(target)
            _write_package_file(table, veiled, sorted_by)
            hidden.append((veiled, target))
        if "MatchClaim" in planned:
            claim_path = resource_path(directory, planned["MatchClaim"].url)
            undo.folder(claim_path.parent)
            undo.keep(claim_path)
            claims.to_parquet(claim_path)
        for veiled, target in hidden:
            os.replace(veiled, target)
    except BaseException:
        undo.restore()
        raise
    self.resources = resources_with(
        {
            kind: Resource.of_file(
                resource_path(directory, resource.url),
                id=resource.id,
                type=kind,
                timelines=list(resource.timelines),
                url=resource.url,
            )
            for kind, resource in planned.items()
        }
    )
    return True


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

#: The URL of a package's registry in ``package.yaml``: one file at its root.
_REGISTRY_URL = REGISTRY_FILE_NAME

#: The URL of a package's Extraction table in ``package.yaml``.
_EXTRACTIONS_URL = f"{_TABLES_FOLDER}/{EXTRACTIONS_FILE_NAME}"

#: The table types of a package that are no pre-parsed table.
_PACKAGE_TYPES = frozenset({"Registry", "Extraction", "MatchClaim"})

#: The columns naming a row's timelines in a MatchClaim table.
_CLAIM_TIMELINES = ("timeline_a_id", "timeline_b_id")

#: The sorting columns of a written MatchClaim file.
_CLAIM_ORDER = ("timeline_a_id", "timeline_b_id", "id")


def _sorted_by(kind: str) -> tuple[str, ...]:
    """The columns a package file of *kind* is sorted by and declares: a
    MatchClaim file's timelines and IDs, none for a pre-parsed table."""
    return _CLAIM_ORDER if kind == "MatchClaim" else ()


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
    return mask.combine_chunks() if isinstance(mask, pa.ChunkedArray) else mask


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
        rows: Its registry's event rows, each pre-parsed row's location the
            folder of its kind's pre-parsed table.
        contents: Its pre-parsed table of each kind.
        extractions: The extractions its rows name.
        claims: Its MatchClaim table as a package file holds it, or None.
        removed: The timelines whose rows the write removes from the files
            on disk: the manifest's and the one it replaces; none for a
            package not written yet.
        carried: Each table the bundle carries, by its resource ID.
    """

    url: str
    bundle: AlignmentBundle
    text: str
    media_type: str
    rows: pa.Table
    contents: dict[str, pa.Table]
    extractions: tuple[Extraction, ...]
    claims: pa.Table | None
    resources: list[Resource]
    timelines: frozenset[str]
    removed: frozenset[str]
    carried: dict[str, _Copy]


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
    if claims:
        if written.claims is None:
            return pa.array([], pa.string()), pa.array([], pa.string())
        table, column = written.claims, "timeline_a_id"
    else:
        table, column = written.rows, "timeline_id"
    return (
        table.column("id").combine_chunks(),
        table.column(column).combine_chunks(),
    )


@dataclass(frozen=True)
class _Level:
    """One sorted run of the IDs an :class:`_Ids` holds.

    Attributes:
        ids: The IDs, sorted (code point order, Arrow's and Python's alike),
            as a NumPy array of strings for binary search.
        timelines: Row for row, the timeline each lies on.
        place: Row for row, the index of its holder's place.
    """

    ids: np.ndarray
    timelines: pa.Array
    place: np.ndarray

    @classmethod
    def of(cls, ids: pa.Array, timelines: pa.Array, place: pa.Array) -> _Level:
        """The level of *ids* (with their timelines and places), sorted."""
        order = pc.sort_indices(ids)
        return cls(
            np.asarray(ids.take(order).to_numpy(zero_copy_only=False), dtype=object),
            timelines.take(order),
            np.asarray(place.take(order).to_numpy(zero_copy_only=False)),
        )

    def __len__(self) -> int:
        return len(self.ids)


class _Ids:
    """IDs held in a package besides a bundle being written, each with the
    timeline it lies on and its holder's place: those of the files on disk,
    or of the bundles staged.

    The IDs are held in sorted levels, whose sizes at least halve from one
    to the next: adding a holder's IDs adds a level and joins the last
    levels while one is at most twice the size of the next, so each ID is
    sorted again a logarithmic number of times, and refusing a bundle's IDs
    searches each level once (a binary search per ID), whatever the number
    of holders. Removing a holder filters every level.
    """

    def __init__(self) -> None:
        self._levels: list[_Level] = []
        self._places: list[str] = []
        self._place_of: dict[str, int] = {}

    @classmethod
    def of(cls, parts: list[tuple[pa.Table, str]], column: str) -> _Ids:
        """The IDs of *parts*, each a table of ``id`` and *column* with the
        place its rows lie in, in one level."""
        held = cls()
        if not parts:
            return held
        places = [
            pa.repeat(pa.scalar(held._place(where), pa.int64()), table.num_rows)
            for table, where in parts
        ]
        held._push(
            _Level.of(
                pa.concat_arrays(
                    [table.column("id").combine_chunks() for table, _ in parts]
                ),
                pa.concat_arrays(
                    [table.column(column).combine_chunks() for table, _ in parts]
                ),
                pa.concat_arrays(places),
            )
        )
        return held

    def _place(self, where: str) -> int:
        """The index of the place *where*, held from now on."""
        if where not in self._place_of:
            self._place_of[where] = len(self._places)
            self._places.append(where)
        return self._place_of[where]

    def _push(self, level: _Level) -> None:
        """Hold *level*, joining the last levels while one is not more than
        twice the next."""
        if not len(level):
            return
        self._levels.append(level)
        while len(self._levels) > 1 and len(self._levels[-2]) <= 2 * len(
            self._levels[-1]
        ):
            last, before = self._levels.pop(), self._levels.pop()
            self._levels.append(
                _Level.of(
                    pa.array(np.concatenate([before.ids, last.ids]), pa.string()),
                    pa.concat_arrays([before.timelines, last.timelines]),
                    pa.array(np.concatenate([before.place, last.place]), pa.int64()),
                )
            )

    def add(self, ids: pa.Array, timelines: pa.Array, where: str) -> None:
        """Hold *ids*, each on the timeline of *timelines*, held at *where*."""
        place = pa.repeat(pa.scalar(self._place(where), pa.int64()), len(ids))
        self._push(_Level.of(ids, timelines, place))

    def remove(self, where: str) -> None:
        """Hold none of the IDs held at *where* any more."""
        place = self._place_of.get(where)
        if place is None:
            return
        levels = self._levels
        self._levels = []
        for level in levels:
            kept = level.place != place
            if kept.all():
                self._levels.append(level)
            elif kept.any():
                self._levels.append(
                    _Level(
                        level.ids[kept],
                        level.timelines.filter(pa.array(kept)),
                        level.place[kept],
                    )
                )

    def refuse(
        self,
        ids: pa.Array,
        timelines: pa.Array,
        what: str,
        url: str,
        *,
        twice: bool = True,
    ) -> None:
        """Refuse *ids* (each on the timeline of *timelines*, of the manifest
        at *url*) where one is held here at another place than the manifest
        at *url*, or (with *twice*) two of them are one.

        Raises:
            ValueError: Naming the first such ID and both holders.
        """
        if twice and len(ids) > 1:
            order = pc.sort_indices(ids).to_numpy()
            ordered = ids.take(order)
            repeated = pc.equal(ordered.slice(1), ordered.slice(0, len(ordered) - 1))
            if pc.any(repeated).as_py():
                at = pc.index(repeated, True).as_py()
                first, again = sorted((int(order[at]), int(order[at + 1])))
                raise _collision(
                    what,
                    ids[again].as_py(),
                    timelines[again].as_py(),
                    url,
                    timelines[first].as_py(),
                    f"the same manifest {url!r}",
                )
        if not self._levels or not len(ids):
            return
        own = np.asarray(ids.to_numpy(zero_copy_only=False), dtype=object)
        itself = self._place_of.get(f"the manifest {url!r}")
        found: tuple[int, _Level, int] | None = None
        for level in self._levels:
            at = np.searchsorted(level.ids, own)
            near = np.minimum(at, len(level) - 1)
            hit = (at < len(level)) & (level.ids[near] == own)
            if itself is not None:
                hit &= level.place[near] != itself
            hits = np.flatnonzero(hit)
            if hits.size and (found is None or hits[0] < found[0]):
                found = (int(hits[0]), level, int(near[hits[0]]))
        if found is None:
            return
        first, level, holder = found
        raise _collision(
            what,
            ids[first].as_py(),
            timelines[first].as_py(),
            url,
            level.timelines[holder].as_py(),
            self._places[int(level.place[holder])],
        )


class _Staged:
    """The IDs the bundles a package stages hold, kept as each is staged:
    staging a bundle checks its own IDs alone, against these, never the
    staged bundles' tables again.

    Attributes:
        owners: The manifest URL owning each timeline ID staged.
        ids: Of the events (``False``) and of the MatchClaims (``True``),
            each ID staged with its timeline, held at the place
            ``the manifest '<url>'`` (:class:`_Ids`).
    """

    def __init__(self) -> None:
        self.owners: dict[str, str] = {}
        self.ids: dict[bool, _Ids] = {False: _Ids(), True: _Ids()}
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
            ids, timelines = _id_columns(written, claims)
            self.ids[claims].refuse(
                ids,
                timelines,
                "MatchClaim" if claims else "event",
                written.url,
                twice=False,
            )

    def add(self, written: _Written) -> _Written | None:
        """Stage *written*, in place of the bundle staged at its URL, which is
        returned (None where none was)."""
        previous = self.remove(written.url)
        self._staged[written.url] = written
        self.owners.update(dict.fromkeys(written.timelines, written.url))
        for claims in (False, True):
            ids, timelines = _id_columns(written, claims)
            self.ids[claims].add(ids, timelines, f"the manifest {written.url!r}")
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
            self.ids[claims].remove(f"the manifest {url!r}")
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
    its tables: the **registry** (``registry.parquet`` at the root, as this
    class writes it: one Parquet file holding every row, the non-event rows,
    one per manifest element, MatchClaim and extraction, then the event
    rows, one per event, each position on the axis of its timeline's root),
    the **Extraction table**
    (``tables/extractions.parquet``), one **pre-parsed table** per kind of
    event (``tables/<Kind>/``, each event's content keyed by its ID, without
    positions) and the **MatchClaim table** (``tables/MatchClaim/``). A
    manifest names the registry and the MatchClaim table in its
    ``resources``, with the timelines whose rows it keeps; nothing in a file
    names a manifest. An event ID is held once among all events of the
    package, every kind; a MatchClaim ID once among its MatchClaims; a
    timeline ID, as this class writes a package, by one manifest.

    :meth:`from_file` opens a package; :meth:`bundle` reads one of its
    manifests as an :class:`~timetoalign.alignment.AlignmentBundle`, every
    table filtered by the timelines its resources list and the events'
    content read when first asked for. :meth:`write_bundle` writes a bundle
    into the package as one of its manifests, and :meth:`to_file` writes a
    new package where it is told.

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
        >>> package.manifests
        ['piece/manifest.yaml']
        >>> [(table["type"], table["url"]) for table in package.tables]
        [('Registry', 'registry.parquet'), ('Extraction', 'tables/extractions.parquet'), ('Note', 'tables/Note/')]
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
        left hidden beside the registry (its name with one or two ``.``
        before it) or in a folder of ``tables/`` (a name starting with ``.``
        and ending in ``.parquet``), which no reader reads, is removed.

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
                    for stray in folder.glob(".*.parquet"):
                        stray.unlink()
        for entry in data.get("tables") or ():
            if entry.get("type") != "Registry" or entry["url"].endswith("/"):
                continue
            registry = _location_path(path.parent, entry["url"])
            if registry is None:
                continue
            for hidden in (f".{registry.name}", f"..{registry.name}"):
                registry.with_name(hidden).unlink(missing_ok=True)
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
        <timetoalign.alignment.AlignmentBundle.from_file>` reads a manifest
        that ``package.yaml`` lists: every table its resources name, a
        single file as a folder, is read filtered, its rows of the timelines
        the resource lists (a MatchClaim's either of whose timelines it
        lists), the other manifests' passed over; each event's content is
        read from its pre-parsed table, by ID, when its family's events are
        first read. A manifest read alone instead refuses a row of a
        timeline it lacks in a single file.

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
        return AlignmentBundle.from_file(resource_path(self._path.parent, url))

    def write_bundle(self, bundle: AlignmentBundle, url: str) -> None:
        """Write *bundle* into the package as its manifest at *url*.

        The manifest lies at *url* under the package's root, its resources
        naming the registry (``registry.parquet`` at the package's root, or
        where ``package.yaml`` places it, relative to the manifest, without
        size or digest, since other manifests' rows share it) with the timelines
        holding its events, and the MatchClaim table (``tables/MatchClaim/``)
        with the timelines its MatchClaims name; a resource that is no table
        (a recording) is kept. A table the bundle carries (a table resource
        of a type it does not read, or a MatchClaim table it holds no rows
        of) is copied into the package with its bytes, at
        ``tables/<Type>/`` under its own name (a name holding other content:
        one seeded from the table's content), listed in ``package.yaml``,
        and its resource names the copy; one lying in the package already
        stays where it lies, and equal content lying at its name is that
        copy. The format reads such a table as the pre-parsed table of its
        type, keyed by ``id``: a Parquet table without an ``id`` column of
        strings is the package check's ``foreign_table``, so writing the
        package raises it and restores the package, while a folder holding
        no Parquet file (scans, recordings) is carried. A table resource of
        a type the bundle holds no rows of and does not carry is refused:
        the writer has no file to copy.

        Its events of every kind (:func:`~timetoalign.timelines.manifest.event_tables`)
        go into the registry, one file holding the rows of every manifest,
        rewritten whole with each write in the order of the format
        (``timetoshare.tables.write_registry``: the non-event rows by
        ``location`` and ``id``, then the event rows by ``timeline_id``,
        exact ``start`` and ``id``). Their extractions go into the Extraction table
        (``tables/extractions.parquet``, each extraction once, its source
        written as a path from the package's root where the file lies under
        it, else as its ``file:`` URI, so that the package names no path of
        the machine that wrote it; read back, the source is the file's path
        again), their
        content into the pre-parsed table of their kind and their
        MatchClaims (each row with its own provenance) into the MatchClaim
        table: each write adds one new file per kind and one of MatchClaims,
        ``tables/<Type>/part_<13 base-36 characters>.parquet``, holding the
        rows of every bundle it writes (:meth:`to_file` writes every bundle
        staged at once), named from its content through the library's ID
        minter, so that a rebuild names its files again; a column two
        bundles hold in two types is held in the one type holding both
        exactly (an ``int8`` and an ``int64`` column as ``int64``), never
        in a type that may change a value. Last, the
        registry's non-event rows are written from the manifests and tables
        (``timetoshare.tables.write_non_event_rows``, which rewrites the
        registry whole once more, its event rows kept). A manifest the package holds
        at *url* already is replaced: the rows of its timelines and of the
        bundle's leave the files holding them (a pre-parsed row by the ID
        of its event), each such file rewritten whole (one left without a
        row removed), the other manifests' rows kept; its MatchClaims must
        be kept by the bundle, as :meth:`MatchClaimField.to_parquet
        <timetoalign.alignment.MatchClaimField.to_parquet>` requires, so a
        bundle holding none while the package holds MatchClaims of that
        manifest is refused, and an extraction once written stays in the
        Extraction table.

        Before anything is written, every ID is checked in memory: each
        event ID among every event of the package, every kind; each
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
                or an opened package is of another format version or names
                a folder for its registry, before anything is written; if
                the package check refuses the package written, with
                everything restored.
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
        tables = event_tables(bundle.registry, events=True)
        claims = bundle.match_claim_table
        claim_table = _claim_file_table(claims) if len(claims) else None
        written_types = {"Registry", *(("MatchClaim",) if len(claims) else ())}
        folder = posixpath.dirname(url) or "."
        carried = self._carried(bundle, written_types, url)
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
        named = {
            "Registry": (unquote(urlsplit(self._registry_url()).path), tables.timelines)
        }
        if claim_table is not None:
            named["MatchClaim"] = (
                f"{_TABLES_FOLDER}/MatchClaim/",
                _claim_timelines(claim_table),
            )
        for kind, (target, timelines) in named.items():
            relative = posixpath.relpath(target, folder)
            location = quote(
                relative + ("/" if kind == "MatchClaim" else ""), safe="/."
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
            rows=tables.located(
                {kind: f"{_TABLES_FOLDER}/{quote(kind)}/" for kind in tables.contents}
            ),
            contents=dict(tables.contents),
            extractions=tables.extractions,
            claims=claim_table,
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

    def _registry_url(self) -> str:
        """The URL of the package's registry: its entry's in ``package.yaml``,
        else ``registry.parquet`` at the root.

        Raises:
            ValueError: If the entry names a folder.
        """
        url = next(
            (
                entry["url"]
                for entry in self._document["tables"]
                if entry["type"] == "Registry"
            ),
            _REGISTRY_URL,
        )
        if url.endswith("/"):
            raise _registry_folder(url)
        return url

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
                if kind != ("MatchClaim" if claims else "Registry"):
                    continue
                schema = pq.read_schema(file)
                wanted = ("id", *(_CLAIM_TIMELINES if claims else (column, "kind")))
                columns = [c for c in wanted if c in schema.names]
                if "id" not in columns or column not in columns:
                    continue
                rows = pq.read_table(file, columns=columns)
                keep = pc.invert(
                    _rows_of(rows, _CLAIM_TIMELINES if claims else (column,), removed)
                )
                if not claims and "kind" in columns:
                    # The non-event rows hold the IDs of elements, MatchClaims
                    # and extractions, not events.
                    keep = pc.and_(keep, pc.invert(_is_non_event(rows)))
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
        """Write every bundle staged, then ``package.yaml``, then the
        registry's non-event rows, then check the package whole; restore the package
        where anything fails.

        In order: the format version of an opened package is checked; the
        registry is written whole, losing the event rows of the timelines
        replaced and taking the bundles' in (:meth:`_registry_events`), then
        the Extraction
        table (:meth:`_extraction_table`), then one new file per pre-parsed
        kind and one of MatchClaims (:func:`_joined_tables`), each under a
        name starting with ``.``, which readers skip; the rows of the
        timelines replaced leave the pre-parsed and MatchClaim files holding
        them (:meth:`_remove_rows`); the new files are renamed into place;
        the tables the bundles carry are copied in and listed; the
        manifests and ``package.yaml`` are written; the non-event rows are
        written from them (``timetoshare.tables.write_non_event_rows``); the package
        is checked. Where a step raises, every file changed gets its bytes
        back, every file and folder created is removed, and the package's
        document is as it was; the bundles stay staged.

        Raises:
            ValueError: If an opened package is of another format version,
                or an extraction is held under its ID with other content
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

        def listed(kind: str, url: str) -> None:
            if not any(
                entry["type"] == kind and entry["url"] == url for entry in entries
            ):
                entries.append(
                    {"type": kind, "url": url, "media_type": PARQUET_MEDIA_TYPE}
                )

        registry_url = self._registry_url()
        listed("Registry", registry_url)
        extractions = self._extraction_table(pending, entries, root)
        undo = _Undo()
        try:
            placed: list[tuple[Path, Path]] = []
            removed_ids = self._registry_events(
                pending, resource_path(root, registry_url), removed, undo, placed
            )
            if extractions is not None:
                url, table = extractions
                listed("Extraction", url)
                target = resource_path(root, url)
                if url.endswith("/"):
                    undo.folder(target)
                    target = target / _file_name("Extraction", table, target)
                undo.folder(target.parent)
                hidden = target.with_name(f".{target.name}")
                undo.keep(hidden)
                undo.keep(target)
                _write_package_file(table, hidden, ())
                placed.append((hidden, target))
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
                listed(kind, folder_url)
            if removed:
                self._remove_rows(removed, removed_ids, undo)
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
            write_non_event_rows(root)
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
            written.bundle._shared_tables = frozenset(
                resource.id
                for resource in written.resources
                if resource.type in _READ_TYPES and resource.id not in written.carried
            )
            written.bundle._carried_tables = {
                resource_id: root / carried.target
                for resource_id, carried in written.carried.items()
            }
        self._pending.clear()
        self._staged = _Staged()

    def _extraction_table(
        self,
        pending: Sequence[_Written],
        entries: Sequence[Mapping[str, Any]],
        root: Path,
    ) -> tuple[str, pa.Table] | None:
        """The Extraction table to write for the bundles *pending*, at its
        URL: for a file (``tables/extractions.parquet`` as this class writes
        it), every extraction it holds and every new one; for a folder, a new
        part of the new ones alone; None where no extraction is new.

        Raises:
            ValueError: If an extraction is held, on disk or by another
                bundle, under its ID with other content (its date aside),
                naming it; nothing is written then.
        """
        url = next(
            (entry["url"] for entry in entries if entry["type"] == "Extraction"),
            _EXTRACTIONS_URL,
        )
        target = resource_path(root, url)
        held: dict[str, Extraction] = {}
        if target.exists():
            for row in read_table(target, type="Extraction").to_pylist():
                held.setdefault(row["id"], Extraction.from_row(row))
        new: dict[str, Extraction] = {}
        for written in pending:
            for extraction in written.extractions:
                extraction = _extraction_written(extraction, root)
                known = held.get(extraction.id) or new.get(extraction.id)
                if known is None:
                    new[extraction.id] = extraction
                elif _undated(known) != _undated(extraction):
                    raise ValueError(
                        f"The extraction {quote_id(extraction.id)} of the manifest "
                        f"{written.url!r} cannot join the package: the package "
                        f"holds an extraction of that ID of other content; "
                        f"nothing is written."
                    )
        if not new:
            return None
        rows = (
            list(new.values()) if url.endswith("/") else [*held.values(), *new.values()]
        )
        table = extraction_table(rows).cast(EXTRACTION_SCHEMA)
        return url, table.replace_schema_metadata({TYPE_METADATA_KEY: "Extraction"})

    @staticmethod
    def _registry_events(
        pending: Sequence[_Written],
        target: Path,
        removed: frozenset[str],
        undo: _Undo,
        placed: list[tuple[Path, Path]],
    ) -> pa.Array:
        """Write the registry file lying at *target* for the bundles *pending*.

        The file keeps every row it holds but the event rows of the timelines
        *removed*; the bundles' event rows join them; the whole is written by
        ``timetoshare.tables.write_registry``, in the order of the format,
        under the hidden name of *target* (recorded in *placed*, renamed into
        place later). Its non-event rows are kept as they stand until
        ``write_non_event_rows`` rewrites them from the manifests written.

        Returns:
            The IDs of the event rows removed, whose pre-parsed rows go too.
        """
        held = (
            read_table(target, type="Registry").replace_schema_metadata(None)
            if target.is_file()
            else REGISTRY_SCHEMA.empty_table()
        )
        gone = _flat(
            pc.and_(
                _rows_of(held, ("timeline_id",), removed),
                pc.invert(_is_non_event(held)),
            )
        )
        removed_ids = pc.filter(held.column("id").combine_chunks(), gone)
        table = pa.concat_tables(
            [
                held.filter(pc.invert(gone)),
                *(written.rows for written in pending if written.rows.num_rows),
            ],
            promote_options="default",
        )
        undo.folder(target.parent)
        undo.keep(target)
        hidden = target.with_name(f".{target.name}")
        undo.keep(hidden)
        write_registry(table, hidden)
        placed.append((hidden, target))
        return removed_ids

    def _remove_rows(
        self, removed: frozenset[str], removed_ids: pa.Array, undo: _Undo
    ) -> None:
        """Remove the rows of the timelines *removed* from every MatchClaim
        file of the package holding one, and the rows of the events
        *removed_ids* from every pre-parsed file holding one, each such file
        recorded in *undo*, rewritten whole and a file left without a row
        deleted; every other row is kept in its order. A MatchClaim file
        stating a provenance its rows share is rewritten with that
        provenance in each row kept, as the package writer writes
        provenance. The registry is :meth:`_registry_events`'; the
        Extraction table keeps every extraction."""
        for kind, file, _ in list(self._files()):
            if kind in ("Registry", "Extraction"):
                continue
            names = pq.read_schema(file).names
            if kind == "MatchClaim":
                present = [name for name in _CLAIM_TIMELINES if name in names]
                if not present:
                    continue
                mask = _rows_of(pq.read_table(file, columns=present), present, removed)
            else:
                if "id" not in names or not len(removed_ids):
                    continue
                mask = _filled(
                    pc.is_in(
                        pq.read_table(file, columns=["id"]).column("id"),
                        value_set=removed_ids,
                    ),
                    pa.scalar(False),
                )
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


def _undated(extraction: Extraction) -> dict[str, Any]:
    """*extraction*'s row but its date: what two runs of one extraction share."""
    row = extraction.to_row()
    del row["date"]
    return row


def _flat(array: Any) -> pa.Array:
    """*array* as one array, its chunks joined."""
    return array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array


def _is_non_event(rows: pa.Table) -> pa.Array:
    """Row for row, whether a registry row is a non-event row (its ``kind``
    one of the registry's non-event kinds)."""
    return _filled(
        pc.is_in(rows.column("kind"), value_set=pa.array(NON_EVENT_KINDS, pa.string())),
        pa.scalar(False),
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
    (:func:`_sorted_by`), so each timeline's rows keep their order; a
    pre-parsed table keeps its order."""
    keys = _sorted_by(kind)
    if not keys:
        return table
    return table.take(
        pc.sort_indices(table, sort_keys=[(name, "ascending") for name in keys])
    )


def _joined_tables(pending: Sequence[_Written]) -> dict[str, pa.Table]:
    """The pre-parsed table of each kind and the MatchClaim table a package
    writes for the bundles *pending*: the rows of every bundle holding one,
    in one table (:func:`~timetoalign.timelines.manifest.joined_parts`, a
    column two bundles hold in two integer widths widened losslessly),
    sorted (:func:`_sorted`). The timelines of two bundles are distinct, so
    each timeline's rows keep the order its bundle's table gives them; a
    further column joins the first bundle's in the order first met.

    Raises:
        ValueError: If two bundles hold a column of one name in two types no
            type holds both of exactly, naming the column and both
            manifests.
    """
    by_kind: dict[str, list[tuple[str, pa.Table]]] = {}
    for written in pending:
        tables = dict(written.contents)
        if written.claims is not None:
            tables["MatchClaim"] = written.claims
        for kind, table in tables.items():
            by_kind.setdefault(kind, []).append(
                (f"the manifest {written.url!r}", table)
            )
    return {
        kind: _sorted(joined_parts(parts, kind), kind)
        for kind, parts in by_kind.items()
    }


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


# region Resolving


def _local_path(target: str | os.PathLike[str]) -> Path:
    """The local path a manifest or package root *target* names: a path, or
    a ``file:`` URL.

    Raises:
        ValueError: If *target* is a URL of another scheme.
    """
    if isinstance(target, str):
        parts = urlsplit(target)
        if parts.scheme == "file":
            if parts.netloc not in ("", "localhost"):
                raise ValueError(
                    f"{target!r} names a file on the host {parts.netloc!r}; "
                    f"resolve reads local files"
                )
            return Path(url2pathname(parts.path))
        if parts.scheme and len(parts.scheme) > 1:
            raise ValueError(
                f"{target!r} is a {parts.scheme}: URL; resolve reads a local "
                f"manifest or package root, a path or a file: URL"
            )
    return Path(target)


#: The element kinds :func:`_resolve` answers with a library object; every
#: other element is answered with its object in the manifest.
_ELEMENT_OBJECTS = frozenset({"Manifest", "Timeline", "Group", "Resource", "Region"})


def _element(
    bundle: AlignmentBundle, manifest: Mapping[str, Any], kind: str, row: Any, id: str
) -> Any:
    """The element *id* of *kind* of the manifest *bundle* was read from."""
    if kind == "Manifest":
        return bundle
    if kind == "Timeline":
        return bundle.get_timeline(id)
    if kind == "Group":
        return bundle.groups[id]
    if kind == "Resource":
        return next(resource for resource in bundle.resources if resource.id == id)
    if kind == "Region":
        holder = None if row is None else row.get("timeline_id")
        if holder is not None:
            return bundle.get_timeline(holder).get_region(id)
    for identifier, path in manifest_ids(manifest):
        if identifier == id:
            element: Any = manifest
            for step in path[:-1]:
                element = element[step]
            return element
    raise KeyError(f"The manifest holds no element {quote_id(id)}.")


def _event_scalar(row: Mapping[str, Any], content: Mapping[str, Any] | None) -> Any:
    """The event of a registry row (in the package form) and its pre-parsed
    content, as its scalar: positions on the axis of its timeline's root, as
    the registry states them."""
    from timetoalign import core

    kind = row["kind"]
    scalar = getattr(core, kind, None)
    if not (isinstance(scalar, type) and issubclass(scalar, core.Event)):
        scalar = core.Event
    memory = registry_rows_in_memory(
        pa.Table.from_pylist([dict(row)], schema=REGISTRY_SCHEMA), "the registry"
    ).to_pylist()[0]
    values: dict[str, Any] = {**(content or {}), **memory}
    if issubclass(scalar, core.Projection):
        # A Projection reads coordinate cells, which name their own axis.
        for name in ("start", "duration"):
            if values.get(name) is not None:
                values[name] = {
                    **values[name],
                    "unit": row["unit"],
                    "number_type": row["number_type"],
                }
        return core.Projection.from_row(values)
    return scalar.from_row(values, unit=row["unit"], number_type=row["number_type"])


def _resolve(target: str | os.PathLike[str], id: str) -> list[Any]:
    """Implement :func:`timetoalign.alignment.resolve`."""
    from timetoalign.alignment.bundle import AlignmentBundle

    path = _local_path(target)
    found = timetoshare_resolve(path, id)
    bare = timetoshare.ids.from_file_id(id) if id.startswith("#") else id
    root = path if path.is_dir() else _package_root(path)[0]
    bundles: dict[str, tuple[AlignmentBundle, Mapping[str, Any]]] = {}

    def bundle_at(location: str) -> tuple[AlignmentBundle, Mapping[str, Any]]:
        if location not in bundles:
            file = resource_path(root, location)
            manifest = timetoshare.with_bare_ids(
                loads(file.read_text(encoding="utf-8"))
            )
            bundles[location] = (AlignmentBundle.from_file(file), manifest)
        return bundles[location]

    answers: list[Any] = []
    for resolution in found:
        kind, location, row = resolution.kind, resolution.location, resolution.row
        if kind in ELEMENT_KINDS:
            bundle, manifest = bundle_at(resolution.manifest or location or "")
            answers.append(_element(bundle, manifest, kind, row, bare))
            continue
        assert row is not None
        if kind == "MatchClaim":
            table = read_table(
                resource_path(root, str(location)), ids=[bare], type="MatchClaim"
            )
            answers.extend(MatchClaimField.from_package_table(table).to_claims())
        elif kind == "Extraction":
            table = read_table(
                resource_path(root, str(location)), ids=[bare], type="Extraction"
            )
            answers.extend(
                _extraction_read(Extraction.from_row(item), root)
                for item in table.to_pylist()
            )
        else:
            content = None
            where = None if location is None else _location_path(root, location)
            if where is not None:
                table = read_table(where, ids=[bare], type=kind)
                if table.num_rows:
                    table = content_in_memory(
                        table,
                        TimeUnit(row["unit"]),
                        NumberType[row["number_type"]],
                        f"{kind} table '{where.name}'",
                    )
                    content = {
                        key: value
                        for key, value in table.to_pylist()[0].items()
                        if key != "id"
                    }
            answers.append(_event_scalar(row, content))
    return answers


# endregion
