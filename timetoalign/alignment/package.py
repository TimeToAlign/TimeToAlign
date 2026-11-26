"""An AlignmentBundle as a dictionary and as a package of files.

The bundle's manifest dictionary (:meth:`AlignmentBundle.to_dict`,
:meth:`AlignmentBundle.from_dict`), the tables written beside it, the merge
of several bundles into one (:meth:`AlignmentBundle.from_bundles`) and the
resources such a merge names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc
from pydantic import ValidationError

from timetoalign.alignment.claim_store import _refuse_shared_claim_ids
from timetoalign.alignment.claims import MatchClaimField
from timetoalign.core import Box
from timetoalign.core.ids import _refuse_held_ids, quote_id
from timetoalign.timelines import TimelineGroup
from timetoalign.timelines.engines.events import _refusal_of
from timetoalign.timelines.groups import _held_ids
from timetoalign.timelines.manifest import (
    BOXES_FILE_NAME,
    PROJECTIONS_FILE_NAME,
    TABLE_RESOURCE_TYPES,
    Manifest,
    Resource,
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
    MatchClaim table holds every source's MatchClaims, the Projections and
    Boxes lie on the timelines they travel with), and a manifest names at
    most one resource of each table type, so the first source's entry of a
    type names the merged table and a later source's entry of that type is
    not carried. :meth:`AlignmentBundle.to_file` writes the merged table at
    that entry's URL under its ID.

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
            if resource.type in TABLE_RESOURCE_TYPES:
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


def _read_tables(
    self: AlignmentBundle, directory: Path, data: Mapping[str, Any]
) -> None:
    """Implement :meth:`AlignmentBundle._read_tables`."""
    named = {r.type: r for r in self.resources if r.type in TABLE_RESOURCE_TYPES}
    if "Projection" in named:
        path = named["Projection"].path_in(directory)
        if path.is_file():
            for timeline_id, projections in read_projection_table(path, data).items():
                self._require_known_timeline(timeline_id).add_projections(projections)
    if "Box" in named:
        path = named["Box"].path_in(directory)
        if path.is_file():
            for timeline_id, rows in read_event_table(path).items():
                try:
                    boxes = [
                        Box.from_row(
                            row, unit=row["unit"], number_type=row["number_type"]
                        )
                        for row in rows.to_pylist()
                    ]
                except ValidationError as error:
                    # The Box's own refusal, as add_events gives it.
                    raise _refusal_of(error) from None
                self._require_known_timeline(timeline_id).add_events(boxes)
    if "MatchClaim" in named:
        path = named["MatchClaim"].path_in(directory)
        if path.is_file():
            self.add_match_claim_field(MatchClaimField.from_parquet(path))


def _write_tables(self: AlignmentBundle, directory: Path) -> bool:
    """Implement :meth:`AlignmentBundle._write_tables`."""
    claims = self.match_claim_table
    event_tables = self._event_tables()
    if len(claims) == 0 and not event_tables:
        return False
    named = {r.type: r for r in self.resources if r.type in TABLE_RESOURCE_TYPES}
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
        claim_table = claims.table
        named_timelines = pc.unique(
            pa.concat_arrays(
                [
                    claim_table.column("timeline_a_id").combine_chunks(),
                    claim_table.column("timeline_b_id").combine_chunks(),
                ]
            )
        )
        tables.append(
            (
                "MatchClaim",
                url,
                path,
                sorted(named_timelines.to_pylist()),
                claims.to_parquet,
            )
        )
    file_names = {"Projection": PROJECTIONS_FILE_NAME, "Box": BOXES_FILE_NAME}
    for kind, table in event_tables.items():
        url, path = place(kind, file_names[kind])
        holders = pc.unique(table.column("timeline_id")).to_pylist()
        tables.append(
            (
                kind,
                url,
                path,
                sorted(holders),
                lambda path, table=table: write_event_table(table, path),
            )
        )
    kept = {
        table_type
        for table_type, held in named.items()
        if resource_path(directory, held.url).is_file()
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
    resources = [
        (
            written.pop(r.type, r if r.type in kept else None)
            if r.type in TABLE_RESOURCE_TYPES
            else r
        )
        for r in self.resources
    ]
    self.resources = [r for r in resources if r is not None] + list(written.values())
    return True


def _to_dict(
    self: AlignmentBundle,
    *,
    minimal: bool = True,
    events: bool = False,
    external_references: bool = False,
) -> dict[str, Any]:
    """Implement :meth:`AlignmentBundle.to_dict`."""
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
        resources=self.resources,
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
