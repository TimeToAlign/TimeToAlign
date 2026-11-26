"""The MatchStamp table of an AlignmentBundle.

:meth:`~timetoalign.alignment.bundle.AlignmentBundle.get_matchstamp_table`
builds one table of MatchStamps, one row per position, one column per
timeline (and C-Map), as an Arrow table of coordinate cells or a DataFrame.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa
from timetoshare import DEFAULT_BELIEFS, BeliefState

from timetoalign.alignment.claim_store import _belief_key
from timetoalign.alignment.claims import MatchClaim, MatchClaimField
from timetoalign.alignment.filters import belief_context, believed
from timetoalign.alignment.stamps import MatchIntervalStamp, MatchStamp
from timetoalign.core import Coordinate, Interval, NumberType, TimeUnit
from timetoalign.core.retrieval import (
    CoordinateCollection,
    CoordinateInput,
    KeyCollection,
    TableFormat,
    reject_dataframe_options,
    validate_table_format,
)
from timetoalign.core.time import CoordinateValue, cell_coordinate
from timetoalign.core.timestamp import (
    ConversionMapsSpec,
    TimestampColumn,
    build_timestamp_table,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.core.enums import ColumnNaming
    from timetoalign.maps.base import ConversionMap


def _table_anchors(
    rows: MatchClaimField,
) -> list[tuple[str, Coordinate, str, Coordinate]]:
    """The alignment anchors of a MatchClaim table, read column by column.

    Per row its start anchor, then its end anchor where it has one; a row
    without anchors gives none. Each coordinate is read from its cell
    (:func:`~timetoalign.core.time.cell_coordinate`) in the number type the
    cell declares, on its row's timeline of that side; no claim is
    materialised.

    Returns:
        ``(timeline_a_id, coordinate_a, timeline_b_id, coordinate_b)`` per
        anchor, in row order.
    """
    data = rows._struct_array()
    anchors: list[tuple[str, Coordinate, str, Coordinate]] = []
    for timeline_a_id, timeline_b_id, start, end in zip(
        data.field("timeline_a_id").to_pylist(),
        data.field("timeline_b_id").to_pylist(),
        data.field("start_anchor").to_pylist(),
        data.field("end_anchor").to_pylist(),
        strict=True,
    ):
        for anchor in (start, end):
            if anchor is None:
                continue
            anchors.append(
                (
                    timeline_a_id,
                    cell_coordinate(anchor["coordinate_a"]),
                    timeline_b_id,
                    cell_coordinate(anchor["coordinate_b"]),
                )
            )
    return anchors


def _get_matchstamp_table(
    self: AlignmentBundle,
    at: (
        CoordinateInput
        | Interval
        | CoordinateCollection
        | Sequence[CoordinateInput | Interval]
        | str
        | KeyCollection
        | None
    ) = None,
    timeline_id: str | None = None,
    *,
    claims: list[MatchClaim] | None = None,
    within: set[str] | None = None,
    from_graph: bool = False,
    conversion_maps: "ConversionMapsSpec" = False,
    format: TableFormat = "table",
    fields: "ColumnNaming | Callable[[str, dict], str] | list[str] | None" = None,
    units: bool | None = None,
    believer: str | None = None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
) -> "pa.Table | pd.DataFrame":
    """Implement :meth:`AlignmentBundle.get_matchstamp_table`."""
    validate_table_format(format)
    reject_dataframe_options(format, fields=fields, units=units)
    if at is not None:
        if claims is not None:
            raise ValueError("Pass either claims or at, not both")
        resolved = self.get_matchstamp(
            at,
            timeline_id,
            conversion_maps=conversion_maps,
            believer=believer,
            beliefs=beliefs,
        )
        stamps = resolved if isinstance(resolved, list) else [resolved]
        if any(isinstance(stamp, MatchIntervalStamp) for stamp in stamps):
            raise NotImplementedError(
                "get_matchstamp_table does not support Interval queries; "
                "there is no interval table form"
            )
        match_stamps: list[MatchStamp] = []
        for stamp in stamps:
            assert isinstance(stamp, MatchStamp)
            ambiguous = stamp._ambiguous_interval_candidates()
            if ambiguous:
                raise ValueError(
                    "get_matchstamp_table cannot materialize ambiguous "
                    "interval-derived coordinates for timelines "
                    f"{list(ambiguous)}"
                )
            match_stamps.append(stamp)
        rows = [dict(stamp.coordinates) for stamp in match_stamps]
        all_tl_ids: set[str] = set()
        for row in rows:
            all_tl_ids.update(row)
        return self._render_matchstamp_table(
            self._assemble_matchstamp_table(
                rows,
                all_tl_ids,
                within,
                conversion_maps=conversion_maps,
            ),
            format=format,
            fields=fields,
            units=units,
        )

    anchors = (
        _table_anchors(self._claims_in_force(_belief_key(believer, beliefs)))
        if claims is None
        else [
            (anchor.timeline_a_id, anchor.coordinate_a)
            + (anchor.timeline_b_id, anchor.coordinate_b)
            for claim in believed(
                claims, believer, beliefs, context=belief_context(self)
            )
            for anchor in claim.anchors
        ]
    )
    if from_graph:
        rows = self._collapsed_claim_rows(anchors, list(self._known_timelines()))
    else:
        rows = [
            {timeline_a: coordinate_a, timeline_b: coordinate_b}
            for timeline_a, coordinate_a, timeline_b, coordinate_b in anchors
        ]
    all_tl_ids = set()
    for row in rows:
        all_tl_ids.update(row)

    return self._render_matchstamp_table(
        self._assemble_matchstamp_table(
            rows,
            all_tl_ids,
            within,
            conversion_maps=conversion_maps,
        ),
        format=format,
        fields=fields,
        units=units,
    )


def _assemble_matchstamp_table(
    self: AlignmentBundle,
    rows: list[dict[str, Coordinate]],
    all_tl_ids: set[str],
    within: set[str] | None,
    conversion_maps: "ConversionMapsSpec" = False,
) -> "pa.Table":
    """Implement :meth:`AlignmentBundle._assemble_matchstamp_table`."""
    n_rows = len(rows)
    if n_rows == 0:
        return pa.table({})

    # Keep only the timelines ``within`` names, if given
    if within is not None:
        all_tl_ids = all_tl_ids & within

    # Build table with consistent fields, scattering each row into its
    # cells rather than probing every field of every row.
    known = self._known_timelines()
    rank = {tl_id: position for position, tl_id in enumerate(known)}
    field_lists: dict[str, list[Coordinate | None]] = {
        tl_id: [None] * n_rows
        for tl_id in sorted(
            all_tl_ids, key=lambda tl_id: (rank.get(tl_id, len(rank)), tl_id)
        )
    }
    for position, row in enumerate(rows):
        for tl_id, value in row.items():
            column = field_lists.get(tl_id)
            if column is not None:
                column[position] = value

    # Each timeline's derived columns, written after the timeline's own.
    derived_columns: dict[str, dict[str, list[Coordinate | None]]] = {}
    if conversion_maps is not False and conversion_maps is not None:
        from timetoalign.core.timestamp import _conversion_map_enabled_for_spec

        collected: list[tuple[TimeUnit, str, "ConversionMap[Any]"]] = []
        for tl_id in list(field_lists.keys()):
            for cmap in self._get_conversion_maps_for_timeline(tl_id):
                target_unit = cmap.target_unit
                if target_unit is None:
                    continue  # numeric unit conversions only in tables
                if not _conversion_map_enabled_for_spec(cmap, conversion_maps):
                    continue
                collected.append((target_unit, tl_id, cmap))
        label_counts: dict[str, int] = {}
        for target_unit, _tl_id, _cmap in collected:
            label_counts[target_unit.value] = label_counts.get(target_unit.value, 0) + 1
        for target_unit, tl_id, cmap in collected:
            label = target_unit.value
            col_name = label if label_counts[label] == 1 else f"{tl_id}:{label}"
            source_timeline = self.get_timeline(tl_id)
            target_type = target_unit.resolve_number_type(
                source_timeline.number_type
                if source_timeline.number_type in target_unit.allowed_number_types
                else None
            )
            derived: list[Coordinate | None] = []
            for coordinate in field_lists[tl_id]:
                if coordinate is None:
                    derived.append(None)
                    continue
                try:
                    derived.append(
                        Coordinate(
                            cmap(coordinate.value),
                            cmap.target_unit,
                            number_type=target_type,
                        )
                    )
                except ValueError:
                    # The map does not reach this position (outside its
                    # domain) or its value cannot be written on the
                    # target unit: the cell stays null. Any other error
                    # is a defect and propagates.
                    derived.append(None)
            derived_columns.setdefault(tl_id, {})[col_name] = derived
    field_lists = {
        name: column
        for tl_id, coordinates in field_lists.items()
        for name, column in (
            (tl_id, coordinates),
            *derived_columns.get(tl_id, {}).items(),
        )
    }

    columns: list[TimestampColumn] = []
    for name, coordinates in field_lists.items():
        exemplar = next(
            (coordinate for coordinate in coordinates if coordinate is not None),
            None,
        )
        if exemplar is None:
            if name in known:
                timeline = known[name]
                unit = timeline.unit
                number_type = timeline.number_type
            else:
                unit = TimeUnit.number
                number_type = NumberType.float
        else:
            unit = exemplar.unit
            number_type = exemplar.number_type
        columns.append(
            TimestampColumn(
                name=name,
                values=coordinates,
                unit=unit,
                number_type=number_type,
                timeline_id=name if name in known else None,
            )
        )
    return build_timestamp_table(columns)


def _collapsed_claim_rows(
    anchors: list[tuple[str, Coordinate, str, Coordinate]],
    registered: list[str],
) -> list[dict[str, Coordinate]]:
    """Implement :meth:`AlignmentBundle._collapsed_claim_rows`."""
    parent: dict[tuple[str, CoordinateValue], tuple[str, CoordinateValue]] = {}
    coordinates: dict[tuple[str, CoordinateValue], Coordinate] = {}

    def node(timeline_id: str, coordinate: Coordinate) -> tuple[str, CoordinateValue]:
        """Return the union-find key of a coordinate, registering it on first sight."""
        key = (timeline_id, coordinate.value)
        parent.setdefault(key, key)
        coordinates.setdefault(key, coordinate)
        return key

    def find(key: tuple[str, CoordinateValue]) -> tuple[str, CoordinateValue]:
        """Return the root of *key*'s component, compressing the path to it."""
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != root:
            parent[key], key = root, parent[key]
        return root

    for timeline_a, coordinate_a, timeline_b, coordinate_b in anchors:
        root_a = find(node(timeline_a, coordinate_a))
        root_b = find(node(timeline_b, coordinate_b))
        if root_a != root_b:
            parent[root_b] = root_a

    components: dict[tuple[str, CoordinateValue], dict[str, Coordinate]] = {}
    for key in parent:
        timeline_id, value = key
        row = components.setdefault(find(key), {})
        carried = row.get(timeline_id)
        if carried is None or value < carried.value:
            row[timeline_id] = coordinates[key]

    rank = {timeline_id: index for index, timeline_id in enumerate(registered)}

    def first(row: dict[str, Coordinate]) -> tuple[int, str]:
        """The row's timeline the bundle knows first, with its rank."""
        return min((rank.get(tl_id, len(rank)), tl_id) for tl_id in row)

    def order(row: dict[str, Coordinate]) -> tuple[int, float]:
        """Sort key: the first timeline's rank, then the row's value on it."""
        position, timeline_id = first(row)
        return position, float(row[timeline_id].value)

    return sorted(components.values(), key=order)
