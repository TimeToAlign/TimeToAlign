"""Factory functions for creating Timelines from various sources.

This module provides the `create_timeline` function and its helpers,
enabling timeline creation from EventStores, EventData, and Loaders.

Design principles:
- Multiple entry points (EventData.create_timeline, EventStore.create_timeline, create_timeline)
- Children maintain their own 0-based coordinate systems
- Parent length equals max of all child lengths
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc

from timetoalign.core import Domain, NumberType, TimeUnit
from timetoalign.core.ids import IdMinter, content_minter, quote_id, timeline_id_prefix
from timetoalign.storage import EventData, EventStore, SingleStore
from timetoalign.storage.events import (
    _ipc_bytes,
)
from timetoalign.storage.events import (
    _register_timeline_factory as _register_data_factory,
)
from timetoalign.storage.store import (
    _register_timeline_factory as _register_store_factory,
)

if TYPE_CHECKING:
    from timetoalign.timelines.base import Timeline

module_logger = logging.getLogger(__name__)


def mint_timeline_id(
    unit: TimeUnit, minter: IdMinter | None = None, *, role: str | None = None
) -> str:
    """Mint a timeline ID: the unit's timeline prefix, led by *role*.

    The ID is ``{prefix}_{suffix}`` drawn from *minter*
    (:func:`~timetoalign.core.ids.timeline_id_prefix` of *unit*), led by
    ``{role}:`` when a role is given (``score:clt_…``). Without a minter, it
    comes from the content minter of the prefix alone, so successive calls
    of one process mint successive IDs.

    Args:
        unit: The timeline's unit.
        minter: The minter the ID is drawn from (a loader's), or None.
        role: A role leading the ID, or None.

    Returns:
        The ID.

    Examples:
        >>> mint_timeline_id(TimeUnit.quarters, IdMinter(0), role="score")
        'score:clt_1be6k8ij6dzad'
    """
    prefix = timeline_id_prefix(unit)
    timeline_id = (minter or content_minter(prefix)).mint(prefix)
    return timeline_id if role is None else f"{role}:{timeline_id}"


def _content_minter_of(
    unit: TimeUnit, tables: Sequence[tuple[str | None, pa.Table]]
) -> IdMinter:
    """The content minter of timelines made of *tables*.

    Its content is the unit's timeline prefix, the unit, and each table's
    name and Arrow IPC bytes, in order.
    """
    prefix = timeline_id_prefix(unit)
    return content_minter(
        prefix,
        unit.value,
        *(part for name, table in tables for part in (name, _ipc_bytes(table))),
    )


def _refuse_repeated_event_ids(data: EventData, timeline_id: str) -> None:
    """Refuse an event ID held by two rows of *data*, one timeline's events.

    Raises:
        ValueError: Naming the first such ID and the timeline.
    """
    counts = pc.value_counts(data.table.column("id"))
    repeated = counts.filter(pc.greater(counts.field("counts"), pa.scalar(1)))
    if len(repeated):
        raise ValueError(
            f"Event id {quote_id(repeated.field('values')[0].as_py())} is held by "
            f"{repeated.field('counts')[0].as_py()} events of timeline "
            f"{quote_id(timeline_id)}"
        )


def create_timeline_from_event_data(
    data: EventData,
    uid: str | None = None,
    filters: dict[str, Any] | None = None,
    *,
    minter: IdMinter | None = None,
) -> "Timeline":
    """Create a timeline containing one EventData table.

    Args:
        data: Source event data.
        uid: Optional timeline ID; without one, it is minted from *minter*
            (:func:`mint_timeline_id`).
        filters: Optional EventData filters.
        minter: The minter of the timeline ID (a loader's); without one,
            the content minter of the prefix, the unit and the selected
            events.

    Returns:
        A timeline containing the selected events, held in its own
        ``EventData``: adding events to the timeline leaves *data* unchanged.
    """
    source = data.filter(**filters) if filters else data
    timeline_class, effective_number_type = _infer_timeline_class_and_number_type(
        data.unit, data.number_type
    )
    if uid is None:
        uid = mint_timeline_id(
            data.unit, minter or _content_minter_of(data.unit, [(None, source.table)])
        )
    coord_range = source.coordinate_range()
    length = coord_range[1] if coord_range else 0
    timeline = timeline_class(
        length=length,
        unit=source.unit,
        number_type=effective_number_type,
        uid=uid,
    )
    # The timeline's own wrapper over the (immutable) table: adding events
    # to it replaces its table and leaves the source and every other
    # timeline made from the source as they are.
    timeline._events = source._derived(source.table)
    return timeline


def _infer_timeline_class_and_number_type(
    unit: TimeUnit,
    number_type: NumberType,
) -> tuple[type[Timeline], NumberType]:
    """Infer the Timeline subclass and representation for a unit.

    Discreteness is a property of the unit, not of how a particular table
    happened to write its numbers, so a tick column is a discrete timeline
    whether it arrived as ints or floats.

    Args:
        unit: The time unit.
        number_type: The number type from the EventData.

    Returns:
        Tuple of (Timeline subclass, effective number_type).
    """
    from timetoalign.timelines.base import Timeline
    from timetoalign.timelines.types import (
        ContinuousGraphicalTimeline,
        ContinuousLogicalTimeline,
        ContinuousPhysicalTimeline,
        DiscreteGraphicalTimeline,
        DiscreteLogicalTimeline,
        DiscretePhysicalTimeline,
    )

    is_discrete = unit.is_discrete
    effective_number_type = unit.resolve_number_type(
        None if is_discrete else number_type
    )

    mapping: dict[tuple[Domain, bool], type[Timeline]] = {
        (Domain.logical, True): DiscreteLogicalTimeline,
        (Domain.logical, False): ContinuousLogicalTimeline,
        (Domain.physical, True): DiscretePhysicalTimeline,
        (Domain.physical, False): ContinuousPhysicalTimeline,
        (Domain.graphical, True): DiscreteGraphicalTimeline,
        (Domain.graphical, False): ContinuousGraphicalTimeline,
    }

    return mapping.get((unit.domain, is_discrete), Timeline), effective_number_type


def _infer_timeline_class(
    unit: TimeUnit,
    number_type: NumberType,
) -> type[Timeline]:
    """Infer the appropriate Timeline subclass from unit and number_type.

    Args:
        unit: The time unit.
        number_type: The number type.

    Returns:
        The most appropriate Timeline subclass.
    """
    timeline_class, _ = _infer_timeline_class_and_number_type(unit, number_type)
    return timeline_class


def _create_grouped_timeline(
    data_to_use: list[tuple[str, "EventData"]],
    group_by: str,
    timeline_class: type["Timeline"],
    effective_number_type: NumberType,
    uid: str | None = None,
    *,
    minter: IdMinter | None,
) -> "Timeline":
    """Create a timeline with children grouped by a field value.

    Args:
        data_to_use: List of (name, EventData) tuples.
        group_by: Field name to group by.
        timeline_class: The Timeline class to use.
        effective_number_type: The number type for the timeline.
        uid: Optional parent timeline ID; without one, it is minted from
            *minter*.
        minter: The minter of the parent's ID, if missing, and the
            children's, in the order of their group values.

    Returns:
        Timeline with children for each unique group value.
    """
    # Merge all data first
    merged_data = data_to_use[0][1]
    for _, data in data_to_use[1:]:
        merged_data = merged_data.concat(data)

    # Check that group_by field exists
    table = merged_data.table
    available_names = table.column_names
    if group_by not in available_names:
        raise ValueError(
            f"group_by field '{group_by}' not found. "
            f"Available fields: {available_names}"
        )

    # Get unique values in the group_by field
    group_arr = table.column(group_by)
    unique_values = group_arr.unique().to_pylist()

    # Sort unique values for consistent ordering
    try:
        unique_values = sorted(unique_values, key=lambda x: (x is None, x))
    except TypeError:
        # If values can't be sorted (mixed types), use original order
        pass

    module_logger.debug(
        f"Grouping {len(merged_data)} events by '{group_by}' "
        f"into {len(unique_values)} groups"
    )

    # Calculate parent length (max of all data)
    coord_range = merged_data.coordinate_range()
    max_length = coord_range[1] if coord_range else 0

    # Create parent timeline
    parent = timeline_class(
        length=max_length,
        unit=merged_data.unit,
        number_type=effective_number_type,
        uid=mint_timeline_id(merged_data.unit, minter) if uid is None else uid,
    )

    # Create child for each unique value
    for group_value in unique_values:
        if group_value is None:
            child_name = "_none_"
            # Filter for null values
            mask = pc.is_null(group_arr)
        else:
            child_name = str(group_value)
            # Filter for exact match
            mask = pc.equal(group_arr, group_value)

        # Apply filter to table
        filtered_table = table.filter(mask)

        if filtered_table.num_rows == 0:
            continue

        # Create EventData for this group
        group_data = EventData._of_reduced(
            filtered_table, merged_data.unit, merged_data.number_type
        )

        # Create child timeline
        child_coord_range = group_data.coordinate_range()
        child_length = child_coord_range[1] if child_coord_range else 0

        # The group value is the child's label; its id is minted.
        child = timeline_class(
            length=child_length,
            unit=group_data.unit,
            number_type=effective_number_type,
            uid=mint_timeline_id(group_data.unit, minter),
            name=child_name,
        )
        child._events = group_data
        parent.add_child(child, offset=0, allow_expansion=True)

    module_logger.debug(
        f"Created grouped timeline '{parent.id}' with {parent.n_children} children"
    )

    return parent


def create_timeline_from_bundle(
    store: "EventStore",
    uid: str | None = None,
    store_filters: dict[str, dict[str, Any]] | None = None,
    include_stores: list[str] | None = None,
    exclude_stores: list[str] | None = None,
    flatten: bool = False,
    group_by: str | None = None,
    *,
    minter: IdMinter | None = None,
) -> "Timeline":
    """Create a Timeline from an EventStore.

    This is the core implementation called by EventStore.create_timeline().

    Behavior depends on the number of data sources:
    - **Single data source**: Events are placed directly on the timeline
      (no children). This is the common case for simple loaders.
    - **Multiple data sources**: Each data becomes a child timeline at
      offset 0. This is used by ScoreStore, MidiStore, etc.
    - **flatten=True**: Always merge all events into a single timeline.
    - **group_by**: Group events by a field value, creating child timelines
      for each unique value. This overrides the default behavior.

    Args:
        store: The source EventStore (container for EventData).
        uid: Unique ID for the timeline. Auto-generated if None.
        store_filters: Per-data filter kwargs to apply before adding.
            Example: {"notes": {"event_type": "Note"}}
        include_stores: Only include these data (default: all non-empty).
        exclude_stores: Exclude these data.
        flatten: If True, merge all events into timeline (no children).
            If False (default), multiple data become children at offset 0.
        group_by: Field name to group events by. Each unique value becomes
            a child timeline. Useful for grouping by image_filename, page, etc.
        minter: The minter of every timeline ID the call mints (a
            loader's), the parent's first, then the children's in order;
            without one, the content minter of the prefix, the unit and the
            data used (each name and table, in order).

    Returns:
        A Timeline with events (single data) or children (multiple data/groups).
        Each event keeps its ID.

    Raises:
        ValueError: If no data remain after filtering, or all are empty; if
            two events merged into one timeline hold one ID.
    """
    store_filters = store_filters or {}
    exclude_set = set(exclude_stores or [])

    # Determine which data to include
    data_names = list(store.keys())
    if include_stores is not None:
        data_names = [n for n in data_names if n in include_stores]
    data_names = [n for n in data_names if n not in exclude_set]

    if not data_names:
        raise ValueError("No data to include after filtering")

    # Collect data (with filters applied), skip empty
    data_to_use: list[tuple[str, EventData]] = []
    for name in data_names:
        data = store[name]
        if name in store_filters:
            data = data.filter(**store_filters[name])
        if len(data) > 0:
            data_to_use.append((name, data))

    if not data_to_use:
        raise ValueError("All data are empty after filtering")

    # Infer timeline class and effective number_type from first data
    first_name, first_data = data_to_use[0]
    timeline_class, effective_number_type = _infer_timeline_class_and_number_type(
        first_data.unit, first_data.number_type
    )
    single = flatten or len(data_to_use) == 1
    if minter is None and (uid is None or group_by is not None or not single):
        minter = _content_minter_of(
            first_data.unit, [(name, data.table) for name, data in data_to_use]
        )

    # Handle group_by: create children from unique field values
    if group_by is not None:
        return _create_grouped_timeline(
            data_to_use,
            group_by=group_by,
            timeline_class=timeline_class,
            effective_number_type=effective_number_type,
            uid=uid,
            minter=minter,
        )
    if uid is None:
        uid = mint_timeline_id(first_data.unit, minter)

    # Single data case: put events directly on the timeline (no children).
    # This applies when flatten=True is explicitly requested OR when there
    # is only a single data source (the common case for simple loaders and
    # SingleStore).  create_timeline() defaults to flatten=False, but
    # the single-data optimisation still fires because len(data_to_use)==1.
    if single:
        # Merge all events into single timeline
        merged_data = first_data
        for _, data in data_to_use[1:]:
            merged_data = merged_data.concat(data)
        # Create timeline with corrected number_type
        coord_range = merged_data.coordinate_range()
        length = coord_range[1] if coord_range else 0
        timeline = timeline_class(
            length=length,
            unit=merged_data.unit,
            number_type=effective_number_type,
            uid=uid,
        )
        if len(data_to_use) > 1:
            _refuse_repeated_event_ids(merged_data, timeline.id)
        timeline._events = merged_data

        module_logger.debug(
            f"Created timeline '{timeline.id}' with {len(merged_data)} events"
        )

        return timeline

    else:
        # Multiple data: create parent with children at offset 0
        # Parent length = max of all child lengths
        max_length: float | Fraction = 0.0
        for _, data in data_to_use:
            coord_range = data.coordinate_range()
            if coord_range:
                max_length = max(max_length, coord_range[1])

        parent = timeline_class(
            length=max_length,
            unit=first_data.unit,
            number_type=effective_number_type,
            uid=uid,
        )

        # Add each data as a child at offset 0
        for name, data in data_to_use:
            # Create child with corrected number_type
            child_coord_range = data.coordinate_range()
            child_length = child_coord_range[1] if child_coord_range else 0
            # The store name is the child's label; its id is minted.
            child = timeline_class(
                length=child_length,
                unit=data.unit,
                number_type=effective_number_type,
                uid=mint_timeline_id(data.unit, minter),
                name=name,
            )
            child._events = data
            parent.add_child(child, offset=0, allow_expansion=True)

        module_logger.debug(
            f"Created timeline '{parent.id}' with {parent.n_children} children"
        )

        return parent


def create_timeline(
    source: EventStore | EventData | Any,
    uid: str | None = None,
    store_filters: dict[str, dict[str, Any]] | None = None,
    include_stores: list[str] | None = None,
    exclude_stores: list[str] | None = None,
    flatten: bool = False,
    group_by: str | None = None,
    *,
    minter: IdMinter | None = None,
) -> "Timeline":
    """Create a Timeline from various source types.

    This is the most flexible timeline creation API, supporting:
    - EventStore (ScoreStore, MidiStore, SingleStore)
    - EventData (creates timeline with events directly on it)
    - Loader (uses loader.store)

    Behavior depends on the number of data sources:
    - **Single data source** (typical for simple loaders): Events are placed
      directly on the timeline. No child timelines are created.
    - **Multiple data sources** (ScoreStore, MidiStore): Each data becomes
      a child timeline at offset 0, maintaining its own coordinate system.
    - **group_by**: Groups events by a field value, creating child timelines
      for each unique value. Overrides default single/multi behavior.

    Args:
        source: The data source (EventStore, EventData, or any loader exposing a
            .store attribute holding an EventStore).
        uid: Unique ID for the timeline. Auto-generated if None.
        store_filters: Per-data filter kwargs to apply before adding.
            Example: {"notes": {"event_type": "Note"}} to exclude rests.
        include_stores: Only include these data (default: all non-empty).
        exclude_stores: Exclude these data from the timeline.
        flatten: If True, merge all events into a single timeline (no children).
            If False (default), multiple data sources become children.
        group_by: Field name to group events by. Each unique value becomes
            a child timeline. Useful for grouping by image_filename, page, etc.
        minter: The minter of every timeline ID the call mints, as
            :func:`create_timeline_from_bundle` says.

    Returns:
        A Timeline with events (single source) or children (multiple sources/groups).

    Raises:
        TypeError: If source is not a supported type.
        ValueError: If no data remain after filtering.

    Examples:
        >>> # From a simple loader (single EventData -> events on timeline)
        >>> loader = TsvLoader()
        >>> loader.load("annotations.tsv")
        >>> timeline = create_timeline(loader, uid="my_timeline")
        >>> timeline.n_events  # Events are directly on the timeline
        100

        >>> # From a score loader (multiple stores -> children)
        >>> loader = PartituraLoader()
        >>> loader.load("score.musicxml")
        >>> timeline = create_timeline(loader, uid="my_score")
        >>> # Each child is named after its store; its id is minted (clt_...)
        >>> children = {timeline.get_child(i).name: i for i in timeline.list_children()}
        >>> notes_tl = timeline.get_child(children["notes"])

        >>> # With filters (only Notes, exclude annotations)
        >>> timeline = create_timeline(
        ...     loader,
        ...     store_filters={"notes": {"event_type": "Note"}},
        ...     exclude_stores=["annotations"],
        ... )

        >>> # Flattened (merge multiple stores into one timeline)
        >>> timeline = create_timeline(loader, flatten=True)

        >>> # Grouped by field (e.g., image_filename)
        >>> timeline = create_timeline(loader, group_by="image_filename")
        >>> # Creates children: page1.png, page2.png, etc.
    """
    # Normalize to EventStore
    if isinstance(source, EventStore):
        store = source
    elif isinstance(source, EventData):
        store = SingleStore(source, name="events")
    elif hasattr(source, "store") and isinstance(source.store, EventStore):
        store = source.store
    else:
        raise TypeError(
            f"source must be EventStore, EventData, or Loader, "
            f"got {type(source).__name__}"
        )

    return create_timeline_from_bundle(
        store,
        uid=uid,
        store_filters=store_filters,
        include_stores=include_stores,
        exclude_stores=exclude_stores,
        flatten=flatten,
        group_by=group_by,
        minter=minter,
    )


_register_data_factory(create_timeline_from_event_data)
_register_store_factory(create_timeline_from_bundle)
