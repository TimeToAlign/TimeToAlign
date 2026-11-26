"""What the timeline engines read on the timeline composing them.

The engine mixins (events, external references, children, conversion maps,
tabular export, regions, segments) are composed into
:class:`~timetoalign.timelines.base.Timeline` only, and read on ``self`` the
timeline's state, its own members and each other's. :class:`TimelineHost`
states those members once, for the type checker: each engine names it as its
base under ``TYPE_CHECKING`` and ``object`` at run time, so the classes, their
method resolution order and their attributes are those of plain mixins.
Nothing imports this module at run time.

A member an engine defines and reads only itself is not listed here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable, Collection, Iterable, Sequence
    from fractions import Fraction

    import pandas as pd
    import pyarrow as pa

    from timetoalign.core import (
        Coordinate,
        CoordinateSpec,
        CoordinateValue,
        IdCoordinate,
        NumberType,
        TimeUnit,
    )
    from timetoalign.core.retrieval import (
        CoordinateFormat,
        CoordinateInput,
        CoordinateResult,
        Rounding,
    )
    from timetoalign.core.timestamp import (
        ConversionMapsSpec,
        TimeIntervalStamp,
        TimeStamp,
    )
    from timetoalign.maps import ConversionMap
    from timetoalign.storage import EventData

    from ..base import Timeline
    from ..flowcontrol import FlowControlRegistry, FlowControlView, Jump
    from ..regions import Region
    from ..structure import MeasureMap


class TimelineHost(Protocol):
    """The members the engines read on ``self``, as the timeline has them."""

    # region State (assigned in Timeline.__init__)

    _id: str
    _unit: TimeUnit
    _number_type: NumberType
    _parent: Timeline | None
    _length: Coordinate | None
    _locked: bool
    _meta: dict[str, Any]
    _events: EventData
    _children: dict[str, Timeline]
    _child_offsets: dict[str, Coordinate]
    _conversion_maps: dict[str, ConversionMap[Any]]
    _unit_maps: dict[TimeUnit, list[ConversionMap[Any]]]
    _regions: dict[str, Region]
    _measure_record_map: ConversionMap[Any] | None
    _flow_control: FlowControlRegistry
    _logger: logging.Logger

    # endregion

    # region Timeline

    @property
    def _timeline(self) -> Timeline:
        """This host as the timeline it is."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name for display."""
        ...

    @property
    def origin(self) -> Coordinate:
        """The start coordinate (always 0)."""
        ...

    @property
    def measure_map(self) -> MeasureMap | None:
        """Authored measure structure; absent on fresh and derived slices."""
        ...

    @property
    def tree_flow_control(self) -> FlowControlView:
        """The flow control of this timeline and of its descendants, on its axis."""
        ...

    def _is_segment_line_instance(self) -> bool:
        """Whether this timeline is a SegmentLine (the dictionary's flag)."""
        ...

    def _root_start(self) -> tuple[Timeline, Fraction]:
        """Return this timeline's root and this timeline's exact start on it."""
        ...

    def _refuse_repeated_jumps(
        self,
        arriving: Iterable[tuple[Fraction, Jump, str]],
        *,
        replaced: Collection[str] = (),
    ) -> None:
        """Refuse a Jump arriving at a position of the root's axis that is taken."""
        ...

    def _manifest_roots(self) -> list[Timeline]:
        """Return the root of every tree of this timeline's manifest, once each."""
        ...

    def _refuse_held(self, entries: Sequence[tuple[str, str]]) -> None:
        """Refuse elements joining this timeline's manifest whose id it holds."""
        ...

    def _refuse_held_element(self, element_id: str, element: str) -> None:
        """Refuse an element joining this timeline whose id the manifest holds."""
        ...

    def _admit_map(self, cmap: ConversionMap[Any]) -> None:
        """Refuse *cmap* where an id of it or of a map nested in it is held."""
        ...

    def _rename_references(self, old_id: str, new_id: str) -> None:
        """Rename *old_id* to *new_id* in every statement of the manifest."""
        ...

    def _make_coordinate(self, value: CoordinateValue | Coordinate) -> Coordinate:
        """Create a Coordinate in this timeline's unit."""
        ...

    def _require_length(self, operation: str) -> Coordinate:
        """Return the length, raising when an operation needs an unknown one."""
        ...

    def _check_not_locked(self, operation: str) -> None:
        """Raise RuntimeError if timeline is locked."""
        ...

    @staticmethod
    def _coord_to_value(value: Any) -> Any:
        """Return a coordinate value without discarding an exact ratio."""
        ...

    # endregion

    # region Events

    @property
    def n_events(self) -> int:
        """Number of events (excluding segment events)."""
        ...

    def _ensure_capacity(
        self,
        required_end: CoordinateValue,
        allow_expansion: bool = False,
    ) -> None:
        """Ensure timeline is long enough, expanding if permitted."""
        ...

    def _add_events_unchecked(self, rows: list[dict[str, Any]]) -> None:
        """Add events without validation (internal use)."""
        ...

    def get_events(
        self,
        temporal_type: Literal["instant", "interval"] | None = None,
        event_type: str | None = None,
        include_children: bool = True,
        min_coord: CoordinateSpec | None = None,
        max_coord: CoordinateSpec | None = None,
        **field_filters: Any,
    ) -> EventData:
        """Filter and retrieve events."""
        ...

    def _extract_coord_value(
        self,
        event: dict[str, Any],
        *keys: str,
    ) -> int | float | Fraction | None:
        """Extract coordinate value from event, trying multiple keys."""
        ...

    def _sorted_event_dicts(self) -> list[dict[str, Any]]:
        """Return events as dicts sorted by start coordinate."""
        ...

    # endregion

    # region Children

    @property
    def _extent(self) -> CoordinateValue | None:
        """This timeline's extent on its own axis, or None if unbounded."""
        ...

    def _holds_column(
        self,
        positions: pa.StructArray,
        number_type: NumberType,
        start: CoordinateValue = 0,
        *,
        end_held: bool | None = None,
    ) -> pa.Array:
        """Apply the position test to a column of positions."""
        ...

    def _set_length(self, length: Coordinate | None) -> None:
        """Set this timeline's length, which its parent must contain."""
        ...

    def _copy_events_to_cut(
        self,
        target: Timeline,
        start: Any,
        end: Any,
        *,
        copy_events: bool = True,
    ) -> None:
        """Copy the events starting in [start, end) onto *target*, cut there."""
        ...

    # endregion

    # region Conversion maps

    def get_conversion_map(
        self, target_unit: TimeUnit | str
    ) -> ConversionMap[Any] | None:
        """Get a conversion map by target unit **or** by name/id."""
        ...

    def _get_number_type_for_timeline(self, timeline_id: str) -> NumberType | None:
        """Get the numeric representation used by this timeline or a descendant."""
        ...

    def _get_unit_for_timeline(self, timeline_id: str) -> TimeUnit | None:
        """Get the TimeUnit for a timeline anywhere in the subtree."""
        ...

    @overload
    def get_coordinate_at(
        self,
        at: CoordinateInput,
        timeline_id: str | None = None,
        *,
        format: Literal["id_coordinate"] = ...,
        rounding: Rounding = ...,
    ) -> IdCoordinate: ...

    @overload
    def get_coordinate_at(
        self,
        at: CoordinateInput,
        timeline_id: str | None = None,
        *,
        format: Literal["coordinate"],
        rounding: Rounding = ...,
    ) -> Coordinate: ...

    @overload
    def get_coordinate_at(
        self,
        at: CoordinateInput,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat,
        rounding: Rounding = ...,
    ) -> CoordinateResult | pd.Series: ...

    @overload
    def get_coordinate_for(
        self,
        key: str,
        timeline_id: str | None = None,
        *,
        format: Literal["id_coordinate"] = ...,
        rounding: Rounding = ...,
    ) -> IdCoordinate: ...

    @overload
    def get_coordinate_for(
        self,
        key: str,
        timeline_id: str | None = None,
        *,
        format: Literal["coordinate"],
        rounding: Rounding = ...,
    ) -> Coordinate: ...

    @overload
    def get_coordinate_for(
        self,
        key: str,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat,
        rounding: Rounding = ...,
    ) -> CoordinateResult | pd.Series: ...

    def _resolve_axis_value(self, coord: CoordinateSpec) -> int | float | Fraction:
        """Resolve a coordinate and return its native numeric value."""
        ...

    def _query_position(self, coord: CoordinateSpec) -> int | float | Fraction:
        """The position a query names on this axis, as a comparison reads it."""
        ...

    # endregion

    # region Tabular export

    def get_timestamp_for(
        self,
        key: str,
        timeline_id: str | None = None,
        *,
        conversion_maps: ConversionMapsSpec = True,
    ) -> TimeStamp | TimeIntervalStamp:
        """Get the timestamp for a specific event by its ID."""
        ...

    def _resolve_conversion_maps(
        self, spec: ConversionMapsSpec
    ) -> list[ConversionMap[Any]]:
        """Resolve a flexible conversion_maps specification to a list of C-Maps."""
        ...

    # endregion

    # region Regions

    def add_region(self, region: Region) -> Region:
        """Attach a pre-existing Region object to this timeline."""
        ...

    def create_region(
        self,
        start: CoordinateSpec,
        end: CoordinateSpec,
        *,
        id: str | None = None,
        name: str | None = None,
        meta: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> Region:
        """Create a new Region and attach it to this timeline."""
        ...

    def _resolve_predicate(
        self,
        predicate: str | dict[str, Any] | Callable[[dict], bool],
    ) -> Callable[[dict], bool]:
        """Convert a predicate specification to a callable."""
        ...

    def get_region(self, region_id: str) -> Region:
        """Get a Region by id."""
        ...

    # endregion
