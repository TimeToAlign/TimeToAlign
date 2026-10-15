"""Provide region operations for timeline instances."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from typing import Any

from timetoalign.core import CoordinateSpec, CoordinateValue
from timetoalign.core.time import coordinate_numeric_value
from timetoalign.core.timestamp import _format_coordinate_value

from ..regions import Region


def _resolve_boundary_names(
    boundaries: Sequence[CoordinateSpec],
    names: Sequence[str] | None,
    name_format: str,
    prefix: str,
    resolve: Callable[[CoordinateSpec], CoordinateValue],
) -> tuple[list[CoordinateValue], list[str]]:
    """Validate boundaries and resolve one name per interval."""
    if len(boundaries) < 2:
        raise ValueError(f"Need at least 2 boundary coordinates, got {len(boundaries)}")

    coords = [resolve(boundary) for boundary in boundaries]
    for i in range(1, len(coords)):
        if coords[i] <= coords[i - 1]:
            raise ValueError(
                f"Boundaries must be monotonically increasing: "
                f"boundaries[{i - 1}]="
                f"{_format_coordinate_value(float(coords[i - 1]))} >= "
                f"boundaries[{i}]={_format_coordinate_value(float(coords[i]))}"
            )

    n_regions = len(coords) - 1
    if names is not None:
        if len(names) != n_regions:
            raise ValueError(
                f"Expected {n_regions} names for {n_regions} regions, got {len(names)}"
            )
        return coords, list(names)

    return coords, [
        name_format.format(prefix=prefix, i=i, n=i + 1) for i in range(n_regions)
    ]


class RegionsMixin:
    """Provide region operations for timeline instances."""

    def add_region(self, region: Region) -> Region:
        """Attach a pre-existing Region object to this timeline.

        The region's id must be unique in this timeline's manifest: a minted
        id already held there is replaced by a freshly minted one, a given
        one raises.

        Args:
            region: The Region to attach.

        Returns:
            The stored Region: *region* itself, or a copy under the newly
            minted id.

        Raises:
            ValueError: If the region's unit does not match the timeline's
                unit, or its given id is already held in this manifest.

        Examples:
            >>> r = Region(Coordinate(10, TimeUnit.seconds),
            ...            Coordinate(30, TimeUnit.seconds), name="Chorus")
            >>> tl.add_region(r)
        """
        if region.unit != self._unit:
            raise ValueError(
                f"Region unit '{region.unit}' does not match "
                f"timeline unit '{self._unit}'"
            )
        region_id = self._admit_element_id(
            region.id,
            minted=region._minted,
            prefix="region",
            element=f"region {region.id!r}",
        )
        if region_id != region.id:
            region = replace(region, id=region_id, _minted=True)

        self._regions[region.id] = region
        self._logger.debug(
            f"Added region '{region.id}' "
            f"[{_format_coordinate_value(float(region.start.value))}, "
            f"{_format_coordinate_value(float(region.end.value))})"
        )
        return region

    def create_region(
        self,
        start: CoordinateSpec,
        end: CoordinateSpec,
        *,
        id: str | None = None,
        name: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Region:
        """Create a new Region and attach it to this timeline.

        Under the unified verb×noun API, ``create`` constructs a new object,
        attaches it, and returns it.

        Args:
            start: Start coordinate.
            end: End coordinate (must be >= start).
            id: The region's id, kept verbatim. When None, ``region_{N}`` is
                minted, unique in this timeline's manifest.
            name: Optional display name; names may repeat.
            meta: Optional metadata dictionary.

        Returns:
            The created Region object.

        Raises:
            ValueError: If the given id is already held in this manifest or
                end < start.

        Examples:
            >>> tl.create_region(10.0, 30.0, name="Chorus")
            >>> tl.create_region(30.0, 50.0, id="verse", meta={"repeat": 2})
        """
        return self.add_region(
            Region(
                self.get_coordinate_at(start, format="coordinate"),
                self.get_coordinate_at(end, format="coordinate"),
                id=id,
                name=name,
                meta=meta or {},
            )
        )

    def _rename_region(self, old_id: str, new_id: str) -> None:
        """Store the region *old_id* under the minted id *new_id*."""
        self._regions = {
            (new_id if key == old_id else key): (
                replace(region, id=new_id, _minted=True) if key == old_id else region
            )
            for key, region in self._regions.items()
        }

    def create_regions_from_boundaries(
        self,
        boundaries: Sequence[CoordinateSpec],
        *,
        names: Sequence[str] | None = None,
        name_format: str = "{prefix}_{n}",
        prefix: str = "section",
    ) -> list[Region]:
        """Create contiguous regions from boundary coordinates.

        Given k+1 sorted boundary coordinates, creates k regions where
        region_i spans [boundaries[i], boundaries[i+1]). The names label the
        regions; their ids are minted.

        Args:
            boundaries: k+1 monotonically increasing coordinates.
            names: Explicit names for the k regions. Mutually exclusive
                with name_format/prefix.
            name_format: Format string. Placeholders: {prefix}, {i} (0-based),
                {n} (1-based).
            prefix: Prefix for auto-generated names.

        Returns:
            List of k Region objects in boundary order.

        Raises:
            ValueError: If fewer than 2 boundaries or not monotonically
                increasing.

        Examples:
            >>> tl.create_regions_from_boundaries(
            ...     [0, 30, 60, 90],
            ...     prefix="movement",
            ... )
            [Region('region_1', name='movement_1', 0-30 quarters),
             Region('region_2', name='movement_2', 30-60 quarters),
             Region('region_3', name='movement_3', 60-90 quarters)]
        """
        coords, region_names = _resolve_boundary_names(
            boundaries,
            names,
            name_format,
            prefix,
            self._resolve_axis_value,
        )

        return [
            self.create_region(start, end, name=name)
            for start, end, name in zip(
                coords[:-1], coords[1:], region_names, strict=True
            )
        ]

    def create_regions_by_grouping(
        self,
        field: str,
        *,
        name_format: str = "{value}",
    ) -> list[Region]:
        """Create regions by grouping *adjacent* events on a field value.

        For each *run* of consecutive events that share the same value in the
        specified field, creates a region spanning the run's coordinate extent
        ``[min_start, max_end)``. Only adjacent events with the same value are
        grouped — non-adjacent occurrences of the same value produce separate
        regions, which may share a name; their ids are minted.

        This "run-length" semantics is essential for musical data where, e.g.,
        the same time signature may recur after a change (4/4 → 3/4 → 4/4)
        and each occurrence should be its own region.

        Args:
            field: Event field name to group by.
            name_format: Format string. Placeholders: {value}, {i} (0-based),
                {n} (1-based), {run} (1-based run index for this value).

        Returns:
            List of Region objects ordered by start coordinate.

        Raises:
            ValueError: If field does not exist in events.

        Examples:
            >>> # Time-signature regions (adjacent grouping)
            >>> tl.create_regions_by_grouping("timesig")
            [Region('region_1', name='4/4', 0-64 quarters),
             Region('region_2', name='3/4', 64-88 quarters),
             Region('region_3', name='4/4', 88-120 quarters)]
        """
        # Collect events sorted by start coordinate
        events_sorted = self._sorted_event_dicts()
        if not events_sorted:
            return []

        # Check field exists
        first_event = events_sorted[0]
        if field not in first_event:
            raise ValueError(
                f"Field '{field}' not found in events. "
                f"Available fields: {list(first_event.keys())}"
            )

        # Build runs of adjacent equal values
        runs: list[tuple[Any, Any, Any]] = []  # (value, start, end)
        value_counts: dict[Any, int] = {}

        current_value = None
        run_start = 0.0
        run_end = 0.0

        for event in events_sorted:
            val = event.get(field)

            ev_start_value = event.get("start", event.get("instant"))
            ev_end_value = event.get("end")
            if ev_end_value is None:
                ev_end_value = ev_start_value
            ev_start = (
                coordinate_numeric_value(ev_start_value)
                if ev_start_value is not None
                else None
            )
            ev_end = (
                coordinate_numeric_value(ev_end_value)
                if ev_end_value is not None
                else ev_start
            )

            if val != current_value:
                # Close previous run
                if current_value is not None:
                    runs.append((current_value, run_start, run_end))
                # Start new run
                current_value = val
                run_start = ev_start if ev_start is not None else 0.0
                run_end = ev_end if ev_end is not None else run_start
            else:
                # Extend current run
                if ev_end is not None:
                    run_end = max(run_end, ev_end)

        # Close last run
        if current_value is not None:
            runs.append((current_value, run_start, run_end))

        # Create regions
        result: list[Region] = []
        for i, (value, start, end) in enumerate(runs):
            value_counts.setdefault(value, 0)
            value_counts[value] += 1
            region_name = name_format.format(
                value=value,
                i=i,
                n=i + 1,
                run=value_counts[value],
            )
            region = self.create_region(start, end, name=region_name)
            result.append(region)

        return result

    def create_regions_by_splitting(
        self,
        predicate: str | dict[str, Any] | Callable[[dict], bool],
        *,
        names: Sequence[str] | None = None,
        name_format: str = "{prefix}_{n}",
        prefix: str = "section",
        include_before_first: bool = True,
        include_after_last: bool = True,
    ) -> list[Region]:
        """Create contiguous regions by splitting at events matching a predicate.

        Finds events matching the predicate, uses their coordinates as split
        points, creates contiguous regions between consecutive split points.

        The predicate can be:
        - A string: field name. Events where this field is truthy (non-null,
          non-empty, non-zero) are split points.
        - A dict: keyword filters in the same style as ``EventData.filter()``.
          For example ``{"breaks": "section"}`` selects events whose ``breaks``
          field equals ``"section"``.
        - A callable: receives event dict, returns True for split points.

        For each matching event the split coordinate is the event's ``end``
        (interval events) or ``start``/``instant`` (instant events).

        Args:
            predicate: Field name, filter dict, or callable identifying
                split-point events.
            names: Explicit region names.
            name_format: Format string. Placeholders: {prefix}, {i}, {n}.
            prefix: Prefix for auto-generated names.
            include_before_first: Create a region from timeline origin to
                first split point.
            include_after_last: Create a region from last split point to
                timeline end.

        Returns:
            List of contiguous Region objects in coordinate order.

        Raises:
            UnknownLengthError: If ``include_after_last`` is set and the
                length is unknown, so that the last region has no end.

        Examples:
            >>> # Split at section breaks
            >>> tl.create_regions_by_splitting("breaks", prefix="movement")

            >>> # Split at specific break types
            >>> tl.create_regions_by_splitting(
            ...     {"breaks": "section"}, prefix="movement"
            ... )
        """
        end = (
            self._require_length("split up to the end").value
            if include_after_last
            else None
        )

        # Resolve predicate to a callable
        match_fn = self._resolve_predicate(predicate)

        # Find split coordinates
        split_coords: list[Any] = []
        events_sorted = self._sorted_event_dicts()

        for event in events_sorted:
            if match_fn(event):
                # Use end coordinate for intervals, start for instants
                coord_value = event.get("end")
                if coord_value is None:
                    coord_value = event.get("start", event.get("instant"))
                if coord_value is not None:
                    split_coords.append(coordinate_numeric_value(coord_value))

        # Deduplicate and sort
        split_coords = sorted(set(split_coords))

        # Build boundary list
        boundaries: list[Any] = []
        if include_before_first:
            boundaries.append(self.origin.value)
        boundaries.extend(split_coords)
        if end is not None:
            boundaries.append(end)

        # Deduplicate again (split point might coincide with origin/end)
        boundaries = sorted(set(boundaries))

        if len(boundaries) < 2:
            return []

        # Determine names
        n_regions = len(boundaries) - 1
        if names is not None:
            if len(names) != n_regions:
                raise ValueError(f"Expected {n_regions} names, got {len(names)}")
            region_names = list(names)
        else:
            region_names = [
                name_format.format(prefix=prefix, i=i, n=i + 1)
                for i in range(n_regions)
            ]

        result: list[Region] = []
        for i in range(n_regions):
            region = self.create_region(
                boundaries[i], boundaries[i + 1], name=region_names[i]
            )
            result.append(region)
        return result

    def _resolve_predicate(
        self,
        predicate: str | dict[str, Any] | Callable[[dict], bool],
    ) -> Callable[[dict], bool]:
        """Convert a predicate specification to a callable.

        Supports three forms:
        - str: field name — truthy test on that field's value.
        - dict: keyword filters (same semantics as ``EventData.filter``).
        - callable: used directly.

        Args:
            predicate: The predicate specification.

        Returns:
            A callable ``(event_dict) -> bool``.
        """
        if callable(predicate) and not isinstance(predicate, (str, dict)):
            return predicate

        if isinstance(predicate, str):
            name = predicate

            def _match_truthy(event: dict) -> bool:
                val = event.get(name)
                if val is None:
                    return False
                if isinstance(val, str):
                    return bool(val.strip())
                return bool(val)

            return _match_truthy

        if isinstance(predicate, dict):
            filters = predicate

            def _match_dict(event: dict) -> bool:
                for key, expected in filters.items():
                    val = event.get(key)
                    if val is None:
                        return False
                    # Support list of acceptable values
                    if isinstance(expected, (list, tuple, set, frozenset)):
                        if val not in expected:
                            return False
                    else:
                        if val != expected:
                            return False
                return True

            return _match_dict

        raise TypeError(
            f"predicate must be str, dict, or callable, got {type(predicate)}"
        )

    def get_region(self, region_id: str) -> Region:
        """Get a Region by id.

        Args:
            region_id: The region's id.

        Returns:
            The Region object.

        Raises:
            KeyError: If no region with that id exists.
        """
        if region_id not in self._regions:
            raise KeyError(
                f"No region with id '{region_id}'. "
                f"Available regions: {list(self._regions.keys())}"
            )
        return self._regions[region_id]

    def get_regions_at(
        self,
        coord: CoordinateSpec,
    ) -> list[Region]:
        """Return all regions containing the given coordinate.

        A region contains coord if region.start <= coord < region.end
        (left-inclusive, right-exclusive).

        Args:
            coord: The coordinate to query.

        Returns:
            List of Region objects containing coord, ordered by start
            coordinate. Empty list if no regions contain coord.

        Examples:
            >>> tl.get_regions_at(75.0)
            [Region('region_1', name='verse', 30-90 quarters),
             Region('region_2', name='chorus', 60-120 quarters)]
        """
        coord_val = float(self._resolve_axis_value(coord))
        matching = [r for r in self._regions.values() if r.contains(coord_val)]
        matching.sort(key=lambda r: float(r.start.value))
        return matching

    def has_region(self, region_id: str) -> bool:
        """Check if a region exists.

        Args:
            region_id: The region's id.

        Returns:
            True if the region exists.
        """
        return region_id in self._regions

    def iter_regions(self) -> Iterator[Region]:
        """Iterate over all regions in insertion order.

        Yields:
            Region objects.
        """
        yield from self._regions.values()

    @property
    def n_regions(self) -> int:
        """Number of regions on this timeline."""
        return len(self._regions)

    def list_regions(self) -> list[str]:
        """List all region ids.

        Returns:
            List of region ids in insertion order.
        """
        return list(self._regions.keys())
