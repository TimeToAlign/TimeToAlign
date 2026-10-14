"""AlignmentBundle - The primary entry point for alignment workflows.

This module implements the `AlignmentBundle` class, a single entry point for
all alignment workflows as described in the API redesign specification. The
bundle manages timelines, groups, and coordinate transfer operations.

Within a group, coordinate transfer uses typed coordinate retrieval.
Across groups, transfer is mediated by
the ``MatchClaim`` -> ``MatchLine`` -> ``WarpMap`` pipeline.
WarpMaps are built lazily and cached for repeated queries.

A bundle is its timelines and its groups. Every timeline is registered
under its own ``id``, which is unique across the bundle's manifest trees.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import pandas as pd

from timetoalign.core import (
    Coordinate,
    CoordinateSpec,
    CoordinateValue,
    IdCoordinate,
    Interval,
    SupportPolicy,
    TimeUnit,
    express_as,
    express_scalar_as,
    resolve_coordinate_spec,
    resolve_id,
)
from timetoalign.core.enums import NumberType
from timetoalign.core.ids import mint_id, validate_id
from timetoalign.core.retrieval import (
    CoordinateCollection,
    CoordinateFormat,
    CoordinateInput,
    CoordinateResult,
    KeyCollection,
    Rounding,
    TableFormat,
    collection_values,
    dispatch_retrieval,
    format_coordinates,
    is_coordinate_input,
    reject_dataframe_options,
    resolve_key_collection,
    validate_coordinate_collection,
    validate_key_collection,
    validate_table_format,
)
from timetoalign.core.timestamp import TimestampColumn, build_timestamp_table
from timetoalign.timelines import TimelineGroup
from timetoalign.timelines.groups import (
    _admit_timeline_tree,
    _held_ids,
    _nested,
    _read_graphical,
    _require_keys,
    _root,
    _roots_from_dicts,
    _tree,
    _write_graphical,
)

from .claims import AlignmentAnchor, MatchClaim, MatchClaimField
from .filters import ClaimFilter
from .graph import MatchGraph, MatchIntervalStamp, MatchStamp
from .matchline import MatchLine
from .warpmap import AmbiguousWarpMapError, WarpMap

if TYPE_CHECKING:
    from collections.abc import Callable

    import pyarrow as pa

    from timetoalign.core.enums import ColumnNaming, Domain
    from timetoalign.core.timestamp import ConversionMapsSpec
    from timetoalign.display.ascii import Diagram
    from timetoalign.maps.base import ConversionMap
    from timetoalign.timelines import Timeline

module_logger = logging.getLogger(__name__)


def _resolve_coordinate_and_timeline(
    coordinate: CoordinateSpec,
    timeline_id: str | None,
) -> tuple[int | float | Fraction, str | None, "TimeUnit | None"]:
    """Decompose a coordinate specification and resolve its timeline.

    Accepts the three canonical coordinate forms and returns the underlying
    value, unit, and timeline the coordinate lives on. When the coordinate is
    an ``IdCoordinate`` it carries its own ``timeline_id``, which fills an
    omitted ``timeline_id`` argument.

    Args:
        coordinate: The query coordinate. One of:

            - int/float/Fraction: Raw value, ``timeline_id`` required.
            - Coordinate: Value with unit, ``timeline_id`` required.
            - IdCoordinate: Value, unit AND timeline_id (``timeline_id`` param
              optional).
        timeline_id: Explicit timeline id, or None to take it from an
            ``IdCoordinate``.

    Returns:
        A ``(coord_value, timeline_id, unit)`` tuple. ``timeline_id`` may
        still be None if a non-Id coordinate was passed without one; callers
        decide whether that is an error.

    Raises:
        TypeError: If ``coordinate`` is not one of the accepted forms.
    """
    resolved = resolve_coordinate_spec(coordinate, timeline_id=timeline_id)
    return resolved.value, resolved.timeline_id, resolved.unit


# region AlignmentBundle


@dataclass
class AlignmentBundle:
    """The primary entry point for all alignment workflows.

    An AlignmentBundle manages timelines and their alignment relationships.
    Within a group, coordinate transfer uses linear interpolation
    (``TimelineGroup.get_coordinate_at()``). Across groups, transfer is mediated
    by the ``MatchClaim`` → ``MatchLine`` → ``WarpMap`` pipeline.

    The bundle provides:

    - Timeline registration and lookup
    - Group management (collections of perfectly aligned timelines)
    - Coordinate transfer between any two timelines (same-group or
      cross-group via ``MatchClaim``/``WarpMap``)

    IMPORTANT: The resulting bundle structure is order-independent. Adding
    timelines in any order produces the same alignment relationships and
    coordinate transfer results.

    Attributes:
        id: Unique identifier for this bundle, ``bundle_N`` unless given.
        name: Optional human-readable name.
        timelines: Dictionary mapping timeline IDs to Timeline objects: every
            timeline added, and the root of every one that is a child.
        groups: Dictionary mapping group IDs to TimelineGroup objects.
        timeline_to_group: Mapping from timeline ID to its containing group ID.
        cross_group_claims: MatchClaims connecting timelines across groups
            (the per-claim Python-list store).
        cross_group_claim_fields: Columnar ``MatchClaimField`` stores of dense
            synchronous-instant pairwise claims, queried vectorized without
            materialising the full claim list.
        styling: The ``styles`` and ``statements`` of the bundle file's
            ``graphical``, kept as given and never interpreted.

    Note:
        The two claim stores are interchangeable as far as queries are
        concerned: every reader consults both, so a bundle whose claims live
        in a ``MatchClaimField`` answers exactly as a bundle holding the same
        claims in the Python list.  What differs is cost — ``MatchClaimField``
        queries stay vectorized, and ``get_claim_fields()`` is the accessor
        that keeps them that way.

    Examples:
        >>> from timetoalign.alignment import AlignmentBundle
        >>> from timetoalign.timelines.types import (
        ...     ContinuousLogicalTimeline,
        ...     ContinuousPhysicalTimeline,
        ... )
        >>> bundle = AlignmentBundle()
        >>> _ = bundle.add_timeline(ContinuousLogicalTimeline(length=200))
        >>> _ = bundle.add_timeline(
        ...     ContinuousPhysicalTimeline(length=91.0), grouped_with="clt_1"
        ... )
        >>> stamp = bundle.get_matchstamp_at(100, "clt_1")
        >>> stamp.get_coordinate_for("cpt_1", format="float")
        45.5
    """

    id: str = field(default="")
    name: str | None = None
    timelines: dict[str, "Timeline"] = field(default_factory=dict)
    groups: dict[str, TimelineGroup] = field(default_factory=dict)
    timeline_to_group: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    cross_group_claims: list[MatchClaim] = field(default_factory=list)
    # Columnar claim stores added additively (e.g. by ListenHereLoader). Each
    # holds a dense set of synchronous instant pairwise claims as a single
    # PyArrow struct column. Queries filter these vectorized and materialise
    # only the matched rows — a million-claim field is never exploded into a
    # Python list. The Python-list ``cross_group_claims`` path is unchanged.
    cross_group_claim_fields: list[MatchClaimField] = field(default_factory=list)

    # Persistent per-bundle setting: how ``get_matchstamp_at`` treats a
    # timeline whose transferred coordinate falls outside alignment support
    # (below the first anchor or beyond the last). The default drops such a
    # timeline; a per-call ``support_policy`` argument overrides it.
    support_policy: SupportPolicy = SupportPolicy.omit

    # The ``styles`` and ``statements`` of the bundle file's ``graphical``,
    # kept as given and never interpreted.
    styling: dict[str, Any] = field(default_factory=dict)

    # Cached WarpMaps: (source_tl_id, target_tl_id) -> WarpMap | None
    # None values are cached as negative results to avoid repeated expensive
    # MatchLine construction for unreachable timeline pairs.
    _warp_map_cache: dict[tuple[str, str], WarpMap | None] = field(
        default_factory=dict, repr=False
    )
    # Cached MatchLines: source_tl_id -> MatchLine (avoids rebuilding the
    # expensive MatchGraph + group extension for every target lookup)
    _matchline_cache: dict[str, MatchLine | None] = field(
        default_factory=dict, repr=False
    )
    # Cached MatchGraphs: (timeline_id, coordinate) -> MatchGraph
    _matchgraph_cache: dict[tuple[str, CoordinateValue], MatchGraph] = field(
        default_factory=dict, repr=False
    )
    # Hash of cross_group_claims list at time of last cache build
    _cache_claims_hash: int = field(default=0, repr=False)

    # Internal logger
    _logger: logging.Logger = field(
        default_factory=lambda: module_logger, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Mint or validate the ID and set up the logger."""
        if not self.id:
            self.id = mint_id("bundle", ())
        else:
            validate_id(self.id)
        self.support_policy = SupportPolicy(self.support_policy)
        self._logger = module_logger.getChild(self.id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlignmentBundle":
        """Rebuild a bundle from the dictionary :meth:`to_dict` writes.

        Every entry of ``"timelines"`` is rebuilt with
        :meth:`Timeline.from_dict <timetoalign.timelines.Timeline.from_dict>`;
        each group is rebuilt from its ``Group`` object, its members resolved
        among those roots and all their descendants and ordered by
        ``graphical.order`` (``members`` order for a group without an entry).
        ``graphical.styles`` and ``graphical.statements`` are kept as
        :attr:`styling`, unread. Every root no group names joins ungrouped.
        Feeding the result back
        through :meth:`to_dict` with the same flags reproduces the input
        dictionary. A ``"meta"`` key — the file's own metadata — is accepted
        and ignored.

        Args:
            data: ``{"timelines": [...], "bundle": {...}, "graphical": {...}}``.

        Returns:
            The rebuilt bundle, without MatchClaims.

        Raises:
            ValueError: If *data* is malformed: a missing or unknown key, an
                unknown ``support_policy``, a timeline id occurring twice, a
                group id occurring twice or equal to a timeline id, a member
                naming no timeline of ``"timelines"``, a timeline named by two
                groups, a member that is an ancestor or descendant of another
                member of its group, or an order that is no permutation of its
                group's members or names no group. The message names the
                problem.

        Examples:
            >>> from fractions import Fraction
            >>> from timetoalign.alignment import AlignmentBundle
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     ContinuousPhysicalTimeline,
            ... )
            >>> bundle = AlignmentBundle(name="study")
            >>> _ = bundle.add_timeline(
            ...     ContinuousLogicalTimeline(length=Fraction(16)), as_group="score"
            ... )
            >>> _ = bundle.add_timeline(ContinuousPhysicalTimeline(length=40.0))
            >>> data = bundle.to_dict()
            >>> AlignmentBundle.from_dict(data).to_dict() == data
            True
        """
        _require_keys(
            data, {"timelines", "bundle"}, {"graphical", "meta"}, "a bundle dictionary"
        )
        timelines = _roots_from_dicts(data["timelines"])
        bundle_data = _require_keys(
            data["bundle"],
            {"id", "name", "meta", "support_policy", "groups"},
            set(),
            "'bundle'",
        )
        if not isinstance(bundle_data["id"], str):
            raise ValueError(f"a bundle id must be a string, got {bundle_data['id']!r}")
        validate_id(bundle_data["id"])
        if bundle_data["name"] is not None and not isinstance(bundle_data["name"], str):
            raise ValueError("the bundle has a non-string name")
        if not isinstance(bundle_data["meta"], dict):
            raise ValueError("the bundle has a non-object meta")
        policies = [policy.value for policy in SupportPolicy]
        if bundle_data["support_policy"] not in policies:
            raise ValueError(
                f"unknown support_policy {bundle_data['support_policy']!r}; "
                f"expected one of {policies}"
            )
        groups = bundle_data["groups"]
        if not isinstance(groups, list):
            raise ValueError("the bundle's 'groups' must be an array")
        orders, styling = _read_graphical(data)
        if orders is None:
            orders = {}
        elif not isinstance(orders, dict):
            raise ValueError("'graphical.order' must map group ids to orders")
        group_ids = [
            group.get("id") if isinstance(group, dict) else None for group in groups
        ]
        unknown = sorted(str(key) for key in orders if key not in group_ids)
        if unknown:
            raise ValueError(f"'graphical.order' names no group: {unknown}")

        bundle = cls(
            id=bundle_data["id"],
            name=bundle_data["name"],
            meta=dict(bundle_data["meta"]),
            support_policy=SupportPolicy(bundle_data["support_policy"]),
            styling=styling,
        )
        for group_data, group_id in zip(groups, group_ids):
            order = orders.get(group_id) if isinstance(group_id, str) else None
            group = TimelineGroup._from_group_object(group_data, timelines, order)
            if group.id in bundle.groups:
                raise ValueError(f"group id {group.id!r} occurs twice")
            bundle.add_group(group)
        for timeline in timelines.values():
            if (
                timeline._parent is None
                and bundle.timelines.get(timeline.id) is not timeline
            ):
                bundle.add_timeline(timeline)
        return bundle

    @classmethod
    def from_bundles(
        cls,
        bundles: list["AlignmentBundle"],
        *,
        id: str = "",
        name: str | None = None,
    ) -> "AlignmentBundle":
        """Merge multiple bundles' groups, standalone timelines, and claims.

        Registers every group and standalone timeline from each source
        bundle into a new bundle, and carries over every cross-group
        ``MatchClaim`` (both the per-claim list and any columnar
        ``MatchClaimField`` stores) unchanged. Each source group is entered as
        a new group with the same members, name and meta; a minted group id
        is minted again in the merged bundle, a given one is kept. Groups from
        different source bundles remain distinct — merging does not itself
        align timelines across bundles. Add cross-group MatchClaims (e.g. via
        :meth:`create_match_claims`) afterwards to align them.

        Args:
            bundles: The bundles to merge, in order.
            id: Explicit ID for the merged bundle. Minted if omitted.
            name: Optional human-readable name for the merged bundle.

        Returns:
            A new AlignmentBundle containing every group, standalone
            timeline, and cross-group claim from ``bundles``.

        Raises:
            ValueError: If two source bundles share a given group ID, or a
                given timeline ID collision would occur across bundles.

        Examples:
            >>> merged = AlignmentBundle.from_bundles([score_bundle, audio_bundle])
            >>> merged.n_groups == score_bundle.n_groups + audio_bundle.n_groups
            True
        """
        merged = cls(id=id, name=name)
        for bundle in bundles:
            for group in bundle.groups.values():
                merged.add_group(
                    TimelineGroup(
                        id=None if group._id_minted else group.id,
                        name=group.name,
                        meta=group.meta,
                        timelines=list(group),
                    )
                )
            for timeline_id, timeline in bundle.timelines.items():
                if timeline_id in bundle.timeline_to_group:
                    continue
                if merged.timelines.get(timeline.id) is timeline:
                    continue
                merged.add_timeline(timeline)
            merged.add_match_claims(bundle.cross_group_claims)
            for claim_field in bundle.cross_group_claim_fields:
                merged.add_match_claim_field(claim_field)
        return merged

    # region Timeline Management

    def add_timeline(
        self,
        timeline: "Timeline",
        *,
        grouped_with: str | None = None,
        as_group: str | None = None,
    ) -> "AlignmentBundle":
        """Add a timeline, optionally grouped with an existing timeline.

        This is the primary method for adding timelines to the bundle.
        Timelines can be standalone or grouped with existing timelines.
        Group membership makes timelines commensurable through
        interpolation; it does not by itself assert any MatchClaim
        between them. A member spans its whole group (see
        :class:`~timetoalign.timelines.TimelineGroup`); to let only part of a
        timeline participate, add the child covering that part.

        A timeline is registered under its own ``id``. An ungrouped
        participant is always a root: a child added without ``grouped_with``
        or ``as_group`` registers its ROOT, and the child stays reachable as
        the root's descendant (exactly what the dictionary form restores). A
        child added to a group joins it, and its root joins the bundle as
        well, ungrouped unless it is grouped itself.

        Args:
            timeline: The Timeline to add.
            grouped_with: ID of an existing timeline to share a group with.
                If provided, both timelines become part of the same group.
                If the target timeline is not yet in a group, a new group
                is created with the target as its first member.
            as_group: Name of the group if a new one is created. The new
                group's id is minted, ``group_N``.

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If the timeline is already a grouped member of the
                bundle, or already in it while no group is requested; if a
                given id of its manifest tree — its own, its root's or any
                descendant's — is held by another timeline of the bundle (a
                minted id held there is minted again instead) or by a group.
            KeyError: If grouped_with references a non-existent timeline.

        Examples:
            Linear (full-extent) alignment:

                >>> from timetoalign.alignment import AlignmentBundle
                >>> from timetoalign.timelines.types import (
                ...     ContinuousPhysicalTimeline,
                ...     DiscreteGraphicalTimeline,
                ...     DiscreteLogicalTimeline,
                ... )
                >>> bundle = AlignmentBundle()
                >>> _ = bundle.add_timeline(DiscreteGraphicalTimeline(length=4000))
                >>> _ = bundle.add_timeline(
                ...     DiscreteLogicalTimeline(length=960), grouped_with="dgt_1"
                ... )
                >>> bundle.groups["group_1"].timeline_ids
                ['dgt_1', 'dlt_1']

            Partial alignment: the participating part is a child.

                >>> image = DiscreteGraphicalTimeline(length=300000)
                >>> holes = DiscreteGraphicalTimeline(length=277776)
                >>> image.add_child(holes, offset=15343)
                >>> audio = ContinuousPhysicalTimeline(length=600.0)
                >>> bundle = AlignmentBundle()
                >>> _ = bundle.add_timeline(audio, as_group="roll")
                >>> _ = bundle.add_timeline(holes, grouped_with="cpt_1")
                >>> sorted(bundle.timelines), bundle.timeline_to_group
                (['cpt_1', 'dgt_1', 'dgt_2'], {'cpt_1': 'group_1', 'dgt_2': 'group_1'})
        """
        if grouped_with is None and as_group is None:
            # An ungrouped participant is a root: a child joins through its
            # root and stays reachable as the root's descendant.
            timeline = _root(timeline)
        if grouped_with is not None and grouped_with not in self.timelines:
            raise KeyError(
                f"Cannot group with '{grouped_with}': not in bundle. "
                f"Available timelines: {list(self.timelines.keys())}"
            )
        if self.timelines.get(timeline.id) is timeline:
            if timeline.id in self.timeline_to_group or (
                grouped_with is None and as_group is None
            ):
                raise ValueError(f"Timeline '{timeline.id}' already exists in bundle")
        # The group refuses nesting; ask before anything changes.
        if grouped_with is not None:
            group_id = self.timeline_to_group.get(grouped_with)
            if group_id is not None:
                self.groups[group_id]._refuse_nested(timeline)
            elif _nested(self.timelines[grouped_with], timeline):
                raise ValueError(
                    f"Timeline {timeline.id!r} cannot be grouped with "
                    f"{grouped_with!r}: one is nested in the other, and a parent "
                    "and its child cannot share one start and end"
                )
        if self.timelines.get(timeline.id) is not timeline:
            self._admit(timeline)

        if grouped_with is not None and grouped_with in self.timeline_to_group:
            group = self.groups[self.timeline_to_group[grouped_with]]
            group.add_timeline(timeline)
        elif grouped_with is not None or as_group is not None:
            founders = [timeline]
            if grouped_with is not None:
                founders.insert(0, self.timelines[grouped_with])
            group = TimelineGroup(name=as_group, timelines=founders)
            group.id = self._free_group_id(group)
            self.groups[group.id] = group
            for founder in founders:
                self.timeline_to_group[founder.id] = group.id
        else:
            group = None

        self._register(timeline)
        if group is not None:
            self.timeline_to_group[timeline.id] = group.id
            self._logger.debug(f"Added timeline '{timeline.id}' to group '{group.id}'")
        else:
            self._logger.debug(f"Added standalone timeline '{timeline.id}'")
        return self

    def add_group(self, group: TimelineGroup) -> "AlignmentBundle":
        """Add a pre-built TimelineGroup with all its timelines at once.

        This is the bulk-registration counterpart of ``add_timeline(..., as_group=...)``.
        It registers the group and every timeline it contains — each under
        its own ``id``, with the root of every member that is a child — into
        the bundle's timeline and group registries.

        Args:
            group: A ``TimelineGroup`` that already contains timelines.

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If the group's given ID is already held in the bundle
                (a minted one is minted again instead); if a member is
                already a member of another group; or if a given id of a
                member's manifest tree is held by another timeline of the
                bundle (a minted id held there is minted again instead) or by
                a group.

        Examples:
            Add a recording group with three DPTs:

                >>> grp = TimelineGroup(timelines=[dpt_1, dpt_2, dpt_3])
                >>> bundle.add_group(grp)
        """
        if any(held is group for held in self.groups.values()):
            raise ValueError(f"Group '{group.id}' already exists in bundle")
        members = list(group)
        for member in members:
            if self.timelines.get(member.id) is member and (
                member.id in self.timeline_to_group
            ):
                raise ValueError(
                    f"Timeline '{member.id}' is already a member of group "
                    f"'{self.timeline_to_group[member.id]}'"
                )
        # Check every member and the group id before anything changes.
        for member in members:
            if self.timelines.get(member.id) is not member:
                self._admit(member, check_only=True)
        group.id = self._free_group_id(group)

        for member in members:
            if self.timelines.get(member.id) is not member:
                self._admit(member)
        group._rekey_members()

        self.groups[group.id] = group
        for member in members:
            self._register(member)
            self.timeline_to_group[member.id] = group.id

        self._logger.debug(f"Added group '{group.id}' with {len(members)} timelines")

        return self

    def _admit(self, timeline: "Timeline", *, check_only: bool = False) -> None:
        """Admit a timeline's manifest tree into the bundle's id scope.

        A given group id is reserved for its group; a minted group id that a
        newcomer's given id takes is minted again for the group.

        Args:
            timeline: The timeline joining the bundle.
            check_only: Raise as admitting would, but change nothing.

        Raises:
            ValueError: If a given id of the tree is held by another timeline
                or a group with a given id (minted timeline ids held there are
                minted again instead).
        """
        held = _held_ids(self.timelines.values())
        _admit_timeline_tree(
            timeline,
            held,
            f"bundle {self.id!r}",
            reserved=(
                group_id
                for group_id, group in self.groups.items()
                if not group._id_minted
            ),
            check_only=check_only,
        )
        if check_only:
            return
        tree_ids = {member.id for member in _tree(_root(timeline))}
        for group_id in [group_id for group_id in self.groups if group_id in tree_ids]:
            group = self.groups.pop(group_id)
            group.id = mint_id("group", set(self.groups) | set(held) | tree_ids)
            self.groups[group.id] = group
            for member_id, owner in self.timeline_to_group.items():
                if owner == group_id:
                    self.timeline_to_group[member_id] = group.id

    def _register(self, timeline: "Timeline") -> None:
        """Register a timeline, and its root when it is a child."""
        self.timelines[timeline.id] = timeline
        root = _root(timeline)
        if self.timelines.get(root.id) is not root:
            self.timelines[root.id] = root

    def _free_group_id(self, group: TimelineGroup) -> str:
        """Return an id for *group* that no group or timeline of the bundle holds.

        A free id is kept and a minted one that is held is minted again.

        Raises:
            ValueError: If the group's given id is held in the bundle.
        """
        taken = set(self.groups) | set(_held_ids(self.timelines.values()))
        if group.id not in taken:
            return group.id
        if not group._id_minted:
            raise ValueError(f"Group '{group.id}' already exists in bundle")
        return mint_id("group", taken | set(_held_ids(group)))

    def get_timeline(self, timeline_id: str) -> "Timeline":
        """Get a timeline by ID.

        Supports partial string and regex matching:
        1. Exact match: If ``timeline_id`` matches an ID exactly, returns it.
        2. Substring match: If ``timeline_id`` is a substring of exactly one ID,
           returns that timeline. If multiple match, returns the first and warns.
        3. Regex match: If ``timeline_id`` is a valid regex, matches via
           ``re.search()``. Same first-match logic with warning.

        Args:
            timeline_id: The timeline's unique identifier, or a partial/regex pattern.

        Returns:
            The Timeline object.

        Raises:
            KeyError: If no timeline matches the pattern.

        Examples:
            >>> bundle.get_timeline("clt_1")          # Exact match
            >>> bundle.get_timeline("score")         # Substring match
            >>> bundle.get_timeline(r"^perf:")       # Regex match
        """
        resolved_id = resolve_id(timeline_id, list(self.timelines.keys()))
        return self.timelines[resolved_id]

    def get_group(self, group_id: str) -> TimelineGroup:
        """Get a group by ID.

        Supports partial string and regex matching:
        1. Exact match: If ``group_id`` matches an ID exactly, returns it.
        2. Substring match: If ``group_id`` is a substring of exactly one ID,
           returns that group. If multiple match, returns the first and warns.
        3. Regex match: If ``group_id`` is a valid regex, matches via
           ``re.search()``. Same first-match logic with warning.

        Args:
            group_id: The group's unique identifier, or a partial/regex pattern.

        Returns:
            The TimelineGroup object.

        Raises:
            KeyError: If no group matches the pattern.

        Examples:
            >>> bundle.get_group("group_1")          # Exact match
            >>> bundle.get_group(r"_1$")             # Regex match
        """
        resolved_id = resolve_id(group_id, list(self.groups.keys()))
        return self.groups[resolved_id]

    def get_group_for_timeline(self, timeline_id: str) -> TimelineGroup | None:
        """Get the group containing a timeline.

        Args:
            timeline_id: The timeline's unique identifier.

        Returns:
            The TimelineGroup containing the timeline, or None if standalone.
        """
        group_id = self.timeline_to_group.get(timeline_id)
        if group_id is None:
            return None
        return self.groups.get(group_id)

    def get_timelines(self, ids: list[str]) -> list["Timeline"]:
        """Get multiple timelines by their IDs.

        Convenience method for retrieving several timelines at once.
        Each ID supports partial string and regex matching (via
        ``get_timeline()``).

        Args:
            ids: List of timeline IDs (or partial/regex patterns).

        Returns:
            List of Timeline objects in the same order as the input IDs.

        Raises:
            KeyError: If any timeline ID is not found.

        Examples:
            >>> timelines = bundle.get_timelines(["score", "audio", "midi"])
            >>> len(timelines)
            3
        """
        return [self.get_timeline(timeline_id) for timeline_id in ids]

    @property
    def n_timelines(self) -> int:
        """Number of timelines in this bundle."""
        return len(self.timelines)

    @property
    def n_groups(self) -> int:
        """Number of groups in this bundle."""
        return len(self.groups)

    @property
    def default_group(self) -> TimelineGroup | None:
        """Get the single group or primary group.

        Returns:
            The first/only group if one exists, None otherwise.
        """
        if not self.groups:
            return None
        return next(iter(self.groups.values()))

    @property
    def timeline_ids(self) -> list[str]:
        """List of all timeline IDs in the bundle."""
        return list(self.timelines.keys())

    @property
    def group_ids(self) -> list[str]:
        """List of all group IDs in the bundle."""
        return list(self.groups.keys())

    # endregion

    # region Coordinate Transfer

    def transfer(
        self,
        coord: CoordinateSpec,
        from_timeline: str,
        to_timeline: str,
    ) -> float | None:
        """Transfer a coordinate from one timeline to another.

        Automatically determines the conversion path:

        1. If both timelines are in the same group: typed group retrieval.
        2. If in different groups with MatchClaims: builds a ``MatchLine``
           and a cached ``WarpMap`` and calls it to
           interpolate the coordinate.
        3. If no path exists: returns ``None``.

        Low-level coordinate transfer utility. For user-facing coordinate
        queries, use ``get_matchstamp_at()`` which returns a full cross-section
        as a MatchStamp.

        Args:
            coord: The coordinate value to transfer.
            from_timeline: ID of the source timeline.
            to_timeline: ID of the target timeline.

        Returns:
            The transferred coordinate, or None if no path exists.

        Raises:
            KeyError: If either timeline is not in the bundle.
        """
        if from_timeline not in self.timelines:
            raise KeyError(
                f"Source timeline '{from_timeline}' not in bundle. "
                f"Available timelines: {list(self.timelines.keys())}"
            )
        if to_timeline not in self.timelines:
            raise KeyError(
                f"Target timeline '{to_timeline}' not in bundle. "
                f"Available timelines: {list(self.timelines.keys())}"
            )

        coord_value, _resolved_timeline_id, _unit = _resolve_coordinate_and_timeline(
            coord, from_timeline
        )
        source_timeline = self.get_timeline(from_timeline)
        if isinstance(coord, IdCoordinate):
            query_input = coord.to_coordinate()
        elif isinstance(coord, Coordinate):
            query_input = coord
        else:
            query_input = Coordinate(
                coord_value,
                source_timeline.unit,
                number_type=source_timeline.number_type,
            )
        query_coordinate = source_timeline.get_coordinate_at(
            query_input,
            format="coordinate",
        )
        assert isinstance(query_coordinate, Coordinate)
        coord = float(query_coordinate.value)

        # Same timeline: no conversion needed
        if from_timeline == to_timeline:
            return coord

        # Check if both are in the same group
        from_group_id = self.timeline_to_group.get(from_timeline)
        to_group_id = self.timeline_to_group.get(to_timeline)

        if from_group_id is not None and from_group_id == to_group_id:
            # Same group: direct conversion via group
            group = self.groups[from_group_id]
            converted = group.get_coordinate_at(
                IdCoordinate(
                    coord,
                    self.get_timeline(from_timeline).unit,
                    from_timeline,
                ),
                timeline_id=to_timeline,
                format="float",
            )
            return float(converted)

        # Cross-group transfer via MatchLine/WarpMap pipeline
        warp = self._get_or_build_warp_map(from_timeline, to_timeline)
        if warp is not None:
            try:
                return warp._interpolate_float(coord)
            except Exception as e:
                self._logger.warning(
                    "WarpMap forward failed for %s -> %s at %s: %s",
                    from_timeline,
                    to_timeline,
                    coord,
                    e,
                )
                return None

        # Try indirect via SOURCE group: convert within source, then warp
        if from_group_id is not None:
            source_group = self.groups[from_group_id]
            for src_other_tl_id in source_group.timeline_ids:
                if src_other_tl_id == from_timeline:
                    continue
                warp = self._get_or_build_warp_map(src_other_tl_id, to_timeline)
                if warp is None:
                    continue
                try:
                    intermediate = source_group.get_coordinate_at(
                        IdCoordinate(
                            coord,
                            self.get_timeline(from_timeline).unit,
                            from_timeline,
                        ),
                        timeline_id=src_other_tl_id,
                        format="coordinate",
                    )
                    assert isinstance(intermediate, Coordinate)
                    return warp._interpolate_float(intermediate.value)
                except Exception:
                    continue

        # Try indirect via TARGET group: warp to reachable group member,
        # then convert within the target group.
        if to_group_id is not None:
            target_group = self.groups[to_group_id]
            for tgt_other_tl_id in target_group.timeline_ids:
                if tgt_other_tl_id == to_timeline:
                    continue
                warp = self._get_or_build_warp_map(from_timeline, tgt_other_tl_id)
                if warp is None:
                    continue
                try:
                    warped = warp._interpolate_float(coord)
                    target_source = self.get_timeline(tgt_other_tl_id)
                    result = target_group.get_coordinate_at(
                        IdCoordinate(warped, target_source.unit, tgt_other_tl_id),
                        timeline_id=to_timeline,
                        format="coordinate",
                    )
                    assert isinstance(result, Coordinate)
                    return float(result.value)
                except Exception:
                    continue

        # A single exact claim cannot construct a WarpMap, but it is still a
        # direct reachability edge. Reuse the match-stamp closure for that
        # exact-anchor case, including chains through standalone timelines.
        stamp = self.get_matchstamp_at(query_coordinate, from_timeline)
        if isinstance(stamp, MatchStamp) and to_timeline in stamp.coordinates:
            return float(stamp.coordinates[to_timeline].value)

        self._logger.debug(
            "No transfer path between '%s' and '%s'",
            from_timeline,
            to_timeline,
        )
        return None

    def transfer_interval(
        self,
        start: CoordinateSpec,
        end: CoordinateSpec,
        from_timeline: str,
        to_timeline: str,
    ) -> tuple[float, float] | None:
        """Transfer an interval from one timeline to another.

        Args:
            start: Start coordinate in source timeline. Accepts a raw
                int/float/Fraction, a Coordinate, or an IdCoordinate.
            end: End coordinate in source timeline. Accepts a raw
                int/float/Fraction, a Coordinate, or an IdCoordinate.
            from_timeline: ID of the source timeline.
            to_timeline: ID of the target timeline.

        Returns:
            Tuple of (start, end) in target timeline, or None if no path.
        """
        start_value, _start_timeline_id, start_unit = _resolve_coordinate_and_timeline(
            start, from_timeline
        )
        end_value, _end_timeline_id, end_unit = _resolve_coordinate_and_timeline(
            end, from_timeline
        )
        source_timeline = self.get_timeline(from_timeline)
        start = float(
            source_timeline.get_coordinate_at(
                Coordinate(start_value, start_unit), format="coordinate"
            ).value
            if start_unit is not None
            else start_value
        )
        end = float(
            source_timeline.get_coordinate_at(
                Coordinate(end_value, end_unit), format="coordinate"
            ).value
            if end_unit is not None
            else end_value
        )
        transferred_start = self.transfer(start, from_timeline, to_timeline)
        transferred_end = self.transfer(end, from_timeline, to_timeline)

        if transferred_start is None or transferred_end is None:
            return None

        return (transferred_start, transferred_end)

    def are_commensurable(self, timeline_a: str, timeline_b: str) -> bool:
        """Check if two timelines can be connected via transfer.

        Two timelines are commensurable if they share the same group
        or if a cross-group path exists via MatchClaims.

        Args:
            timeline_a: First timeline ID.
            timeline_b: Second timeline ID.

        Returns:
            True if coordinates can be transferred between them.
        """
        if timeline_a == timeline_b:
            return True

        group_a = self.timeline_to_group.get(timeline_a)
        group_b = self.timeline_to_group.get(timeline_b)

        # Same group: commensurable
        if group_a is not None and group_a == group_b:
            return True

        # Check for cross-group path via claims
        if self.n_cross_group_claims == 0:
            return False

        # Check if any synchronous claim connects groups containing a and b
        group_a_tl_ids = (
            set(self.groups[group_a].timeline_ids) if group_a else {timeline_a}
        )
        group_b_tl_ids = (
            set(self.groups[group_b].timeline_ids) if group_b else {timeline_b}
        )

        for claim in self.cross_group_claims:
            if not claim.is_synchronous or claim.start_anchor is None:
                continue
            anchor = claim.start_anchor
            # Does this claim connect the two groups?
            a_side = (
                anchor.timeline_a_id in group_a_tl_ids
                or anchor.timeline_b_id in group_a_tl_ids
            )
            b_side = (
                anchor.timeline_a_id in group_b_tl_ids
                or anchor.timeline_b_id in group_b_tl_ids
            )
            if a_side and b_side:
                return True

        # Columnar stores answer the same question with one vectorized pass,
        # without materialising a claim.
        return any(
            claim_field.connects_groups(group_a_tl_ids, group_b_tl_ids)
            for claim_field in self.cross_group_claim_fields
        )

    # endregion

    # region Cross-Group Claims

    def create_match_claims(
        self,
        event_pairs: list[tuple[dict | str | None, str, dict | str | None, str]],
        *,
        synchronous: bool = True,
        agent: str = "user",
        agent_identifier: str = "manual",
    ) -> list[MatchClaim]:
        """Create MatchClaims from a list of event pairs.

        Convenience factory for creating multiple MatchClaims from paired
        events. Each tuple specifies two events and their timeline IDs.

        Args:
            event_pairs: List of tuples, each containing:
                ``(event_a, timeline_a_id, event_b, timeline_b_id)``.
                ``event_a``/``event_b`` are each one of:
                an event dict (must have at least a ``start`` key with a
                coordinate; an ``end`` key on both sides produces an
                interval match), the ``id`` string of an existing event on
                the paired timeline (resolved via ``Timeline.get_event()``),
                or ``None``. Exactly one of ``event_a``/``event_b`` may be
                ``None``, producing a NOMATCH claim for the other side's
                event. The forms may be mixed within a pair.
            synchronous: Whether the matches are temporally synchronous.
                Ignored for NOMATCH pairs, which are always non-synchronous.
            agent: Name of the agent creating the claims (for provenance).
            agent_identifier: The agent's stable identifier — a version string
                for a software agent or a URI for a human agent (e.g.
                ``"manual"``, ``"dynamic_time_warping"``). Stored as
                ``Agent.identifier``.

        Returns:
            List of MatchClaim objects. Also automatically adds them to
            the bundle's ``cross_group_claims``.

        Raises:
            ValueError: If event dicts are missing required keys, an event
                id string doesn't resolve to an existing event, or both
                ``event_a`` and ``event_b`` are ``None``.

        Examples:
            >>> pairs = [
            ...     ({"start": 0.0}, "score", {"start": 45.5}, "audio"),
            ...     ("note:000010", "score", "note:000042", "audio"),
            ...     ("note:000099", "score", None, "audio"),  # NOMATCH
            ... ]
            >>> claims = bundle.create_match_claims(pairs, agent="manual_alignment")
            >>> len(claims)
            3
        """
        from timetoalign.core import AgentType

        from .claims import Agent, MatchClaim, MatchMetadata

        claims = []
        for event_a, tl_a, event_b, tl_b in event_pairs:
            timeline_a = self.timelines[tl_a]
            timeline_b = self.timelines[tl_b]
            event_a = self._resolve_event(event_a, timeline_a, tl_a)
            event_b = self._resolve_event(event_b, timeline_b, tl_b)

            if event_a is None and event_b is None:
                raise ValueError("event_a and event_b cannot both be None")

            metadata = MatchMetadata(
                agent=Agent(
                    name=agent,
                    type=AgentType.software,
                    identifier=agent_identifier,
                ),
            )

            if event_b is None:
                claim = MatchClaim.nomatch(
                    event=event_a,
                    source_tl_id=tl_a,
                    target_tl_id=tl_b,
                    unit=timeline_a.unit,
                    metadata=metadata,
                )
            elif event_a is None:
                claim = MatchClaim.nomatch(
                    event=event_b,
                    source_tl_id=tl_b,
                    target_tl_id=tl_a,
                    unit=timeline_b.unit,
                    metadata=metadata,
                )
            else:
                # Auto-detect interval events: if both events have a non-null
                # "end" key, create an interval match with both start and end
                # anchors.  Event dicts from PyArrow may store coordinates as
                # structs ``{"value": float, ...}``; a None value means no end.
                end_key = None
                end_a = event_a.get("end")
                end_b = event_b.get("end")
                if end_a is not None and end_b is not None:
                    # Handle struct dicts: {"value": float, ...}
                    val_a = end_a["value"] if isinstance(end_a, dict) else end_a
                    val_b = end_b["value"] if isinstance(end_b, dict) else end_b
                    if val_a is not None and val_b is not None:
                        end_key = "end"
                claim = MatchClaim.from_events(
                    event_a=event_a,
                    tl_a_id=tl_a,
                    event_b=event_b,
                    tl_b_id=tl_b,
                    unit_a=timeline_a.unit,
                    unit_b=timeline_b.unit,
                    end_coord_key=end_key,
                    is_synchronous=synchronous,
                    metadata=metadata,
                )
            claims.append(claim)

        if claims:
            self.add_match_claims(claims)

        return claims

    @staticmethod
    def _resolve_event(
        event: dict | str | None, timeline: "Timeline", timeline_id: str
    ) -> dict | None:
        """Resolve an event-dict-or-id-or-None argument to an event dict.

        Args:
            event: An event dict, the ``id`` of an existing event, or
                ``None``.
            timeline: The timeline ``event`` belongs to.
            timeline_id: The timeline's id as given by the caller (for
                error messages).

        Returns:
            The event dict, or ``None`` if ``event`` was ``None``.

        Raises:
            ValueError: If ``event`` is an id string with no matching event.
        """
        if not isinstance(event, str):
            return event
        resolved = timeline.get_event(event)
        if resolved is None:
            raise ValueError(f"No event with id {event!r} on timeline {timeline_id!r}")
        return resolved

    def add_match_claims(
        self,
        claims: Any,
    ) -> "AlignmentBundle":
        """Add MatchClaims connecting timelines across different groups.

        MatchClaims encode coordinate correspondences between timelines in
        different groups (e.g., EEP recording notes matched to ABC score
        notes).  They enable cross-group coordinate transfer via
        ``MatchLine`` → ``WarpMap``.

        WarpMaps are built lazily on first ``transfer()`` or
        ``get_matchstamp_at()`` call, so adding claims is cheap.

        Args:
            claims: MatchClaim objects, claim tuples, or lists of claim tuples.
                A tuple is ``(timeline_a_id, interval_a, timeline_b_id,
                interval_b)``.

        Returns:
            self (for method chaining)
        """
        materialized = list(claims)
        flattened: list[Any] = []
        for item in materialized:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)

        normalized: list[MatchClaim] = []
        for item in flattened:
            if isinstance(item, MatchClaim):
                normalized.append(item)
                continue
            if not isinstance(item, tuple) or len(item) != 4:
                raise TypeError("Claims must be MatchClaims or four-item claim tuples")
            timeline_a_id, interval_a, timeline_b_id, interval_b = item
            start_anchor = AlignmentAnchor(
                timeline_a_id=str(timeline_a_id),
                coordinate_a=interval_a.start,
                timeline_b_id=str(timeline_b_id),
                coordinate_b=interval_b.start,
            )
            end_anchor = AlignmentAnchor(
                timeline_a_id=str(timeline_a_id),
                coordinate_a=interval_a.end,
                timeline_b_id=str(timeline_b_id),
                coordinate_b=interval_b.end,
            )
            normalized.append(
                MatchClaim(
                    timeline_a_id=str(timeline_a_id),
                    timeline_b_id=str(timeline_b_id),
                    start_anchor=start_anchor,
                    end_anchor=end_anchor,
                )
            )

        for claim in normalized:
            claim.set_bundle(self)
        self.cross_group_claims.extend(normalized)
        self._invalidate_warp_cache()
        return self

    def add_match_claim_field(
        self,
        claim_field: MatchClaimField,
    ) -> "AlignmentBundle":
        """Add a columnar ``MatchClaimField`` of cross-group claims.

        This is the columnar counterpart of :meth:`add_match_claims`.  The
        field is stored as-is (one PyArrow struct column) and queried
        vectorized by :meth:`get_matchstamp_at` /
        :meth:`_get_or_build_matchgraph`, which filter the struct column and
        materialise only the handful of claims at the queried coordinate.  A
        dense whole-work field (on the order of a million claims) is therefore
        never exploded into a Python list.

        It complements the per-claim list path: a bundle may hold both a
        Python-list ``cross_group_claims`` and one or more
        ``cross_group_claim_fields`` at once.

        Args:
            claim_field: A :class:`MatchClaimField` of synchronous pairwise
                claims.

        Returns:
            self (for method chaining)
        """
        self.cross_group_claim_fields.append(claim_field)
        self._invalidate_warp_cache()
        return self

    @property
    def n_cross_group_claims(self) -> int:
        """Total cross-group claims across BOTH claim stores.

        Sums the per-claim Python list and the row counts of every columnar
        ``MatchClaimField``.  The field contribution is a row count, so no
        claim is materialised — this is the cheap way to ask "does this
        bundle carry any alignment at all?".
        """
        return len(self.cross_group_claims) + sum(
            len(claim_field) for claim_field in self.cross_group_claim_fields
        )

    def _registered_id_pattern(self, id_pattern: str | None) -> str | None:
        """Narrow a timeline-id regex to the bundle's registered timelines.

        Claims may name timelines the bundle does not hold; a pattern only
        ever selects among the registered ones, so it is rewritten as an
        exact alternation of the registered ids it matches.
        """
        if id_pattern is None:
            return None

        pattern = re.compile(id_pattern)
        matched = {
            timeline_id for timeline_id in self.timelines if pattern.search(timeline_id)
        }
        if not matched:
            return r"(?!)"
        alternatives = "|".join(re.escape(timeline_id) for timeline_id in matched)
        return rf"^(?:{alternatives})\Z"

    def get_match_claims(
        self,
        *,
        timeline_id: str | None = None,
        timeline_ids: set[str] | None = None,
        id_pattern: str | None = None,
        between: tuple[str, str] | None = None,
        synchronous_only: bool = False,
        nomatch_only: bool = False,
        include_domains: set["Domain"] | None = None,
        include_units: set["TimeUnit"] | None = None,
    ) -> list[MatchClaim]:
        """Query MatchClaims connecting timelines across groups.

        This is the primary interface for accessing alignment information.
        All parameters are optional; when none are provided, returns all
        claims.

        Filters are combined with AND logic: a claim must satisfy every
        non-None criterion. Uses the Unified Filter API.

        Both claim stores are queried: the per-claim Python list and every
        columnar ``MatchClaimField``. The columnar matches are filtered
        vectorized but then **materialised**, one ``MatchClaim`` per surviving
        row — for a dense pairwise alignment that is O(n) Python objects. Use
        :meth:`get_claim_fields` when the columnar answer suffices, or narrow
        the query first.

        Args:
            timeline_id: Return claims involving this timeline ID.
            timeline_ids: Return claims involving any of these timeline IDs.
            id_pattern: Regex pattern matched against timeline IDs via
                ``re.search()``. Example: ``r"^perf:"`` matches all
                performance timelines.
            between: Return claims connecting exactly these two timeline IDs
                (order-independent).
            synchronous_only: Exclude non-synchronous (NOMATCH) claims.
            nomatch_only: Return only non-synchronous (NOMATCH) claims.
            include_domains: Only timelines in these domains.
            include_units: Only timelines with these units.

        Returns:
            Filtered list of MatchClaims.

        Examples:
            Get all synchronous claims for a performer::

                >>> claims = bundle.get_match_claims(
                ...     id_pattern=r"dlt_1$", synchronous_only=True
                ... )

            Get NOMATCH claims for a specific pair::

                >>> nomatches = bundle.get_match_claims(
                ...     between=("score:clt_1", "perf:dlt_5"),
                ...     nomatch_only=True,
                ... )
        """
        filt = self._claim_filter(
            timeline_id=timeline_id,
            timeline_ids=timeline_ids,
            id_pattern=id_pattern,
            between=between,
            synchronous_only=synchronous_only,
            nomatch_only=nomatch_only,
            include_domains=include_domains,
            include_units=include_units,
        )
        if filt is None:
            return []

        timeline_lookup = self.timelines
        claims = [
            c
            for c in self.cross_group_claims
            if filt.matches_claim(c, timelines=timeline_lookup)
        ]
        for claim_field in self._filtered_claim_fields(filt, timeline_lookup):
            claims.extend(claim_field.to_claims())
        return claims

    def get_claim_fields(
        self,
        *,
        timeline_id: str | None = None,
        timeline_ids: set[str] | None = None,
        id_pattern: str | None = None,
        between: tuple[str, str] | None = None,
        synchronous_only: bool = False,
        nomatch_only: bool = False,
        include_domains: set["Domain"] | None = None,
        include_units: set["TimeUnit"] | None = None,
    ) -> list[MatchClaimField]:
        """Query the columnar claim stores, materialising nothing.

        This is the scalable counterpart of :meth:`get_match_claims` for
        bundles whose claims live in a ``MatchClaimField`` — a dense pairwise
        alignment can hold hundreds of thousands of claims, and turning each
        one into a Python object costs orders of magnitude more than the
        columnar answer.  Every filter is applied as a vectorized PyArrow
        mask, and the result is a list of *filtered fields*, one per store
        that still has rows.

        The filter parameters are exactly those of :meth:`get_match_claims`
        and carry the same meaning.  Because a ``MatchClaimField`` holds
        synchronous claims only, ``synchronous_only`` is a no-op here and
        ``nomatch_only`` returns nothing.

        Args:
            timeline_id: Keep claims involving this timeline ID.
            timeline_ids: Keep claims involving any of these timeline IDs.
            id_pattern: Regex matched against timeline IDs via ``re.search()``.
            between: Keep claims connecting exactly these two timeline IDs
                (order-independent).
            synchronous_only: No-op (every stored claim is synchronous).
            nomatch_only: Returns an empty list.
            include_domains: Only timelines in these domains.
            include_units: Only timelines with these units.

        Returns:
            Filtered ``MatchClaimField`` objects, empty stores dropped.

        Examples:
            >>> fields = bundle.get_claim_fields(timeline_id="rec-a:cpt_1")
            >>> sum(len(f) for f in fields)
            66780
        """
        filt = self._claim_filter(
            timeline_id=timeline_id,
            timeline_ids=timeline_ids,
            id_pattern=id_pattern,
            between=between,
            synchronous_only=synchronous_only,
            nomatch_only=nomatch_only,
            include_domains=include_domains,
            include_units=include_units,
        )
        if filt is None:
            return []
        return self._filtered_claim_fields(filt, self.timelines)

    def _claim_filter(
        self,
        *,
        timeline_id: str | None,
        timeline_ids: set[str] | None,
        id_pattern: str | None,
        between: tuple[str, str] | None,
        synchronous_only: bool,
        nomatch_only: bool,
        include_domains: set["Domain"] | None,
        include_units: set["TimeUnit"] | None,
    ) -> ClaimFilter | None:
        """Build the ``ClaimFilter`` for the registered timelines.

        Every ID-shaped criterion selects among the bundle's registered
        timelines, once, for both claim stores.

        Returns:
            The filter, or ``None`` when a required timeline is unknown to the
            bundle and no claim can possibly match.
        """
        if timeline_id is not None and timeline_id not in self.timelines:
            return None
        if between is not None and any(
            between_id not in self.timelines for between_id in between
        ):
            return None
        return ClaimFilter.from_kwargs(
            timeline_id=timeline_id,
            timeline_ids=(
                {selected for selected in timeline_ids if selected in self.timelines}
                if timeline_ids is not None
                else None
            ),
            id_pattern=self._registered_id_pattern(id_pattern),
            between=between,
            synchronous_only=synchronous_only,
            nomatch_only=nomatch_only,
            include_domains=include_domains,
            include_units=include_units,
        )

    def _filtered_claim_fields(
        self,
        filt: ClaimFilter,
        timeline_lookup: dict[str, "Timeline"],
    ) -> list[MatchClaimField]:
        """Apply a ``ClaimFilter`` to every columnar claim store.

        The domain/unit criteria are resolved into a set of passing timeline
        IDs (the fields hold IDs, not timeline metadata); everything else maps
        straight onto ``MatchClaimField.filter``.  Stores that filter down to
        zero rows are dropped.
        """
        if not self.cross_group_claim_fields:
            return []
        within = filt.domain_unit_timeline_ids(timeline_lookup)
        filtered = (
            claim_field.filter(
                timeline_id=filt.timeline_id,
                timeline_ids=filt.timeline_ids,
                id_pattern=filt.id_pattern,
                between=filt.between,
                within=within,
                synchronous_only=filt.synchronous_only,
                nomatch_only=filt.nomatch_only,
            )
            for claim_field in self.cross_group_claim_fields
        )
        return [claim_field for claim_field in filtered if len(claim_field) > 0]

    def _axis_on(self, coordinate: float, timeline_id: str) -> int | float | Fraction:
        """Express a query coordinate the way *timeline_id* writes numbers.

        The bundle's own lookups — graph node keys, WarpMap feeds, anchor
        comparisons — run on floats, so a query stays float internally. The
        axis a stamp reports is a different thing: it has crossed onto a
        timeline's axis, and an axis writes every number in its declared
        representation. Without this, the same position answered differently
        depending on whether the caller happened to pass a float or a
        Fraction, which is the argument-type inference the boundary rule
        exists to remove.
        """
        timeline = self.timelines.get(timeline_id)
        if timeline is None:
            return coordinate
        return express_as(coordinate, timeline.number_type)

    @staticmethod
    def _claim_is_relevant_at(
        claim: MatchClaim, timeline_id: str, coordinate: Coordinate
    ) -> bool:
        """Return whether one native-unit claim is relevant to a query."""
        if not claim.is_synchronous or claim.start_anchor is None:
            return False
        if timeline_id not in claim.timelines:
            return False
        if claim.is_interval:
            interval = claim.get_interval_for(timeline_id)
            return interval.start.value <= coordinate.value <= interval.end.value
        stored = claim.start_anchor.get_coordinate_for(timeline_id, format="coordinate")
        assert isinstance(stored, Coordinate)
        return stored.value == coordinate.value

    def _relevant_claims_at(
        self, timeline_id: str, coordinate: Coordinate
    ) -> list[MatchClaim]:
        """Find relevant claims in deterministic store and row order."""
        relevant: list[MatchClaim] = []
        for raw_claim in self.cross_group_claims:
            claim = self._claim_in_native_units(raw_claim)
            if self._claim_is_relevant_at(claim, timeline_id, coordinate):
                relevant.append(claim)
        for claim_field in self.cross_group_claim_fields:
            for raw_claim in claim_field.at(timeline_id, coordinate):
                claim = self._claim_in_native_units(raw_claim)
                if self._claim_is_relevant_at(claim, timeline_id, coordinate):
                    relevant.append(claim)
        return relevant

    def _interval_claim_candidate(
        self,
        claim: MatchClaim,
        queried_timeline_id: str,
        query: Coordinate,
    ) -> tuple[str, Coordinate | None, bool]:
        """Map one query through an interval claim onto its counterpart axis.

        Exact endpoints are returned directly. Interior arithmetic stays in the
        operands' numeric domain, then crosses the counterpart timeline's axis
        boundary exactly once. A point on the queried side paired with a
        nonzero counterpart interval has no single mapped coordinate.

        Returns:
            Counterpart timeline ID, mapped coordinate or ``None``, and whether
            the query is one of the claim's stored anchors.
        """
        if not claim.is_interval or queried_timeline_id not in claim.timelines:
            raise ValueError("Interval claim does not contain the queried timeline")
        queried_interval = claim.get_interval_for(queried_timeline_id)
        counterpart_id = (
            claim.timeline_b_id
            if queried_timeline_id == claim.timeline_a_id
            else claim.timeline_a_id
        )
        counterpart_interval = claim.get_interval_for(counterpart_id)
        query_value = query.value
        queried_start = queried_interval.start.value
        queried_end = queried_interval.end.value
        counterpart_start = counterpart_interval.start.value
        counterpart_end = counterpart_interval.end.value
        is_anchor = query_value == queried_start or query_value == queried_end

        if queried_start == queried_end:
            if counterpart_start != counterpart_end:
                return counterpart_id, None, is_anchor
            mapped = counterpart_start
        elif query_value == queried_start:
            mapped = counterpart_start
        elif query_value == queried_end:
            mapped = counterpart_end
        else:
            mapped = counterpart_start + (
                (query_value - queried_start) / (queried_end - queried_start)
            ) * (counterpart_end - counterpart_start)

        timeline = self.get_timeline(counterpart_id)
        canonical = self._axis_on(mapped, counterpart_id)
        return (
            counterpart_id,
            Coordinate(
                canonical,
                timeline.unit,
                number_type=timeline.number_type,
            ),
            is_anchor,
        )

    def _get_or_build_matchgraph(
        self,
        timeline_id: str,
        coordinate: CoordinateValue,
    ) -> MatchGraph:
        """Get or build the MatchGraph containing (timeline_id, coordinate).

        Checks the cache first. On cache miss, finds all synchronous
        cross-group claims whose anchors touch this coordinate on this
        timeline (or any timeline connected at this coordinate via
        transitive closure), builds a MatchGraph from them, and caches
        it keyed by ALL (timeline_id, coordinate) nodes in the resulting
        graph.

        Args:
            timeline_id: A timeline in the bundle.
            coordinate: The coordinate to look up.

        Returns:
            The MatchGraph for this coordinate's connected component.

        Raises:
            ValueError: If no synchronous claims touch this node.
        """
        key = (timeline_id, coordinate)
        if key in self._matchgraph_cache:
            return self._matchgraph_cache[key]

        timeline = self.get_timeline(timeline_id)
        query = Coordinate(coordinate, timeline.unit, number_type=timeline.number_type)
        relevant_claims = [
            claim
            for claim in self._relevant_claims_at(timeline_id, query)
            if not claim.is_interval
        ]

        if not relevant_claims:
            raise ValueError(
                f"No synchronous claims touch ({timeline_id!r}, {coordinate}) "
                f"in this bundle."
            )

        # Build graph — transitive closure connects all timelines at
        # this coordinate
        mg = MatchGraph(
            claims=relevant_claims,
            units=self._get_unit_map(),
            axis=self._axis_on(coordinate, timeline_id),
            source=self,
            source_id=timeline_id,
        )

        # Cache keyed by ALL nodes in the graph
        for node in mg._graph.nodes():
            node_tl, node_coord = node
            self._matchgraph_cache[(node_tl, node_coord)] = mg

        return mg

    def get_coordinate_at(
        self,
        at: CoordinateInput,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | pd.Series:
        """Resolve one coordinate onto a requested timeline axis.

        Args:
            at: Raw, plain, or ID-bearing coordinate input.
            timeline_id: Requested result timeline ID.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            One coordinate projection or a length-one Series.
        """
        if not (
            not isinstance(at, bool)
            and isinstance(at, (int, float, Fraction, Coordinate))
        ):
            raise TypeError("get_coordinate_at requires one scalar coordinate input")
        if isinstance(at, IdCoordinate):
            source_id = at.timeline_id
            result_id = timeline_id or source_id
        else:
            if timeline_id is None:
                raise ValueError(
                    "timeline_id is required for raw or plain bundle coordinate queries"
                )
            source_id = timeline_id
            result_id = timeline_id
        if source_id not in self.timelines:
            raise KeyError(f"Unknown source timeline ID {source_id!r} in {self.id!r}")
        if result_id not in self.timelines:
            raise KeyError(f"Unknown result timeline ID {result_id!r} in {self.id!r}")
        if source_id == result_id:
            plain = at.to_coordinate() if isinstance(at, IdCoordinate) else at
            result = self.get_timeline(result_id).get_coordinate_at(
                plain,
                timeline_id=self.get_timeline(result_id).id,
                format="coordinate",
                rounding=rounding,
            )
            assert isinstance(result, Coordinate)
            identified = IdCoordinate.from_coordinate(result, result_id)
            return format_coordinates(
                [identified],
                format=format,
                rounding=rounding,
                scalar=True,
                series_name=result_id,
            )
        stamp = self.get_matchstamp_at(at, source_id)
        return stamp.get_coordinate_for(result_id, format=format, rounding=rounding)

    def get_coordinates_at(
        self,
        at: CoordinateCollection,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> list[CoordinateResult] | pd.Series:
        """Resolve a coordinate collection onto one timeline axis.

        Args:
            at: Coordinate positions to resolve atomically.
            timeline_id: Requested result timeline ID.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            A list of projections or canonical-value Series.
        """
        values, index = validate_coordinate_collection(at)
        if not values and timeline_id is None:
            raise ValueError("timeline_id is required for an empty bundle query")
        if timeline_id is not None and timeline_id not in self.timelines:
            raise KeyError(f"Unknown result timeline ID {timeline_id!r} in {self.id!r}")
        results: list[IdCoordinate] = []
        for value in values:
            result = self.get_coordinate_at(
                value,
                timeline_id=timeline_id,
                format="id_coordinate",
                rounding=rounding,
            )
            assert isinstance(result, IdCoordinate)
            results.append(result)
        return format_coordinates(
            results,
            format=format,
            rounding=rounding,
            scalar=False,
            index=index,
            series_name=timeline_id
            or (
                results[0].timeline_id
                if results
                and all(
                    result.timeline_id == results[0].timeline_id for result in results
                )
                else "coordinate"
            ),
            empty_number_type=(
                self.get_timeline(timeline_id).number_type
                if timeline_id is not None
                else None
            ),
        )

    def _event_owners(self, key: str) -> list[str]:
        """Return the IDs of the registered timelines owning an event key."""
        return [
            timeline_id
            for timeline_id, timeline in self.timelines.items()
            if timeline.get_event(key) is not None
        ]

    def get_coordinate_for(
        self,
        key: str,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | pd.Series:
        """Return a uniquely owned event's start on a selected bundle axis.

        Args:
            key: Event ID with exactly one owning registered timeline.
            timeline_id: Requested result timeline ID, or the owner by default.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            One event-start projection or a length-one Series.
        """
        if not isinstance(key, str):
            raise TypeError("get_coordinate_for requires an event-ID string")
        owners = self._event_owners(key)
        if not owners:
            raise KeyError(f"Event {key!r} not found in bundle {self.id!r}")
        if len(owners) != 1:
            raise ValueError(f"Event {key!r} has competing owners {owners}")
        owner = owners[0]
        source = self.get_timeline(owner).get_coordinate_for(
            key, format="coordinate", rounding=rounding
        )
        assert isinstance(source, Coordinate)
        result = self.get_coordinate_at(
            IdCoordinate.from_coordinate(source, owner),
            timeline_id=timeline_id or owner,
            format="id_coordinate",
            rounding=rounding,
        )
        assert isinstance(result, IdCoordinate)
        return format_coordinates(
            [result],
            format=format,
            rounding=rounding,
            scalar=True,
            index=pd.Index([key]) if format == "series" else None,
            series_name="coordinate",
        )

    def get_coordinates_for(
        self,
        keys: KeyCollection,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> list[CoordinateResult] | pd.Series:
        """Return event-start coordinates for a key collection.

        Args:
            keys: Event IDs to resolve atomically.
            timeline_id: Requested result timeline ID, or each owner by default.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            A list of projections or canonical-value Series.
        """
        values, index = validate_key_collection(keys)
        results: list[IdCoordinate] = []
        for key in values:
            result = self.get_coordinate_for(
                key,
                timeline_id=timeline_id,
                format="id_coordinate",
                rounding=rounding,
            )
            assert isinstance(result, IdCoordinate)
            results.append(result)
        if format == "series" and index is None:
            index = pd.Index(values)
        return format_coordinates(
            results,
            format=format,
            rounding=rounding,
            scalar=False,
            index=index,
            series_name="coordinate",
            empty_number_type=(
                self.get_timeline(timeline_id).number_type
                if timeline_id is not None
                else None
            ),
        )

    def get_coordinate(
        self,
        at: CoordinateInput | CoordinateCollection | str | KeyCollection,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | list[CoordinateResult] | pd.Series:
        """Dispatch a positional or event-key bundle coordinate query.

        Args:
            at: Scalar or plural coordinate position or event key.
            timeline_id: Requested result timeline ID when required.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            The selected precise-getter result.
        """
        return dispatch_retrieval(
            self,
            "get_coordinate",
            "get_coordinates",
            at,
            timeline_id,
            format=format,
            rounding=rounding,
        )

    def get_matchstamp_at(
        self,
        at: CoordinateSpec | Interval,
        timeline_id: str | None = None,
        *,
        support_policy: "SupportPolicy | str | None" = None,
        conversion_maps: "ConversionMapsSpec" = False,
        timeline_ids: set[str] | None = None,
        id_pattern: str | None = None,
        include_domains: set["Domain"] | None = None,
        include_units: set["TimeUnit"] | None = None,
    ) -> MatchStamp | MatchIntervalStamp:
        """Get a cross-group match stamp for a position on a timeline.

        Coordinate input is the primary interface for cross-domain coordinate
        transfer and returns coordinates on every connected timeline. Interval
        input resolves its endpoints through that same coordinate lane and
        combines the two results.

        A coordinate stamp is the **transitive cross-group union** reachable
        from the query. Exact-anchor coordinates from the coordinate's
        MatchGraph are overlaid first (exact wins over interpolated), then the
        assembly walks outward across groups to closure: every reached timeline
        expands into its own group (by interpolation) and warps into
        not-yet-reached groups (by WarpMap). ``is_interpolated`` is ``False``
        exactly when the query node itself carries an explicit anchor.

        A timeline whose transferred coordinate falls outside alignment
        support — the entering coordinate lies outside the transferring
        WarpMap's source-anchor hull, or the produced coordinate would fall
        outside ``[0, length]`` — is handled by ``support_policy``. No policy
        ever yields a negative coordinate or one beyond a timeline's length.

        Args:
            at: The query position. Can be:

                - int/float/Fraction: Raw value, ``timeline_id`` required.
                - Coordinate: Value with unit, ``timeline_id`` required.
                - IdCoordinate: Value with unit AND timeline_id
                  (``timeline_id`` param optional).
                - Interval: Two coordinate positions, ``timeline_id`` required.
            timeline_id: ID of the source timeline. Required unless
                ``at`` is an ``IdCoordinate``, in which case the
                coordinate's own ``timeline_id`` is used.
            support_policy: How to treat out-of-support timelines
                (``omit`` / ``clamp`` / ``extrapolate``). Accepts a
                :class:`~timetoalign.core.SupportPolicy` or its string name.
                ``None`` (the default) uses the bundle's ``support_policy``
                setting, itself ``omit`` by default. The query timeline's own
                coordinate is never dropped, clamped, or altered.
            conversion_maps: C-map conversions available through unit lookup
                and display. Opt-in: defaults to ``False``.
            timeline_ids: Only include these timeline IDs in the result.
            id_pattern: Regex filter for timeline IDs in the result.
            include_domains: Only these domains in the result.
            include_units: Only these units in the result.

        Returns:
            A coordinate query returns a MatchStamp. An Interval query returns
            a MatchIntervalStamp combining its independently resolved endpoint
            MatchStamps.

        Raises:
            TypeError: If ``at`` is not a supported coordinate or Interval.
            ValueError: If timeline_id is None and ``at`` is not an
                IdCoordinate.
            KeyError: If timeline_id is not in the bundle.

        Examples:
            >>> ms = bundle.get_matchstamp_at(10.0, "score:clt_1")
            >>> ms.n_timelines
            23  # score + 22 performers
        """
        if isinstance(at, Interval):
            if timeline_id is None:
                raise ValueError(
                    "timeline_id is required unless coordinate is an IdCoordinate"
                )
            start_stamp = self.get_matchstamp_at(
                at.start,
                timeline_id,
                support_policy=support_policy,
                conversion_maps=conversion_maps,
                timeline_ids=timeline_ids,
                id_pattern=id_pattern,
                include_domains=include_domains,
                include_units=include_units,
            )
            end_stamp = self.get_matchstamp_at(
                at.end,
                timeline_id,
                support_policy=support_policy,
                conversion_maps=conversion_maps,
                timeline_ids=timeline_ids,
                id_pattern=id_pattern,
                include_domains=include_domains,
                include_units=include_units,
            )
            assert type(start_stamp) is MatchStamp
            assert type(end_stamp) is MatchStamp
            queried_interval = Interval(
                start_stamp.coordinates[timeline_id],
                end_stamp.coordinates[timeline_id],
            )
            return MatchIntervalStamp(
                source_id=timeline_id,
                interval=queried_interval,
                start_stamp=start_stamp,
                end_stamp=end_stamp,
            )

        coordinate_value, timeline_id, _unit = _resolve_coordinate_and_timeline(
            at, timeline_id
        )
        if timeline_id is None:
            raise ValueError(
                "timeline_id is required unless coordinate is an IdCoordinate"
            )

        if timeline_id not in self.timelines:
            raise KeyError(
                f"Timeline '{timeline_id}' not in bundle. "
                f"Available timelines: {list(self.timelines.keys())}"
            )

        source_timeline = self.get_timeline(timeline_id)
        if isinstance(at, IdCoordinate):
            query_input = at.to_coordinate()
        elif isinstance(at, Coordinate):
            query_input = at
        else:
            query_input = Coordinate(
                coordinate_value,
                source_timeline.unit,
                number_type=source_timeline.number_type,
            )
        query_coordinate = source_timeline.get_coordinate_at(
            query_input,
            format="coordinate",
        )
        assert isinstance(query_coordinate, Coordinate)
        queried = query_coordinate.value

        relevant_claims = self._relevant_claims_at(timeline_id, query_coordinate)
        interval_claims = [claim for claim in relevant_claims if claim.is_interval]
        candidate_groups: dict[
            str, list[tuple[MatchClaim, Coordinate | None, bool]]
        ] = {}
        for claim in interval_claims:
            target_id, candidate, is_anchor = self._interval_claim_candidate(
                claim, timeline_id, query_coordinate
            )
            candidate_groups.setdefault(target_id, []).append(
                (claim, candidate, is_anchor)
            )

        interval_coordinates: dict[str, CoordinateValue] = {}
        interval_anchor_edges: list[tuple[str, str]] = []
        interval_inferred_edges: list[tuple[str, str]] = []
        for target_id, candidates in candidate_groups.items():
            values = [candidate for _, candidate, _ in candidates]
            if any(candidate is None for candidate in values):
                continue
            unique: list[Coordinate] = []
            for candidate in values:
                assert candidate is not None
                if candidate not in unique:
                    unique.append(candidate)
            if len(unique) != 1:
                continue
            interval_coordinates[target_id] = unique[0].value
            edge = (timeline_id, target_id)
            if any(is_anchor for _, _, is_anchor in candidates):
                interval_anchor_edges.append(edge)
            else:
                interval_inferred_edges.append(edge)

        policy = (
            self.support_policy
            if support_policy is None
            else self._coerce_support_policy(support_policy)
        )

        coordinates, anchor_edges, inferred_edges, query_has_anchor = (
            self._assemble_matchstamp(
                queried,
                timeline_id,
                support_policy=policy,
                conversion_maps=conversion_maps,
                interval_coordinates=interval_coordinates,
                interval_anchor_edges=interval_anchor_edges,
                interval_inferred_edges=interval_inferred_edges,
                query_has_interval_anchor=any(
                    query_coordinate.value
                    in (
                        claim.get_interval_for(timeline_id).start.value,
                        claim.get_interval_for(timeline_id).end.value,
                    )
                    for claim in interval_claims
                ),
            )
        )

        values = {
            coordinate_id: value
            for coordinate_id, value in coordinates.items()
            if value is not None
        }
        declared = [held for held in self.timelines if held in values]
        ordered_ids = [timeline_id]
        ordered_ids.extend(held for held in declared if held != timeline_id)
        ordered_ids.extend(sorted(set(values).difference(ordered_ids)))
        typed_coordinates: dict[str, Coordinate] = {}
        for ordered_id in ordered_ids:
            value = queried if ordered_id == timeline_id else values[ordered_id]
            timeline = self.get_timeline(ordered_id)
            typed_coordinates[ordered_id] = Coordinate(
                value,
                timeline.unit,
                number_type=timeline.number_type,
            )
        stamp = MatchStamp(
            coordinates=typed_coordinates,
            source_id=timeline_id,
            anchor_edges=list(anchor_edges),
            inferred_edges=list(inferred_edges),
            source=self,
            is_interpolated=not query_has_anchor,
            conversion_maps=conversion_maps,
            interval_claims=list(interval_claims),
        )

        # Apply post-hoc filtering if requested
        has_filter = (
            timeline_ids is not None
            or id_pattern is not None
            or include_domains is not None
            or include_units is not None
        )
        if has_filter:
            filt = ClaimFilter.from_kwargs(
                timeline_ids=(
                    {
                        selected
                        for selected in timeline_ids
                        if selected in self.timelines
                    }
                    if timeline_ids is not None
                    else None
                ),
                id_pattern=self._registered_id_pattern(id_pattern),
                include_domains=include_domains,
                include_units=include_units,
            )
            filter_timelines = self.timelines
            filtered_coords = {
                tl_id: coord
                for tl_id, coord in stamp.coordinates.items()
                if filt.matches_timeline(
                    tl_id,
                    timelines=filter_timelines,
                )
            }
            remaining = set(filtered_coords.keys())
            stamp = MatchStamp(
                coordinates=filtered_coords,
                source_id=(
                    stamp.source_id
                    if stamp.source_id in filtered_coords
                    else next(iter(filtered_coords))
                ),
                anchor_edges=[
                    (a, b)
                    for a, b in stamp.anchor_edges
                    if a in remaining and b in remaining
                ],
                inferred_edges=[
                    (a, b)
                    for a, b in stamp.inferred_edges
                    if a in remaining and b in remaining
                ],
                source=stamp.source,
                is_interpolated=stamp.is_interpolated,
                conversion_maps=stamp.conversion_maps,
                interval_claims=(
                    stamp.interval_claims if stamp.source_id in filtered_coords else []
                ),
            )

        return stamp

    @staticmethod
    def _coerce_support_policy(
        policy: "SupportPolicy | str",
    ) -> SupportPolicy:
        """Coerce a policy argument to a :class:`SupportPolicy` member."""
        return policy if isinstance(policy, SupportPolicy) else SupportPolicy(policy)

    def get_matchstamps_at(
        self,
        at: CoordinateCollection | Sequence[CoordinateInput | Interval],
        timeline_id: str | None = None,
        *,
        support_policy: "SupportPolicy | str | None" = None,
        conversion_maps: "ConversionMapsSpec" = False,
        timeline_ids: set[str] | None = None,
        id_pattern: str | None = None,
        include_domains: set["Domain"] | None = None,
        include_units: set["TimeUnit"] | None = None,
    ) -> list[MatchStamp | MatchIntervalStamp]:
        """Get cross-group MatchStamps at a collection of positions.

        Every element is resolved through :meth:`get_matchstamp_at`, so each
        one yields exactly one full transitive cross-section, and input order
        is preserved.

        Args:
            at: Coordinate or Interval positions to resolve. Each element is a raw
                ``int``/``float``/``Fraction`` or ``Coordinate`` (needing
                ``timeline_id``), an ``IdCoordinate`` (carrying its own), or an
                ``Interval`` (needing ``timeline_id``).
            timeline_id: ID of the source timeline.
            support_policy: How to treat out-of-support timelines.
            conversion_maps: C-map conversions available through the stamps.
            timeline_ids: Only include these timeline IDs in each result.
            id_pattern: Regex filter for timeline IDs in each result.
            include_domains: Only these domains in each result.
            include_units: Only these units in each result.

        Returns:
            One stamp per input position. Coordinate elements produce
            MatchStamps and Interval elements produce MatchIntervalStamps.

        Examples:
            >>> stamps = bundle.get_matchstamps_at([0.0, 50.0], "score:clt_1")
            >>> stamps[1].get_coordinate_for("score:clt_1", format="float")
            50.0
        """
        values, _ = collection_values(at)
        for position, value in enumerate(values):
            if not is_coordinate_input(value) and not isinstance(value, Interval):
                raise TypeError(
                    f"Matchstamp collection element {position} has unsupported "
                    f"type {type(value).__name__}"
                )
        return [
            self.get_matchstamp_at(
                value,
                timeline_id,
                support_policy=support_policy,
                conversion_maps=conversion_maps,
                timeline_ids=timeline_ids,
                id_pattern=id_pattern,
                include_domains=include_domains,
                include_units=include_units,
            )
            for value in values
        ]

    def get_matchstamp_for(
        self,
        key: str,
        timeline_id: str | None = None,
        *,
        support_policy: "SupportPolicy | str | None" = None,
        conversion_maps: "ConversionMapsSpec" = False,
        timeline_ids: set[str] | None = None,
        id_pattern: str | None = None,
        include_domains: set["Domain"] | None = None,
        include_units: set["TimeUnit"] | None = None,
    ) -> MatchStamp:
        """Get the cross-group MatchStamp at a uniquely owned event.

        Args:
            key: Event ID with exactly one owning registered timeline.
            timeline_id: ID of the timeline the stamp is anchored on; by default the
                event's owner.
            support_policy: How to treat out-of-support timelines.
            conversion_maps: C-map conversions available through the stamp.
            timeline_ids: Only include these timeline IDs in the result.
            id_pattern: Regex filter for timeline IDs in the result.
            include_domains: Only these domains in the result.
            include_units: Only these units in the result.

        Returns:
            MatchStamp spanning all connected timelines.

        Raises:
            KeyError: If no registered timeline owns the event.
            ValueError: If several registered timelines own it.

        Examples:
            >>> ms = bundle.get_matchstamp_for("note:000001")
            >>> ms.get_coordinate_for(ms.source_id)  # the event's own owner axis
        """
        if not isinstance(key, str):
            raise TypeError("get_matchstamp_for requires an event-ID string")
        owners = self._event_owners(key)
        if not owners:
            raise KeyError(f"Event {key!r} not found in bundle {self.id!r}")
        if len(owners) != 1:
            raise ValueError(f"Event {key!r} has competing owners {owners}")
        owner = owners[0]
        source = self.get_timeline(owner).get_coordinate_for(key, format="coordinate")
        assert isinstance(source, Coordinate)
        return self.get_matchstamp_at(
            IdCoordinate.from_coordinate(source, owner),
            timeline_id or owner,
            support_policy=support_policy,
            conversion_maps=conversion_maps,
            timeline_ids=timeline_ids,
            id_pattern=id_pattern,
            include_domains=include_domains,
            include_units=include_units,
        )

    def get_matchstamps_for(
        self,
        keys: KeyCollection,
        timeline_id: str | None = None,
        *,
        support_policy: "SupportPolicy | str | None" = None,
        conversion_maps: "ConversionMapsSpec" = False,
        timeline_ids: set[str] | None = None,
        id_pattern: str | None = None,
        include_domains: set["Domain"] | None = None,
        include_units: set["TimeUnit"] | None = None,
    ) -> list[MatchStamp]:
        """Get cross-group MatchStamps for a collection of event IDs.

        Args:
            keys: Event IDs to resolve, each with exactly one owner.
            timeline_id: ID of the timeline every stamp is anchored on; by default
                each event's own owner.
            support_policy: How to treat out-of-support timelines.
            conversion_maps: C-map conversions available through the stamps.
            timeline_ids: Only include these timeline IDs in each result.
            id_pattern: Regex filter for timeline IDs in each result.
            include_domains: Only these domains in each result.
            include_units: Only these units in each result.

        Returns:
            One MatchStamp per key, in input order; an empty collection gives
            an empty list.

        Raises:
            KeyError: If any key is unknown. The batch answers completely or
                not at all.
        """
        stamps, _ = resolve_key_collection(
            keys,
            lambda key: self.get_matchstamp_for(
                key,
                timeline_id,
                support_policy=support_policy,
                conversion_maps=conversion_maps,
                timeline_ids=timeline_ids,
                id_pattern=id_pattern,
                include_domains=include_domains,
                include_units=include_units,
            ),
        )
        return stamps

    def get_matchstamp(
        self,
        at: (
            CoordinateInput
            | Interval
            | CoordinateCollection
            | Sequence[CoordinateInput | Interval]
            | str
            | KeyCollection
        ),
        timeline_id: str | None = None,
        *,
        support_policy: "SupportPolicy | str | None" = None,
        conversion_maps: "ConversionMapsSpec" = False,
        timeline_ids: set[str] | None = None,
        id_pattern: str | None = None,
        include_domains: set["Domain"] | None = None,
        include_units: set["TimeUnit"] | None = None,
    ) -> MatchStamp | MatchIntervalStamp | list[MatchStamp | MatchIntervalStamp]:
        """Dispatch a positional or event-key match-stamp query.

        Args:
            at: Scalar or plural coordinate or Interval position, or event key.
            timeline_id: ID of the timeline the query is expressed on.
            support_policy: How to treat out-of-support timelines.
            conversion_maps: C-map conversions available through the stamps.
            timeline_ids: Only include these timeline IDs in each result.
            id_pattern: Regex filter for timeline IDs in each result.
            include_domains: Only these domains in each result.
            include_units: Only these units in each result.

        Returns:
            The selected precise-getter result.

        Raises:
            TypeError: If ``at`` mixes keys with position forms or is an
                unsupported runtime form.
        """
        options = {
            "support_policy": support_policy,
            "conversion_maps": conversion_maps,
            "timeline_ids": timeline_ids,
            "id_pattern": id_pattern,
            "include_domains": include_domains,
            "include_units": include_units,
        }
        if isinstance(at, str):
            return self.get_matchstamp_for(at, timeline_id, **options)
        if is_coordinate_input(at) or isinstance(at, Interval):
            return self.get_matchstamp_at(at, timeline_id, **options)
        values, _ = collection_values(at)
        if all(isinstance(value, str) for value in values):
            return self.get_matchstamps_for(values, timeline_id, **options)
        if all(
            is_coordinate_input(value) or isinstance(value, Interval)
            for value in values
        ):
            return self.get_matchstamps_at(values, timeline_id, **options)
        raise TypeError(
            "Matchstamp dispatcher collections must contain only string keys or "
            "coordinate and Interval positions"
        )

    def _assemble_matchstamp(
        self,
        coordinate: CoordinateValue,
        actual_timeline_id: str,
        *,
        support_policy: SupportPolicy,
        conversion_maps: "ConversionMapsSpec",
        interval_coordinates: dict[str, CoordinateValue] | None = None,
        interval_anchor_edges: list[tuple[str, str]] | None = None,
        interval_inferred_edges: list[tuple[str, str]] | None = None,
        query_has_interval_anchor: bool = False,
    ) -> tuple[
        dict[str, CoordinateValue],
        list[tuple[str, str]],
        list[tuple[str, str]],
        bool,
    ]:
        """Assemble the transitive cross-group union reachable from a query.

        Seeds from the query node, overlays exact-anchor coordinates from the
        query's MatchGraph (exact wins over interpolated), then walks outward
        across groups to closure: every reached timeline expands into its own
        group (interpolation) and warps into not-yet-reached groups (WarpMap),
        each transferred coordinate governed by ``support_policy``. The query
        timeline's own coordinate is never dropped, clamped, or altered.

        Args:
            coordinate: The query coordinate in the query timeline's unit.
            actual_timeline_id: Actual timeline id of the query.
            support_policy: How out-of-support transfers are handled.
            conversion_maps: C-map spec forwarded to group timestamps.
            interval_coordinates: Unambiguous interval-derived seed values.
            interval_anchor_edges: Interval-claim edges reached at an anchor.
            interval_inferred_edges: Interval-claim edges reached in an interior.
            query_has_interval_anchor: Whether an interval claim explicitly
                anchors the query position.

        Returns:
            ``(coordinates, anchor_edges, inferred_edges, query_has_anchor)``.
            ``query_has_anchor`` is True when the query node carries an
            explicit anchor (making the stamp non-interpolated).
        """
        coordinates: dict[str, CoordinateValue] = {}
        anchor_edges: list[tuple[str, str]] = []
        inferred_edges: list[tuple[str, str]] = []

        def add_inferred_edge(timeline_a: str, timeline_b: str) -> None:
            """Record an inferred relationship once, regardless of orientation."""
            if timeline_a == timeline_b:
                return
            if (timeline_a, timeline_b) in inferred_edges:
                return
            if (timeline_b, timeline_a) in inferred_edges:
                return
            if (timeline_a, timeline_b) in anchor_edges:
                return
            if (timeline_b, timeline_a) in anchor_edges:
                return
            inferred_edges.append((timeline_a, timeline_b))

        # 1. Exact-anchor seed: the query's connected component of explicit
        #    (and any graph-inferred) claims.
        try:
            mg = self._get_or_build_matchgraph(actual_timeline_id, coordinate)
        except ValueError as error:
            if "No synchronous claims touch" not in str(error):
                raise
            mg = None

        query_has_anchor = mg is not None or query_has_interval_anchor
        if query_has_anchor:
            if mg is not None:
                graph_stamp = mg.get_matchstamp()
                coordinates.update(
                    {
                        timeline_id: stored.value
                        for timeline_id, stored in graph_stamp.coordinates.items()
                    }
                )
                anchor_edges.extend(graph_stamp.anchor_edges)
                inferred_edges.extend(graph_stamp.inferred_edges)
        for timeline_id, value in (interval_coordinates or {}).items():
            coordinates.setdefault(timeline_id, value)
        for edge in interval_anchor_edges or []:
            if edge not in anchor_edges and edge[::-1] not in anchor_edges:
                anchor_edges.append(edge)
        for edge in interval_inferred_edges or []:
            add_inferred_edge(*edge)
        # The query timeline's coordinate is authoritative and never altered.
        coordinates[actual_timeline_id] = coordinate

        # 2. Transitive closure across groups (BFS worklist).
        worklist: list[str] = list(coordinates.keys())
        processed: set[str] = set()

        while worklist:
            node_id = worklist.pop()
            if node_id in processed:
                continue
            processed.add(node_id)
            node_coordinate = coordinates[node_id]
            node_timeline = self.get_timeline(node_id)

            # Direct instant claims are reachability edges in their own right.
            # This precedes group handling so standalone timelines participate
            # in the same transitive closure.
            query = Coordinate(
                node_coordinate,
                node_timeline.unit,
                number_type=node_timeline.number_type,
            )
            for claim in self._relevant_claims_at(node_id, query):
                if claim.is_interval:
                    continue
                anchor = claim.start_anchor
                assert anchor is not None
                if anchor.timeline_a_id == node_id:
                    target_id = anchor.timeline_b_id
                    target_coordinate = anchor.coordinate_b.value
                else:
                    target_id = anchor.timeline_a_id
                    target_coordinate = anchor.coordinate_a.value
                if (node_id, target_id) not in anchor_edges and (
                    target_id,
                    node_id,
                ) not in anchor_edges:
                    anchor_edges.append((node_id, target_id))
                if target_id in coordinates:
                    continue
                coordinates[target_id] = target_coordinate
                worklist.append(target_id)

            # A registered timeline reaches the others of its manifest tree
            # (its root and the root's descendants) by exact offsets.
            for relative_id, relative_coordinate in self._manifest_relatives(
                node_timeline, node_coordinate
            ).items():
                if relative_id in coordinates:
                    continue
                coordinates[relative_id] = relative_coordinate
                add_inferred_edge(node_id, relative_id)
                worklist.append(relative_id)

            group_id = self.timeline_to_group.get(node_id)
            if group_id is None:
                continue
            source_group = self.groups.get(group_id)
            if source_group is None:
                continue

            # 2a. Within-group conversion to the node's own group members:
            # exact, and always within each member, since members share one
            # start and one end.
            if len(source_group.timeline_ids) > 1:
                grouped = self._get_group_timestamp(
                    source_group,
                    node_coordinate,
                    node_id,
                    conversion_maps=conversion_maps,
                )
                for member_id, member_coordinate in grouped.items():
                    if member_id in coordinates or member_coordinate is None:
                        continue
                    coordinates[member_id] = member_coordinate
                    add_inferred_edge(node_id, member_id)
                    worklist.append(member_id)

            # 2b. Cross-group warp into every not-yet-reached group.
            for other_group_id, other_group in self.groups.items():
                if other_group_id == group_id:
                    continue
                if all(
                    member_id in coordinates for member_id in other_group.timeline_ids
                ):
                    continue
                transferred = self._transfer_to_group(
                    float(node_coordinate),
                    node_id,
                    source_group,
                    other_group,
                    support_policy,
                )
                if transferred is None:
                    continue
                target_id, target_coordinate = transferred
                if target_id in coordinates:
                    continue
                coordinates[target_id] = target_coordinate
                add_inferred_edge(node_id, target_id)
                worklist.append(target_id)

        return coordinates, anchor_edges, inferred_edges, query_has_anchor

    def get_matchstamps(
        self,
        claims: list[MatchClaim] | None = None,
        *,
        from_graph: bool = True,
        conversion_maps: "ConversionMapsSpec" = False,
    ) -> list[MatchStamp]:
        """Get MatchStamps for the bundle's claims.

        The bare plural means "stamps for the claims this bundle holds" — it
        is the claim-driven listing, not the batch sibling of
        :meth:`get_matchstamps_at`. To resolve a batch of *positions*, use
        :meth:`get_matchstamps_at`; to resolve event IDs, use
        :meth:`get_matchstamps_for`.

        Uses the bundle's caching mechanism for efficient retrieval.

        Args:
            claims: List of MatchClaims to get stamps for. If None, uses
                every cross-group claim in the bundle — the per-claim list
                and every columnar ``MatchClaimField`` (see
                :meth:`get_match_claims`, which materialises the columnar
                rows).
            from_graph: If True (default), return full MatchStamps from the
                MatchGraph (all connected timelines). If False, return
                reduced 2-timeline stamps.
            conversion_maps: C-map conversions available through unit lookup
                and display. Opt-in: defaults to ``False``.

        Returns:
            List of MatchStamp objects. Non-synchronous claims yield None
            entries (filtered out).

        Examples:
            >>> stamps = bundle.get_matchstamps()
            >>> len(stamps)
            100
            >>> stamps[0].n_timelines
            23
        """
        if claims is None:
            claims = self.get_match_claims()

        stamps = []
        for claim in claims:
            stamp = claim.get_matchstamp(
                bundle=self, from_graph=from_graph, conversion_maps=conversion_maps
            )
            if stamp is not None:
                stamps.append(stamp)

        return stamps

    def get_matchstamp_table(
        self,
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
        timeline_filter: set[str] | None = None,
        from_graph: bool = False,
        conversion_maps: "ConversionMapsSpec" = False,
        format: TableFormat = "table",
        fields: "ColumnNaming | Callable[[str, dict], str] | list[str] | None" = None,
        units: bool | None = None,
    ) -> "pa.Table | pd.DataFrame":
        """Get a table of MatchStamps for alignment queries.

        Analogous to ``get_timestamp_table()`` but for cross-group alignment.
        Fields are timeline IDs holding their coordinate values; what a *row*
        is depends on ``from_graph``:

        - ``from_graph=False`` (default) — one row per synchronous claim.  A
          pairwise claim fills exactly two cells and leaves the rest null, so
          a dense pairwise alignment produces one sparse row per claim.
        - ``from_graph=True`` — one row per connected component of the
          ``(timeline_id, coordinate)`` graph the claims induce.  The pairwise
          rows above collapse into the *cross-section* they describe: one row
          per aligned instant, every participating timeline filled.

        When ``claims`` is None both claim stores are read: the per-claim
        Python list and every columnar ``MatchClaimField``.  The columnar
        stores are read four Arrow columns at a time — no ``MatchClaim`` is
        ever materialised — which is what keeps a hundreds-of-thousands-of-rows
        alignment tabulable.

        When ``at`` is given instead of ``claims``, each position or event ID
        is resolved through :meth:`get_matchstamp_at` and becomes exactly one
        row — a full transitive cross-section, every reached timeline filled —
        in input order. ``from_graph`` does not apply on this path (the stamps
        are already collapsed cross-sections) and is ignored.

        Args:
            at: Query positions or event IDs to resolve through
                :meth:`get_matchstamp_at`, one row each in input order. Each
                element is a raw ``int``/``float``/``Fraction`` or
                ``Coordinate`` (needing ``timeline_id``) or an ``IdCoordinate``
                (carrying its own timeline). An ``Interval`` also needs
                ``timeline_id`` and has no table form. Mutually exclusive with
                ``claims``.
            timeline_id: Source timeline for the ``at`` batch; may be
                ``None`` when every element is an ``IdCoordinate`` or an event
                ID. Ignored on the ``claims`` path.
            claims: List of MatchClaims to tabulate.  If None, uses every
                cross-group claim in the bundle (both stores).  When given,
                only those claims are tabulated.  Mutually exclusive with
                ``at``.
            timeline_filter: Only include these timeline fields.
            from_graph: Collapse claims into one row per connected component
                instead of one row per claim.  Ignored on the ``at`` path.
            conversion_maps: C-map conversions to add as derived columns, one
                per (timeline, enabled unit-conversion map). Opt-in: defaults
                to ``False``. Only numeric unit-conversion maps (a
                ``target_unit`` set) become columns; label/structured maps
                appear in stamp display but never as table columns.
            format: ``"table"`` (default) for a PyArrow table, ``"dataframe"``
                for a pandas DataFrame.
            fields: How to name the DataFrame fields (``format="dataframe"``).
            units: If True (the DataFrame default), append units to field
                names like "name (unit)".

        Returns:
            One field per timeline, each a coordinate struct carrying its
            number twice.  Non-synchronous claims are excluded.  Empty input
            yields an empty table.

        Raises:
            ValueError: If both ``claims`` and ``at`` are given, on an unknown
                ``format``, or when a DataFrame-shaping option is supplied for
                an Arrow result.

        Note:
            Collapsed rows are ordered by the coordinate on the
            lexicographically smallest timeline ID present in the component,
            then by that ID — a total order, since two components can never
            share a ``(timeline_id, coordinate)`` node.  A component that
            somehow carries two coordinates for one timeline (which a
            well-formed alignment never does) keeps the smaller one.

        Examples:
            >>> table = bundle.get_matchstamp_table()
            >>> table.num_rows
            100
            >>> table.column_names
            ['score:clt_1', 'perf:dlt_1', 'perf:dlt_2', ...]
            >>> bundle.get_matchstamp_table(from_graph=True).num_rows
            25
            >>> bundle.get_matchstamp_table([0.0, 50.0], "score:clt_1").num_rows
            2
        """
        validate_table_format(format)
        reject_dataframe_options(format, fields=fields, units=units)
        if at is not None:
            if claims is not None:
                raise ValueError("Pass either claims or at, not both")
            resolved = self.get_matchstamp(
                at,
                timeline_id,
                conversion_maps=conversion_maps,
            )
            stamps = resolved if isinstance(resolved, list) else [resolved]
            if any(isinstance(stamp, MatchIntervalStamp) for stamp in stamps):
                raise NotImplementedError(
                    "get_matchstamp_table does not support Interval queries; "
                    "there is no interval table form"
                )
            for stamp in stamps:
                assert isinstance(stamp, MatchStamp)
                ambiguous = stamp._ambiguous_interval_candidates()
                if ambiguous:
                    raise ValueError(
                        "get_matchstamp_table cannot materialize ambiguous "
                        "interval-derived coordinates for timelines "
                        f"{list(ambiguous)}"
                    )
            rows = [dict(stamp.coordinates) for stamp in stamps]
            all_tl_ids: set[str] = set()
            for row in rows:
                all_tl_ids.update(row)
            return self._render_matchstamp_table(
                self._assemble_matchstamp_table(
                    rows,
                    [],
                    all_tl_ids,
                    timeline_filter,
                    conversion_maps=conversion_maps,
                ),
                format=format,
                fields=fields,
                units=units,
            )

        list_claims = self.cross_group_claims if claims is None else claims
        claim_fields = [] if claims is not None else self.cross_group_claim_fields

        all_tl_ids = set()
        rows = []
        # Bulk (timeline_a_id, timeline_b_id, coordinate_a, coordinate_b)
        # column reads, one per columnar store, scattered without dicts.
        bulk_columns: list[tuple[list[str], list[str], list[float], list[float]]] = []

        if from_graph:
            rows = self._collapsed_claim_rows(list_claims, claim_fields)
            for row in rows:
                all_tl_ids.update(row)
        else:
            for claim in list_claims:
                if not claim.is_synchronous or claim.start_anchor is None:
                    continue

                # Get reduced stamp (2 timelines only, for efficiency)
                stamp = claim.get_matchstamp(bundle=self, from_graph=False)
                if stamp is None:
                    continue

                row = dict(stamp.coordinates)
                rows.append(row)
                all_tl_ids.update(stamp.coordinates.keys())

            bulk_columns = [
                claim_field.coordinate_pairs() for claim_field in claim_fields
            ]
            for ids_a, ids_b, _, _ in bulk_columns:
                all_tl_ids.update(ids_a)
                all_tl_ids.update(ids_b)
            all_tl_ids.discard(None)

        return self._render_matchstamp_table(
            self._assemble_matchstamp_table(
                rows,
                bulk_columns,
                all_tl_ids,
                timeline_filter,
                conversion_maps=conversion_maps,
            ),
            format=format,
            fields=fields,
            units=units,
        )

    @staticmethod
    def _render_matchstamp_table(
        table: "pa.Table",
        *,
        format: TableFormat,
        fields: "ColumnNaming | Callable[[str, dict], str] | list[str] | None",
        units: bool | None,
    ) -> "pa.Table | pd.DataFrame":
        """Return the assembled table in the requested output shape."""
        if format == "table":
            return table
        from timetoalign.core.timestamp import timestamp_table_to_dataframe

        return timestamp_table_to_dataframe(
            table=table,
            fields=fields,
            units=True if units is None else units,
        )

    def _assemble_matchstamp_table(
        self,
        rows: list[dict[str, Coordinate | float | None]],
        bulk_columns: list[tuple[list[str], list[str], list[float], list[float]]],
        all_tl_ids: set[str],
        timeline_filter: set[str] | None,
        conversion_maps: "ConversionMapsSpec" = False,
    ) -> "pa.Table":
        """Scatter per-row dicts and bulk columnar reads into one PyArrow table.

        Shared assembly tail for :meth:`get_matchstamp_table`.  Each ``rows``
        entry contributes one row (its timeline keys scattered into the
        matching fields); each ``bulk_columns`` entry contributes
        ``len(ids_a)`` further rows read four Arrow columns at a time — no
        ``MatchClaim`` materialised.  ``all_tl_ids`` is the field universe
        before filtering; ``timeline_filter`` (when given) intersects it.

        Args:
            rows: One dict per already-materialised stamp/claim row.
            bulk_columns: ``(ids_a, ids_b, coordinates_a, coordinates_b)``
                reads, one per columnar store, each contributing one row per
                position.
            all_tl_ids: Union of every timeline ID appearing in the inputs.
            timeline_filter: Only include these timeline fields, if given.
            conversion_maps: C-map conversions to add as derived columns, one
                per (timeline, enabled unit-conversion map). Opt-in: defaults
                to ``False``. Only numeric unit-conversion maps (a
                ``target_unit`` set) become columns.

        Returns:
            PyArrow Table with one field per (filtered) timeline ID, followed
            by any derived conversion columns.  Empty input yields an empty
            table.
        """
        import pyarrow as pa

        n_rows = len(rows) + sum(len(ids_a) for ids_a, _, _, _ in bulk_columns)
        if n_rows == 0:
            return pa.table({})

        # Apply timeline filter if provided
        if timeline_filter is not None:
            all_tl_ids = all_tl_ids & timeline_filter

        # Build table with consistent fields, scattering each row into its
        # cells rather than probing every field of every row.
        field_lists: dict[str, list[Coordinate | None]] = {
            tl_id: [None] * n_rows for tl_id in sorted(all_tl_ids)
        }
        for position, row in enumerate(rows):
            for tl_id, value in row.items():
                column = field_lists.get(tl_id)
                if column is not None:
                    column[position] = self._canonical_table_coordinate(tl_id, value)

        offset = len(rows)
        for ids_a, ids_b, coordinates_a, coordinates_b in bulk_columns:
            for position in range(len(ids_a)):
                column = field_lists.get(ids_a[position])
                if column is not None:
                    column[offset + position] = self._canonical_table_coordinate(
                        ids_a[position], coordinates_a[position]
                    )
                column = field_lists.get(ids_b[position])
                if column is not None:
                    column[offset + position] = self._canonical_table_coordinate(
                        ids_b[position], coordinates_b[position]
                    )
            offset += len(ids_a)

        if conversion_maps is not False and conversion_maps is not None:
            from timetoalign.core.timestamp import _conversion_map_enabled_for_spec

            collected: list[tuple[str, str, "ConversionMap[Any]"]] = []
            for tl_id in list(field_lists.keys()):
                for cmap in self._get_conversion_maps_for_timeline(tl_id):
                    if cmap.target_unit is None:
                        continue  # numeric unit conversions only in tables
                    if not _conversion_map_enabled_for_spec(cmap, conversion_maps):
                        continue
                    collected.append((cmap.target_unit.value, tl_id, cmap))
            label_counts: dict[str, int] = {}
            for label, _tl_id, _cmap in collected:
                label_counts[label] = label_counts.get(label, 0) + 1
            for label, tl_id, cmap in collected:
                col_name = label if label_counts[label] == 1 else f"{tl_id}:{label}"
                source_timeline = self.get_timeline(tl_id)
                target_type = cmap.target_unit.resolve_number_type(
                    source_timeline.number_type
                    if source_timeline.number_type
                    in cmap.target_unit.allowed_number_types
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
                    except Exception:
                        derived.append(None)
                field_lists[col_name] = derived

        columns: list[TimestampColumn] = []
        for name, coordinates in field_lists.items():
            exemplar = next(
                (coordinate for coordinate in coordinates if coordinate is not None),
                None,
            )
            if exemplar is None:
                if name in self.timelines:
                    timeline = self.get_timeline(name)
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
                    timeline_id=name if name in self.timelines else None,
                )
            )
        return build_timestamp_table(columns)

    def _canonical_table_coordinate(
        self, timeline_id: str, value: Coordinate | float | None
    ) -> Coordinate | None:
        """Return one match-stamp table cell in its timeline's canonical type.

        Args:
            timeline_id: ID of a bundle timeline.
            value: Stored coordinate or a numeric value from a columnar claim.

        Returns:
            Canonical coordinate, or ``None`` for an absent cell.

        Raises:
            KeyError: If ``timeline_id`` is not registered in the bundle.
        """
        if value is None:
            return None
        timeline = self.get_timeline(timeline_id)
        raw_value = value.value if isinstance(value, Coordinate) else value
        return Coordinate(
            raw_value,
            timeline.unit,
            number_type=timeline.number_type,
        )

    def _collapsed_claim_rows(
        self,
        claims: list[MatchClaim],
        claim_fields: list[MatchClaimField],
    ) -> list[dict[str, CoordinateValue | None]]:
        """Collapse pairwise claims into one row per aligned instant.

        Each synchronous claim is an edge between the nodes
        ``(timeline_a_id, coordinate_a)`` and ``(timeline_b_id, coordinate_b)``.
        A union-find over those edges recovers the connected components, and
        each component becomes one row mapping every participating timeline to
        its coordinate.  Columnar stores contribute their edges through a bulk
        four-column read, so no ``MatchClaim`` is materialised.

        Args:
            claims: Per-claim edge source (non-synchronous claims ignored).
            claim_fields: Columnar edge sources.

        Returns:
            One row per component, ordered as documented on
            :meth:`get_matchstamp_table`.
        """
        parent: dict[tuple[str, float], tuple[str, float]] = {}

        def find(node: tuple[str, float]) -> tuple[str, float]:
            root = node
            while parent[root] != root:
                root = parent[root]
            while parent[node] != root:
                parent[node], node = root, parent[node]
            return root

        def union(node_a: tuple[str, float], node_b: tuple[str, float]) -> None:
            parent.setdefault(node_a, node_a)
            parent.setdefault(node_b, node_b)
            root_a, root_b = find(node_a), find(node_b)
            if root_a != root_b:
                parent[root_b] = root_a

        # The union runs on float keys, because two claims meet at a node only
        # when they name the same position and a float is the one spelling both
        # sides always have. What the row REPORTS is the exact coordinate
        # recorded for that node, so an anchor authored at 5/3 is emitted as
        # 5/3 rather than as the double it was joined on.
        exact: dict[tuple[str, float], CoordinateValue] = {}

        def record(timeline_id: str, coordinate: CoordinateValue) -> tuple[str, float]:
            node = (timeline_id, float(coordinate))
            exact.setdefault(node, coordinate)
            return node

        for claim in claims:
            if not claim.is_synchronous or claim.start_anchor is None:
                continue
            anchor = claim.start_anchor
            union(
                record(claim.timeline_a_id, anchor.coordinate_a.value),
                record(claim.timeline_b_id, anchor.coordinate_b.value),
            )
        for claim_field in claim_fields:
            ids_a, ids_b, coordinates_a, coordinates_b = claim_field.coordinate_pairs()
            for position in range(len(ids_a)):
                union(
                    record(ids_a[position], coordinates_a[position]),
                    record(ids_b[position], coordinates_b[position]),
                )

        components: dict[tuple[str, float], dict[str, CoordinateValue | None]] = {}
        for node in parent:
            timeline_id, coordinate = node
            row = components.setdefault(find(node), {})
            carried = row.get(timeline_id)
            if carried is None or coordinate < carried:
                row[timeline_id] = exact[node]

        return sorted(
            components.values(),
            key=lambda row: (float(row[min(row)]), min(row)),
        )

    def _invalidate_warp_cache(self) -> None:
        """Clear the WarpMap, MatchLine, and MatchGraph caches, forcing rebuild on next access."""
        self._warp_map_cache.clear()
        self._matchline_cache.clear()
        self._matchgraph_cache.clear()
        self._cache_claims_hash = 0

    def _claim_in_native_units(self, claim: MatchClaim) -> MatchClaim:
        """Return a claim whose anchors use each timeline's native axis.

        Native means both halves of what an axis declares: its unit and the
        representation it writes numbers in. An anchor on its own can only
        reach the unit's default, so the bundle is the first place a timeline
        declaring something other than that default can be honoured, and
        every reader downstream -- MatchGraph, MatchLine, WarpMap inference --
        sees the timeline's own answer rather than the claim author's.
        """

        def _native(timeline_id: str, coordinate: Coordinate) -> Coordinate:
            timeline = self.timelines.get(timeline_id)
            if timeline is None:
                return coordinate
            if coordinate.unit == TimeUnit.number:
                coordinate = Coordinate(coordinate.value, timeline.unit)
            elif coordinate.unit != timeline.unit:
                result = timeline.get_coordinate_at(coordinate, format="coordinate")
                assert isinstance(result, Coordinate)
                coordinate = result
            return express_scalar_as(coordinate, timeline.number_type)

        updates: dict[str, Any] = {}
        for name in ("start_anchor", "end_anchor"):
            anchor = getattr(claim, name)
            if anchor is None:
                continue
            coordinate_a = _native(anchor.timeline_a_id, anchor.coordinate_a)
            coordinate_b = _native(anchor.timeline_b_id, anchor.coordinate_b)
            if (
                coordinate_a is not anchor.coordinate_a
                or coordinate_b is not anchor.coordinate_b
            ):
                updates[name] = anchor.model_copy(
                    update={
                        "coordinate_a": coordinate_a,
                        "coordinate_b": coordinate_b,
                    }
                )
        # A NOMATCH claim keeps its position outside an anchor; it is on a
        # named axis all the same, so it gets the same treatment rather than
        # a second, quieter set of rules.
        if claim.source_coordinate is not None:
            source_coordinate = _native(claim.timeline_a_id, claim.source_coordinate)
            if source_coordinate is not claim.source_coordinate:
                updates["source_coordinate"] = source_coordinate
        return claim.model_copy(update=updates) if updates else claim

    def _get_or_build_match_line(self, source_tl_id: str) -> MatchLine | None:
        """Get or lazily build a MatchLine for the given source timeline.

        The MatchLine is cached per ``source_tl_id`` and shared across
        all target lookups from the same source. This avoids rebuilding
        the expensive ``MatchGraph`` for every ``(source, target)`` pair.

        Group extension is deliberately NOT applied here. When groups
        contain timelines at different sampling rates (e.g. 44.1 kHz
        audio alongside 42 Hz features), within-group interpolation maps
        many distinct coordinates to the same low-resolution value,
        creating spurious edges that collapse ALL graph components into
        one.  WarpMap construction needs clean per-claim pairs, not the
        merged supergraph.  Group extension is only appropriate for
        MatchStamp display (``get_matchstamp_at``).

        Args:
            source_tl_id: ID of the source timeline.

        Returns:
            MatchLine for the source, or None if construction fails.
        """
        if source_tl_id in self._matchline_cache:
            return self._matchline_cache[source_tl_id]

        # Columnar stores contribute only the rows that actually touch this
        # source. ``connecting`` is a vectorized mask, so a dense pairwise
        # field is cut to its source slice — for R recordings that is 2/R of
        # the rows — before anything is materialised, and the resulting
        # MatchLine is then cached per source.
        claims = list(self.cross_group_claims)
        for claim_field in self.cross_group_claim_fields:
            claims.extend(claim_field.connecting(source_tl_id).to_claims())
        claims = [self._claim_in_native_units(claim) for claim in claims]

        try:
            match_line = MatchLine.from_claims(
                claims=claims,
                source_timeline_id=source_tl_id,
            )
        except Exception as e:
            self._logger.debug(
                "Failed to build MatchLine for source '%s': %s",
                source_tl_id,
                e,
            )
            self._matchline_cache[source_tl_id] = None
            return None

        self._matchline_cache[source_tl_id] = match_line
        return match_line

    def _get_or_build_warp_map(
        self, source_tl_id: str, target_tl_id: str
    ) -> WarpMap | None:
        """Get or lazily build a WarpMap for the given timeline pair.

        Builds a ``MatchLine`` from ``cross_group_claims`` with group
        extension and constructs a ``WarpMap`` via
        ``WarpMap.from_match_line()``.  Both negative results (unreachable
        pairs) and positive results are cached; the cache is invalidated
        when ``add_match_claims()`` is called.

        Args:
            source_tl_id: ID of the source timeline.
            target_tl_id: ID of the target timeline.

        Returns:
            WarpMap for the pair, or None if insufficient data.
        """
        if self.n_cross_group_claims == 0:
            return None

        # Check cache validity. The hash must move when EITHER store changes,
        # so it folds in the total claim count and the identity of each
        # columnar store alongside the list's own identity.
        claims_hash = id(self.cross_group_claims) + self.n_cross_group_claims
        for claim_field in self.cross_group_claim_fields:
            claims_hash += id(claim_field)
        if claims_hash != self._cache_claims_hash:
            self._warp_map_cache.clear()
            self._matchline_cache.clear()
            self._cache_claims_hash = claims_hash

        cache_key = (source_tl_id, target_tl_id)
        if cache_key in self._warp_map_cache:
            return self._warp_map_cache[cache_key]

        # Build or retrieve cached MatchLine for this source
        match_line = self._get_or_build_match_line(source_tl_id)
        if match_line is None:
            self._warp_map_cache[cache_key] = None
            return None

        # Check if target timeline is reachable
        if target_tl_id not in match_line.target_timeline_ids():
            self._warp_map_cache[cache_key] = None
            return None

        # Build WarpMap
        try:
            warp = WarpMap.from_match_line(match_line, target_tl_id)
        except AmbiguousWarpMapError:
            raise
        except ValueError as e:
            self._logger.debug(
                "Failed to build WarpMap %s -> %s: %s",
                source_tl_id,
                target_tl_id,
                e,
            )
            self._warp_map_cache[cache_key] = None
            return None

        self._warp_map_cache[cache_key] = warp
        self._logger.debug(
            "Built WarpMap %s -> %s (%d anchors)",
            source_tl_id,
            target_tl_id,
            warp.n_anchors,
        )
        return warp

    # endregion

    # region Grouped Timestamp Helpers

    def _get_group_timestamp(
        self,
        group: TimelineGroup,
        coordinate: CoordinateValue,
        timeline_id: str,
        *,
        conversion_maps: "ConversionMapsSpec" = True,
    ) -> dict[str, CoordinateValue | None]:
        """Get the group members' exact coordinates at a position on one member.

        Args:
            group: The TimelineGroup.
            coordinate: Coordinate in the source timeline's native unit,
                or in a unit convertible via C-Map (see
                ``_resolve_claim_coordinate``).
            timeline_id: ID of the member the coordinate is on.

        Returns:
            Dict mapping member IDs to coordinates, each in its member's
            declared number type; a position outside the source member gives
            the source alone.
        """
        result: dict[str, CoordinateValue | None] = {}

        try:
            ts = group.get_timestamp_at(
                coordinate,
                timeline_id,
                conversion_maps=conversion_maps,
            )
            for tl_id in group.timeline_ids:
                try:
                    coord = ts.get_coordinate_for(tl_id, format="coordinate")
                except KeyError:
                    result[tl_id] = None
                    continue
                assert isinstance(coord, Coordinate)
                result[tl_id] = coord.value
        except Exception as e:
            self._logger.debug(
                f"Failed to get group timestamp at {coordinate} on {timeline_id}: {e}"
            )
            # Return what we can: source coordinate at least
            result[timeline_id] = coordinate

        return result

    def _manifest_relatives(
        self, timeline: "Timeline", coordinate: CoordinateValue
    ) -> dict[str, CoordinateValue]:
        """Carry a position onto the registered timelines of its manifest tree.

        A child sits at an exact offset inside its root, so a position on any
        timeline of the tree is a position on the root, and from there on
        every registered timeline of the tree whose closed span ``[0, length]``
        contains it.

        Args:
            timeline: A registered timeline.
            coordinate: A position on it.

        Returns:
            The coordinates of the other registered timelines of the tree that
            reach the position, by id.
        """
        root = _root(timeline)
        relatives = [
            held
            for held in self.timelines.values()
            if held is not timeline and _root(held) is root
        ]
        if not relatives:
            return {}
        root_coordinate = (
            coordinate
            if timeline is root
            else root._get_parent_coordinate_from_child(timeline.id, coordinate)
        )
        reached: dict[str, CoordinateValue] = {}
        for relative in relatives:
            origin = (
                0
                if relative is root
                else root._get_parent_coordinate_from_child(relative.id, 0)
            )
            value = root_coordinate - origin
            if 0 <= value <= relative.length.value:
                reached[relative.id] = express_as(value, relative.number_type)
        return reached

    def _get_unit_map(self) -> dict[str, str]:
        """Build a mapping from timeline IDs to their coordinate unit strings.

        Returns:
            Dict mapping timeline IDs to unit strings (e.g. "samples", "quarters").
        """
        unit_map: dict[str, str] = {}
        for timeline_id, tl in self.timelines.items():
            if hasattr(tl, "unit") and tl.unit is not None:
                unit_map[timeline_id] = str(tl.unit)
        return unit_map

    def _get_conversion_maps_for_timeline(
        self, timeline_id: str
    ) -> "list[ConversionMap[Any]]":
        """Get every conversion map attached to a timeline in the bundle.

        Returns C-Maps of all
        kinds (including unitless label/structured maps), so MatchStamp display
        surfaces them — the bundle-level counterpart of the TimelineGroup method
        of the same name.
        """
        tl = self.timelines.get(timeline_id)
        if tl is None:
            return []
        return list(tl._conversion_maps.values())

    def _timeline_length(self, timeline_id: str) -> CoordinateValue | None:
        """Return a timeline's exact length in its native unit, or None."""
        tl = self.timelines.get(timeline_id)
        if tl is None or tl.length is None:
            return None
        return tl.length.value

    def _clip_to_length(
        self, timeline_id: str, coordinate: CoordinateValue
    ) -> CoordinateValue:
        """Clip a coordinate into ``[0, length]`` for a target timeline.

        A clipped coordinate is the timeline's own exact bound, never a float
        rendering of it.
        """
        if coordinate < 0:
            return 0
        length = self._timeline_length(timeline_id)
        if length is not None and coordinate > length:
            return length
        return coordinate

    def _transfer_coordinate(
        self,
        warp: WarpMap,
        entering: float,
        target_tl_id: str,
        support_policy: SupportPolicy,
    ) -> CoordinateValue | None:
        """Warp one coordinate into a target timeline under a support policy.

        The coordinate is out of support when it lies outside the source hull
        or when the produced coordinate falls outside the target timeline's
        ``[0, length]`` range.

        Args:
            warp: The source → target WarpMap.
            entering: The source coordinate to transfer.
            target_tl_id: Actual timeline id of the target.
            support_policy: The active policy.

        Returns:
            The target coordinate (always within ``[0, length]``), or None when
            out-of-support under the ``omit`` policy.
        """
        source_coords = warp._source_float_array
        hull_low = float(source_coords[0])
        hull_high = float(source_coords[-1])
        entering = float(entering)
        length = self._timeline_length(target_tl_id)

        def _produce(value: float) -> float:
            return warp._interpolate_float(value)

        produced = _produce(entering)
        within_hull = hull_low <= entering <= hull_high
        within_length = produced >= 0.0 and (length is None or produced <= length)
        if within_hull and within_length:
            return produced

        if support_policy is SupportPolicy.omit:
            return None
        if support_policy is SupportPolicy.clamp:
            clamped = min(max(entering, hull_low), hull_high)
            return self._clip_to_length(target_tl_id, _produce(clamped))
        # extrapolate: keep the linear extrapolation, clipped into range.
        return self._clip_to_length(target_tl_id, produced)

    def _transfer_to_group(
        self,
        coordinate: float,
        source_tl_id: str,
        source_group: TimelineGroup,
        target_group: TimelineGroup,
        support_policy: SupportPolicy,
    ) -> tuple[str, float] | None:
        """Transfer a coordinate from one group to another via WarpMap.

        Searches for a WarpMap connecting any timeline in the source
        group to any timeline in the target group.  Tries direct maps
        first, then indirect (convert within source group, then warp).
        Each transfer is governed by ``support_policy``
        (see :meth:`_transfer_coordinate`); a transfer omitted under the
        ``omit`` policy is skipped so another timeline in the target group may
        still be reached.

        Args:
            coordinate: Source coordinate.
            source_tl_id: Source timeline ID.
            source_group: Source group.
            target_group: Target group.
            support_policy: How out-of-support transfers are handled.

        Returns:
            ``(target_timeline_id, transferred_coordinate)`` or ``None``.
        """
        # Try direct: source_tl_id -> any timeline in target group
        for target_tl_id in target_group.timeline_ids:
            warp = self._get_or_build_warp_map(source_tl_id, target_tl_id)
            if warp is None:
                continue
            try:
                transferred = self._transfer_coordinate(
                    warp,
                    coordinate,
                    target_tl_id,
                    support_policy,
                )
            except Exception:
                continue
            if transferred is not None:
                return (target_tl_id, transferred)

        # Try indirect: convert within source group, then warp
        for src_other_tl_id in source_group.timeline_ids:
            if src_other_tl_id == source_tl_id:
                continue
            # Convert within source group
            try:
                intermediate = source_group.get_coordinate_at(
                    IdCoordinate(
                        coordinate,
                        self.get_timeline(source_tl_id).unit,
                        source_tl_id,
                    ),
                    timeline_id=src_other_tl_id,
                    format="coordinate",
                )
                assert isinstance(intermediate, Coordinate)
            except (KeyError, ValueError):
                continue

            # Try WarpMap from intermediate to target group
            for target_tl_id in target_group.timeline_ids:
                warp = self._get_or_build_warp_map(src_other_tl_id, target_tl_id)
                if warp is None:
                    continue
                try:
                    transferred = self._transfer_coordinate(
                        warp,
                        float(intermediate.value),
                        target_tl_id,
                        support_policy,
                    )
                except Exception:
                    continue
                if transferred is not None:
                    return (target_tl_id, transferred)

        return None

    # endregion

    # region Summary and Serialization

    def summary(self) -> dict[str, Any]:
        """Get a summary of the bundle contents.

        Returns a deterministic representation suitable for comparison.
        Keys and timeline lists are sorted for order-independence.

        Returns:
            Dictionary with bundle information.
        """
        timeline_info = {}
        for tl_id in sorted(self.timelines.keys()):
            tl = self.timelines[tl_id]
            group_id = self.timeline_to_group.get(tl_id)
            timeline_info[tl_id] = {
                "name": tl.name,
                "length": (
                    float(tl.length.value) if hasattr(tl.length, "value") else tl.length
                ),
                "unit": str(tl.unit) if hasattr(tl, "unit") else None,
                "group": group_id,
            }

        group_info = {}
        for grp_id in sorted(self.groups.keys()):
            grp = self.groups[grp_id]
            group_info[grp_id] = {
                "name": grp.name,
                "n_timelines": grp.n_timelines,
                "timeline_ids": sorted(grp.timeline_ids),
            }

        return {
            "id": self.id,
            "name": self.name,
            "n_timelines": len(self.timelines),
            "n_groups": len(self.groups),
            "timelines": timeline_info,
            "groups": group_info,
            "meta": self.meta,
        }

    def to_dict(
        self,
        *,
        events: bool = False,
        external_references: bool = False,
    ) -> dict[str, Any]:
        """Convert the bundle to a dictionary for serialization.

        The bundle is written as its timelines and its groups.
        ``"timelines"`` holds every root timeline of the bundle — every
        registered timeline that is no child, which includes the root of each
        one that is — each exactly
        :meth:`Timeline.to_dict <timetoalign.timelines.Timeline.to_dict>`,
        sorted by id. ``"bundle"`` carries the groups as ``Group`` objects
        sorted by id, each naming its members by id; ``"graphical"`` keeps
        every group's member order, followed by the ``styles`` and
        ``statements`` of :attr:`styling` that are present. A root no group
        names is ungrouped. MatchClaims are not written. The file's own
        ``meta`` is the writer's concern.

        Args:
            events: Passed to every root's ``to_dict``.
            external_references: Passed to every root's ``to_dict``.

        Returns:
            ``{"timelines": [...], "bundle": {"id", "name", "meta",
            "support_policy", "groups"}, "graphical": {"order": {...},
            "styles"?, "statements"?}}``.

        Examples:
            >>> from fractions import Fraction
            >>> from timetoalign.alignment import AlignmentBundle
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     ContinuousPhysicalTimeline,
            ... )
            >>> bundle = AlignmentBundle()
            >>> _ = bundle.add_timeline(
            ...     ContinuousLogicalTimeline(length=Fraction(16)), as_group="score"
            ... )
            >>> _ = bundle.add_timeline(
            ...     ContinuousPhysicalTimeline(length=40.0), grouped_with="clt_1"
            ... )
            >>> data = bundle.to_dict()
            >>> [timeline["id"] for timeline in data["timelines"]]
            ['clt_1', 'cpt_1']
            >>> data["bundle"]["groups"]
            [{'id': 'group_1', 'name': 'score', 'meta': {}, 'members': ['clt_1', 'cpt_1']}]
            >>> data["graphical"]
            {'order': {'group_1': ['clt_1', 'cpt_1']}}
        """
        roots: dict[str, Timeline] = {}
        for timeline in self.timelines.values():
            root = _root(timeline)
            roots.setdefault(root.id, root)
        groups = [self.groups[group_id] for group_id in sorted(self.groups)]
        return {
            "timelines": [
                roots[root_id].to_dict(
                    events=events, external_references=external_references
                )
                for root_id in sorted(roots)
            ],
            "bundle": {
                "id": self.id,
                "name": self.name,
                "meta": dict(self.meta),
                "support_policy": SupportPolicy(self.support_policy).value,
                "groups": [group._to_group_object() for group in groups],
            },
            "graphical": _write_graphical(
                {group.id: group.timeline_ids for group in groups}, self.styling
            ),
        }

    def __repr__(self) -> str:
        name_str = f", name={self.name!r}" if self.name else ""
        return (
            f"AlignmentBundle(id={self.id!r}{name_str}, "
            f"timelines={len(self.timelines)}, groups={len(self.groups)})"
        )

    def __str__(self) -> str:
        """Return human-readable ASCII diagram of the bundle.

        Uses the diagram() method to generate a visual representation
        showing all groups and their member timelines.
        """
        return str(self.diagram())

    # endregion

    # region Display

    def diagram(
        self,
        width: int = 80,
        show_children: bool = True,
        max_children: int = 6,
        max_standalone: int = 6,
        unicode: bool = True,
        depth: bool | int = True,
    ) -> "Diagram":
        """Generate ASCII diagram for this bundle.

        Args:
            width: Total width of the diagram in characters.
            show_children: Whether to expand child timelines.
            max_children: Maximum children per timeline.
            max_standalone: Maximum standalone timelines to display
                before truncating with an ellipsis.
            unicode: Use Unicode characters (True) or ASCII fallback (False).
            depth: Child levels to render in nested timeline diagrams. ``True``
                renders all levels, ``False`` renders direct children only,
                and a non-negative integer renders at most that many levels.
                In particular, ``0`` renders no child rows.

        Returns:
            Diagram object (displays as ASCII in terminal, rich HTML in Jupyter).

        Raises:
            ValueError: If ``depth`` is a negative integer.

        Examples:
            >>> print(bundle.diagram())
            AlignmentBundle[thoresen_alignment]

              TimelineGroup[dgt1_group] (2 timelines)
              ┌──────────────────────────────────────────────────────┐
              │ DiscreteGraphicalTimeline[dgt_1:1] (11 events)        │
              │ 0 ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶ 4835 pixels    │
              └──────────────────────────────────────────────────────┘

              MatchClaims: 5
        """
        from timetoalign.display.ascii import bundle_diagram

        return bundle_diagram(
            self,
            width=width,
            show_children=show_children,
            max_children=max_children,
            max_standalone=max_standalone,
            unicode=unicode,
            depth=depth,
        )

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebooks.

        Displays the ASCII diagram in a monospace pre block so it
        renders correctly in notebook output cells.
        """
        return self.diagram()._repr_html_()

    # endregion


# endregion
