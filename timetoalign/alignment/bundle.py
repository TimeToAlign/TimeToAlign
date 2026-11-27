"""AlignmentBundle - The primary entry point for alignment workflows.

This module implements the `AlignmentBundle` class, a single entry point for
all alignment workflows. The bundle manages timelines, groups, and coordinate
transfer operations.

Within a group, coordinate transfer uses typed coordinate retrieval.
Across groups, transfer is mediated by
the ``MatchClaim`` -> ``MatchLine`` -> ``WarpMap`` pipeline.
WarpMaps are built lazily and cached for repeated queries.

A bundle is its timelines and its groups. Every timeline is registered
under its own ``id``. The bundle's trees, groups and given id make one
manifest, in which every id is unique whatever element holds it.
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, overload

import pandas as pd
import pyarrow as pa
from timetoshare import DEFAULT_BELIEFS, BeliefState

from timetoalign.core import (
    Box,
    Coordinate,
    CoordinateSpec,
    CoordinateValue,
    Interval,
    Provenance,
    SupportPolicy,
    TimeUnit,
    express_as,
    resolve_id,
)
from timetoalign.core.ids import (
    BUNDLE_PREFIX,
    IdMinter,
    _OpenObject,
    _refuse_held_ids,
    content_minter,
    quote_id,
    validate_id,
)
from timetoalign.core.provenance import BeliefValue
from timetoalign.core.retrieval import (
    CoordinateCollection,
    CoordinateFormat,
    CoordinateInput,
    CoordinateResult,
    KeyCollection,
    Rounding,
    TableFormat,
)
from timetoalign.timelines import TimelineGroup
from timetoalign.timelines.groups import (
    _groups_by_timeline,
    _IdScope,
    _nested,
    _root,
)
from timetoalign.timelines.manifest import (
    ManifestFileMixin,
    Resource,
    event_table,
    projection_table,
)
from timetoalign.timelines.statements import (
    MutableStatementsMixin,
)

from . import claim_events, claim_store, closure, package, stamp_table
from .claim_store import (  # noqa: F401
    _ALL_BELIEFS,
    _belief_key,
    _BeliefKey,
    _refuse_shared_claim_ids,
)
from .claims import (
    MatchClaim,
    MatchClaimField,
)
from .closure import (  # noqa: F401
    _all_keys,
    _all_positions,
    _resolve_coordinate_and_timeline,
)
from .filters import ClaimFilter
from .graph import MatchGraph, MatchIntervalStamp, MatchStamp
from .matchline import MatchLine
from .package import _merged_resources  # noqa: F401
from .stamp_table import _table_anchors  # noqa: F401
from .warpmap import WarpMap

if TYPE_CHECKING:
    from collections.abc import Callable
    from os import PathLike

    from timetoshare.serialization import Serialization

    from timetoalign.core.enums import ColumnNaming, Domain
    from timetoalign.core.timestamp import ConversionMapsSpec
    from timetoalign.display.ascii import Diagram
    from timetoalign.maps.base import ConversionMap
    from timetoalign.timelines import Timeline

module_logger = logging.getLogger(__name__)


# region AlignmentBundle


@dataclass
class AlignmentBundle(_IdScope, MutableStatementsMixin, ManifestFileMixin, _OpenObject):
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

    The alignment relationships and coordinates do not depend on the order
    in which timelines are added; the order of table rows and columns and
    of a stamp's coordinates follows registration.

    Attributes:
        id: Unique identifier for this bundle, minted with the prefix
            ``bundle`` unless given.
        name: Optional human-readable name.
        timelines: Dictionary mapping timeline IDs to Timeline objects: every
            timeline added, and the root of every one that is a child. The
            other descendants of these roots are not listed, yet every
            getter answers for them as for the listed ones.
        groups: Dictionary mapping group IDs to TimelineGroup objects. The
            groups are the one statement of membership; a timeline may be a
            member of several (see :attr:`timeline_to_groups`).
        match_claim_table: The bundle's MatchClaims, every kind, in one
            :class:`~timetoalign.alignment.claims.MatchClaimField` (the
            MatchClaim table), queried vectorized; a claim is materialised
            only where a reader needs the object. A MatchClaim is
            addressed by its id alone.
        styling: The members of the manifest's ``graphical`` other than
            ``order`` (``styles``, ``rules``, ``timestamps`` and any
            other), kept as given and never interpreted.
        statements: What the manifest stands for, as property–value pairs,
            with their :attr:`provenance` (see
            :mod:`~timetoalign.timelines.statements`).
        resources: The files the manifest names (its ``resources``), each a
            :class:`~timetoalign.timelines.manifest.Resource`: the
            MatchClaim table and the Projections table that :meth:`to_file`
            writes beside the manifest, and any other kept as read.

    Note:
        Every reader of the MatchClaims (queries, the matchstamp getters,
        the table getter :meth:`get_matchstamp_table`, the MatchLines and
        WarpMaps behind transfer) reads the one table, through the belief
        filter (``believer``, ``beliefs``) where the reader takes it. Two
        bundles are equal when their claims are too: the tables are compared
        row by row, provenance included (:class:`MatchClaimField` equality).

    Examples:
        >>> from timetoalign.alignment import AlignmentBundle
        >>> from timetoalign.timelines.types import (
        ...     ContinuousLogicalTimeline,
        ...     ContinuousPhysicalTimeline,
        ... )
        >>> bundle = AlignmentBundle()
        >>> _ = bundle.add_timeline(ContinuousLogicalTimeline(length=200, uid="score"))
        >>> _ = bundle.add_timeline(
        ...     ContinuousPhysicalTimeline(length=91.0, uid="audio"), grouped_with="score"
        ... )
        >>> stamp = bundle.get_matchstamp_at(100, "score")
        >>> stamp.get_coordinate_for("audio", format="float")
        45.5
    """

    _scope_kind: ClassVar[str] = "bundle"

    id: str = field(default="")
    name: str | None = None
    timelines: dict[str, "Timeline"] = field(default_factory=dict)
    groups: dict[str, TimelineGroup] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    # The one store of the bundle's MatchClaims, changed only through
    # ``add_match_claims``, ``add_match_claim_field`` and ``from_bundles``,
    # which check what joins it. Compared row by row, provenance included.
    match_claim_table: MatchClaimField = field(
        default_factory=lambda: MatchClaimField.from_claims([]),
        init=False,
    )

    # Persistent per-bundle setting: how ``get_matchstamp_at`` treats a
    # timeline whose transferred coordinate falls outside alignment support
    # (below the first anchor or beyond the last). The default drops such a
    # timeline; a per-call ``support_policy`` argument overrides it.
    support_policy: SupportPolicy = SupportPolicy.omit

    # The members of the manifest's ``graphical`` other than ``order``,
    # kept as given and never interpreted.
    styling: dict[str, Any] = field(default_factory=dict)

    # The files the manifest names; ``to_file`` rewrites the entries of the
    # MatchClaim table and the Projections table it writes.
    resources: list[Resource] = field(default_factory=list)

    # The manifest's statements and provenance as the file writes them, and
    # the prefixes its ``@context`` declares beyond the catalogue's.
    _statement_state: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _prefixes: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    # Cached WarpMaps: (belief filter, source_tl_id, target_tl_id) ->
    # WarpMap | None. None values are cached as negative results to avoid
    # repeated expensive MatchLine construction for unreachable pairs.
    _warp_map_cache: dict[tuple[_BeliefKey, str, str], WarpMap | None] = field(
        default_factory=dict, repr=False, compare=False
    )
    # Cached MatchLines: (belief filter, source_tl_id) -> MatchLine (avoids
    # rebuilding the MatchGraph for every target lookup)
    _matchline_cache: dict[tuple[_BeliefKey, str], MatchLine | None] = field(
        default_factory=dict, repr=False, compare=False
    )
    # Cached MatchGraphs: (belief filter, timeline_id, coordinate) -> MatchGraph
    _matchgraph_cache: dict[tuple[_BeliefKey, str, CoordinateValue], MatchGraph] = (
        field(default_factory=dict, repr=False, compare=False)
    )
    # The claims in force under each belief filter, and those of them naming
    # each timeline; emptied whenever the table changes.
    _in_force_cache: dict[_BeliefKey, MatchClaimField] = field(
        default_factory=dict, repr=False, compare=False
    )
    _naming_cache: dict[tuple[_BeliefKey, str], MatchClaimField] = field(
        default_factory=dict, repr=False, compare=False
    )
    # The seed of the minter of the id when none is given; not stored.
    seed: InitVar[int | None] = field(default=None, kw_only=True)
    # Whether the id was given, so that the manifest written carries it.
    _id_given: bool = field(default=False, init=False, repr=False, compare=False)
    # Whether the manifest read held a ``graphical`` object, written back
    # even with every member at its default.
    _has_graphical: bool = field(default=False, init=False, repr=False, compare=False)
    # The band orders of the manifest read, keyed by timeline id, kept as
    # given beside the styling and written back with their timelines.
    _band_orders: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    # The measure maps of the manifest read that no timeline names, written
    # back with the ones the timelines name.
    _measure_maps: list[Any] = field(
        default_factory=list, init=False, repr=False, compare=False
    )

    # Internal logger
    _logger: logging.Logger = field(
        default_factory=lambda: module_logger, init=False, repr=False
    )

    def __post_init__(self, seed: int | None = None) -> None:
        """Mint or validate the ID and set up the logger.

        Without an id, one with the prefix ``bundle`` is minted: from
        ``IdMinter(seed)`` with *seed*, else from the content minter of the
        name and the ids of the timelines and groups given. A minted id is
        never written to a manifest.
        """
        self._id_given = bool(self.id)
        if not self.id:
            minter = (
                IdMinter(seed)
                if seed is not None
                else content_minter(
                    BUNDLE_PREFIX, self.name, list(self.timelines), list(self.groups)
                )
            )
            self.id = minter.mint(BUNDLE_PREFIX)
        else:
            validate_id(self.id)
        self.support_policy = SupportPolicy(self.support_policy)
        self._logger = module_logger.getChild(self.id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlignmentBundle":
        """Read a bundle from any TimeToShare manifest.

        The manifest's ``id`` becomes the bundle's (a manifest without ``id``,
        named or anonymous, gives a bundle whose minted id :meth:`to_dict`
        never writes), its ``name`` and ``meta`` the bundle's. Every root timeline is rebuilt;
        each group is rebuilt from its ``Group`` object, its members resolved
        among the roots and all their descendants and ordered by its
        ``graphical.order`` entry (``members`` order without one).
        The manifest's ``resources`` are kept as :attr:`resources`; the
        tables they name are read by :meth:`from_file`, which knows where
        the manifest lies. ``graphical``'s other keys are kept as :attr:`styling`, and the
        entries of ``graphical.order`` keyed by a timeline, its band orders,
        are kept too, both unread. The manifest's statements and provenance
        become the bundle's, with every prefix its ``@context`` declares
        beyond the catalogue's, used or not: they are the context under which
        the MatchClaims' actors compare (a prefix declared only for the
        actors of the MatchClaim table expands).
        Every root no group names joins
        ungrouped. A timeline several groups name is a member of each, one
        object shared by them. Keys the library does not know are kept on the
        object read from the JSON object they were found in, and written back.

        Args:
            data: A TimeToShare manifest.

        Returns:
            The rebuilt bundle, without MatchClaims.

        Raises:
            ValueError: If *data* is no TimeToShare manifest of version
                0.2.0 or is malformed: a timeline id occurring twice, a group
                id occurring twice or equal to a timeline id, a member naming
                no timeline of ``"timelines"``, a group naming one member
                twice, a member that is an ancestor or descendant of another
                member of its group, an order that is no permutation of its
                group's members or names no group or timeline. The message
                names the problem.

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
        return package._from_dict(cls, data)

    @classmethod
    def from_bundles(
        cls,
        bundles: list["AlignmentBundle"],
        *,
        id: str = "",
        name: str | None = None,
        seed: int | None = None,
    ) -> "AlignmentBundle":
        """Merge multiple bundles' groups, standalone timelines, claims and resources.

        Registers every group and standalone timeline from each source
        bundle into a new bundle, and carries over every ``MatchClaim``
        unchanged, its id included: a MatchClaim's id never changes and
        addresses it alone, so the merged :attr:`match_claim_table` holds
        every row once, and two sources holding one id are refused before
        anything is merged (one of them is to be stated again under other
        ids first). The statements of the
        timelines and regions travel with them, and their prefixes; the
        source bundles' own statements describe their manifests and are not
        carried. The sources' :attr:`resources` are carried, each under
        its own ID and as it stands: the merge has one MatchClaim table, one
        Projections table and one Box table, so the first source's entry of
        each table type names the merged table (:meth:`to_file` writes it
        there under that entry's ID) and a later source's entry of that type
        is not carried; every other resource, such as an audio file a
        resource names, is. Each source group is entered as
        a new group with the same id, members, name and meta, each member
        keeping its own lock. Groups from
        different source bundles remain distinct — merging does not itself
        align timelines across bundles. Add cross-group MatchClaims (e.g. via
        :meth:`create_match_claims`) afterwards to align them.

        Known limit: the timelines are not copied, the merged bundle shares
        them with the source bundles.

        Args:
            bundles: The bundles to merge, in order.
            id: Explicit ID for the merged bundle. Minted if omitted.
            name: Optional human-readable name for the merged bundle.
            seed: The seed of the minter of the merged bundle's id when *id*
                is omitted; not stored.

        Returns:
            A new AlignmentBundle containing every group, standalone
            timeline, MatchClaim and resource from ``bundles``.

        Raises:
            ValueError: If two source bundles share a group ID, a timeline
                ID collision would occur across bundles,
                two source bundles hold a MatchClaim under one id (the message
                names the id and both bundles), or the ID of a resource
                carried is held by a resource of another source, by an
                element of a source's timelines or groups or by the merged
                bundle (the message names the ID and both holders); nothing
                is merged then.

        Examples:
            >>> from timetoalign.alignment import AlignmentBundle
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     ContinuousPhysicalTimeline,
            ... )
            >>> score_bundle = AlignmentBundle()
            >>> _ = score_bundle.add_timeline(
            ...     ContinuousLogicalTimeline(length=16, uid="score"), as_group="score"
            ... )
            >>> audio_bundle = AlignmentBundle()
            >>> _ = audio_bundle.add_timeline(
            ...     ContinuousPhysicalTimeline(length=40.0, uid="audio"), as_group="audio"
            ... )
            >>> merged = AlignmentBundle.from_bundles([score_bundle, audio_bundle])
            >>> merged.n_groups == score_bundle.n_groups + audio_bundle.n_groups
            True
            >>> sorted(merged.timelines)
            ['audio', 'score']
        """
        return package._from_bundles(cls, bundles, id=id, name=name, seed=seed)

    # region Timeline Management

    def add_timeline(
        self,
        timeline: "Timeline",
        *,
        grouped_with: str | None = None,
        as_group: str | None = None,
        seed: int | None = None,
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

        A timeline may be a member of several groups: one already in the
        bundle, grouped or not, joins a further group through
        ``grouped_with`` or ``as_group``.

        Args:
            timeline: The Timeline to add.
            grouped_with: ID of a timeline of the bundle's trees to share a
                group with, registered or nested in one; it is registered as
                a member. If provided, both timelines become part of the same
                group.
                If the target timeline is not yet in a group, a new group
                is created with the target as its first member. A target in
                several groups is refused, since the group to join would be a
                guess.
            as_group: Name of the group if a new one is created. The new
                group's id is minted (:class:`TimelineGroup`).
            seed: The seed of the minter of a new group's id; not stored.

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If the timeline is already in the bundle while no
                group is requested, or already in the group it would join; if
                ``grouped_with`` names a timeline of several groups; if an id
                of its manifest tree — of any timeline, conversion map,
                region or flow-control element of it — is held by another
                element of the bundle's trees, by a group or by the bundle.
            KeyError: If grouped_with names no timeline of the bundle's trees,
                naming every timeline the bundle knows.

        Examples:
            Linear (full-extent) alignment:

                >>> from timetoalign.alignment import AlignmentBundle
                >>> from timetoalign.timelines.types import (
                ...     ContinuousPhysicalTimeline,
                ...     DiscreteGraphicalTimeline,
                ...     DiscreteLogicalTimeline,
                ... )
                >>> bundle = AlignmentBundle()
                >>> _ = bundle.add_timeline(DiscreteGraphicalTimeline(length=4000, uid="scan"))
                >>> _ = bundle.add_timeline(
                ...     DiscreteLogicalTimeline(length=960, uid="midi"), grouped_with="scan"
                ... )
                >>> bundle.get_group_for_timeline("scan").timeline_ids
                ['scan', 'midi']

            Partial alignment: the participating part is a child.

                >>> image = DiscreteGraphicalTimeline(length=300000, uid="image")
                >>> holes = DiscreteGraphicalTimeline(length=277776, uid="holes")
                >>> image.add_child(holes, offset=15343)
                >>> audio = ContinuousPhysicalTimeline(length=600.0, uid="audio")
                >>> bundle = AlignmentBundle()
                >>> _ = bundle.add_timeline(audio, as_group="roll")
                >>> _ = bundle.add_timeline(holes, grouped_with="audio")
                >>> roll = bundle.get_group_for_timeline("audio")
                >>> sorted(bundle.timelines), roll.timeline_ids
                (['audio', 'holes', 'image'], ['audio', 'holes'])
        """
        if grouped_with is None and as_group is None:
            # An ungrouped participant is a root: a child joins through its
            # root and stays reachable as the root's descendant.
            timeline = _root(timeline)
        # The timeline to group with may be any timeline of the bundle's
        # trees, registered or nested in one.
        target = (
            self._require_known_timeline(grouped_with, role="Timeline to group with")
            if grouped_with is not None
            else None
        )
        registered = self.timelines.get(timeline.id) is timeline
        if registered and grouped_with is None and as_group is None:
            raise ValueError(f"Timeline '{timeline.id}' already exists in bundle")
        target_groups = (
            self.get_groups_for_timeline(grouped_with)
            if grouped_with is not None
            else ()
        )
        if grouped_with is not None and len(target_groups) > 1:
            raise ValueError(
                f"Cannot group with {quote_id(grouped_with)}: it is a member of "
                f"several groups {[group.id for group in target_groups]}, so the group "
                "to join is ambiguous"
            )
        group: TimelineGroup | None = None
        if target_groups:
            # The group refuses a member it holds already and nesting, and
            # admits the newcomer into every bundle holding it, this one
            # included, before anything changes.
            (group,) = target_groups
            group.add_timeline(timeline)
        else:
            if (
                target is not None
                and grouped_with is not None
                and _nested(target, timeline)
            ):
                raise ValueError(
                    f"Timeline {quote_id(timeline.id)} cannot be grouped with "
                    f"{quote_id(grouped_with)}: one is nested in the other, and a parent "
                    "and its child cannot share one start and end"
                )
            founders = [timeline] if target is None else [target, timeline]
            group_id = None
            if target is not None or as_group is not None:
                # The new group's id is checked before anything changes.
                group_id = TimelineGroup._minted_id(
                    as_group, [founder.id for founder in founders], seed
                )
                self._refuse_group_id(group_id, None if registered else timeline)
            if not registered:
                self._admit(timeline)
            if group_id is not None:
                group = TimelineGroup(name=as_group, timelines=founders, id=group_id)
                self._hold_group(group)
                if target is not None:
                    # A member is registered, a nested one too.
                    self._register(target)

        self._register(timeline)
        if group is not None:
            self._logger.debug(f"Added timeline '{timeline.id}' to group '{group.id}'")
        else:
            self._logger.debug(f"Added standalone timeline '{timeline.id}'")
        return self

    def add_group(self, group: TimelineGroup) -> "AlignmentBundle":
        """Add a pre-built TimelineGroup with all its timelines at once.

        This is the bulk-registration counterpart of ``add_timeline(..., as_group=...)``.
        It registers the group and every timeline it contains — each under
        its own ``id``, with the root of every member that is a child — into
        the bundle's timeline and group registries. A member may already be a
        member of another group of the bundle: a timeline may belong to
        several groups. The group keeps its id.

        Args:
            group: A ``TimelineGroup`` that already contains timelines.

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If the group's ID is already held in the bundle, or
                if an id of a member's manifest tree is held by another
                element of the bundle's trees, by a group or by the bundle.

        Examples:
            Add a recording group with three DPTs:

                >>> from timetoalign.alignment import AlignmentBundle
                >>> from timetoalign.timelines import TimelineGroup
                >>> from timetoalign.timelines.types import DiscretePhysicalTimeline
                >>> grp = TimelineGroup(
                ...     id="recording",
                ...     timelines=[
                ...         DiscretePhysicalTimeline(length=44100, uid=f"mic {n}")
                ...         for n in range(1, 4)
                ...     ],
                ... )
                >>> bundle = AlignmentBundle().add_group(grp)
                >>> list(bundle.timelines), list(bundle.groups)
                (['mic 1', 'mic 2', 'mic 3'], ['recording'])
        """
        self._take_group(group)
        return self

    def _take_group(self, group: TimelineGroup) -> None:
        """Admit *group* and its members.

        Raises:
            ValueError: As :meth:`add_group`; nothing has changed then.
        """
        if any(held is group for held in self.groups.values()):
            raise ValueError(f"Group {quote_id(group.id)} already exists in bundle")
        members = list(group)
        # Check every member and the group id before anything changes.
        for member in members:
            if self.timelines.get(member.id) is not member:
                self._admit(member)
        self._refuse_group_id(group.id)

        self._hold_group(group)
        for member in members:
            self._register(member)

        self._logger.debug(f"Added group '{group.id}' with {len(members)} timelines")

    def _admit(self, timeline: "Timeline") -> None:
        """Check a timeline's manifest tree joining the bundle's id scope.

        Every id is unique in the bundle, whatever element holds it, the
        groups' ids and the bundle's own included. Nothing changes.

        Args:
            timeline: The timeline joining the bundle.

        Raises:
            ValueError: If an id of the tree is held by another element of
                the bundle's trees, a group or the bundle.
        """
        self._admit_tree(timeline)

    def _scope_members(self) -> list["Timeline"]:
        """Return the registered timelines."""
        return list(self.timelines.values())

    def _sub_scopes(self) -> dict[str, TimelineGroup]:
        """Return the groups, keyed by id."""
        return self.groups

    def _roots(self) -> list["Timeline"]:
        """Return the root of every tree the bundle holds, once each."""
        roots: dict[str, Timeline] = {}
        for timeline in self.timelines.values():
            root = _root(timeline)
            roots.setdefault(root.id, root)
        return list(roots.values())

    def _known_timelines(self) -> dict[str, "Timeline"]:
        """Return every timeline the bundle answers for, by id.

        A bundle knows every timeline of its trees, whatever its depth and
        whether or not it was added, registered or grouped: the registered
        timelines first, in registration order, then every other descendant
        of each root, depth-first. Registering is how a tree enters the
        bundle, never a condition for finding one of its timelines; the
        registry itself (:attr:`timelines`) is unchanged by a lookup.

        Every getter addresses timelines through this mapping (or
        :meth:`_known_timeline` for one exact id), and a stamp reaches each
        of them from any other timeline of its tree by exact offsets
        (:meth:`_manifest_relatives`).
        """
        known: dict[str, Timeline] = dict(self.timelines)
        for root in self._roots():
            for _, descendant in root._iter_descendants():
                known.setdefault(descendant.id, descendant)
        return known

    def _known_timeline(self, timeline_id: str) -> "Timeline | None":
        """Return the timeline of the bundle's trees with this exact id, else None.

        The same rule as :meth:`_known_timelines`, answered from the registry
        first so the common case walks no tree.
        """
        held = self.timelines.get(timeline_id)
        if held is not None:
            return held
        return self._known_timelines().get(timeline_id)

    def _require_known_timeline(
        self, timeline_id: str, *, role: str = "Timeline"
    ) -> "Timeline":
        """Return the timeline of the bundle's trees with this exact id.

        Args:
            timeline_id: The exact id of a timeline of the bundle's trees.
            role: How the message names the timeline (``"Source timeline"``).

        Raises:
            KeyError: If no timeline of the bundle's trees has this id, naming
                every timeline the bundle knows (:meth:`_known_timelines`).
        """
        held = self._known_timeline(timeline_id)
        if held is None:
            raise KeyError(
                f"{role} '{timeline_id}' not in bundle. "
                f"Available timelines: {list(self._known_timelines())}"
            )
        return held

    def _statements_holder(self) -> "AlignmentBundle":
        """Return the bundle: it holds the statements of its manifest itself."""
        return self

    def _statement_holder_name(self) -> str:
        """Return the bundle as a message names it."""
        return f"bundle {quote_id(self.id)}"

    def _register(self, timeline: "Timeline") -> None:
        """Register a timeline, and its root when it is a child."""
        self.timelines[timeline.id] = timeline
        timeline._join_scope(self)
        root = _root(timeline)
        if self.timelines.get(root.id) is not root:
            self.timelines[root.id] = root
            root._join_scope(self)

    def _hold_group(self, group: TimelineGroup) -> None:
        """Hold *group* under its id and record the bundle on it."""
        self.groups[group.id] = group
        group._hold_in(self)

    def _refuse_group_id(
        self, group_id: str, joining: "Timeline | None" = None
    ) -> None:
        """Refuse a group joining the bundle where an element holds its id.

        Args:
            group_id: The group's id.
            joining: A timeline joining the bundle with the group, whose
                manifest tree counts as held.

        Raises:
            ValueError: If *group_id*, given or minted, is held in the
                bundle or by the tree of *joining*.
        """
        taken = self._scope_ids()
        if joining is not None:
            taken = {**_root(joining)._tree_ids(), **taken}
        _refuse_held_ids(
            [(group_id, f"group {quote_id(group_id)}")],
            taken,
            scope=f"bundle {quote_id(self.id)}",
        )

    def get_timeline(self, timeline_id: str) -> "Timeline":
        """Get a timeline of the bundle's trees by ID.

        Every timeline of the trees the bundle holds is found, registered or
        not and at any depth; registered timelines come first when a pattern
        matches several.

        Takes an exact ID, else a regular expression matching whole IDs
        (:func:`~timetoalign.core.ids.resolve_id`): when several match, the
        first is returned with a warning. A fragment of an ID matches
        nothing.

        Args:
            timeline_id: The timeline's ID, or a regular expression matching
                whole IDs.

        Returns:
            The Timeline object.

        Raises:
            KeyError: If no timeline matches the pattern.

        Examples:
            >>> from timetoalign.alignment import AlignmentBundle
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     ContinuousPhysicalTimeline,
            ... )
            >>> bundle = AlignmentBundle()
            >>> _ = bundle.add_timeline(ContinuousLogicalTimeline(length=16, uid="score:clt_1"))
            >>> _ = bundle.add_timeline(ContinuousPhysicalTimeline(length=40.0, uid="perf:cpt_1"))
            >>> bundle.get_timeline("score:clt_1").id  # Exact match
            'score:clt_1'
            >>> bundle.get_timeline("perf:.*").id  # Regular expression
            'perf:cpt_1'
        """
        held = self.timelines.get(timeline_id)
        if held is not None:
            return held
        known = self._known_timelines()
        return known[resolve_id(timeline_id, list(known))]

    def get_group(self, group_id: str) -> TimelineGroup:
        """Get a group by ID.

        Takes an exact ID, else a regular expression matching whole IDs
        (:func:`~timetoalign.core.ids.resolve_id`): when several match, the
        first is returned with a warning. A fragment of an ID matches
        nothing.

        Args:
            group_id: The group's ID, or a regular expression matching whole
                IDs.

        Returns:
            The TimelineGroup object.

        Raises:
            KeyError: If no group matches the pattern.

        Examples:
            >>> from timetoalign.alignment import AlignmentBundle
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     ContinuousPhysicalTimeline,
            ... )
            >>> bundle = AlignmentBundle()
            >>> _ = bundle.add_timeline(
            ...     ContinuousLogicalTimeline(length=16, uid="score"), as_group="score"
            ... )
            >>> _ = bundle.add_timeline(
            ...     ContinuousPhysicalTimeline(length=40.0, uid="audio"), as_group="audio"
            ... )
            >>> group_id = bundle.get_group_for_timeline("score").id
            >>> bundle.get_group(group_id).timeline_ids
            ['score']
        """
        resolved_id = resolve_id(group_id, list(self.groups.keys()))
        return self.groups[resolved_id]

    def get_group_for_timeline(self, timeline_id: str) -> TimelineGroup | None:
        """Get the one group containing a timeline.

        Args:
            timeline_id: The timeline's unique identifier.

        Returns:
            The TimelineGroup containing the timeline, or None if standalone.

        Raises:
            ValueError: If the timeline is a member of several groups; use
                :meth:`get_groups_for_timeline` for all of them.
        """
        groups = self.get_groups_for_timeline(timeline_id)
        if len(groups) > 1:
            raise ValueError(
                f"Timeline {timeline_id!r} is a member of several groups "
                f"{[group.id for group in groups]}; use get_groups_for_timeline"
            )
        return groups[0] if groups else None

    def get_groups_for_timeline(self, timeline_id: str) -> tuple[TimelineGroup, ...]:
        """Get every group containing a timeline, sorted by group id.

        Args:
            timeline_id: The timeline's unique identifier.

        Returns:
            The groups holding the timeline; empty if it is standalone.
        """
        return tuple(
            self.groups[group_id]
            for group_id in self.timeline_to_groups.get(timeline_id, ())
        )

    def _shared_group(self, timeline_a: str, timeline_b: str) -> TimelineGroup | None:
        """Return the first group, by sorted id, holding both timelines.

        Members of one group share one start and one end, so every group two
        timelines share converts between them alike; the first stands for all.
        """
        membership = self.timeline_to_groups
        shared = set(membership.get(timeline_a, ())) & set(
            membership.get(timeline_b, ())
        )
        return self.groups[min(shared)] if shared else None

    def _warp_units(
        self, belief_key: _BeliefKey = _ALL_BELIEFS
    ) -> list[tuple[TimelineGroup | None, tuple[str, ...]]]:
        """Return the units a matchstamp warps between, each with its members.

        Every group is one unit, and so is every other registered timeline,
        with ``None`` for its group: it is warped from and into exactly as a
        group of that one member would be. A timeline of the bundle's trees
        that is neither registered nor grouped (:meth:`_known_timelines`) is
        a unit of its own only when a claim names it.

        A claim names the timeline it was made on: claims on a child build
        that child's ``MatchLine``, never its root's, so the root's offsets
        carry a position to the child (:meth:`_manifest_relatives`) but only
        the child's unit warps it on through those claims. A unit no claim
        reaches warps nothing, so a descendant no claim names is left out
        rather than tried on every step of the closure. The claims are those
        in force under the belief filter *belief_key*.
        """
        return closure._warp_units(self, belief_key)

    def _group_component(self, timeline_id: str) -> set[str]:
        """Return the timelines reached from one through shared groups.

        A timeline of several groups joins them: every member of a group it
        belongs to is reached, and so on through those members' groups. The
        timeline itself is included.
        """
        return closure._group_component(self, timeline_id)

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
            >>> from timetoalign.alignment import AlignmentBundle
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     ContinuousPhysicalTimeline,
            ...     DiscretePhysicalTimeline,
            ... )
            >>> bundle = AlignmentBundle()
            >>> _ = bundle.add_timeline(ContinuousLogicalTimeline(length=16, uid="score"))
            >>> _ = bundle.add_timeline(ContinuousPhysicalTimeline(length=40.0, uid="audio"))
            >>> _ = bundle.add_timeline(DiscretePhysicalTimeline(length=88200, uid="midi"))
            >>> timelines = bundle.get_timelines(["score", "audio", "midi"])
            >>> [timeline.id for timeline in timelines]
            ['score', 'audio', 'midi']
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
    def timeline_to_groups(self) -> dict[str, tuple[str, ...]]:
        """Every grouped timeline's id, mapped to the ids of its groups.

        Read from :attr:`groups`, the one statement of membership. A
        timeline may be a member of several groups; its group ids are
        sorted. A standalone timeline is absent.

        Examples:
            >>> from timetoalign.alignment import AlignmentBundle
            >>> from timetoalign.timelines import TimelineGroup
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     ContinuousPhysicalTimeline,
            ... )
            >>> score = ContinuousLogicalTimeline(length=16, uid="score")
            >>> audio = ContinuousPhysicalTimeline(length=40.0, uid="audio")
            >>> video = ContinuousPhysicalTimeline(length=40.0, uid="video")
            >>> bundle = AlignmentBundle()
            >>> _ = bundle.add_group(TimelineGroup(id="g1", timelines=[score, audio]))
            >>> _ = bundle.add_group(TimelineGroup(id="g2", timelines=[score, video]))
            >>> bundle.timeline_to_groups
            {'score': ('g1', 'g2'), 'audio': ('g1',), 'video': ('g2',)}
        """
        return _groups_by_timeline(self.groups)

    @property
    def group_ids(self) -> list[str]:
        """List of all group IDs in the bundle."""
        return list(self.groups.keys())

    # endregion

    # region Coordinate Transfer

    transfer = closure.transfer

    def transfer_interval(
        self,
        start: CoordinateSpec,
        end: CoordinateSpec,
        from_timeline: str,
        to_timeline: str,
        *,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> tuple[float, float] | None:
        """Transfer an interval from one timeline to another.

        Args:
            start: Start coordinate in source timeline. Accepts a raw
                int/float/Fraction, a Coordinate, or an IdCoordinate.
            end: End coordinate in source timeline. Accepts a raw
                int/float/Fraction, a Coordinate, or an IdCoordinate.
            from_timeline: ID of the source timeline.
            to_timeline: ID of the target timeline.
            believer: The actor whose beliefs filter the MatchClaims read
                (the belief filter); ``None`` reads every claim.
            beliefs: The states read for the believer (``"true"``,
                ``"undecided"``, ``"unlikely"``, ``"false"``); default all
                but ``"false"``.

        Returns:
            Tuple of (start, end) in target timeline, or None if no path.
        """
        return closure._transfer_interval(
            self,
            start,
            end,
            from_timeline,
            to_timeline,
            believer=believer,
            beliefs=beliefs,
        )

    def are_commensurable(
        self,
        timeline_a: str,
        timeline_b: str,
        *,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> bool:
        """Check if two timelines can be connected via transfer.

        Two timelines are commensurable if groups connect them — they share
        a group, or a chain of groups sharing members leads from one to the
        other — or if a synchronous MatchClaim in force under the belief
        filter connects the timelines so connected to each.

        Args:
            timeline_a: First timeline ID.
            timeline_b: Second timeline ID.
            believer: The actor whose beliefs filter the claims (the belief
                filter); ``None`` reads every claim.
            beliefs: The states read for the believer (``"true"``,
                ``"undecided"``, ``"unlikely"``, ``"false"``); default all
                but ``"false"``.

        Returns:
            True if coordinates can be transferred between them.
        """
        if timeline_a == timeline_b:
            return True
        group_a_tl_ids = self._group_component(timeline_a)
        if timeline_b in group_a_tl_ids:
            return True
        # One vectorized pass over the claims in force; none is materialised.
        return self._claims_in_force(_belief_key(believer, beliefs)).connects_groups(
            group_a_tl_ids, self._group_component(timeline_b)
        )

    # endregion

    # region MatchClaims

    def create_match_claims(
        self,
        event_pairs: Sequence[tuple[dict | str, str, dict | str, str]],
        *,
        synchronous: bool = True,
        certainty: float = 1.0,
        provenance: Provenance | None = None,
        actor: str | None = None,
        date: str | datetime.date | None = None,
    ) -> list[MatchClaim]:
        """Create MatchClaims between events, projecting a side that has none.

        A MatchClaim connects two sides. Each tuple ``(event_a,
        timeline_a_id, event_b, timeline_b_id)`` names one side per
        timeline, each side one of:

        - the ``id`` of an event its timeline (or a descendant of it) holds:
          the timeline's own event answers for its ID; an ID the timeline does
          not hold and two of its descendants do names no one event and is
          refused (name the descendant holding the event instead);
        - an event dict whose ``id`` its timeline holds: that event;
        - an event dict stating a position: ``start``, and ``end`` for a
          span, each a number on the timeline's axis or a
          :class:`~timetoalign.core.Coordinate` in any unit one of the
          timeline's C-Maps converts, and optionally ``name``. On a
          timeline counting samples a position is an ``int`` or a
          ``Coordinate``, never a bare float. A span runs forward (``end``
          at or after ``start``). A **Projection** stands for the side
          there, on the timeline's axis
          (:meth:`Timeline.add_projections`): under the dict's ``id`` (an
          event the bundle does not hold, ``projection_of`` null), or,
          without one, naming the other side's event in ``projection_of``
          when the other side is an event its timeline holds (null when the
          other side is ``NOMATCH`` or both sides are positions of a
          conceptual pair). The timeline keeps one
          Projection per event and position: a second call with the same
          pairs reuses every Projection the first added. An ``id`` the
          timeline already holds names that event; given with
          ``start``/``end``, they must be the event's own span. The ``name``
          is the claim's name of the Projection: the name the first claim on
          a Projection recorded is the one every later claim on it shows.
          Where both sides of a synchronous pair are dicts without ``id``,
          no Projection stands for either: both sides are
          :data:`~timetoalign.core.ids.CHUNK`, the claim's anchor holding
          the two positions (an interval where both state an end; a
          ``name`` names the stated position and is kept as that side's
          ``event_X_name``);
        - :data:`~timetoalign.core.ids.NOMATCH`, the NOMATCH sentinel, on
          one side at most.

        The MatchClaim is then made between the two sides, on the two
        timelines' own axes (:meth:`MatchClaim.from_events`); an end on both
        sides makes an interval claim. A pair with ``NOMATCH`` makes a
        NOMATCH claim for the other side's event
        (:meth:`MatchClaim.nomatch`), its id minted at its construction. The
        claims join the bundle (:meth:`add_match_claims`).

        Every pair is checked before any Projection is added, so a refusal
        leaves the bundle and its timelines unchanged: the timelines are the
        bundle's, an id names an event of its timeline, at most one side is
        ``NOMATCH``, every position is placed on its timeline's axis. A
        position, a Projection's or a CHUNK side's, never lengthens a locked
        timeline (a loader locks a timeline whose source states its length,
        such as a recording's): a position past its end is refused. An
        unlocked timeline grows to hold it where its parent contains the new
        end (a parent never grows
        with its child); a position past an end the parent cannot contain is
        refused, as is one past where the parent ends a child of unknown
        length.

        Args:
            event_pairs: The pairs, ``(event_a, timeline_a_id, event_b,
                timeline_b_id)`` each.
            synchronous: Whether the matches are temporally synchronous; a
                non-synchronous (conceptual) claim holds no anchors. Ignored
                for NOMATCH pairs, which are always non-synchronous.
            certainty: How certain every claim is, in [0, 1].
            provenance: How the claims were made and who believes them
                (:class:`~timetoalign.core.provenance.Provenance`), shared
                by every claim.
            actor: The person stating the claims, an IRI: recorded in
                their provenance with the belief ``"true"``, replacing that
                actor's belief in *provenance* (actors compared as IRIs
                under the bundle's ``@context``). A loader or a service
                records how the claims were made and states no belief.
            date: The day of *actor*'s belief, ``YYYY-MM-DD`` or a
                :class:`datetime.date`; today when None.

        Returns:
            The claims made, in pair order, each holding the id the bundle
            gave it.

        Raises:
            ValueError: If both sides of a pair are ``NOMATCH``, a side is
                ``CHUNK`` or a dict whose ``id`` is ``NOMATCH`` or ``CHUNK``
                (a CHUNK side is a dict stating a position without ``id``,
                a NOMATCH side is ``NOMATCH``), a pair with CHUNK on both
                sides states an end on one side only, an event ID
                names no one event (the timeline does not hold it and two
                of its descendants do), a dict naming no held event states
                no ``start``, a position is a float on a timeline counting
                samples, a span runs backwards, a position lies past where
                its timeline can reach (the end of a locked timeline, or an
                end its parent cannot contain), an ``id`` the timeline holds
                is given with a span not its own, a position cannot be placed
                on its timeline's axis, a float start, end or duration is
                one its timeline cannot store (not finite, or too large for
                a number cell), or *date* is given without *actor* or is no
                calendar date.
            KeyError: If a timeline id names no timeline of the bundle's
                trees (a nested timeline is found whether or not it was
                registered), or an id string names no event of its timeline.

        Examples:
            >>> from timetoalign import NOMATCH
            >>> from timetoalign.alignment import AlignmentBundle
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     ContinuousPhysicalTimeline,
            ... )
            >>> score = ContinuousLogicalTimeline(length=16, uid="score")
            >>> score.add_events(
            ...     [
            ...         {"id": "n10", "event_type": "Note", "instant": 4},
            ...         {"id": "n99", "event_type": "Note", "instant": 8},
            ...     ]
            ... )
            ['n10', 'n99']
            >>> audio = ContinuousPhysicalTimeline(length=60.0, uid="audio")
            >>> audio.add_events([{"id": "o42", "event_type": "Note", "instant": 50.5}])
            ['o42']
            >>> bundle = AlignmentBundle()
            >>> _ = bundle.add_timeline(score, as_group="score")
            >>> _ = bundle.add_timeline(audio, as_group="audio")
            >>> claims = bundle.create_match_claims(
            ...     [
            ...         ("n10", "score", "o42", "audio"),
            ...         ("n99", "score", {"start": 55.0}, "audio"),  # projected
            ...         ("n99", "score", NOMATCH, "audio"),
            ...     ]
            ... )
            >>> [(claim.event_a_id, claim.event_b_id) for claim in claims[::2]]
            [('n10', 'o42'), ('n99', 'NOMATCH_SENTINEL')]
            >>> projection = claims[1].event_b_id
            >>> audio.get_event(projection)["projection_of"]
            {'timeline_id': 'score', 'event_id': 'n99'}
            >>> again = bundle.create_match_claims([("n99", "score", {"start": 55.0}, "audio")])
            >>> again[0].event_b_id == projection, again[0].id == claims[1].id
            (True, False)
            >>> stated = bundle.create_match_claims(
            ...     [("n10", "score", "o42", "audio")],
            ...     actor="https://orcid.org/0000-0002-1825-0097",
            ...     date="2026-11-21",
            ... )
            >>> stated[0].provenance.beliefs[0].belief
            'true'
        """
        return claim_events._create_match_claims(
            self,
            event_pairs,
            synchronous=synchronous,
            certainty=certainty,
            provenance=provenance,
            actor=actor,
            date=date,
        )

    def add_match_claims(self, claims: Iterable[MatchClaim]) -> "AlignmentBundle":
        """Add MatchClaims to the bundle's MatchClaim table.

        A claim names timelines of the bundle's trees, and each of its
        coordinates is on its timeline's own axis, its unit and declared
        number type (:meth:`MatchClaim.check_on_timelines`): a position in
        another unit is converted when the claim is made
        (:meth:`create_match_claims`, :meth:`MatchClaim.from_events`), never
        here, and a claim holding one is refused.

        A MatchClaim is addressed by its id alone, and a claim's id never
        changes: an id the bundle holds, or one held by two of the claims,
        is refused, whether it was given or minted. Every claim holds its id
        from its construction, and joining mints none. Each claim records the bundle
        (:attr:`MatchClaim.bundle`); a claim another bundle holds joins as a
        copy. The table holds the claims' rows, so a claim read back from
        the bundle equals the one added but is another object. Every check
        precedes any change.

        WarpMaps are built lazily on first transfer or ``get_matchstamp_at()``
        call, so adding claims is cheap.

        Args:
            claims: The MatchClaims.

        Returns:
            self (for method chaining)

        Raises:
            TypeError: If an item is not a MatchClaim.
            KeyError: If a claim names a timeline the bundle does not hold.
            ValueError: If a claim's id is held by another claim of the
                bundle or by two of the claims, or a coordinate is not on
                its timeline's axis.
        """
        return claim_store._add_match_claims(self, claims)

    def add_match_claim_field(self, claim_field: MatchClaimField) -> "AlignmentBundle":
        """Add the rows of a :class:`MatchClaimField` to the bundle's table.

        The columnar counterpart of :meth:`add_match_claims`, for a bulk
        producer (an alignment loader, a MatchClaim table read from its
        file): no claim is materialised. The same checks hold: every row
        names timelines of the bundle's trees with coordinates on their
        axes, and no row's id is held by the bundle (ids never change). The field passed in is left as it is.

        Args:
            claim_field: The MatchClaims.

        Returns:
            self (for method chaining)

        Raises:
            KeyError: If a row names a timeline the bundle does not hold.
            ValueError: If a row's id is held by another claim of the
                bundle, or a coordinate is not on its timeline's axis.
        """
        self._check_rows(claim_field)
        self._append_rows(
            claim_field._joined(self.match_claim_table._ids, holder="bundle")
        )
        return self

    def add_match_claim_belief(
        self,
        claim_id: str,
        *,
        actor: str,
        belief: BeliefValue,
        date: str | datetime.date | None = None,
        note: str | None = None,
    ) -> "AlignmentBundle":
        """Record *actor*'s belief in the MatchClaim *claim_id*.

        A MatchClaim is never removed and never changed: invalidating one is
        believing it ``"false"``, and a correction is a new MatchClaim. Only
        the claim's provenance changes. The belief replaces the actor's
        earlier one in that claim, actors compared as IRIs under the
        bundle's ``@context``, every prefix it declares (a compact and an
        absolute spelling of one actor are one actor); every other actor's
        belief stands. A claim stated without provenance gets an entry
        holding the belief alone. The belief filter (``believer=``,
        ``beliefs=`` on every query) reads it at once; the MatchClaim table
        written by :meth:`to_file` succeeds the one written before. A belief
        in a pair of the bundle's own statements is recorded by
        :meth:`add_belief`.

        Args:
            claim_id: The MatchClaim's id.
            actor: Who holds the belief, an IRI (compact or absolute).
            belief: ``"true"``, ``"false"`` or ``"unlikely"``.
            date: The day, ``YYYY-MM-DD`` or a :class:`datetime.date`; today
                when None.
            note: A remark.

        Returns:
            self (for method chaining)

        Raises:
            KeyError: If the bundle holds no MatchClaim *claim_id*.
            ValueError: If *belief* is none of the three words, or *date* is
                no calendar date.

        Examples:
            >>> from timetoalign.alignment import AlignmentBundle
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     DiscretePhysicalTimeline,
            ... )
            >>> bundle = AlignmentBundle()
            >>> _ = bundle.add_timeline(ContinuousLogicalTimeline(length=16, uid="score"))
            >>> _ = bundle.add_timeline(DiscretePhysicalTimeline(length=705600, uid="pcm"))
            >>> (claim,) = bundle.create_match_claims(
            ...     [({"start": 0}, "score", {"start": 0}, "pcm")]
            ... )
            >>> _ = bundle.add_match_claim_belief(
            ...     claim.id, actor="https://orcid.org/0000-0001", belief="false",
            ...     date="2026-11-21",
            ... )
            >>> len(bundle.get_match_claims(believer="https://orcid.org/0000-0001"))
            0
        """
        return claim_store._add_match_claim_belief(
            self, claim_id, actor=actor, belief=belief, date=date, note=note
        )

    def _check_rows(self, rows: MatchClaimField) -> None:
        """Refuse rows naming a timeline the bundle lacks or one off its axis.

        Raises:
            KeyError: Naming the timelines the bundle does not hold.
            ValueError: If a coordinate is not on its timeline's axis.
        """
        return claim_store._check_rows(self, rows)

    def _append_rows(self, rows: MatchClaimField) -> None:
        """Append checked, admitted rows to the table."""
        if len(rows) == 0:
            return
        self.match_claim_table = self.match_claim_table._concatenated(rows)
        self._invalidate_warp_cache()

    def _take_claims_of(self, bundle: "AlignmentBundle") -> None:
        """Append another bundle's claims, every id kept.

        The merge refused sources sharing a claim id before it began
        (:func:`_refuse_shared_claim_ids`), so the ids are not checked again.
        """
        rows = bundle.match_claim_table
        if len(rows) == 0:
            return
        self._check_rows(rows)
        self._append_rows(rows)

    def _claims_in_force(
        self, belief_key: _BeliefKey = _ALL_BELIEFS
    ) -> MatchClaimField:
        """The claims the belief filter keeps: the one point every reader reads.

        Every reader of the bundle's MatchClaims — the queries, the
        matchstamp getters, the table getter, the MatchGraphs, MatchLines and
        WarpMaps behind them — takes its claims from here, so the belief
        filter (:func:`~timetoalign.core.provenance.belief_mask`, through
        :meth:`MatchClaimField.filter`) is applied once, vectorized, per
        state of the table, actors compared as IRIs under the bundle's
        manifest ``@context`` (:func:`~timetoalign.alignment.filters.belief_context`),
        as its statements compare them.

        Args:
            belief_key: ``(believer, beliefs)`` (:func:`_belief_key`); no
                believer keeps every claim.
        """
        return claim_store._claims_in_force(self, belief_key)

    def _claims_naming(
        self, timeline_id: str, belief_key: _BeliefKey = _ALL_BELIEFS
    ) -> MatchClaimField:
        """The claims in force naming *timeline_id*, cut once per table state.

        A timeline no claim names (a loader's category child, an unfolded
        span) costs one lookup, however many claims the bundle holds.
        """
        key = (belief_key, timeline_id)
        naming = self._naming_cache.get(key)
        if naming is None:
            naming = self._claims_in_force(belief_key).connecting(timeline_id)
            self._naming_cache[key] = naming
        return naming

    def _materialized(self, rows: MatchClaimField) -> list[MatchClaim]:
        """The rows as MatchClaims, each recording this bundle."""
        claims = rows.to_claims()
        for claim in claims:
            claim.set_bundle(self)
        return claims

    def _known_id_pattern(self, id_pattern: str | None) -> str | None:
        """Narrow a timeline-id regex to the timelines the bundle knows.

        A pattern only ever selects among the known timelines
        (:meth:`_known_timelines`), so it is rewritten as an exact
        alternation of the known ids it matches.
        """
        return claim_store._known_id_pattern(self, id_pattern)

    def get_match_claims(
        self,
        *,
        timeline_id: str | None = None,
        timeline_ids: set[str] | None = None,
        id_pattern: str | None = None,
        between: tuple[str, str] | None = None,
        within: set[str] | None = None,
        synchronous_only: bool = False,
        nomatch_only: bool = False,
        include_domains: set["Domain"] | None = None,
        include_units: set["TimeUnit"] | None = None,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> list[MatchClaim]:
        """Query the bundle's MatchClaims.

        All parameters are optional; when none are provided, returns all
        claims, in table order. Filters are combined with AND logic: a claim
        must satisfy every criterion set (the unified filter parameters).

        The table is filtered vectorized (:meth:`get_claim_field`) and the
        rows kept are then **materialised**, one ``MatchClaim`` each,
        recording this bundle — for a dense pairwise alignment that is O(n)
        Python objects. Use :meth:`get_claim_field` when the columnar answer
        suffices.

        Args:
            timeline_id: Return claims involving this timeline ID.
            timeline_ids: Return claims involving any of these timeline IDs.
            id_pattern: Regex pattern matched against timeline IDs via
                ``re.search()``. Example: ``r"^perf:"`` matches all
                performance timelines.
            between: Return claims connecting exactly these two timeline IDs
                (order-independent).
            within: Return claims whose both timelines are in this set.
            synchronous_only: Only the synchronous claims (with anchors).
            nomatch_only: Only the NOMATCH claims.
            include_domains: Only timelines in these domains.
            include_units: Only timelines with these units.
            believer: The actor whose beliefs filter the claims (the belief
                filter); ``None`` filters nothing.
            beliefs: The states kept for the believer (``"true"``,
                ``"undecided"``, ``"unlikely"``, ``"false"``); default all
                but ``"false"``.

        Returns:
            Filtered list of MatchClaims.

        Examples:
            A score aligned with two performances, one of which skips a
            note::

                >>> from timetoalign import NOMATCH
                >>> from timetoalign.alignment import AlignmentBundle
                >>> from timetoalign.timelines.types import (
                ...     ContinuousLogicalTimeline,
                ...     DiscreteLogicalTimeline,
                ... )
                >>> bundle = AlignmentBundle()
                >>> _ = bundle.add_timeline(
                ...     ContinuousLogicalTimeline(length=16, uid="score:clt_1"), as_group="score"
                ... )
                >>> for n in (1, 5):
                ...     _ = bundle.add_timeline(
                ...         DiscreteLogicalTimeline(length=7680, uid=f"perf:dlt_{n}"),
                ...         as_group=f"perf_{n}",
                ...     )
                >>> _ = bundle.create_match_claims(
                ...     [
                ...         ({"start": 0}, "score:clt_1", {"start": 0}, "perf:dlt_1"),
                ...         ({"start": 8}, "score:clt_1", {"start": 3840}, "perf:dlt_1"),
                ...         ({"start": 0}, "score:clt_1", {"start": 0}, "perf:dlt_5"),
                ...         ({"start": 8}, "score:clt_1", NOMATCH, "perf:dlt_5"),
                ...     ]
                ... )

            Get all synchronous claims for a performer::

                >>> claims = bundle.get_match_claims(
                ...     id_pattern=r"dlt_1$", synchronous_only=True
                ... )
                >>> len(claims)
                2

            Get NOMATCH claims for a specific pair::

                >>> nomatches = bundle.get_match_claims(
                ...     between=("score:clt_1", "perf:dlt_5"),
                ...     nomatch_only=True,
                ... )
                >>> len(nomatches)
                1
        """
        return claim_store._get_match_claims(
            self,
            timeline_id=timeline_id,
            timeline_ids=timeline_ids,
            id_pattern=id_pattern,
            between=between,
            within=within,
            synchronous_only=synchronous_only,
            nomatch_only=nomatch_only,
            include_domains=include_domains,
            include_units=include_units,
            believer=believer,
            beliefs=beliefs,
        )

    def get_claim_field(
        self,
        *,
        timeline_id: str | None = None,
        timeline_ids: set[str] | None = None,
        id_pattern: str | None = None,
        between: tuple[str, str] | None = None,
        within: set[str] | None = None,
        synchronous_only: bool = False,
        nomatch_only: bool = False,
        include_domains: set["Domain"] | None = None,
        include_units: set["TimeUnit"] | None = None,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> MatchClaimField:
        """Query the MatchClaim table, materialising nothing.

        The columnar counterpart of :meth:`get_match_claims`, with the same
        filter parameters and meaning: every criterion is a vectorized
        PyArrow mask over :attr:`match_claim_table`, and the answer is the
        filtered table (empty when nothing passes).

        Args:
            timeline_id: Keep claims involving this timeline ID.
            timeline_ids: Keep claims involving any of these timeline IDs.
            id_pattern: Regex matched against timeline IDs via ``re.search()``.
            between: Keep claims connecting exactly these two timeline IDs
                (order-independent).
            within: Keep claims whose both timelines are in this set.
            synchronous_only: Only the synchronous claims (with anchors).
            nomatch_only: Only the NOMATCH claims.
            include_domains: Only timelines in these domains.
            include_units: Only timelines with these units.
            believer: The actor whose beliefs filter the claims (the belief
                filter); ``None`` filters nothing.
            beliefs: The states kept for the believer; default all but
                ``"false"``.

        Returns:
            The filtered :class:`MatchClaimField`.

        Examples:
            >>> from timetoalign.alignment import AlignmentBundle, MatchClaimField
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     ContinuousPhysicalTimeline,
            ... )
            >>> score = ContinuousLogicalTimeline(length=16, uid="score:clt_1")
            >>> audio = ContinuousPhysicalTimeline(length=40.0, uid="rec-a:cpt_1")
            >>> bundle = AlignmentBundle()
            >>> _ = bundle.add_timeline(score, as_group="score")
            >>> _ = bundle.add_timeline(audio, as_group="rec-a")
            >>> _ = bundle.add_match_claim_field(
            ...     MatchClaimField.from_columns(
            ...         ["score:clt_1"] * 3,
            ...         ["rec-a:cpt_1"] * 3,
            ...         ["n1", "n2", "n3"],
            ...         ["o1", "o2", "o3"],
            ...         [0, 8, 16],
            ...         [0.0, 20.0, 40.0],
            ...         timelines=[score, audio],
            ...     )
            ... )
            >>> len(bundle.get_claim_field(timeline_id="rec-a:cpt_1"))
            3
        """
        return claim_store._get_claim_field(
            self,
            timeline_id=timeline_id,
            timeline_ids=timeline_ids,
            id_pattern=id_pattern,
            between=between,
            within=within,
            synchronous_only=synchronous_only,
            nomatch_only=nomatch_only,
            include_domains=include_domains,
            include_units=include_units,
            believer=believer,
            beliefs=beliefs,
        )

    def _claim_filter(
        self,
        *,
        timeline_id: str | None,
        timeline_ids: set[str] | None,
        id_pattern: str | None,
        between: tuple[str, str] | None,
        within: set[str] | None,
        synchronous_only: bool,
        nomatch_only: bool,
        include_domains: set["Domain"] | None,
        include_units: set["TimeUnit"] | None,
        believer: str | None,
        beliefs: Collection[BeliefState],
    ) -> ClaimFilter | None:
        """Build the ``ClaimFilter`` for the timelines the bundle knows.

        Every ID-shaped criterion selects among the bundle's known timelines
        (:meth:`_known_timelines`).

        Returns:
            The filter, or ``None`` when a required timeline is unknown to the
            bundle and no claim can possibly match.
        """
        return claim_store._claim_filter(
            self,
            timeline_id=timeline_id,
            timeline_ids=timeline_ids,
            id_pattern=id_pattern,
            between=between,
            within=within,
            synchronous_only=synchronous_only,
            nomatch_only=nomatch_only,
            include_domains=include_domains,
            include_units=include_units,
            believer=believer,
            beliefs=beliefs,
        )

    def _axis_on(
        self, coordinate: CoordinateValue, timeline_id: str
    ) -> int | float | Fraction:
        """Express a computed coordinate the way *timeline_id* writes numbers.

        The WarpMap feeds run on floats; a position mapped through an
        interval claim is exact whenever its inputs are. The axis a stamp
        reports is a different thing from either: it has crossed onto a
        timeline's axis, and an axis writes every number in its declared
        representation. Without this, the same position answered differently
        depending on whether the caller happened to pass a float or a
        Fraction, which is the argument-type inference the boundary rule
        exists to remove.
        """
        timeline = self._known_timeline(timeline_id)
        if timeline is None:
            return coordinate
        return express_as(coordinate, timeline.number_type)

    def _relevant_claims_at(
        self,
        timeline_id: str,
        coordinate: Coordinate,
        belief_key: _BeliefKey = _ALL_BELIEFS,
    ) -> list[MatchClaim]:
        """The claims in force relevant at a position, in table order.

        An instant claim is relevant where one of its coordinates on the
        timeline equals the position, an interval claim where its span on
        the timeline holds it (closed); a claim without anchors never is
        (:meth:`MatchClaimField.at`). Only the claims naming the timeline
        are compared, so a timeline no claim names costs nothing. Every
        coordinate is on its timeline's axis since the claim was added, so
        none is converted here.
        """
        return self._materialized(
            self._claims_naming(timeline_id, belief_key).at(timeline_id, coordinate)
        )

    def _group_positions(
        self, timeline_id: str, query: Coordinate
    ) -> dict[str, Coordinate]:
        """A position on one timeline and on every member of its groups.

        The members are reached through shared groups as
        :meth:`_group_component` reaches them, the groups of each member in
        id order and their members in group order; every position is the
        group's exact conversion (:meth:`_get_group_timestamp`), on the
        member's own axis. A member the position lies outside of is left
        out.

        Returns:
            The queried timeline first, then each member reached, in that
            order.
        """
        return closure._group_positions(self, timeline_id, query)

    @staticmethod
    def _interval_claim_reading(
        claim: MatchClaim, positions: Mapping[str, Coordinate]
    ) -> tuple[str, Coordinate]:
        """The timeline an interval claim is read on, with the position there.

        The first of *positions* (see :meth:`_group_positions`) that the
        claim names and whose span on that timeline holds the position,
        closed, as :meth:`_relevant_claims_at` selects it.

        Raises:
            ValueError: If no position is held by the claim.
        """
        return closure._interval_claim_reading(claim, positions)

    def _interval_readings(
        self,
        timeline_id: str,
        query: Coordinate,
        belief_key: _BeliefKey = _ALL_BELIEFS,
    ) -> list[tuple[MatchClaim, str, Coordinate]]:
        """The interval claims a matchstamp reads, each with where it reads it.

        A position on one member of a group is the same instant on every
        other member, so the claims relevant at the query are those relevant
        at the queried position and those relevant at its position on each
        member of the queried timeline's groups (:meth:`_group_positions`).
        Each claim is listed once, at the first of those timelines it is
        relevant on (:meth:`_interval_claim_reading`): the queried
        timeline's claims first, in table order, then each member's. The
        members' positions are computed and read only when an interval claim
        names one of them, so a bundle of instant claims does no more work.

        Returns:
            ``(claim, timeline_id, position)`` per claim: the timeline the
            claim is read on and the query's position there.
        """
        return closure._interval_readings(self, timeline_id, query, belief_key)

    def _interval_claim_candidate(
        self,
        claim: MatchClaim,
        queried_timeline_id: str,
        query: Coordinate,
    ) -> tuple[str, Coordinate | None, bool]:
        """Map one query through an interval claim onto its counterpart axis.

        Exact endpoints are returned directly. An interior position is mapped
        affinely, in exact rational arithmetic when the query and the four
        span bounds are all exact (ints, Fractions), so that 44/87 of a span
        of 7 quarters is 308/87, never the binary ratio of a double; a float
        among them makes it double arithmetic. The result crosses the
        counterpart timeline's axis boundary exactly once, written as that
        axis declares. A point on the queried side paired with a nonzero
        counterpart interval has no single mapped coordinate.

        Returns:
            Counterpart timeline ID, mapped coordinate or ``None``, and whether
            the query is one of the claim's stored anchors.
        """
        return closure._interval_claim_candidate(
            self, claim, queried_timeline_id, query
        )

    def _get_or_build_matchgraph(
        self,
        timeline_id: str,
        coordinate: CoordinateValue,
        belief_key: _BeliefKey = _ALL_BELIEFS,
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
            belief_key: The belief filter the claims are read under.

        Returns:
            The MatchGraph for this coordinate's connected component.

        Raises:
            ValueError: If no synchronous claims touch this node.
        """
        return closure._get_or_build_matchgraph(
            self, timeline_id, coordinate, belief_key
        )

    def get_coordinate_at(
        self,
        at: CoordinateInput,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> CoordinateResult | pd.Series:
        """Resolve one coordinate onto a requested timeline axis.

        Another timeline is reached through the MatchStamp at the position
        (:meth:`get_matchstamp_at`), read from the MatchClaims the belief
        filter keeps, as every query of the bundle reads them.

        Args:
            at: Raw, plain, or ID-bearing coordinate input.
            timeline_id: Requested result timeline ID.
            format: Requested coordinate output format.
            rounding: Integral projection mode.
            believer: The actor whose beliefs filter the MatchClaims read
                (the belief filter); ``None`` reads every claim.
            beliefs: The states read for the believer (``"true"``,
                ``"undecided"``, ``"unlikely"``, ``"false"``); default all
                but ``"false"``.

        Returns:
            One coordinate projection or a length-one Series.
        """
        return closure._get_coordinate_at(
            self,
            at,
            timeline_id,
            format=format,
            rounding=rounding,
            believer=believer,
            beliefs=beliefs,
        )

    def get_coordinates_at(
        self,
        at: CoordinateCollection,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> list[CoordinateResult] | pd.Series:
        """Resolve a coordinate collection onto one timeline axis.

        Args:
            at: Coordinate positions to resolve atomically.
            timeline_id: Requested result timeline ID.
            format: Requested coordinate output format.
            rounding: Integral projection mode.
            believer: The actor whose beliefs filter the MatchClaims read
                (the belief filter); ``None`` reads every claim.
            beliefs: The states read for the believer (``"true"``,
                ``"undecided"``, ``"unlikely"``, ``"false"``); default all
                but ``"false"``.

        Returns:
            A list of projections or canonical-value Series.
        """
        return closure._get_coordinates_at(
            self,
            at,
            timeline_id,
            format=format,
            rounding=rounding,
            believer=believer,
            beliefs=beliefs,
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
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> CoordinateResult | pd.Series:
        """Return a uniquely owned event's start on a selected bundle axis.

        Args:
            key: Event ID with exactly one owning registered timeline.
            timeline_id: Requested result timeline ID, or the owner by default.
            format: Requested coordinate output format.
            rounding: Integral projection mode.
            believer: The actor whose beliefs filter the MatchClaims read
                (the belief filter); ``None`` reads every claim.
            beliefs: The states read for the believer (``"true"``,
                ``"undecided"``, ``"unlikely"``, ``"false"``); default all
                but ``"false"``.

        Returns:
            One event-start projection or a length-one Series.
        """
        return closure._get_coordinate_for(
            self,
            key,
            timeline_id,
            format=format,
            rounding=rounding,
            believer=believer,
            beliefs=beliefs,
        )

    def get_coordinates_for(
        self,
        keys: KeyCollection,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> list[CoordinateResult] | pd.Series:
        """Return event-start coordinates for a key collection.

        Args:
            keys: Event IDs to resolve atomically.
            timeline_id: Requested result timeline ID, or each owner by default.
            format: Requested coordinate output format.
            rounding: Integral projection mode.
            believer: The actor whose beliefs filter the MatchClaims read
                (the belief filter); ``None`` reads every claim.
            beliefs: The states read for the believer (``"true"``,
                ``"undecided"``, ``"unlikely"``, ``"false"``); default all
                but ``"false"``.

        Returns:
            A list of projections or canonical-value Series.
        """
        return closure._get_coordinates_for(
            self,
            keys,
            timeline_id,
            format=format,
            rounding=rounding,
            believer=believer,
            beliefs=beliefs,
        )

    def get_coordinate(
        self,
        at: CoordinateInput | CoordinateCollection | str | KeyCollection,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> CoordinateResult | list[CoordinateResult] | pd.Series:
        """Dispatch a positional or event-key bundle coordinate query.

        Args:
            at: Scalar or plural coordinate position or event key.
            timeline_id: Requested result timeline ID when required.
            format: Requested coordinate output format.
            rounding: Integral projection mode.
            believer: The actor whose beliefs filter the MatchClaims read
                (the belief filter); ``None`` reads every claim.
            beliefs: The states read for the believer (``"true"``,
                ``"undecided"``, ``"unlikely"``, ``"false"``); default all
                but ``"false"``.

        Returns:
            The selected precise-getter result.
        """
        return closure._get_coordinate(
            self,
            at,
            timeline_id,
            format=format,
            rounding=rounding,
            believer=believer,
            beliefs=beliefs,
        )

    @overload
    def get_matchstamp_at(
        self,
        at: CoordinateSpec,
        timeline_id: str | None = None,
        *,
        support_policy: "SupportPolicy | str | None" = None,
        conversion_maps: "ConversionMapsSpec" = False,
        within: set[str] | None = None,
        id_pattern: str | None = None,
        include_domains: set["Domain"] | None = None,
        include_units: set["TimeUnit"] | None = None,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> MatchStamp: ...

    @overload
    def get_matchstamp_at(
        self,
        at: Interval,
        timeline_id: str | None = None,
        *,
        support_policy: "SupportPolicy | str | None" = None,
        conversion_maps: "ConversionMapsSpec" = False,
        within: set[str] | None = None,
        id_pattern: str | None = None,
        include_domains: set["Domain"] | None = None,
        include_units: set["TimeUnit"] | None = None,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> MatchIntervalStamp: ...

    @overload
    def get_matchstamp_at(
        self,
        at: CoordinateSpec | Interval,
        timeline_id: str | None = None,
        *,
        support_policy: "SupportPolicy | str | None" = None,
        conversion_maps: "ConversionMapsSpec" = False,
        within: set[str] | None = None,
        id_pattern: str | None = None,
        include_domains: set["Domain"] | None = None,
        include_units: set["TimeUnit"] | None = None,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> MatchStamp | MatchIntervalStamp: ...

    def get_matchstamp_at(
        self,
        at: CoordinateSpec | Interval,
        timeline_id: str | None = None,
        *,
        support_policy: "SupportPolicy | str | None" = None,
        conversion_maps: "ConversionMapsSpec" = False,
        within: set[str] | None = None,
        id_pattern: str | None = None,
        include_domains: set["Domain"] | None = None,
        include_units: set["TimeUnit"] | None = None,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
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
        exactly when the query node itself carries an anchor a claim states.

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
            within: Only these timelines' coordinates in the result, beside
                the queried timeline's, which the narrowing (this,
                ``id_pattern``, ``include_domains``, ``include_units``)
                always keeps.
            id_pattern: Regex filter for timeline IDs in the result.
            include_domains: Only these domains in the result.
            include_units: Only these units in the result.
            believer: The actor whose beliefs filter the MatchClaims read
                (the belief filter); ``None`` reads every claim.
            beliefs: The states read for the believer (``"true"``,
                ``"undecided"``, ``"unlikely"``, ``"false"``); default all
                but ``"false"``.

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
            >>> from timetoalign.alignment import AlignmentBundle
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     ContinuousPhysicalTimeline,
            ... )
            >>> bundle = AlignmentBundle()
            >>> _ = bundle.add_timeline(
            ...     ContinuousLogicalTimeline(length=16, uid="score:clt_1"), as_group="score"
            ... )
            >>> for n, seconds in ((1, 40.0), (2, 32.0)):
            ...     _ = bundle.add_timeline(
            ...         ContinuousPhysicalTimeline(length=seconds, uid=f"perf:cpt_{n}"),
            ...         as_group=f"perf_{n}",
            ...     )
            ...     _ = bundle.create_match_claims(
            ...         [
            ...             ({"start": 0}, "score:clt_1", {"start": 0.0}, f"perf:cpt_{n}"),
            ...             ({"start": 16}, "score:clt_1", {"start": seconds}, f"perf:cpt_{n}"),
            ...         ]
            ...     )
            >>> ms = bundle.get_matchstamp_at(10, "score:clt_1")
            >>> ms.n_timelines  # score + 2 performers
            3
            >>> ms.get_coordinate_for("perf:cpt_2", format="float")
            20.0
        """
        return closure._get_matchstamp_at(
            self,
            at,
            timeline_id,
            support_policy=support_policy,
            conversion_maps=conversion_maps,
            within=within,
            id_pattern=id_pattern,
            include_domains=include_domains,
            include_units=include_units,
            believer=believer,
            beliefs=beliefs,
        )

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
        within: set[str] | None = None,
        id_pattern: str | None = None,
        include_domains: set["Domain"] | None = None,
        include_units: set["TimeUnit"] | None = None,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
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
            within: Only these timelines' coordinates in each result, beside
                the queried timeline's, which the narrowing (this,
                ``id_pattern``, ``include_domains``, ``include_units``)
                always keeps.
            id_pattern: Regex filter for timeline IDs in each result.
            include_domains: Only these domains in each result.
            include_units: Only these units in each result.
            believer: The actor whose beliefs filter the MatchClaims read
                (the belief filter); ``None`` reads every claim.
            beliefs: The states read for the believer (``"true"``,
                ``"undecided"``, ``"unlikely"``, ``"false"``); default all
                but ``"false"``.

        Returns:
            One stamp per input position. Coordinate elements produce
            MatchStamps and Interval elements produce MatchIntervalStamps.

        Examples:
            >>> from timetoalign.alignment import AlignmentBundle
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     ContinuousPhysicalTimeline,
            ... )
            >>> bundle = AlignmentBundle()
            >>> _ = bundle.add_timeline(
            ...     ContinuousLogicalTimeline(length=100, uid="score:clt_1"), as_group="score"
            ... )
            >>> _ = bundle.add_timeline(
            ...     ContinuousPhysicalTimeline(length=200.0, uid="perf:cpt_1"), as_group="perf"
            ... )
            >>> _ = bundle.create_match_claims(
            ...     [
            ...         ({"start": 0}, "score:clt_1", {"start": 0.0}, "perf:cpt_1"),
            ...         ({"start": 100}, "score:clt_1", {"start": 200.0}, "perf:cpt_1"),
            ...     ]
            ... )
            >>> stamps = bundle.get_matchstamps_at([0, 50], "score:clt_1")
            >>> stamps[1].get_coordinate_for("score:clt_1", format="float")
            50.0
            >>> stamps[1].get_coordinate_for("perf:cpt_1", format="float")
            100.0
        """
        return closure._get_matchstamps_at(
            self,
            at,
            timeline_id,
            support_policy=support_policy,
            conversion_maps=conversion_maps,
            within=within,
            id_pattern=id_pattern,
            include_domains=include_domains,
            include_units=include_units,
            believer=believer,
            beliefs=beliefs,
        )

    def get_matchstamp_for(
        self,
        key: str,
        timeline_id: str | None = None,
        *,
        support_policy: "SupportPolicy | str | None" = None,
        conversion_maps: "ConversionMapsSpec" = False,
        within: set[str] | None = None,
        id_pattern: str | None = None,
        include_domains: set["Domain"] | None = None,
        include_units: set["TimeUnit"] | None = None,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> MatchStamp:
        """Get the cross-group MatchStamp at a uniquely owned event.

        Args:
            key: Event ID with exactly one owning registered timeline.
            timeline_id: ID of the timeline the stamp is anchored on; by default the
                event's owner.
            support_policy: How to treat out-of-support timelines.
            conversion_maps: C-map conversions available through the stamp.
            within: Only these timelines' coordinates in the result, beside
                the queried timeline's, which the narrowing (this,
                ``id_pattern``, ``include_domains``, ``include_units``)
                always keeps.
            id_pattern: Regex filter for timeline IDs in the result.
            include_domains: Only these domains in the result.
            include_units: Only these units in the result.
            believer: The actor whose beliefs filter the MatchClaims read
                (the belief filter); ``None`` reads every claim.
            beliefs: The states read for the believer (``"true"``,
                ``"undecided"``, ``"unlikely"``, ``"false"``); default all
                but ``"false"``.

        Returns:
            MatchStamp spanning all connected timelines.

        Raises:
            KeyError: If no registered timeline owns the event.
            ValueError: If several registered timelines own it.

        Examples:
            >>> from timetoalign.alignment import AlignmentBundle
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     ContinuousPhysicalTimeline,
            ... )
            >>> score = ContinuousLogicalTimeline(length=16, uid="score:clt_1")
            >>> _ = score.add_events([{"id": "n1", "event_type": "Note", "instant": 4}])
            >>> bundle = AlignmentBundle()
            >>> _ = bundle.add_timeline(score, as_group="score")
            >>> _ = bundle.add_timeline(
            ...     ContinuousPhysicalTimeline(length=40.0, uid="perf:cpt_1"), as_group="perf"
            ... )
            >>> _ = bundle.create_match_claims(
            ...     [
            ...         ({"start": 0}, "score:clt_1", {"start": 0.0}, "perf:cpt_1"),
            ...         ({"start": 16}, "score:clt_1", {"start": 40.0}, "perf:cpt_1"),
            ...     ]
            ... )
            >>> ms = bundle.get_matchstamp_for("n1")
            >>> ms.get_coordinate_for(ms.source_id)  # the event's own owner axis
            IdCoordinate(Fraction(4, 1), quarters, 'score:clt_1')
            >>> ms.get_coordinate_for("perf:cpt_1", format="float")
            10.0
        """
        return closure._get_matchstamp_for(
            self,
            key,
            timeline_id,
            support_policy=support_policy,
            conversion_maps=conversion_maps,
            within=within,
            id_pattern=id_pattern,
            include_domains=include_domains,
            include_units=include_units,
            believer=believer,
            beliefs=beliefs,
        )

    def get_matchstamps_for(
        self,
        keys: KeyCollection,
        timeline_id: str | None = None,
        *,
        support_policy: "SupportPolicy | str | None" = None,
        conversion_maps: "ConversionMapsSpec" = False,
        within: set[str] | None = None,
        id_pattern: str | None = None,
        include_domains: set["Domain"] | None = None,
        include_units: set["TimeUnit"] | None = None,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> list[MatchStamp]:
        """Get cross-group MatchStamps for a collection of event IDs.

        Args:
            keys: Event IDs to resolve, each with exactly one owner.
            timeline_id: ID of the timeline every stamp is anchored on; by default
                each event's own owner.
            support_policy: How to treat out-of-support timelines.
            conversion_maps: C-map conversions available through the stamps.
            within: Only these timelines' coordinates in each result, beside
                the queried timeline's, which the narrowing (this,
                ``id_pattern``, ``include_domains``, ``include_units``)
                always keeps.
            id_pattern: Regex filter for timeline IDs in each result.
            include_domains: Only these domains in each result.
            include_units: Only these units in each result.
            believer: The actor whose beliefs filter the MatchClaims read
                (the belief filter); ``None`` reads every claim.
            beliefs: The states read for the believer (``"true"``,
                ``"undecided"``, ``"unlikely"``, ``"false"``); default all
                but ``"false"``.

        Returns:
            One MatchStamp per key, in input order; an empty collection gives
            an empty list.

        Raises:
            KeyError: If any key is unknown. The batch answers completely or
                not at all.
        """
        return closure._get_matchstamps_for(
            self,
            keys,
            timeline_id,
            support_policy=support_policy,
            conversion_maps=conversion_maps,
            within=within,
            id_pattern=id_pattern,
            include_domains=include_domains,
            include_units=include_units,
            believer=believer,
            beliefs=beliefs,
        )

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
        within: set[str] | None = None,
        id_pattern: str | None = None,
        include_domains: set["Domain"] | None = None,
        include_units: set["TimeUnit"] | None = None,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> (
        MatchStamp
        | MatchIntervalStamp
        | list[MatchStamp]
        | list[MatchStamp | MatchIntervalStamp]
    ):
        """Dispatch a positional or event-key match-stamp query.

        Args:
            at: Scalar or plural coordinate or Interval position, or event key.
            timeline_id: ID of the timeline the query is expressed on.
            support_policy: How to treat out-of-support timelines.
            conversion_maps: C-map conversions available through the stamps.
            within: Only these timelines' coordinates in each result, beside
                the queried timeline's, which the narrowing (this,
                ``id_pattern``, ``include_domains``, ``include_units``)
                always keeps.
            id_pattern: Regex filter for timeline IDs in each result.
            include_domains: Only these domains in each result.
            include_units: Only these units in each result.
            believer: The actor whose beliefs filter the MatchClaims read
                (the belief filter); ``None`` reads every claim.
            beliefs: The states read for the believer (``"true"``,
                ``"undecided"``, ``"unlikely"``, ``"false"``); default all
                but ``"false"``.

        Returns:
            The selected precise-getter result.

        Raises:
            TypeError: If ``at`` mixes keys with position forms or is an
                unsupported runtime form.
        """
        return closure._get_matchstamp(
            self,
            at,
            timeline_id,
            support_policy=support_policy,
            conversion_maps=conversion_maps,
            within=within,
            id_pattern=id_pattern,
            include_domains=include_domains,
            include_units=include_units,
            believer=believer,
            beliefs=beliefs,
        )

    def _assemble_matchstamp(
        self,
        coordinate: CoordinateValue,
        actual_timeline_id: str,
        *,
        support_policy: SupportPolicy,
        conversion_maps: "ConversionMapsSpec",
        belief_key: _BeliefKey = _ALL_BELIEFS,
        interval_coordinates: dict[str, CoordinateValue] | None = None,
        interval_stated_pairs: list[tuple[str, str]] | None = None,
        interval_inferred_pairs: list[tuple[str, str]] | None = None,
        interval_withheld: set[str] | None = None,
        query_has_interval_anchor: bool = False,
    ) -> tuple[
        dict[str, CoordinateValue | None],
        list[tuple[str, str]],
        list[tuple[str, str]],
        bool,
    ]:
        """Assemble the transitive cross-group union reachable from a query.

        Seeds from the query node, overlays exact-anchor coordinates from the
        query's MatchGraph (exact wins over interpolated), then walks outward
        across groups to closure: every reached timeline expands into its own
        group (interpolation) and warps into not-yet-reached groups (WarpMap),
        an ungrouped timeline warping from and into a group of its own,
        each transferred coordinate governed by ``support_policy``. The query
        timeline's own coordinate is never dropped, clamped, or altered.

        Args:
            coordinate: The query coordinate in the query timeline's unit.
            actual_timeline_id: Actual timeline id of the query.
            support_policy: How out-of-support transfers are handled.
            conversion_maps: C-map spec forwarded to group timestamps.
            belief_key: The belief filter every claim is read under.
            interval_coordinates: Unambiguous interval-derived seed values.
            interval_stated_pairs: Interval-claim edges reached at an anchor.
            interval_inferred_pairs: Interval-claim edges reached in an interior.
            interval_withheld: Timelines the query's interval claims map to
                no single coordinate. Each stays unresolved unless an exact
                anchor of the query's MatchGraph gives it a coordinate, which
                then wins; no interpolation or warp fills it, so no transfer
                collapses the claims' alternatives into one value.
            query_has_interval_anchor: Whether an interval claim's anchor
                names the query position.

        Returns:
            ``(coordinates, stated_pairs, inferred_pairs, query_has_anchor)``.
            ``query_has_anchor`` is True when the query node carries an
            anchor a claim states (making the stamp non-interpolated).
        """
        return closure._assemble_matchstamp(
            self,
            coordinate,
            actual_timeline_id,
            support_policy=support_policy,
            conversion_maps=conversion_maps,
            belief_key=belief_key,
            interval_coordinates=interval_coordinates,
            interval_stated_pairs=interval_stated_pairs,
            interval_inferred_pairs=interval_inferred_pairs,
            interval_withheld=interval_withheld,
            query_has_interval_anchor=query_has_interval_anchor,
        )

    def get_matchstamps(
        self,
        claims: list[MatchClaim] | None = None,
        *,
        from_graph: bool = True,
        conversion_maps: "ConversionMapsSpec" = False,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> list[MatchStamp]:
        """Get MatchStamps for the bundle's claims.

        The bare plural means "stamps for the claims this bundle holds" — it
        is the claim-driven listing, not the batch sibling of
        :meth:`get_matchstamps_at`. To resolve a batch of *positions*, use
        :meth:`get_matchstamps_at`; to resolve event IDs, use
        :meth:`get_matchstamps_for`.

        Uses the bundle's caching mechanism for efficient retrieval.

        Args:
            claims: List of MatchClaims to get stamps for. If None, every
                MatchClaim of the bundle (:meth:`get_match_claims`, which
                materialises the rows).
            from_graph: If True (default), return full MatchStamps
                (:meth:`get_matchstamp_at` at each claim's start anchor, all
                connected timelines). If False, return reduced 2-timeline
                stamps.
            conversion_maps: C-map conversions available through unit lookup
                and display. Opt-in: defaults to ``False``.
            believer: The actor whose beliefs filter the claims, those
                listed and those every full stamp reads (the belief filter);
                ``None`` keeps every claim.
            beliefs: The states kept for the believer; default all but
                ``"false"``.

        Returns:
            List of MatchStamp objects. Non-synchronous claims yield None
            entries (filtered out).

        Examples:
            >>> from timetoalign.alignment import AlignmentBundle
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     ContinuousPhysicalTimeline,
            ... )
            >>> bundle = AlignmentBundle()
            >>> _ = bundle.add_timeline(
            ...     ContinuousLogicalTimeline(length=16, uid="score:clt_1"), as_group="score"
            ... )
            >>> for n, seconds in ((1, 40.0), (2, 32.0)):
            ...     _ = bundle.add_timeline(
            ...         ContinuousPhysicalTimeline(length=seconds, uid=f"perf:cpt_{n}"),
            ...         as_group=f"perf_{n}",
            ...     )
            ...     _ = bundle.create_match_claims(
            ...         [
            ...             ({"start": 0}, "score:clt_1", {"start": 0.0}, f"perf:cpt_{n}"),
            ...             ({"start": 16}, "score:clt_1", {"start": seconds}, f"perf:cpt_{n}"),
            ...         ]
            ...     )
            >>> stamps = bundle.get_matchstamps()
            >>> len(stamps)  # one per claim
            4
            >>> stamps[0].n_timelines  # score + 2 performers
            3
        """
        return closure._get_matchstamps(
            self,
            claims,
            from_graph=from_graph,
            conversion_maps=conversion_maps,
            believer=believer,
            beliefs=beliefs,
        )

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
        within: set[str] | None = None,
        from_graph: bool = False,
        conversion_maps: "ConversionMapsSpec" = False,
        format: TableFormat = "table",
        fields: "ColumnNaming | Callable[[str, dict], str] | list[str] | None" = None,
        units: bool | None = None,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> "pa.Table | pd.DataFrame":
        """Get a table of MatchStamps for alignment queries.

        Analogous to ``get_timestamp_table()`` but for cross-group alignment.
        Fields are timeline IDs holding their coordinate values; what a *row*
        is depends on ``from_graph``:

        - ``from_graph=False`` (default) — one row per alignment anchor: an
          instant claim gives one, an interval claim two (its start anchor,
          then its end anchor). A row fills exactly two cells and leaves the
          rest null, so a dense pairwise alignment produces sparse rows.
        - ``from_graph=True`` — one row per connected component of the
          ``(timeline_id, coordinate)`` graph the anchors induce, end anchors
          included.  The pairwise rows above collapse into the
          *cross-section* they describe: one row per aligned instant, every
          participating timeline filled.

        When ``claims`` is None the claims are the bundle's
        :attr:`match_claim_table`, read column by column — no ``MatchClaim``
        is materialised. Every coordinate is written as the claim holds it,
        on its timeline's axis (a claim is checked against its timelines
        when it is added), so no cell is converted here. The belief filter
        (``believer``, ``beliefs``) selects the claims either way.

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
            within: Only these timelines' coordinates in the table, one
                column each; ``None`` keeps every timeline.
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
            believer: The actor whose beliefs filter the claims (the belief
                filter); ``None`` keeps every claim.
            beliefs: The states kept for the believer; default all but
                ``"false"``.

        Returns:
            One field per timeline, each a coordinate struct carrying its
            number twice, in the order the bundle registered the timelines
            (:meth:`_known_timelines`; a timeline it does not know after
            them, by ID), each followed by its derived C-Map fields.
            Non-synchronous claims are excluded.  Empty input yields an
            empty table.

        Raises:
            ValueError: If both ``claims`` and ``at`` are given, on an unknown
                ``format``, when a DataFrame-shaping option is supplied for
                an Arrow result, or when an ``at`` position resolves to
                ambiguous interval-derived coordinates — several candidates
                for one timeline, as legato note ends and spread chords
                produce where a match file gives one interval claim per note.
            NotImplementedError: If ``at`` holds an ``Interval``; there is no
                interval table form.

        Note:
            Collapsed rows are ordered by the timeline the bundle registered
            first among those present in the component, then by the
            coordinate on it — a total order, since two components can never
            share a ``(timeline_id, coordinate)`` node: the rows on the
            first-registered timeline come first, in the order of its axis,
            then those not on it, by the next timeline.  A component that
            somehow carries two coordinates for one timeline (which a
            well-formed alignment never does) keeps the smaller one.

        Examples:
            >>> from timetoalign.alignment import AlignmentBundle
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     ContinuousPhysicalTimeline,
            ... )
            >>> bundle = AlignmentBundle()
            >>> _ = bundle.add_timeline(
            ...     ContinuousLogicalTimeline(length=16, uid="score:clt_1"), as_group="score"
            ... )
            >>> for n, seconds in ((1, 40.0), (2, 32.0)):
            ...     _ = bundle.add_timeline(
            ...         ContinuousPhysicalTimeline(length=seconds, uid=f"perf:cpt_{n}"),
            ...         as_group=f"perf_{n}",
            ...     )
            ...     _ = bundle.create_match_claims(
            ...         [
            ...             ({"start": 0}, "score:clt_1", {"start": 0.0}, f"perf:cpt_{n}"),
            ...             ({"start": 16}, "score:clt_1", {"start": seconds}, f"perf:cpt_{n}"),
            ...         ]
            ...     )
            >>> table = bundle.get_matchstamp_table()
            >>> table.num_rows  # one per anchor
            4
            >>> table.column_names
            ['score:clt_1', 'perf:cpt_1', 'perf:cpt_2']
            >>> bundle.get_matchstamp_table(from_graph=True).num_rows  # one per instant
            2
            >>> bundle.get_matchstamp_table([0, 8], "score:clt_1").num_rows
            2
        """
        return stamp_table._get_matchstamp_table(
            self,
            at,
            timeline_id,
            claims=claims,
            within=within,
            from_graph=from_graph,
            conversion_maps=conversion_maps,
            format=format,
            fields=fields,
            units=units,
            believer=believer,
            beliefs=beliefs,
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
        rows: list[dict[str, Coordinate]],
        all_tl_ids: set[str],
        within: set[str] | None,
        conversion_maps: "ConversionMapsSpec" = False,
    ) -> "pa.Table":
        """Scatter per-row dicts into one PyArrow table.

        Shared assembly tail for :meth:`get_matchstamp_table`.  Each ``rows``
        entry contributes one row, its timeline keys scattered into the
        matching fields, each cell the coordinate as given, on its
        timeline's axis. ``all_tl_ids`` is the field universe before
        filtering; ``within`` (when given) intersects it.

        Args:
            rows: One dict per row, timeline id to coordinate.
            all_tl_ids: Union of every timeline ID appearing in the inputs.
            within: Only these timelines' columns, if given.
            conversion_maps: C-map conversions to add as derived columns, one
                per (timeline, enabled unit-conversion map). Opt-in: defaults
                to ``False``. Only numeric unit-conversion maps (a
                ``target_unit`` set) become columns.

        Returns:
            PyArrow Table with one field per (filtered) timeline ID, in
            registration order (:meth:`_known_timelines`; an unknown
            timeline after them, by ID), each followed by its derived
            conversion columns.  Empty input yields an empty table.
        """
        return stamp_table._assemble_matchstamp_table(
            self, rows, all_tl_ids, within, conversion_maps
        )

    @staticmethod
    def _collapsed_claim_rows(
        anchors: list[tuple[str, Coordinate, str, Coordinate]],
        registered: list[str],
    ) -> list[dict[str, Coordinate]]:
        """Collapse alignment anchors into one row per aligned instant.

        Each anchor is an edge between the nodes ``(timeline_a_id,
        coordinate_a)`` and ``(timeline_b_id, coordinate_b)``. A union-find
        over those edges recovers the connected components, and each
        component becomes one row mapping every participating timeline to
        its coordinate. Nodes are keyed by their exact values: every
        coordinate of a timeline is on its axis, so two anchors meet at a
        node exactly when they name the same position.

        Args:
            anchors: ``(timeline_a_id, coordinate_a, timeline_b_id,
                coordinate_b)`` per anchor, end anchors included.
            registered: The bundle's timeline ids in the order the bundle
                knows them, registered ones first.

        Returns:
            One row per component, ordered as documented on
            :meth:`get_matchstamp_table`.
        """
        return stamp_table._collapsed_claim_rows(anchors, registered)

    def _invalidate_warp_cache(self) -> None:
        """Clear every cache built from the claims, forcing rebuild on next access."""
        self._warp_map_cache.clear()
        self._matchline_cache.clear()
        self._matchgraph_cache.clear()
        self._in_force_cache.clear()
        self._naming_cache.clear()

    def _set_statement_state(
        self, state: Mapping[str, Any], prefixes: Mapping[str, str]
    ) -> None:
        """Store the statements and prefixes; new prefixes clear the claim caches.

        The belief filter compares actors under the bundle's ``@context``,
        whose prefixes these are, so a filtered table cached under the old
        ones would answer for a context the bundle no longer has.
        """
        changed = dict(prefixes) != self._prefixes
        super()._set_statement_state(state, prefixes)
        if changed:
            self._invalidate_warp_cache()

    def _get_or_build_match_line(
        self, source_tl_id: str, belief_key: _BeliefKey = _ALL_BELIEFS
    ) -> MatchLine | None:
        """Get or lazily build a MatchLine for the given source timeline.

        The MatchLine is built from the claims in force naming the source
        (:meth:`_claims_naming`, a vectorized cut of the table, so a dense
        pairwise table is cut to its source slice before anything is
        materialised), cached per source and belief filter and shared across
        all target lookups from the same source. A source no claim names
        has no line. What lies further than one claim from the source is
        reached by the matchstamp closure, unit by unit.

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
            belief_key: The belief filter the claims are read under.

        Returns:
            MatchLine for the source, or None where no claim names it.
        """
        return closure._get_or_build_match_line(self, source_tl_id, belief_key)

    def _get_or_build_warp_map(
        self,
        source_tl_id: str,
        target_tl_id: str,
        belief_key: _BeliefKey = _ALL_BELIEFS,
    ) -> WarpMap | None:
        """Get or lazily build a WarpMap for the given timeline pair.

        Builds the source's ``MatchLine`` (:meth:`_get_or_build_match_line`)
        and constructs a ``WarpMap`` via ``WarpMap.from_match_line()``. Both
        negative results (unreachable pairs) and positive results are
        cached per belief filter; the cache is invalidated whenever the
        table changes.

        Args:
            source_tl_id: ID of the source timeline.
            target_tl_id: ID of the target timeline.
            belief_key: The belief filter the claims are read under.

        Returns:
            WarpMap for the pair, or None if insufficient data.
        """
        return closure._get_or_build_warp_map(
            self, source_tl_id, target_tl_id, belief_key
        )

    # endregion

    # region Grouped Timestamp Helpers

    _get_group_timestamp = closure._get_group_timestamp

    def _manifest_relatives(
        self, timeline: "Timeline", coordinate: CoordinateValue
    ) -> dict[str, CoordinateValue]:
        """Carry a position onto every other timeline of its manifest tree.

        A child sits at an exact offset inside its parent, so a position on
        any timeline of the tree is a position on each of the others. It is
        carried from the queried timeline along the tree's edges, one step
        per edge: up to a parent by adding the child's offset, down to a
        child by subtracting it (:func:`~timetoalign.core.time.shift_number`,
        the parent-child mechanism of :class:`Timeline` itself). Going
        through the root instead (adding the queried timeline's origin on
        the root, subtracting the other's) is two roundings on a float axis
        where one step is none: a child at offset 0 got another value than
        its parent. Step by step, a zero offset is the identity, and an
        exact axis stays exact.

        Which timelines appear follows the timeline's own parent-to-child
        rule (``Timeline._holds_position``, which its stamps and tables read
        too). The walk starts only from a position on the queried timeline,
        closed ``[0, length]``: its own end is its own whatever its siblings
        do, since the position was asked of it. From there the position is
        carried up to every ancestor, since an ancestor contains its
        descendant's whole span, end included. A position outside the queried
        timeline lies on no ancestor and reaches no relative: none is
        returned, and the stamp holds the queried timeline with whatever the
        other mechanisms reach, as a timeline's own stamp and a group's
        answer outside their span. From the queried timeline and from each
        ancestor it is carried down only into the children that hold it, ``[0, length)`` and their end where
        no sibling starts there, and so on at every depth: a timeline
        appears only if every timeline on the way down holds the position,
        exactly as in the root's own stamp, so a boundary shared by two
        siblings reaches the one starting there and its descendants, never
        a descendant of the other. A length that is not known bounds below
        only. Every timeline of the tree takes part, whatever its depth and
        whether or not it is registered or grouped
        (:meth:`_known_timelines`): a middle ancestor between a registered
        root and a grouped descendant is reached like either of them.

        Args:
            timeline: A timeline of one of the bundle's trees.
            coordinate: A position on it.

        Returns:
            The coordinates of the other timelines of the tree that reach the
            position, by id, the root first and then depth-first; empty for a
            position outside the queried timeline.
        """
        return closure._manifest_relatives(self, timeline, coordinate)

    def _get_unit_map(self) -> dict[str, str]:
        """Build a mapping from timeline IDs to their coordinate unit strings.

        Returns:
            Dict mapping timeline IDs to unit strings (e.g. "samples", "quarters").
        """
        unit_map: dict[str, str] = {}
        for timeline_id, tl in self._known_timelines().items():
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
        tl = self._known_timeline(timeline_id)
        if tl is None:
            return []
        return list(tl._conversion_maps.values())

    def _timeline_length(self, timeline_id: str) -> CoordinateValue | None:
        """Return a timeline's exact length in its native unit, or None."""
        tl = self._known_timeline(timeline_id)
        if tl is None or tl.length is None:
            return None
        return tl.length.value

    def _clip_to_length(
        self, timeline_id: str, coordinate: CoordinateValue
    ) -> CoordinateValue:
        """Clip a coordinate into ``[0, length]`` for a target timeline.

        A clipped coordinate is the timeline's own exact bound, never a float
        rendering of it. A timeline of unknown length is bounded below only.
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
        return closure._transfer_coordinate(
            self, warp, entering, target_tl_id, support_policy
        )

    _transfer_to_group = closure._transfer_to_group

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
        membership = self.timeline_to_groups
        for tl_id in sorted(self.timelines.keys()):
            tl = self.timelines[tl_id]
            length = tl.length
            timeline_info[tl_id] = {
                "name": tl.name,
                "length": float(length.value) if length is not None else length,
                "unit": str(tl.unit) if hasattr(tl, "unit") else None,
                "groups": list(membership.get(tl_id, ())),
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

    def _read_tables(self, directory: Path, data: Mapping[str, Any]) -> None:
        """Read the package tables the manifest names, lying in *directory*.

        The Projections table first: each Projection is added to the
        timeline its row names (:meth:`Timeline.add_projections
        <timetoalign.timelines.Timeline.add_projections>`, which keeps the
        IDs and reuses a Projection the timeline already holds, as one read
        from the manifest's event rows), after TimeToShare's check of the
        table against the manifest. Then the Box table
        (:func:`~timetoalign.timelines.manifest.read_event_table`): each row
        read as a :class:`~timetoalign.core.Box` on the axis it states and
        added to its timeline (:meth:`Timeline.add_events
        <timetoalign.timelines.Timeline.add_events>`), so each Box is
        checked against its timeline's axis and image as any Box added is.
        Then the MatchClaim table, added as
        :meth:`add_match_claim_field` adds a field, every ID kept, its
        further columns kept as read (:meth:`MatchClaimField.from_package_table`).
        Every resource of a type is read, in the order :attr:`resources`
        lists them: a manifest may keep the rows of one type in several
        files, each listing its own timelines. A resource naming a single
        file reads the whole file, every row of which is the manifest's; one
        naming a folder (its URL ends in ``/``) reads the folder's files
        through ``timetoshare.tables.read_table``, keeping the rows of the
        timelines it lists (a MatchClaim's either of whose timelines it
        lists), since a folder may hold other manifests' rows too; the rows
        kept are then read as a file's are. A table the manifest names that
        does not lie there is not read: the manifest loads without it, and
        its entry stays in :attr:`resources`. A table is read as it lies,
        whatever it was when its entry's size and digest were stated.

        Raises:
            ValueError: If a resource lies elsewhere than on disk relative
                to the manifest, or a table breaks a rule (a
                ``timetoshare.tables.TableError`` for the Projections table
                and for a folder whose files state another type or cannot be
                read as one table; as :meth:`Timeline.add_events
                <timetoalign.timelines.Timeline.add_events>` refuses a Box;
                as :meth:`MatchClaimField.from_package_table` and
                :meth:`add_match_claim_field` for the MatchClaim table).
            KeyError: If a row of a file names a timeline the bundle does
                not hold.
        """
        return package._read_tables(self, directory, data)

    def _event_tables(self) -> dict[str, pa.Table]:
        """The package's event tables by resource type, each holding a row.

        The Projections table
        (:func:`~timetoalign.timelines.manifest.projection_table`) and the
        Box table (:func:`~timetoalign.timelines.manifest.event_table`) of
        every timeline of the bundle's trees, sorted by ID; a table without
        a row is left out.

        Raises:
            ValueError: As :func:`~timetoalign.timelines.manifest.projection_table`
                refuses a Projection ID held by two timelines.
        """
        known = self._known_timelines()
        timelines = [known[tid] for tid in sorted(known)]
        tables = {
            "Projection": projection_table(timelines),
            "Box": event_table(timelines, Box),
        }
        return {kind: table for kind, table in tables.items() if table.num_rows}

    def to_file(
        self,
        path: str | PathLike[str],
        *,
        minimal: bool | None = None,
        styling: bool | None = None,
        serialization: Serialization | None = None,
        events: bool | None = None,
        external_references: bool | None = None,
    ) -> Path:
        """Write the package: the manifest and, beside it, the bundle's tables.

        As :meth:`ManifestFileMixin.to_file
        <timetoalign.timelines.manifest.ManifestFileMixin.to_file>` writes
        a manifest, with the MatchClaim table and the event tables, of
        Projections and of Boxes (:meth:`_write_tables`), written between a
        first writing of the manifest, which validates and serializes it
        before any file is written, and the writing that names them. A
        manifest never names an event's ID: where the bundle writes a table,
        its Projections and Boxes lie in their tables alone and the events
        of its timelines are not written into the manifest, so writing event
        rows (``events``, given or as read) is refused before any file is
        written.

        Raises:
            ValueError: If event rows would be written beside a table, or as
                :meth:`ManifestFileMixin.to_file
                <timetoalign.timelines.manifest.ManifestFileMixin.to_file>`
                and :meth:`_write_tables` refuse.
            NotImplementedError: As :meth:`_write_tables` refuses a table
                resource naming a folder, before any file is written.
        """
        writes_events = (
            events
            if events is not None
            else (self._file_form is not None and self._file_form.events)
        )
        if writes_events and (len(self.match_claim_table) or self._event_tables()):
            raise ValueError(
                "A package's manifest names no event ID: the bundle writes its "
                "MatchClaim table and its event tables beside the manifest, so "
                "its event rows are not written into it; write it with "
                "events=False."
            )
        return super().to_file(
            path,
            minimal=minimal,
            styling=styling,
            serialization=serialization,
            events=events,
            external_references=external_references,
        )

    def _write_tables(self, directory: Path) -> bool:
        """Write the package tables into *directory* and name them in :attr:`resources`.

        A bundle holding MatchClaims writes its MatchClaim table
        (:attr:`match_claim_table`, :meth:`MatchClaimField.to_parquet`, so
        that a table already lying there must be succeeded). A bundle one
        of whose timelines holds Projections, with MatchClaims or without,
        writes the Projections table of every timeline of its trees, sorted
        by timeline ID
        (:func:`~timetoalign.timelines.manifest.projection_table`), and one
        of whose timelines holds Boxes writes their Box table likewise
        (:func:`~timetoalign.timelines.manifest.event_table`). Each
        lies where its entry read with the manifest places it, else beside
        the manifest (``match_claims.parquet``, ``projections.parquet``,
        ``boxes.parquet``);
        its entry keeps its ID (a new entry's is minted,
        :meth:`Resource.of_file <timetoalign.timelines.manifest.Resource.of_file>`) and
        names the timelines it holds data for (sorted, or in the order the
        entry read lists them where it lists the same), the Parquet media
        type, the file's size and its SHA-256 digest. An entry of either
        type whose table is not written stays where its file lies in
        *directory* and goes where it does not. A bundle holding neither
        MatchClaims nor Projections nor Boxes writes no table and keeps its
        :attr:`resources` as they are: a manifest read without its tables
        names them still.

        The bundle writes one table of each type. Where :attr:`resources`
        holds several entries of one type (a manifest keeping that type's
        rows in several files), the table, holding every row of the type,
        lies at the first entry's URL under its ID, and the other entries
        of that type are left out of the manifest written; their files are
        neither read again nor removed. A table resource naming a folder is
        refused: the library writes no table into a folder, whose files
        other manifests of a package may share.

        Returns:
            Whether a table was written.

        Raises:
            NotImplementedError: If an entry of a table type names a
                folder, naming the entry, before any file is written.
            ValueError: As :meth:`MatchClaimField.to_parquet` refuses the
                MatchClaim table, or as
                :func:`~timetoalign.timelines.manifest.projection_table`
                refuses a Projection ID held by two timelines, or if a new
                entry's id is held in the manifest, before any file is
                written.
        """
        return package._write_tables(self, directory)

    def to_dict(
        self,
        *,
        minimal: bool = True,
        events: bool = False,
        external_references: bool = False,
    ) -> dict[str, Any]:
        """Write the bundle as a TimeToShare manifest.

        The manifest carries the bundle's ``id`` only when one was given
        (a minted id is never written), its ``name`` and ``meta``, every
        root timeline of the bundle — every registered timeline that is no
        child, which includes the root of each one that is — as
        :meth:`Timeline.to_dict <timetoalign.timelines.Timeline.to_dict>`
        writes it, sorted by id, and the groups sorted by id, each naming its
        members by id. ``graphical.order`` keeps every group's member order
        under its id and, unchanged, the band order read with the manifest
        of every timeline written, beside the members of :attr:`styling`. A
        root no group names is ungrouped. The bundle's statements and
        provenance are the manifest's, and the manifest carries its inline
        JSON-LD ``@context`` with the prefixes its statements use and those
        its MatchClaim table's provenance uses (its actors: the table
        :meth:`to_file` writes beside the manifest is read under this
        context), each standing for the namespace the bundle declares
        (:attr:`prefixes`), else the catalogue's; a declared prefix neither
        uses is not written. The manifest carries its
        :attr:`resources`. MatchClaims are not written into the manifest:
        :meth:`to_file` writes them into the MatchClaim table beside it. The
        support policy is a setting of the bundle, not part of the manifest.

        Args:
            minimal: If False, write every optional key, its default value
                included (the complete form); by default every key at its
                default value is left out (the minimal form).
            events: Write every timeline's event rows.
            external_references: Write every timeline's external references.

        Returns:
            ``{"@context": {...}, "timetoshare": "0.2.0", "id"?, "name",
            "meta", "timelines": [...], "groups": [...], "measure_maps":
            [...], "statements"?, "provenance"?, "graphical"?: {"order":
            {...}, ...}}``.

        Examples:
            >>> from fractions import Fraction
            >>> from timetoalign.alignment import AlignmentBundle
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     ContinuousPhysicalTimeline,
            ... )
            >>> bundle = AlignmentBundle()
            >>> _ = bundle.add_timeline(
            ...     ContinuousLogicalTimeline(length=Fraction(16), uid="score"),
            ...     as_group="score",
            ...     seed=7,
            ... )
            >>> _ = bundle.add_timeline(
            ...     ContinuousPhysicalTimeline(length=40.0, uid="audio"), grouped_with="score"
            ... )
            >>> data = bundle.to_dict()
            >>> "id" in data, [timeline["id"] for timeline in data["timelines"]]
            (False, ['#audio', '#score'])
            >>> data["groups"]
            [{'id': '#group_1hvif5xc2r1fv', 'name': 'score', 'members': ['#audio', '#score']}]
            >>> data["graphical"]["order"]
            {'#group_1hvif5xc2r1fv': ['#score', '#audio']}
        """
        return package._to_dict(
            self,
            minimal=minimal,
            events=events,
            external_references=external_references,
        )

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
            >>> from timetoalign.alignment import AlignmentBundle
            >>> from timetoalign.timelines.types import (
            ...     ContinuousPhysicalTimeline,
            ...     DiscreteGraphicalTimeline,
            ... )
            >>> bundle = AlignmentBundle(id="thoresen_alignment")
            >>> _ = bundle.add_timeline(
            ...     DiscreteGraphicalTimeline(length=4835, uid="scan"), as_group="image", seed=7
            ... )
            >>> _ = bundle.add_timeline(
            ...     ContinuousPhysicalTimeline(length=60.0, uid="audio"), grouped_with="scan"
            ... )
            >>> print(bundle.diagram(width=60))
            AlignmentBundle[thoresen_alignment]
            <BLANKLINE>
              TimelineGroup[group_1hvif5xc2r1fv] (2 timelines)
              ┌──────────────────────────────────────────────────────────┐
              │ DiscreteGraphicalTimeline[scan]                          │
              │                       0 ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶ 4835 pixels │
              │                                                          │
              │ ContinuousPhysicalTimeline[audio]                        │
              │                       0 ~~~~~~~~~~~~~~~~~~~~ 60 seconds  │
              └──────────────────────────────────────────────────────────┘
            <BLANKLINE>
              MatchClaims: 0
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
