"""Unified filter API for alignment queries.

This module provides the ``ClaimFilter`` dataclass used consistently across
the alignment API to filter MatchClaims, MatchGraph nodes, and MatchLine
construction.

The canonical filter parameters are:

- ``timeline_id``: Exact single ID match.
- ``timeline_ids``: Exact set-of-IDs match.
- ``id_pattern``: Regex pattern matched via ``re.search()`` against timeline IDs.
- ``between``: Tuple of two timeline IDs; matches claims connecting exactly those two.
- ``synchronous_only``: Exclude non-synchronous (NOMATCH) claims.
- ``nomatch_only``: Return only non-synchronous (NOMATCH) claims.
- ``include_domains``: Only timelines in these domains.
- ``include_units``: Only timelines with these units.
- ``believer``, ``belief``: The belief filter: for one actor, every claim the
  actor does not believe ``false`` (``belief="not_false"``, the default) or
  only those the actor believes ``true`` (``"true"``); without a believer
  nothing is filtered. Decided by
  :func:`~timetoalign.core.provenance.belief_mask` alone.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from timetoalign.core.enums import Domain, TimeUnit
from timetoalign.core.provenance import (
    BeliefFilter,
    belief_mask,
    check_belief_filter,
    provenance_array,
)

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.alignment.claims import MatchClaim
    from timetoalign.timelines import Timeline

module_logger = logging.getLogger(__name__)


def believed(
    claims: Sequence[MatchClaim],
    believer: str | None,
    belief: BeliefFilter = "not_false",
    *,
    context: Mapping[str, Any] | None = None,
) -> list[MatchClaim]:
    """Return the claims the belief filter keeps for *believer*, in order.

    The list form of the belief filter: it asks
    :func:`~timetoalign.core.provenance.belief_mask` once over the claims'
    provenance, the one place the filter is decided. A claim without
    provenance holds no belief. Without a believer every claim is kept.

    Args:
        claims: The claims to filter.
        believer: The actor whose beliefs filter; ``None`` keeps every claim.
        belief: ``"not_false"`` keeps every claim the believer does not
            believe ``false``; ``"true"`` only those the believer believes
            ``true``.
        context: The JSON-LD context under which the believer and the
            actors compare as IRIs (the bundle's manifest ``@context``,
            :func:`belief_context`); ``None`` declares no prefix.

    Returns:
        The kept claims, in their order.

    Raises:
        ValueError: If *belief* is neither ``"not_false"`` nor ``"true"``.
    """
    claims = list(claims)
    if believer is None:
        check_belief_filter(belief)
        return claims
    mask = belief_mask(
        provenance_array([claim.provenance for claim in claims]),
        believer,
        belief,
        context=context,
    )
    return [claim for claim, kept in zip(claims, mask.to_pylist(), strict=True) if kept]


def belief_context(bundle: AlignmentBundle | None) -> dict[str, Any] | None:
    """The context under which a bundle's MatchClaims compare actors.

    The ``@context`` of the bundle's manifest (the format's terms, the
    catalogue's prefixes and the bundle's own), as the bundle's statements
    use it, so a believer named by a compact IRI matches a belief recorded
    under the full IRI; ``None`` for no bundle (no prefix declared).

    Args:
        bundle: The bundle holding the claims, or ``None``.
    """
    return None if bundle is None else bundle._statement_context()


@dataclass(frozen=True)
class ClaimFilter:
    """Reusable filter for querying MatchClaims and related objects.

    All fields are optional. When multiple fields are set, they are combined
    with AND logic: a claim must satisfy every non-None criterion.

    The ``matches_claim()`` method tests a single MatchClaim against all
    criteria. The ``matches_timeline()`` method tests a single timeline ID
    (used by MatchGraph node-level filtering).

    Examples:
        Filter for claims involving a specific performer::

            f = ClaimFilter(id_pattern=r"^perf:")
            filtered = [c for c in claims if f.matches_claim(c)]

        Filter for synchronous claims between score and one performer::

            f = ClaimFilter(
                between=("score:clt_1", "perf:dlt_1"),
                synchronous_only=True,
            )

    Attributes:
        timeline_id: Return claims involving this exact timeline ID.
        timeline_ids: Return claims involving any of these timeline IDs.
        id_pattern: Regex pattern matched against timeline IDs via
            ``re.search()``. Example: ``r"^perf:"`` matches all
            performance timelines.
        between: Return claims connecting exactly these two timelines
            (order-independent).
        synchronous_only: If True, exclude non-synchronous (NOMATCH) claims.
        nomatch_only: If True, return only non-synchronous (NOMATCH) claims.
        include_domains: Only include timelines in these domains.
            Requires a ``timelines`` dict for resolution.
        include_units: Only include timelines with these units.
            Requires a ``timelines`` dict for resolution.
        believer: The actor whose beliefs filter claims (the belief
            filter); ``None`` filters nothing.
        belief: ``"not_false"`` keeps every claim the believer does not
            believe ``false``; ``"true"`` only those believed ``true``.
    """

    timeline_id: str | None = None
    timeline_ids: set[str] | None = field(default=None)
    id_pattern: str | None = None
    between: tuple[str, str] | None = None
    synchronous_only: bool = False
    nomatch_only: bool = False
    include_domains: set[Domain] | None = field(default=None)
    include_units: set[TimeUnit] | None = field(default=None)
    believer: str | None = None
    belief: BeliefFilter = "not_false"

    def __post_init__(self) -> None:
        """Validate mutually exclusive options and the belief filter."""
        if self.synchronous_only and self.nomatch_only:
            raise ValueError("synchronous_only and nomatch_only are mutually exclusive")
        check_belief_filter(self.belief)

    @property
    def _compiled_pattern(self) -> re.Pattern[str] | None:
        """Lazily compiled regex from ``id_pattern``."""
        if self.id_pattern is None:
            return None
        return re.compile(self.id_pattern)

    def matches_claim(
        self,
        claim: MatchClaim,
        *,
        timelines: dict[str, Timeline] | None = None,
    ) -> bool:
        """Test whether a MatchClaim passes all filter criteria.

        Args:
            claim: The MatchClaim to test.
            timelines: Dict of timeline_id -> Timeline for resolving
                domain/unit filters. Required only when ``include_domains``
                or ``include_units`` are set.

        Returns:
            True if the claim passes all filters.
        """
        # Synchronous / NOMATCH filters
        if self.synchronous_only and not claim.is_synchronous:
            return False
        if self.nomatch_only and claim.is_synchronous:
            return False

        # timeline_id: claim must involve this timeline
        if self.timeline_id is not None:
            if not claim.connects(self.timeline_id):
                return False

        # timeline_ids: claim must involve at least one
        if self.timeline_ids is not None:
            if not any(claim.connects(tid) for tid in self.timeline_ids):
                return False

        # id_pattern: at least one of the claim's timelines must match
        pat = self._compiled_pattern
        if pat is not None:
            if not (pat.search(claim.timeline_a_id) or pat.search(claim.timeline_b_id)):
                return False

        # between: must connect exactly these two (order-independent)
        if self.between is not None:
            if not claim.connects_both(self.between[0], self.between[1]):
                return False

        # Domain / unit filters: BOTH timelines must pass
        if self.include_domains is not None or self.include_units is not None:
            if not self._timeline_passes_domain_unit(claim.timeline_a_id, timelines):
                return False
            if not self._timeline_passes_domain_unit(claim.timeline_b_id, timelines):
                return False

        # The belief filter, decided where every consumer decides it.
        if self.believer is not None:
            context = belief_context(claim.bundle)
            if not believed([claim], self.believer, self.belief, context=context):
                return False

        return True

    def matches_timeline(
        self,
        timeline_id: str,
        *,
        timelines: dict[str, Timeline] | None = None,
    ) -> bool:
        """Test whether a single timeline ID passes the filter criteria.

        Used for node-level filtering in MatchGraph. Only the timeline-ID
        related filters are applied (``timeline_id``, ``timeline_ids``,
        ``id_pattern``, ``include_domains``, ``include_units``). The
        claim-level filters (``synchronous_only``, ``nomatch_only``,
        ``between``, the belief filter) are ignored.

        Args:
            timeline_id: The timeline ID to test.
            timelines: Dict of timeline_id -> Timeline for resolving
                domain/unit filters.

        Returns:
            True if the timeline passes all applicable filters.
        """
        # timeline_id filter
        if self.timeline_id is not None:
            if timeline_id != self.timeline_id:
                return False

        # timeline_ids filter
        if self.timeline_ids is not None:
            if timeline_id not in self.timeline_ids:
                return False

        # id_pattern filter
        pat = self._compiled_pattern
        if pat is not None:
            if not pat.search(timeline_id):
                return False

        # Domain / unit filters
        if not self._timeline_passes_domain_unit(timeline_id, timelines):
            return False

        return True

    def domain_unit_timeline_ids(
        self,
        timelines: dict[str, Timeline] | None,
    ) -> set[str] | None:
        """Resolve the domain/unit criteria into the set of ids that pass.

        ``include_domains`` / ``include_units`` are the only criteria that
        need per-timeline metadata. Resolving them once into a plain set of
        timeline ids lets a columnar store — which holds ids but no timeline
        metadata — apply the same restriction vectorized.

        The set carries the same semantics ``matches_claim`` applies: a claim
        passes only when **both** of its timelines are in it.

        Args:
            timelines: Dict of timeline_id -> Timeline for resolution.

        Returns:
            ``None`` when neither criterion is set (no restriction), else the
            set of timeline ids satisfying both criteria (empty when
            ``timelines`` is None, mirroring ``matches_claim``).
        """
        if self.include_domains is None and self.include_units is None:
            return None
        if timelines is None:
            return set()
        return {
            timeline_id
            for timeline_id in timelines
            if self._timeline_passes_domain_unit(timeline_id, timelines)
        }

    def _timeline_passes_domain_unit(
        self,
        timeline_id: str,
        timelines: dict[str, Timeline] | None,
    ) -> bool:
        """Check whether a timeline passes domain/unit filters.

        Domain and unit criteria require metadata for the timeline.

        Args:
            timeline_id: The timeline ID to check.
            timelines: Optional timeline lookup dict.

        Returns:
            True if the timeline passes domain/unit filters.
        """
        if self.include_domains is None and self.include_units is None:
            return True

        if timelines is None:
            return False

        tl = timelines.get(timeline_id)
        if tl is None:
            return False

        tl_unit = getattr(tl, "unit", None)
        if self.include_domains is not None:
            tl_domain = getattr(tl_unit, "domain", None)
            if tl_domain is None or tl_domain not in self.include_domains:
                return False

        if self.include_units is not None:
            if tl_unit is None or tl_unit not in self.include_units:
                return False

        return True

    @classmethod
    def from_kwargs(
        cls,
        *,
        timeline_id: str | None = None,
        timeline_ids: set[str] | None = None,
        id_pattern: str | None = None,
        between: tuple[str, str] | None = None,
        synchronous_only: bool = False,
        nomatch_only: bool = False,
        include_domains: set[Domain] | None = None,
        include_units: set[TimeUnit] | None = None,
        believer: str | None = None,
        belief: BeliefFilter = "not_false",
    ) -> ClaimFilter:
        """Create a ClaimFilter from keyword arguments.

        Convenience constructor mirroring the canonical filter parameter
        signature used across the API.

        Args:
            timeline_id: Return claims involving this timeline.
            timeline_ids: Return claims involving any of these timelines.
            id_pattern: Regex pattern matched against timeline IDs.
            between: Return claims connecting exactly these two timelines.
            synchronous_only: Exclude non-synchronous claims.
            nomatch_only: Return only non-synchronous claims.
            include_domains: Only timelines in these domains.
            include_units: Only timelines with these units.
            believer: The actor whose beliefs filter claims.
            belief: ``"not_false"`` or ``"true"`` (the belief filter).

        Returns:
            A new ClaimFilter.
        """
        return cls(
            timeline_id=timeline_id,
            timeline_ids=timeline_ids,
            id_pattern=id_pattern,
            between=between,
            synchronous_only=synchronous_only,
            nomatch_only=nomatch_only,
            include_domains=include_domains,
            include_units=include_units,
            believer=believer,
            belief=belief,
        )

    def __repr__(self) -> str:
        parts = []
        if self.timeline_id is not None:
            parts.append(f"timeline_id={self.timeline_id!r}")
        if self.timeline_ids is not None:
            parts.append(f"timeline_ids={self.timeline_ids!r}")
        if self.id_pattern is not None:
            parts.append(f"id_pattern={self.id_pattern!r}")
        if self.between is not None:
            parts.append(f"between={self.between!r}")
        if self.synchronous_only:
            parts.append("synchronous_only=True")
        if self.nomatch_only:
            parts.append("nomatch_only=True")
        if self.include_domains is not None:
            parts.append(f"include_domains={self.include_domains!r}")
        if self.include_units is not None:
            parts.append(f"include_units={self.include_units!r}")
        if self.believer is not None:
            parts.append(f"believer={self.believer!r}")
        if self.belief != "not_false":
            parts.append(f"belief={self.belief!r}")
        return f"ClaimFilter({', '.join(parts)})"
