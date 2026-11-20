"""Alignment data structures for TimeToAlign!

This module provides the core alignment infrastructure for cross-timeline
synchronization. See the Conceptual Model documentation
(https://timetoalign.github.io/concepts.html).

Public API:
- AlignmentBundle: Primary entry point for alignment workflows
- Agent: The human or software author of a match claim
- AlignmentAnchor: Atomic coordinate pair claim
- MatchClaim: Low-level match between two events
- MatchClaimField: ``SemanticField[MatchClaim]`` columnar store for
  synchronous-instant pairwise claims
- MatchMetadata: Provenance information for matches (agent + certainty)
- MatchGraph: Graph of MatchClaims (networkx integration)
- MatchStamp: Cross-group timestamp at a single coordinate
- MatchIntervalStamp: Combined endpoint resolutions for an interval query
- MatchLine: Ordered sequence of MatchStamps for WarpMap generation
- WarpMap: Bidirectional coordinate warping from alignment data
"""

from __future__ import annotations

from .bundle import AlignmentBundle
from .claims import (
    Agent,
    AlignmentAnchor,
    ClaimType,
    MatchClaim,
    MatchClaimField,
    MatchMetadata,
)
from .filters import ClaimFilter
from .graph import MatchGraph, MatchIntervalStamp, MatchStamp
from .match_format import MatchFileContext, NoteRecord, SnoteRecord
from .matching import (
    MatchResult,
    match_notes_by_attributes,
    prepare_abc_notes_for_matching,
    prepare_eep_notes_for_matching,
)
from .matchline import MatchLine
from .skeleton import TimeSkeleton
from .warpmap import WarpMap

__all__ = [
    # Bundle (Primary Entry Point)
    "AlignmentBundle",
    "TimeSkeleton",
    # Anchors and Claims
    "Agent",
    "AlignmentAnchor",
    "ClaimType",
    "MatchClaim",
    "MatchClaimField",
    "MatchMetadata",
    # Filters
    "ClaimFilter",
    # Graph, Stamps, and Lines
    "MatchGraph",
    "MatchIntervalStamp",
    "MatchStamp",
    "MatchLine",
    "WarpMap",
    # Match Format (export)
    "MatchFileContext",
    "SnoteRecord",
    "NoteRecord",
    # Matching
    "MatchResult",
    "match_notes_by_attributes",
    "prepare_abc_notes_for_matching",
    "prepare_eep_notes_for_matching",
]
