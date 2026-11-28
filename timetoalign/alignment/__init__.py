"""Alignment data structures for TimeToAlign!

This module provides the core alignment infrastructure for cross-timeline
synchronization. See the Conceptual Model documentation
(https://timetoalign.github.io/concepts.html).

Public API:
- AlignmentBundle: Primary entry point for alignment workflows
- Package: A folder of manifests sharing their tables (package.yaml)
- resolve: Everything an ID names, through a manifest or a package root
- AlignmentAnchor: Atomic coordinate pair claim
- MatchClaim: Low-level match between two events
- MatchClaimField: The MatchClaim table, ``SemanticField[MatchClaim]``
- Provenance, Belief: How a MatchClaim was made, and who believes it
- MatchGraph: Graph of MatchClaims (networkx integration)
- InferredEdge: An anchor a MatchGraph infers through group membership
- MatchStamp: Cross-group timestamp at a single coordinate
- MatchIntervalStamp: Combined endpoint resolutions for an interval query
- MatchLine: Ordered sequence of MatchStamps for WarpMap generation
- WarpMap: Bidirectional coordinate warping from alignment data
"""

from __future__ import annotations

from timetoalign.core import Belief, Provenance

from .bundle import AlignmentBundle, resolve
from .claims import AlignmentAnchor, ClaimType, MatchClaim, MatchClaimField
from .filters import ClaimFilter
from .graph import InferredEdge, MatchGraph, MatchIntervalStamp, MatchStamp
from .match_format import MatchFileContext, NoteRecord, SnoteRecord
from .matching import (
    MatchResult,
    match_notes_by_attributes,
    prepare_abc_notes_for_matching,
    prepare_eep_notes_for_matching,
)
from .matchline import MatchLine
from .package import Package
from .warpmap import WarpMap

__all__ = [
    # Bundle (Primary Entry Point)
    "AlignmentBundle",
    "Package",
    "resolve",
    # Anchors and Claims
    "AlignmentAnchor",
    "Belief",
    "ClaimType",
    "MatchClaim",
    "MatchClaimField",
    "Provenance",
    # Filters
    "ClaimFilter",
    # Graph, Stamps, and Lines
    "InferredEdge",
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
