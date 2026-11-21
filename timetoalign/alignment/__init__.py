"""Alignment data structures for TimeToAlign!

This module provides the core alignment infrastructure for cross-timeline
synchronization. See the Conceptual Model documentation
(https://timetoalign.github.io/concepts.html).

Public API:
- AlignmentBundle: Primary entry point for alignment workflows
- AlignmentAnchor: Atomic coordinate pair claim
- MatchClaim: Low-level match between two events
- MatchClaimField: The MatchClaim table, ``SemanticField[MatchClaim]``
- Provenance, Belief: How a MatchClaim was made, and who believes it
- MatchGraph: Graph of MatchClaims (networkx integration)
- MatchStamp: Cross-group timestamp at a single coordinate
- MatchIntervalStamp: Combined endpoint resolutions for an interval query
- MatchLine: Ordered sequence of MatchStamps for WarpMap generation
- WarpMap: Bidirectional coordinate warping from alignment data
"""

from __future__ import annotations

from importlib import import_module

from timetoalign.core.provenance import Belief, Provenance

from .bundle import AlignmentBundle
from .claims import AlignmentAnchor, ClaimType, MatchClaim, MatchClaimField
from .filters import ClaimFilter
from .graph import MatchGraph, MatchIntervalStamp, MatchStamp
from .match_format import MatchFileContext, NoteRecord, SnoteRecord
from .matchline import MatchLine
from .skeleton import TimeSkeleton
from .warpmap import WarpMap

__all__ = [
    # Bundle (Primary Entry Point)
    "AlignmentBundle",
    "TimeSkeleton",
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

#: Exports read from their module on first use: note matching pulls in its
#: own preparation code, which the alignment structures above never need.
_DEFERRED_EXPORTS: dict[str, str] = {
    "MatchResult": ".matching",
    "match_notes_by_attributes": ".matching",
    "prepare_abc_notes_for_matching": ".matching",
    "prepare_eep_notes_for_matching": ".matching",
}


def __getattr__(name: str) -> object:
    """Resolve and cache a deferred export."""
    module_name = _DEFERRED_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return eager and deferred public names."""
    return sorted(set(globals()) | set(__all__))
