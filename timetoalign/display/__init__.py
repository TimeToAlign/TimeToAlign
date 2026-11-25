"""Display module for TimeToAlign! ASCII visualization.

This module provides pure ASCII/Unicode terminal display for timelines,
timeline groups and alignment bundles.
Zero external dependencies.

Public API:
    - timeline_diagram(): Generate ASCII diagram for a Timeline
    - group_diagram(): Generate ASCII diagram for a TimelineGroup
    - bundle_diagram(): Generate ASCII diagram for an AlignmentBundle
    - TIMELINE_CHARS: Character mapping for the 6 timeline types
    - REGION_CHARS: Character mapping for region display
"""

from __future__ import annotations

from timetoalign.display.ascii import (
    REGION_CHARS,
    REGION_CHARS_ASCII,
    TIMELINE_CHARS,
    TIMELINE_CHARS_ASCII,
    Diagram,
    bundle_diagram,
    group_diagram,
    timeline_diagram,
)
from timetoalign.display.html import affordance_html, affordance_line, code

__all__ = [
    "Diagram",
    "timeline_diagram",
    "group_diagram",
    "bundle_diagram",
    "TIMELINE_CHARS",
    "TIMELINE_CHARS_ASCII",
    "REGION_CHARS",
    "REGION_CHARS_ASCII",
    "affordance_html",
    "affordance_line",
    "code",
]
