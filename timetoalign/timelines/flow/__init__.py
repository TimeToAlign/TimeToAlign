"""Play flow control, state flows in measures and map positions along them.

The elements of flow control and their registry live in
:mod:`timetoalign.timelines.flowcontrol` and are exported from here and
from :mod:`timetoalign.timelines` as well.
"""

from __future__ import annotations

from ..flowcontrol import (
    Barline,
    Break,
    DestinationReference,
    FlowCondition,
    FlowControlElement,
    FlowControlRegistry,
    Jump,
    JumpDestination,
    JumpTarget,
    PassesFrom,
    SectionBreak,
)
from .engine import AtomicSection, FlowFinding, FlowResult, atomic_sections, play
from .flowmap import (
    ComputedFlows,
    FlowMap,
    PlacedChunk,
    PlacedGap,
    PlayedFlow,
    ResolvedFlowMap,
    StatedFlow,
    VirtualTimeline,
)
from .measures import (
    FlowControlDraft,
    MeasureFinding,
    MeasureRange,
    flow_control_from_measures,
    measure_keys_from_flow_control,
    measure_order_from_successors,
    measure_ranges,
    read_flow_csv,
    repeat_end_findings,
)

__all__ = [
    "AtomicSection",
    "Barline",
    "Break",
    "ComputedFlows",
    "DestinationReference",
    "FlowCondition",
    "FlowControlDraft",
    "FlowControlElement",
    "FlowControlRegistry",
    "FlowFinding",
    "FlowMap",
    "FlowResult",
    "Jump",
    "JumpDestination",
    "JumpTarget",
    "MeasureFinding",
    "MeasureRange",
    "PassesFrom",
    "PlacedChunk",
    "PlacedGap",
    "PlayedFlow",
    "ResolvedFlowMap",
    "SectionBreak",
    "StatedFlow",
    "VirtualTimeline",
    "atomic_sections",
    "flow_control_from_measures",
    "measure_keys_from_flow_control",
    "measure_order_from_successors",
    "measure_ranges",
    "play",
    "read_flow_csv",
    "repeat_end_findings",
]
