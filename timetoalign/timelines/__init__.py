"""Timeline classes for TimeToAlign!

This module provides the 6 timeline types (3 domains x 2 modalities):

Domains:
- Logical: Symbolic/musical time (beats, quarters, ticks)
- Physical: Acoustic/real time (seconds, samples)
- Graphical: Visual/spatial (pixels, coordinates)

Modalities:
- Continuous: Float or Fraction coordinates
- Discrete: Integer coordinates

Specialized timelines:
- SegmentLine: Timeline with contiguous child segments

Beat grids:
- BeatGrid: Tempo segments generating a labelled lattice of beats
- BeatGridSegment: One tempo segment of a beat grid
- GridBeat: One labelled beat of a grid

Statements (what a bundle, a timeline or a region stands for):
- StatementsMixin: Property–value pairs and their provenance, read side
- MutableStatementsMixin: The same, changed in place (timeline, bundle)
- LIBRARY_AGENT: The agent this library's importers name by default

Package resources (the files a manifest names beside it):
- Resource: One entry of a manifest's ``resources``, a file and its timelines
- Integrity: The digest a resource states of its file's bytes

Structural components:
- Region: Named TimeInterval (not a timeline itself)
- MeasureMap: Immutable printed-order measure structure
- MetricNode: One timespan in a packed metrical forest
- Expansion: An attributed partition of a metric node
- Reading: Provenance and vocabulary for one forest reading
- TempoEntry: One symbolic-axis tempo or observation fact
- TempoMap: An independent sequence of tempo facts
- MetricHierarchy: Packed metrical forest with independent tempo maps

Flow:
- FlowMap: One order in which stretches of a root timeline are played
- ResolvedFlowMap: A FlowMap as placed stretches; converts positions both ways
- PlacedChunk, PlacedGap: One stretch or one gap of a resolved FlowMap
- VirtualTimeline: The statement that a map was applied to a timeline
- ComputedFlows: The atomic sections and FlowMaps written from flow control
- StatedFlow: The Regions and the FlowMap written from an order a source states
- Gap: A stretch of a FlowMap in which nothing of the timeline is played

Flow control:
- FlowControlRegistry: The flow-control elements of one timeline
- FlowControlView: Flow-control elements read on one axis, such as a tree's
- FlowControlElement: What the five kinds of element share
- Jump, JumpTarget, Break, SectionBreak, Barline: The five kinds of element
- JumpDestination: One place a Jump may go on at, and when it does
- FlowCondition, DestinationReference, PassesFrom: The conditions of a
  destination and of a Break
- play, atomic_sections: The engine that plays flow control over a length
- FlowResult, FlowFinding, AtomicSection: What the engine returns
- PlayedFlow: The flow a timeline's flow control plays, with its findings

The reading of flow control from measures lives in
``timetoalign.timelines.flow``.

Public API:
- Timeline: Base class
- ContinuousLogicalTimeline, DiscreteLogicalTimeline
- ContinuousPhysicalTimeline, DiscretePhysicalTimeline
- ContinuousGraphicalTimeline, DiscreteGraphicalTimeline
- BeatGrid: Tempo segments generating a labelled lattice of beats
- SegmentLine: Timeline where all children are contiguous
- Region: Named TimeInterval for partitioning
- create_timeline: Factory function for creating timelines from bundles/stores
- get_timeline_class: Factory for getting timeline class by domain/modality
"""

from __future__ import annotations

from timetoalign.core import Gap

from .base import Timeline, UnknownLengthError
from .beatgrid import BeatGrid, BeatGridSegment, GridBeat
from .factory import create_timeline
from .flow import (
    AtomicSection,
    Barline,
    Break,
    ComputedFlows,
    DestinationReference,
    FlowCondition,
    FlowControlElement,
    FlowControlRegistry,
    FlowControlView,
    FlowFinding,
    FlowMap,
    FlowResult,
    Jump,
    JumpDestination,
    JumpTarget,
    PassesFrom,
    PlacedChunk,
    PlacedGap,
    PlayedFlow,
    ResolvedFlowMap,
    SectionBreak,
    StatedFlow,
    VirtualTimeline,
    atomic_sections,
    play,
)
from .groups import TimelineGroup
from .manifest import Integrity, Resource
from .regions import Region
from .statements import LIBRARY_AGENT, MutableStatementsMixin, StatementsMixin
from .structure import (
    Expansion,
    MeasureMap,
    MetricHierarchy,
    MetricNode,
    Reading,
    TempoEntry,
    TempoMap,
)
from .types import (
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    DiscretePhysicalTimeline,
    GraphicalTimeline,
    LogicalTimeline,
    PhysicalTimeline,
    SegmentLine,
    get_timeline_class,
)

__all__ = [
    # Base
    "Timeline",
    "TimelineGroup",
    "UnknownLengthError",
    # Factory
    "create_timeline",
    "get_timeline_class",
    # Domain base classes
    "LogicalTimeline",
    "PhysicalTimeline",
    "GraphicalTimeline",
    # Concrete types (6)
    "ContinuousLogicalTimeline",
    "DiscreteLogicalTimeline",
    "ContinuousPhysicalTimeline",
    "DiscretePhysicalTimeline",
    "ContinuousGraphicalTimeline",
    "DiscreteGraphicalTimeline",
    # Specialized timelines
    "SegmentLine",
    # Beat grids
    "BeatGrid",
    "BeatGridSegment",
    "GridBeat",
    # Statements
    "LIBRARY_AGENT",
    "MutableStatementsMixin",
    "StatementsMixin",
    # Package resources
    "Integrity",
    "Resource",
    # Structural components
    "Region",
    "MeasureMap",
    "MetricNode",
    "Expansion",
    "Reading",
    "TempoEntry",
    "TempoMap",
    "MetricHierarchy",
    # Flow
    "FlowMap",
    "ResolvedFlowMap",
    "PlacedChunk",
    "PlacedGap",
    "ComputedFlows",
    "StatedFlow",
    "VirtualTimeline",
    "Gap",
    # Flow control
    "FlowControlRegistry",
    "FlowControlView",
    "FlowControlElement",
    "Jump",
    "JumpTarget",
    "Break",
    "SectionBreak",
    "Barline",
    "JumpDestination",
    "FlowCondition",
    "DestinationReference",
    "PassesFrom",
    "play",
    "atomic_sections",
    "FlowResult",
    "FlowFinding",
    "AtomicSection",
    "PlayedFlow",
]
