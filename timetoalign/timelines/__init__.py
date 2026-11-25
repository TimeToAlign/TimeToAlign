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
- FlowMap, FlowMapSection: The played spans of a timeline, mapping its
  coordinates to those of its unfolding and back
- Gap: A hole between two played spans

Flow control (the elements, their registry, the engine that plays them and
their reading from measures) lives in ``timetoalign.timelines.flowcontrol``
and ``timetoalign.timelines.flow``.

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

from .base import Timeline, UnknownLengthError
from .beatgrid import BeatGrid, BeatGridSegment, GridBeat
from .factory import create_timeline
from .flow import FlowMap, FlowMapSection, Gap
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
    "FlowMapSection",
    "Gap",
]
