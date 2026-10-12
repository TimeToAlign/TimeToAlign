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

Structural components:
- Region: Named TimeInterval (not a timeline itself)
- MeasureMap: Immutable printed-order measure structure
- SectionHierarchy: Section partition over one MeasureMap
- MetricNode: One timespan in a packed metrical forest
- Expansion: An attributed partition of a metric node
- Reading: Provenance and vocabulary for one forest reading
- TempoEntry: One symbolic-axis tempo or observation fact
- TempoMap: An independent sequence of tempo facts
- MetricHierarchy: Packed metrical forest with independent tempo maps

Flow API (MeasureUnit architecture):
- MeasureUnit: Fundamental building block (one per MeasureData row)
- Typed MeasureUnit subclasses (Typing step):
  - IncompleteMeasure: Measure shorter than expected (anacrusis, final, split)
  - CompleteMeasure: Measure matching expected duration
  - OverlengthMeasure: Measure exceeding expected duration (fermata, cadenza)
  - TypedMeasure: Type alias for any typed measure
- MeasureGroup hierarchy (Grouping step):
  - MeasureGroup: Base class for groupings of typed measures
  - SplitMeasure: IncompleteMeasures that together form a complete unit
  - IncompleteGroup: Isolated IncompleteMeasures (will merge in playthrough)
  - VoltaGroup: Measures under same volta bracket
  - CompleteMeasureGroup: Adjacent CompleteMeasures
  - OverlengthGroup: OverlengthMeasures
- AtomicSection: Smallest indivisible traversal unit (with typed_measures)
- FlowDiagnostic: A structural-invariant violation found in an atomic flow graph
- PlaythroughSection: Contiguous group of atomic sections (with typed_measures)
- Flow: A computed flow (sequence of measure visitations)
- FlowMap: Attached to timelines for coordinate transformation
- ScoreFlowController: Compute Flow paths from MeasureData
- load_valid_flows: Load all valid flows from a .flow.csv file

Note:
    MC ranges in AtomicSection and PlaythroughSection use the right-open
    interval convention [mc_start, mc_end), consistent with partitura and
    TimeToAlign's TimeInterval model.

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

from .base import Timeline
from .beatgrid import BeatGrid, BeatGridSegment, GridBeat
from .factory import create_timeline
from .flow import (
    AtomicSection,
    CompleteMeasure,
    CompleteMeasureGroup,
    Flow,
    FlowControllerBase,
    FlowDiagnostic,
    FlowMap,
    FlowMapSection,
    Gap,
    IncompleteGroup,
    IncompleteMeasure,
    MeasureGroup,
    MeasureUnit,
    OverlengthGroup,
    OverlengthMeasure,
    PlaythroughSection,
    ScoreFlowController,
    SegmentNameGenerator,
    SplitMeasure,
    TypedMeasure,
    VoltaGroup,
    create_unfolded_timeline,
    load_valid_flows,
)
from .groups import TimelineGroup
from .regions import Region
from .structure import (
    Expansion,
    MeasureMap,
    MetricHierarchy,
    MetricNode,
    Reading,
    SectionHierarchy,
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
    # Structural components
    "Region",
    "MeasureMap",
    "SectionHierarchy",
    "MetricNode",
    "Expansion",
    "Reading",
    "TempoEntry",
    "TempoMap",
    "MetricHierarchy",
    # Flow API
    "MeasureUnit",
    # Typed MeasureUnit subclasses (Typing step)
    "IncompleteMeasure",
    "CompleteMeasure",
    "OverlengthMeasure",
    "TypedMeasure",
    # MeasureGroup hierarchy (Grouping step)
    "MeasureGroup",
    "SplitMeasure",
    "IncompleteGroup",
    "VoltaGroup",
    "CompleteMeasureGroup",
    "OverlengthGroup",
    # Sections and Flow
    "Flow",
    "FlowMap",
    "FlowMapSection",
    "Gap",
    "FlowControllerBase",  # Abstract base class
    "ScoreFlowController",  # Concrete implementation for score data
    "SegmentNameGenerator",  # Customizable atomic-section labelling
    "AtomicSection",
    "FlowDiagnostic",
    "PlaythroughSection",
    "load_valid_flows",
    "create_unfolded_timeline",  # Create unfolded timeline from flow
]
