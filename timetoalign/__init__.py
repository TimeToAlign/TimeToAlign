"""TimeToAlign! — A library for representing and aligning musical timelines.

This library provides tools for:

- Representing musical timelines across physical, logical, and graphical domains
- Converting between different coordinate systems using ConversionMaps
- Aligning events across timelines using Match objects
- Loading and saving alignment data in Parquet and JSON formats

Basic usage::

    >>> import timetoalign as tta
    >>> from timetoalign.core import Coordinate, TimeUnit, Domain
    >>>
    >>> # Create a coordinate
    >>> c = Coordinate(120, TimeUnit.ticks)
    >>> print(c)
    120 ticks

Installation & Optional Dependencies
=====================================

The core install (``pip install -e .``) is deliberately lightweight:
only PyArrow, pandas, NetworkX, typing_extensions, pydantic and timetoshare
(the TimeToShare manifest format: its forms, files and IDs) are
required.  This
gives you the full timeline/map/alignment framework but **no file-format-
specific loaders**.  Loader backends and other features are available through
optional extras.

Atomic extras — one concern each
--------------------------------

Install with ``pip install -e ".[<extra>]"`` from the repository root.

+---------------+--------------------------------------+----------------------------------------------+
| Extra         | Packages                             | Purpose                                      |
+===============+======================================+==============================================+
| ``midi``      | ``mido``                             | MIDI file loading (PerformanceMidiLoader)    |
+---------------+--------------------------------------+----------------------------------------------+
| ``partitura`` | ``partitura``                        | Score parsing via partitura                  |
|               |                                      | (PartituraLoader, ScoreMidiLoader)           |
+---------------+--------------------------------------+----------------------------------------------+
| ``music21``   | ``music21``                          | Score parsing via music21 (Music21Loader)    |
+---------------+--------------------------------------+----------------------------------------------+
| ``ms3``       | ``ms3``                              | DCML TSV score parsing (Ms3Loader)           |
+---------------+--------------------------------------+----------------------------------------------+
| ``verovio``   | ``verovio``                          | Engraved-score loading via Verovio           |
|               |                                      | (VerovioLoader)                              |
+---------------+--------------------------------------+----------------------------------------------+
| ``audio``     | ``soundfile``, ``mutagen``           | Audio file loading + MP3/M4A metadata        |
+---------------+--------------------------------------+----------------------------------------------+
| ``graphical`` | ``pymupdf``, ``pillow``              | PDF/image loading & drawing                  |
+---------------+--------------------------------------+----------------------------------------------+
| ``plot``      | ``matplotlib``                       | Visualisation                                |
+---------------+--------------------------------------+----------------------------------------------+
| ``delta``     | ``deltalake``                        | Delta Lake columnar storage (future)         |
+---------------+--------------------------------------+----------------------------------------------+
| ``rdf``       | ``rdflib``                           | MeLODy Turtle import (``MelodyLoader``)      |
+---------------+--------------------------------------+----------------------------------------------+

Composite extras — convenience bundles
--------------------------------------

+---------------+-----------------------------------------+----------------------------------------------+
| Extra         | Includes                                | Purpose                                      |
+===============+=========================================+==============================================+
| ``scores``    | ``partitura``, ``music21``, ``ms3``     | All score-loader backends                    |
+---------------+-----------------------------------------+----------------------------------------------+
| ``loaders``   | ``midi``, ``scores``, ``audio``,        | Every loader dependency                      |
|               | ``graphical``                           |                                              |
+---------------+-----------------------------------------+----------------------------------------------+
| ``tutorial``  | ``loaders``, ``plot``, plus             | Everything needed for the tutorial notebooks |
|               | ``jupytext``, ``jupyter``               |                                              |
+---------------+-----------------------------------------+----------------------------------------------+
| ``all``       | ``tutorial``, ``delta``, ``rdf``        | All runtime features                         |
+---------------+-----------------------------------------+----------------------------------------------+
| ``dev``       | ``all``, plus ``pre-commit``,           | All features + development / CI tooling      |
|               | ``pytest``, ``pytest-cov``,             |                                              |
|               | ``pytest-benchmark``, ``hypothesis``,   |                                              |
|               | ``ruff``                                |                                              |
+---------------+-----------------------------------------+----------------------------------------------+

The inclusion chain is::

    dev  ⊃  all  ⊃  tutorial  ⊃  loaders  ⊃  { midi, scores, audio, graphical }
                                            +  plot, jupytext, jupyter
                               +  delta, rdf

Examples (from the repository root)::

    pip install -e .                         # Core only
    pip install -e ".[midi]"                 # Core + MIDI loading
    pip install -e ".[scores]"               # Core + all score-loader backends
    pip install -e ".[loaders]"              # Core + every loader
    pip install -e ".[tutorial]"             # Loaders + plotting + Jupyter
    pip install -e ".[all]"                  # All runtime features
    pip install -e ".[dev]"                  # Everything + dev tooling

For more information, see the documentation site.
"""

from __future__ import annotations

from importlib import import_module

from timetoalign.core import (
    Belief,
    BoundingBox,
    BoundingBoxField,
    Box,
    BoxField,
    ClaimType,
    ColumnNaming,
    ColumnRole,
    Coordinate,
    CoordinateField,
    CoordinateSpec,
    CoordinateValue,
    Domain,
    Event,
    EventField,
    EventReference,
    EventType,
    IdCoordinate,
    Interval,
    IntervalField,
    NumberType,
    OptionalCoordinate,
    Point,
    PointField,
    Projection,
    Provenance,
    Quadrilateral,
    QuadrilateralField,
    ResolvedCoordinate,
    ScopedId,
    Shape,
    ShapeField,
    TimeIntervalStamp,
    TimeStamp,
    TimeUnit,
    resolve_coordinate_spec,
)
from timetoalign.maps import (
    ChainMap,
    ConstantMap,
    ConversionMap,
    FloatingMeasureMap,
    InterpolationMap,
    IntervalToConstantMap,
    LinearMap,
    MeasureBeatMap,
    MeasureNumberMap,
    MeasureRecordMap,
    PiecewiseMap,
    QuartersToFloatingMeasures,
    QuartersToMeasureNumber,
    ScalarMap,
    SecondsToSamples,
    ShiftMap,
    TableMap,
)
from timetoalign.storage import DictStore, EventData, EventStore, SingleStore

_DEFERRED_EXPORTS = {
    # Foundation structure
    "BeatPolicy": ("timetoalign.core", "BeatPolicy"),
    "Duration": ("timetoalign.core", "Duration"),
    "Expansion": ("timetoalign.timelines", "Expansion"),
    "Measure": ("timetoalign.core", "Measure"),
    "MeasureMap": ("timetoalign.timelines", "MeasureMap"),
    "MetricHierarchy": ("timetoalign.timelines", "MetricHierarchy"),
    "MetricNode": ("timetoalign.timelines", "MetricNode"),
    "Reading": ("timetoalign.timelines", "Reading"),
    "Tempo": ("timetoalign.core", "Tempo"),
    "TempoEntry": ("timetoalign.timelines", "TempoEntry"),
    "TempoMap": ("timetoalign.timelines", "TempoMap"),
    # Loader layer
    "AudioInfo": ("timetoalign.loader", "AudioInfo"),
    "AudioLoader": ("timetoalign.loader", "AudioLoader"),
    "RekordboxLoader": ("timetoalign.loader", "RekordboxLoader"),
    "EepNotesLoader": ("timetoalign.loader", "EepNotesLoader"),
    "Loader": ("timetoalign.loader", "Loader"),
    "EventLoader": ("timetoalign.loader", "EventLoader"),
    "MatchfileLoader": ("timetoalign.loader", "MatchfileLoader"),
    "Ms3Loader": ("timetoalign.loader.score", "Ms3Loader"),
    "TsvLoader": ("timetoalign.loader.tabular", "TsvLoader"),
    "CsvLoader": ("timetoalign.loader.tabular", "CsvLoader"),
    "LabLoader": ("timetoalign.loader.tabular", "LabLoader"),
    "SoloLoader": ("timetoalign.loader.tabular", "SoloLoader"),
    "TabularLoader": ("timetoalign.loader.tabular", "TabularLoader"),
    "MeasureMapLoader": ("timetoalign.loader.score", "MeasureMapLoader"),
    "Music21Loader": ("timetoalign.loader.score", "Music21Loader"),
    "PartituraLoader": ("timetoalign.loader.score", "PartituraLoader"),
    "VerovioLoader": ("timetoalign.loader.score", "VerovioLoader"),
    "JsonLoader": ("timetoalign.loader.format", "JsonLoader"),
    "XmlLoader": ("timetoalign.loader.format", "XmlLoader"),
    "MidiLoader": ("timetoalign.loader.midi", "MidiLoader"),
    "PerformanceMidiLoader": (
        "timetoalign.loader.midi.performance",
        "PerformanceMidiLoader",
    ),
    "ScoreMidiLoader": ("timetoalign.loader.midi.score", "ScoreMidiLoader"),
    "ParangonadaLoader": ("timetoalign.loader.alignment", "ParangonadaLoader"),
    "MpmLoader": ("timetoalign.loader.alignment", "MpmLoader"),
    "PerformancePrecisionLoader": (
        "timetoalign.loader.alignment",
        "PerformancePrecisionLoader",
    ),
    "ListenHereLoader": ("timetoalign.loader.alignment", "ListenHereLoader"),
    "MapsLoader": ("timetoalign.loader.alignment", "MapsLoader"),
    "GraphicalLoader": ("timetoalign.loader.graphical", "GraphicalLoader"),
    "ATONLoader": ("timetoalign.loader.graphical.aton", "ATONLoader"),
    "IIIFManifestLoader": (
        "timetoalign.loader.graphical",
        "IIIFManifestLoader",
    ),
    "ScoreWarpLoader": ("timetoalign.loader.graphical", "ScoreWarpLoader"),
    "DmosLoader": ("timetoalign.loader.graphical", "DmosLoader"),
    "MeiFacsimileLoader": ("timetoalign.loader.graphical", "MeiFacsimileLoader"),
    "RepoVizzInfo": ("timetoalign.loader", "RepoVizzInfo"),
    "RepoVizzLoader": ("timetoalign.loader", "RepoVizzLoader"),
    "TiliaDictStore": ("timetoalign.loader", "TiliaDictStore"),
    "TiliaJsonLoader": ("timetoalign.loader", "TiliaJsonLoader"),
    # Timeline layer
    "BeatGrid": ("timetoalign.timelines", "BeatGrid"),
    "BeatGridSegment": ("timetoalign.timelines", "BeatGridSegment"),
    "GridBeat": ("timetoalign.timelines", "GridBeat"),
    "Integrity": ("timetoalign.timelines", "Integrity"),
    "Resource": ("timetoalign.timelines", "Resource"),
    "ContinuousGraphicalTimeline": (
        "timetoalign.timelines",
        "ContinuousGraphicalTimeline",
    ),
    "ContinuousLogicalTimeline": ("timetoalign.timelines", "ContinuousLogicalTimeline"),
    "ContinuousPhysicalTimeline": (
        "timetoalign.timelines",
        "ContinuousPhysicalTimeline",
    ),
    "DiscreteGraphicalTimeline": ("timetoalign.timelines", "DiscreteGraphicalTimeline"),
    "DiscreteLogicalTimeline": ("timetoalign.timelines", "DiscreteLogicalTimeline"),
    "DiscretePhysicalTimeline": ("timetoalign.timelines", "DiscretePhysicalTimeline"),
    "Gap": ("timetoalign.timelines", "Gap"),
    "GraphicalTimeline": ("timetoalign.timelines", "GraphicalTimeline"),
    "LogicalTimeline": ("timetoalign.timelines", "LogicalTimeline"),
    "PhysicalTimeline": ("timetoalign.timelines", "PhysicalTimeline"),
    "Timeline": ("timetoalign.timelines", "Timeline"),
    "TimelineGroup": ("timetoalign.timelines", "TimelineGroup"),
    "UnknownLengthError": ("timetoalign.timelines", "UnknownLengthError"),
    # Flow control
    "FlowControlRegistry": ("timetoalign.timelines", "FlowControlRegistry"),
    "FlowControlView": ("timetoalign.timelines", "FlowControlView"),
    "FlowControlElement": ("timetoalign.timelines", "FlowControlElement"),
    "Jump": ("timetoalign.timelines", "Jump"),
    "JumpTarget": ("timetoalign.timelines", "JumpTarget"),
    "Break": ("timetoalign.timelines", "Break"),
    "SectionBreak": ("timetoalign.timelines", "SectionBreak"),
    "Barline": ("timetoalign.timelines", "Barline"),
    "JumpDestination": ("timetoalign.timelines", "JumpDestination"),
    "FlowCondition": ("timetoalign.timelines", "FlowCondition"),
    "DestinationReference": ("timetoalign.timelines", "DestinationReference"),
    "PassesFrom": ("timetoalign.timelines", "PassesFrom"),
    "play": ("timetoalign.timelines", "play"),
    "atomic_sections": ("timetoalign.timelines", "atomic_sections"),
    "FlowResult": ("timetoalign.timelines", "FlowResult"),
    "FlowFinding": ("timetoalign.timelines", "FlowFinding"),
    "AtomicSection": ("timetoalign.timelines", "AtomicSection"),
    "PlayedFlow": ("timetoalign.timelines", "PlayedFlow"),
    # Alignment layer
    "AlignmentAnchor": ("timetoalign.alignment", "AlignmentAnchor"),
    "AlignmentBundle": ("timetoalign.alignment", "AlignmentBundle"),
    "ClaimFilter": ("timetoalign.alignment", "ClaimFilter"),
    "InferredEdge": ("timetoalign.alignment", "InferredEdge"),
    "MatchClaim": ("timetoalign.alignment", "MatchClaim"),
    "MatchClaimField": ("timetoalign.alignment", "MatchClaimField"),
    "MatchFileContext": ("timetoalign.alignment", "MatchFileContext"),
    "MatchGraph": ("timetoalign.alignment", "MatchGraph"),
    "MatchIntervalStamp": ("timetoalign.alignment", "MatchIntervalStamp"),
    "MatchLine": ("timetoalign.alignment", "MatchLine"),
    "MatchStamp": ("timetoalign.alignment", "MatchStamp"),
    "NoteRecord": ("timetoalign.alignment", "NoteRecord"),
    "SnoteRecord": ("timetoalign.alignment", "SnoteRecord"),
    "WarpMap": ("timetoalign.alignment", "WarpMap"),
}


def __getattr__(name: str) -> object:
    """Resolve and cache a deferred public export."""
    target = _DEFERRED_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return eager and deferred public module names."""
    return sorted(set(globals()) | set(__all__) | set(_DEFERRED_EXPORTS))


# User-friendly alias for Coordinate
Coord = Coordinate

__version__ = "1.2.0"

__all__ = [
    # Version
    "__version__",
    # Enums
    "ClaimType",
    "ColumnNaming",
    "ColumnRole",
    "Domain",
    "TimeUnit",
    "NumberType",
    "EventType",
    # Types
    "BeatPolicy",
    "Coordinate",
    "CoordinateField",
    "Coord",  # Alias for Coordinate
    "EventReference",
    "Event",
    "EventField",
    "Projection",
    "Box",
    "BoxField",
    "Point",
    "PointField",
    "Shape",
    "ShapeField",
    "BoundingBox",
    "BoundingBoxField",
    "Quadrilateral",
    "QuadrilateralField",
    "IdCoordinate",
    "Interval",
    "IntervalField",
    "CoordinateValue",
    "CoordinateSpec",
    "Duration",
    "Expansion",
    "Measure",
    "MeasureMap",
    "MetricHierarchy",
    "MetricNode",
    "Reading",
    "Tempo",
    "TempoEntry",
    "TempoMap",
    "OptionalCoordinate",
    "ResolvedCoordinate",
    "resolve_coordinate_spec",
    # Timestamps (first-class citizens)
    "TimeStamp",
    "TimeIntervalStamp",
    # IDs
    "ScopedId",
    # Loader - Event-based (Type 2)
    "EventData",
    "EventStore",
    "SingleStore",
    "DictStore",
    "Loader",
    "EventLoader",
    # Loader - Manifest-based (Type 1)
    "AudioLoader",
    "AudioInfo",
    "RepoVizzLoader",
    "RepoVizzInfo",
    # Loader - Physical event-based
    "EepNotesLoader",
    "TsvLoader",
    "CsvLoader",
    "LabLoader",
    "SoloLoader",
    "TabularLoader",
    "MeasureMapLoader",
    "Music21Loader",
    "PartituraLoader",
    "VerovioLoader",
    "JsonLoader",
    "XmlLoader",
    "MidiLoader",
    "PerformanceMidiLoader",
    "ScoreMidiLoader",
    # Loader - Alignment
    "MatchfileLoader",
    "Ms3Loader",
    "TiliaJsonLoader",
    "TiliaDictStore",
    "ParangonadaLoader",
    "MpmLoader",
    "PerformancePrecisionLoader",
    "ListenHereLoader",
    "MapsLoader",
    "GraphicalLoader",
    "ATONLoader",
    "IIIFManifestLoader",
    "ScoreWarpLoader",
    "DmosLoader",
    "MeiFacsimileLoader",
    # Timelines
    "Timeline",
    "LogicalTimeline",
    "PhysicalTimeline",
    "GraphicalTimeline",
    "ContinuousLogicalTimeline",
    "DiscreteLogicalTimeline",
    "ContinuousPhysicalTimeline",
    "DiscretePhysicalTimeline",
    "ContinuousGraphicalTimeline",
    "DiscreteGraphicalTimeline",
    "BeatGrid",
    "BeatGridSegment",
    "GridBeat",
    # Maps
    "ConstantMap",
    "ConversionMap",
    "IntervalToConstantMap",
    "InterpolationMap",
    "LinearMap",
    "MeasureRecordMap",
    "MeasureNumberMap",
    "FloatingMeasureMap",
    "MeasureBeatMap",
    "PiecewiseMap",
    "QuartersToFloatingMeasures",
    "QuartersToMeasureNumber",
    "ScalarMap",
    "SecondsToSamples",
    "ShiftMap",
    "TableMap",
    "ChainMap",
    # Alignment
    "AlignmentBundle",
    "Gap",
    "TimelineGroup",
    "Resource",
    "Integrity",
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
    "AlignmentAnchor",
    "Belief",
    "ClaimFilter",
    "InferredEdge",
    "MatchClaim",
    "MatchClaimField",
    "MatchFileContext",
    "MatchGraph",
    "MatchIntervalStamp",
    "MatchLine",
    "MatchStamp",
    "NoteRecord",
    "Provenance",
    "SnoteRecord",
    "WarpMap",
]
