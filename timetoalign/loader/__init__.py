"""Loader package for TimeToAlign!

This package provides the infrastructure for loading music representations
and storing events in efficient PyArrow tables.

Loader Taxonomy:
    The package contains four categories of loaders:

    **Loader** - Returns EventStore (events only):
        - ScoreLoader subclasses: Music21Loader, PartituraLoader, Ms3Loader,
          VerovioLoader
        - MidiLoader subclasses: ScoreMidiLoader, PerformanceMidiLoader
        - TabularLoader subclasses: CsvLoader, TsvLoader

    **ManifestLoader** - Returns ManifestData (dimensions + metadata only):
        - AudioLoader: Audio file -> DiscretePhysicalTimeline (samples)
        - IIIFManifestLoader: IIIF manifest -> Graphical dimensions

    **Readers of areas on page images** - Return timelines holding Box events
    (in ``timetoalign.loader.graphical``):
        - DmosLoader: DMOS page layout -> pages, systems, measures
        - MeiFacsimileLoader: MEI facsimile -> surfaces, zones, ``@facs`` links

    **AlignmentLoader** - Returns AlignmentStore (events + C-maps):
        - MatchfileLoader: Score-to-performance alignment
        - MapsLoader: MAPS score-to-performance alignment (MatchClaims)
        - TiliaJsonLoader: TiLiA hierarchical annotations
        - MpmLoader, ParangonadaLoader, PerformancePrecisionLoader,
          ListenHereLoader: multimodal alignment exports

Classes:
    EventData: PyArrow-based bulk event storage.
    EventStore: Abstract base class for collections of EventData.
    SingleStore: Store wrapper for a single EventData.
    AlignmentStore: Container for aligned multimodal data.
    ManifestData: Container for manifest/metadata.
    Loader: Abstract base class for event-based loaders.
    ManifestLoader: Abstract base class for manifest loaders.
    AlignmentLoader: Abstract base class for alignment loaders.
    AudioLoader: Manifest loader for audio files.
    AudioInfo: Metadata container for audio files.

NOTE: Class naming was updated in the 2026-01 API refactoring:
- EventStore -> EventData (PyArrow table storage)
- EventBundle -> EventStore (container for EventData)
- SingleStoreBundle -> SingleStore

The package also exports schema utilities for working with coordinate
structures and table metadata.
"""

from __future__ import annotations

from timetoalign.core.time import (
    coordinate_to_struct,
)
from timetoalign.storage.schema import (
    TEMPORAL_TYPE_INSTANT,
    TEMPORAL_TYPE_INTERVAL,
    ComputedField,
    Field,
    extend_schema,
    get_base_field_names,
    get_unit_from_schema,
    make_base_schema,
    make_coordinate_field,
    make_coordinate_type,
    make_table_metadata,
    parse_json_to_struct,
    parse_table_metadata,
)

from ._optional import import_optional, resolve_optional
from .alignment import MapsLoader, TiliaDictStore, TiliaJsonLoader
from .base import AlignmentLoader, EventLoader, Loader, ManifestData, ManifestLoader
from .metadata import MeiHeadLoader, MelodyLoader
from .physical import (
    AudioInfo,
    AudioLoader,
    EepNotesLoader,
    RekordboxLoader,
    RepoVizzInfo,
    RepoVizzLoader,
)

# Optional loaders: importing the package never fails because one loader's
# dependency is absent; asking for that loader names its extra.
_OPTIONAL_LOADERS = {"MatchfileLoader": ("alignment", "partitura")}

__all__ = [
    "ManifestData",
    # Loader ABCs
    "Loader",
    "EventLoader",
    "ManifestLoader",
    "AlignmentLoader",
    # Alignment loaders
    "MapsLoader",
    *import_optional(__name__, _OPTIONAL_LOADERS, globals()),
    "TiliaDictStore",
    "TiliaJsonLoader",
    # Physical domain loaders
    "AudioLoader",
    "AudioInfo",
    "RepoVizzLoader",
    "RepoVizzInfo",
    "EepNotesLoader",
    "RekordboxLoader",
    # Metadata importers
    "MeiHeadLoader",
    "MelodyLoader",
    "Field",
    "ComputedField",
    "parse_json_to_struct",
    # Schema utilities
    "make_coordinate_type",
    "make_coordinate_field",
    "coordinate_to_struct",
    "make_base_schema",
    "get_base_field_names",
    "extend_schema",
    "get_unit_from_schema",
    "make_table_metadata",
    "parse_table_metadata",
    # Constants
    "TEMPORAL_TYPE_INSTANT",
    "TEMPORAL_TYPE_INTERVAL",
]
__getattr__ = resolve_optional(__name__, _OPTIONAL_LOADERS)
