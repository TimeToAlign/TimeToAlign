"""Alignment loader sub-package for TimeToAlign!

This sub-package contains loaders for alignment file formats that encode
score-performance correspondences, cross-domain alignments, and related
multimodal data.

Loaders:
    MatchfileLoader: Vienna Match (.match) score-to-performance alignment files.
    TiliaJsonLoader: TiLiA JSON annotation exports (.tla/.json).
    PerformancePrecisionLoader: CAAMP audio-to-score alignment specimens
        (.solo score + Verovio timemap + per-performer alignment CSVs).
    ParangonadaLoader: parangonada CSV exports (part/ppart/align triples)
        ingested as one multimodal AlignmentBundle.
    ListenHereLoader: Listen Here! alignment JSON exports (many recordings of
        one work warped onto a shared reference grid) ingested as one
        audio-to-audio AlignmentBundle with complete-topology pairwise claims.
    MpmLoader: MPM-Toolbox MSM+MPM+MPR triples (notated score + modelled
        performance markup + observed alignment) ingested as one
        multimodal AlignmentBundle.
    MapsLoader: MAPS score-to-performance alignment JSON (TROMPA's
        trompa-align), stated as MatchClaims between a score timeline keyed
        by MEI ``xml:id`` and a performance timeline.
    Ieee1599Loader: IEEE 1599 XML documents (spine + logic / notational /
        audio layers) ingested as one multimodal AlignmentBundle in which
        every layer is aligned to the spine.

Stores:
    TiliaDictStore: DictStore subclass with TiLiA timeline type properties.
"""

from __future__ import annotations

from .ieee1599 import Ieee1599Loader
from .listen_here import ListenHereLoader
from .matchfile import MatchfileLoader
from .mpm import MpmLoader
from .parangonada import ParangonadaLoader
from .performance_precision import PerformancePrecisionLoader
from .tilia import TiliaDictStore, TiliaJsonLoader
from .trompa_maps import MapsLoader

__all__ = [
    "Ieee1599Loader",
    "ListenHereLoader",
    "MapsLoader",
    "MatchfileLoader",
    "MpmLoader",
    "ParangonadaLoader",
    "PerformancePrecisionLoader",
    "TiliaDictStore",
    "TiliaJsonLoader",
]
