"""Alignment loader sub-package for TimeToAlign!

This sub-package contains loaders for alignment file formats that encode
score-performance correspondences, cross-domain alignments, and related
multimodal data.

Loaders:
    MatchfileLoader: Vienna Match (.match) score-to-performance alignment files;
        requires the ``partitura`` extra.
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
        multimodal AlignmentBundle; requires the ``mpm`` extra (``lxml``).
    MapsLoader: MAPS score-to-performance alignment JSON (TROMPA's
        trompa-align), stated as MatchClaims between a score timeline keyed
        by MEI ``xml:id`` and a performance timeline.
    MtdLoader: Musical Theme Dataset (MTD) themes, read from the archive or
        an extracted folder: per CD track a seconds timeline with an excerpt
        per theme, per theme a quarters timeline of its encoding, and the
        alignment as MatchClaims from the theme's notes to the excerpt.
    Ieee1599Loader: IEEE 1599 XML documents (spine + logic / notational /
        audio layers) ingested as one multimodal AlignmentBundle in which
        every layer is aligned to the spine.

Stores:
    TiliaDictStore: DictStore subclass with TiLiA timeline type properties.

Asking for a loader whose dependency is not installed raises an
:exc:`ImportError` naming the extra to install.
"""

from __future__ import annotations

from timetoalign.loader._optional import import_optional, resolve_optional

from .ieee1599 import Ieee1599Loader
from .listen_here import ListenHereLoader
from .mtd import MtdLoader
from .parangonada import ParangonadaLoader
from .performance_precision import PerformancePrecisionLoader
from .tilia import TiliaDictStore, TiliaJsonLoader
from .trompa_maps import MapsLoader

# Optional loaders: importing the package never fails because one loader's
# dependency is absent; asking for that loader names its extra.
_OPTIONAL_LOADERS = {
    "MatchfileLoader": ("matchfile", "partitura"),
    "MpmLoader": ("mpm", "mpm"),
}

__all__ = [
    "Ieee1599Loader",
    "ListenHereLoader",
    "MapsLoader",
    "MtdLoader",
    *import_optional(__name__, _OPTIONAL_LOADERS, globals()),
    "ParangonadaLoader",
    "PerformancePrecisionLoader",
    "TiliaDictStore",
    "TiliaJsonLoader",
]
__getattr__ = resolve_optional(__name__, _OPTIONAL_LOADERS)
