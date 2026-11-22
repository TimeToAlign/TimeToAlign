"""Score loading module for TimeToAlign!

This package provides loaders for symbolic music scores.  The core classes
(:class:`ScoreStore`, :class:`ScoreEventType`, :class:`MeasureMapLoader`) are
always available.  Individual loader backends are guarded behind optional
dependencies:

* :class:`PartituraLoader` — requires the ``partitura`` extra
* :class:`Music21Loader` — requires the ``music21`` extra
* :class:`Ms3Loader` — requires the ``ms3`` extra
* :class:`VerovioLoader` — requires the ``verovio`` extra

If you attempt to import a loader whose dependency is not installed, an
:exc:`ImportError` is raised with installation instructions.
"""

from __future__ import annotations

from timetoalign.loader._optional import import_optional, resolve_optional

from .events import ScoreEventType
from .measuremap import MeasureMapLoader
from .store import ScoreStore

__all__ = [
    "MeasureMapLoader",
    "ScoreStore",
    "ScoreEventType",
]

# Optional loader backends: importing the package never fails because one
# backend's dependency is absent; asking for that backend names its extra.
_OPTIONAL_LOADERS = {
    "PartituraLoader": ("partitura", "partitura"),
    "Music21Loader": ("music21", "music21"),
    "Ms3Loader": ("ms3", "ms3"),
    "VerovioLoader": ("verovio", "verovio"),
}

__all__ += import_optional(__name__, _OPTIONAL_LOADERS, globals())
__getattr__ = resolve_optional(__name__, _OPTIONAL_LOADERS)
