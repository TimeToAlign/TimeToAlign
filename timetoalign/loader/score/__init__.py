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

from importlib import import_module
from typing import Any

from .events import ScoreEventType
from .measuremap import MeasureMapLoader
from .store import ScoreStore

__all__ = [
    "MeasureMapLoader",
    "ScoreStore",
    "ScoreEventType",
]

# ---------------------------------------------------------------------------
# Optional loader backends — guarded so that importing the package never
# fails merely because one backend's dependency is absent.
# ---------------------------------------------------------------------------

try:
    from .partitura import PartituraLoader

    __all__.append("PartituraLoader")
except ImportError:
    pass


_OPTIONAL_LOADERS = {
    "PartituraLoader": ("partitura", "partitura"),
    "Music21Loader": ("music21", "music21"),
    "Ms3Loader": ("ms3", "ms3"),
    "VerovioLoader": ("verovio", "verovio"),
}


def __getattr__(name: str) -> Any:
    """Resolve optional score loaders with actionable dependency errors."""
    target = _OPTIONAL_LOADERS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, extra = target
    try:
        return getattr(import_module(f"{__name__}.{module_name}"), name)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"{name} requires its optional dependency; install "
            f"timetoalign[{extra}] to use this loader."
        ) from exc


try:
    from .music21 import Music21Loader

    __all__.append("Music21Loader")
except ImportError:
    pass

try:
    from .ms3 import Ms3Loader

    __all__.append("Ms3Loader")
except ImportError:
    pass

try:
    from .verovio import VerovioLoader

    __all__.append("VerovioLoader")
except ImportError:
    pass
