"""MIDI loading and storage for TimeToAlign!

The core classes (:class:`MidiLoader`, :class:`MidiStore`,
:class:`MidiEventData`, :data:`CC_PURPOSE`, :class:`MidiEventType`) are
always available.  Individual loader backends are guarded behind optional
dependencies:

* :class:`PerformanceMidiLoader` — requires the ``midi`` extra (``mido``)
* :class:`ScoreMidiLoader` — requires the ``partitura`` extra

If you attempt to import a loader whose dependency is not installed, an
:exc:`ImportError` is raised with installation instructions.
"""

from __future__ import annotations

from timetoalign.loader._optional import import_optional, resolve_optional

from .base import MidiLoader
from .constants import CC_PURPOSE, MidiEventType
from .events import MidiEventData, ScoreMidiEventData
from .store import MidiStore

__all__ = [
    "CC_PURPOSE",
    "MidiEventData",
    "MidiEventType",
    "MidiLoader",
    "MidiStore",
    "ScoreMidiEventData",
]

# Optional loader backends: importing the package never fails because one
# backend's dependency is absent; asking for that backend names its extra.
_OPTIONAL_LOADERS = {
    "PerformanceMidiLoader": ("performance", "midi"),
    "ScoreMidiLoader": ("score", "partitura"),
}

__all__ += import_optional(__name__, _OPTIONAL_LOADERS, globals())
__getattr__ = resolve_optional(__name__, _OPTIONAL_LOADERS)
