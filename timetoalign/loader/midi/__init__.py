"""MIDI loading and storage for TimeToAlign!

The core classes (:class:`MidiLoader`, :class:`MidiStore`,
:class:`MidiEventData`, :data:`CC_PURPOSE`, :class:`MidiEventType`) are
always available.  The loaders need optional dependencies:

* :class:`PerformanceMidiLoader` — requires the ``midi`` extra (``mido``)
  to load a file; importing it does not.
* :class:`ScoreMidiLoader` — requires the ``partitura`` extra.

If you attempt to import :class:`ScoreMidiLoader`, or to load a file with
:class:`PerformanceMidiLoader`, without its dependency, an
:exc:`ImportError` is raised with installation instructions.
"""

from __future__ import annotations

from timetoalign.loader._optional import import_optional, resolve_optional

from .base import MidiLoader
from .constants import CC_PURPOSE, MidiEventType
from .events import MidiEventData, ScoreMidiEventData
from .performance import PerformanceMidiLoader
from .store import MidiStore

__all__ = [
    "CC_PURPOSE",
    "MidiEventData",
    "MidiEventType",
    "MidiLoader",
    "MidiStore",
    "ScoreMidiEventData",
    "PerformanceMidiLoader",
]

# Optional loader backends: importing the package never fails because one
# backend's dependency is absent; asking for that backend names its extra.
_OPTIONAL_LOADERS = {
    "ScoreMidiLoader": ("score", "partitura"),
}

__all__ += import_optional(__name__, _OPTIONAL_LOADERS, globals())
__getattr__ = resolve_optional(__name__, _OPTIONAL_LOADERS)
