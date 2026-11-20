"""Physical domain loaders for TimeToAlign!

This package provides loaders for the physical (acoustic/time) domain:

Manifest Loaders (dimensions + metadata only):
    - AudioLoader: Load audio file metadata (WAV, MP3, FLAC, etc.)
    - RepoVizzLoader: Load RepoVizz 2-line CSV sensor data (MoCap, descriptors)

Event Loaders (events with coordinates):
    - EepNotesLoader: Load EEP .notes alignment files (onset/offset/pitch)

Design Philosophy:
    Manifest loaders extract dimensions and metadata from files without loading
    the actual sample data. The resulting timelines have their length set in
    samples, with automatic C-maps for sample-to-seconds conversion.

    Event loaders produce EventData that can populate timelines with events.
"""

from __future__ import annotations

from .audio import AudioInfo, AudioLoader
from .eep_notes import EepNotesLoader
from .rekordbox import RekordboxLoader
from .repovizz import RepoVizzInfo, RepoVizzLoader

__all__ = [
    "AudioLoader",
    "AudioInfo",
    "RepoVizzLoader",
    "RepoVizzInfo",
    "EepNotesLoader",
    "RekordboxLoader",
]
