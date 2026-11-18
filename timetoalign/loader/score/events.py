"""Score event-type constants.

This module defines :class:`ScoreEventType`, the shared vocabulary of
``event_type`` / ``event_category`` string constants used across the
score loaders.  Note storage itself lives in
:class:`~timetoalign.loader.score.stores.notes.NoteEventData`, which
represents pitch exactly once.
"""

from __future__ import annotations


class ScoreEventType:
    """Constants for score event types."""

    # Categories
    CAT_MEASURE = "measure"
    CAT_NOTE = "note"
    CAT_CONTROL = "control"
    CAT_ANNOTATION = "annotation"

    # Common Event Types
    NOTE = "Note"
    REST = "Rest"
    CHORD = "Chord"
    MEASURE = "Measure"

    # Control/Direction Types
    TIME_SIGNATURE = "TimeSignature"
    KEY_SIGNATURE = "KeySignature"
    TEMPO = "Tempo"
    TEMPO_INDICATION = "tempo indication"
    METRONOME = "Metronome"
    DYNAMIC = "Dynamic"
    DIRECTION = "Direction"
    WEDGE = "Wedge"  # Hairpins
    pedal = "Pedal"
    OCTAVE_SHIFT = "OctaveShift"

    # Annotation
    TEXT_BOX = "TextBox"
    TEXT_EXPRESSION = "TextExpression"
