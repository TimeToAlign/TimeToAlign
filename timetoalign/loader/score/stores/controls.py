"""ControlEventData: Storage for control events (dynamics, tempo, etc.)."""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa

from timetoalign.loader.score.stores.base import ScoreEventData
from timetoalign.storage.schema import make_rational_field


class ControlEventData(ScoreEventData):
    """EventData for control events.

    Subtypes include:
    - Dynamic: velocity/dynamics markings (pp, ff, mf)
    - Tempo: tempo changes (MetronomeMark, TempoIndication)
    - Pedal: sustain/soft pedal events
    - OctaveShift: 8va/8vb markings
    - Wedge: crescendo/decrescendo hairpins

    Schema:
    - quarterbeats: Event position (Fraction)
    - duration_qb: Event duration if applicable (Fraction)
    - subtype: Control category (Dynamic, Tempo, Pedal, etc.)
    - value: Numeric value (BPM, velocity level)
    - text: Textual representation
    """

    _extra_fields: ClassVar[list[pa.Field]] = [
        # Measure context
        pa.field("mc", pa.int64(), nullable=True, metadata={"number_type": "int64"}),
        pa.field("mn", pa.string(), nullable=True),
        make_rational_field("mc_onset", nullable=True),
        make_rational_field("mn_onset", nullable=True),
        # Control specifics
        pa.field("subtype", pa.string(), nullable=False),  # Dynamic, Tempo, Pedal, etc.
        pa.field("value", pa.float64(), nullable=True),  # BPM, velocity, etc.
        pa.field("text", pa.string(), nullable=True),  # "ff", "Allegro", etc.
        # Context
        pa.field("voice", pa.int64(), nullable=True),
        pa.field("staff", pa.int64(), nullable=True),
        pa.field("part_id", pa.string(), nullable=True),
    ]
