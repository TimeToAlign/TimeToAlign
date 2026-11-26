"""AnnotationEventData: Storage for text annotations and directions."""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa

from timetoalign.loader.score.stores.base import ScoreEventData
from timetoalign.storage.mixins import HarmonyAccessMixin
from timetoalign.storage.schema import make_rational_field


class AnnotationEventData(ScoreEventData, HarmonyAccessMixin):
    """EventData for text annotations.

    Subtypes include:
    - TextBox: Free text annotations
    - TextExpression: Expression text (Allegro con brio)
    - Rehearsal: Rehearsal marks (A, B, etc.)
    - Direction: Performance directions

    Schema:
    - quarterbeats: Annotation position (Fraction)
    - text: Annotation content
    - subtype: Annotation category
    """

    _extra_fields: ClassVar[list[pa.Field]] = [
        # Measure context
        pa.field("mc", pa.int64(), nullable=True, metadata={"number_type": "int64"}),
        pa.field("mn", pa.string(), nullable=True),
        make_rational_field("mc_onset", nullable=True),
        make_rational_field("mn_onset", nullable=True),
        # Annotation specifics
        pa.field("subtype", pa.string(), nullable=True),  # TextBox, Rehearsal, etc.
        pa.field("text", pa.string(), nullable=False),
        # Context
        pa.field("staff", pa.int64(), nullable=True),
        pa.field("part_id", pa.string(), nullable=True),
    ]
