"""EventData subclasses for MIDI loaders.

Two concrete subclasses model the cross-loader / loader-specific split:

* :class:`MidiEventData` carries the **seven cross-loader columns**
  produced by both mido (``PerformanceMidiLoader``) and partitura
  (``ScoreMidiLoader``).  Performance MIDI uses this class directly —
  there is no notion of voice, staff or part on a raw MIDI track, so
  storing always-null columns for those names would be a redundant
  schema.
* :class:`ScoreMidiEventData` extends the base with the three
  partitura-only columns (``voice``, ``staff``, ``part_id``).  It is
  the storage class for ``ScoreMidiLoader``.

The two classes pair with the :class:`MidiEvent` and
:class:`ScoreMidiEvent` pydantic scalars in ``core/events.py``: each
``_extra_fields`` list mirrors the additional sub-fields the scalar
exposes.  ``MidiStore`` works against either concrete class
transparently via ``type(data).empty(...)`` so that filter / merge
operations preserve the wider schema when one is in play.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pyarrow as pa

from timetoalign.core.events import EnharmonicPitchField
from timetoalign.core.fields import SemanticField
from timetoalign.storage.events import EventData


class MidiEventData(EventData):
    """EventData for performance MIDI events (mido cross-loader columns).

    Carries the seven columns mido can produce on its own: ``pitch``,
    ``velocity``, ``channel``, ``track``, ``control``, ``value``,
    ``program``.  Stored as nullable atomic columns (NOT a struct) for
    fast columnar access.

    A MIDI source is *number-only* — it carries a bare MIDI pitch with no
    enharmonic spelling — so its most-expressive faithful pitch type is
    :class:`~timetoalign.core.events.EnharmonicPitch`.  The raw ``pitch``
    column stays a plain integer (a faithful Layer-0 column); the
    EventData *affords* the ``EnharmonicPitchField`` view over it on
    request via :attr:`_afforded_fields`, honouring the
    scalar↔EventData contract that :class:`MidiEvent.pitch` promises
    without storing a redundant pitch struct.  ``pitch`` is stored as
    ``int64`` to match the ``{midi_number: int64}`` shape that view
    materialises into, keeping the affordance coherent across loaders.

    For score-side MIDI loaded via partitura, see
    :class:`ScoreMidiEventData`, which adds ``voice``, ``staff`` and
    ``part_id`` on top.
    """

    _extra_fields: ClassVar[list[pa.Field]] = [
        # Note fields (required for Notes).  ``pitch`` is int64 so it packs
        # cleanly into the EnharmonicPitch view (``{midi_number: int64}``).
        pa.field("pitch", pa.int64(), nullable=True),
        pa.field("velocity", pa.int8(), nullable=True),
        # MIDI routing
        pa.field("channel", pa.int8(), nullable=True),
        pa.field("track", pa.int16(), nullable=True),
        # Control Change fields
        pa.field("control", pa.int8(), nullable=True),  # CC number
        pa.field("value", pa.int8(), nullable=True),  # CC/Program value
        # Program Change
        pa.field("program", pa.int8(), nullable=True),
    ]

    # The raw ``pitch`` integer affords the most-expressive faithful pitch
    # view a number-only source supports: EnharmonicPitch.
    _afforded_fields: dict[str, type[SemanticField[Any]]] = {
        "pitch": EnharmonicPitchField,
    }


class ScoreMidiEventData(MidiEventData):
    """EventData for score MIDI events (partitura).

    Extends :class:`MidiEventData` with the three partitura-only
    columns: ``voice``, ``staff``, ``part_id``.  ``_extra_fields`` is
    redeclared (not appended) because the base ``EventData.get_schema``
    reads ``cls._extra_fields`` directly as the canonical column list.

    Score MIDI is still number-only at the pitch level (partitura's MIDI
    export carries no enharmonic spelling), so the inherited ``pitch`` →
    ``EnharmonicPitchField`` affordance applies unchanged.
    """

    _extra_fields: ClassVar[list[pa.Field]] = [
        # Inherited cross-loader columns
        pa.field("pitch", pa.int64(), nullable=True),
        pa.field("velocity", pa.int8(), nullable=True),
        pa.field("channel", pa.int8(), nullable=True),
        pa.field("track", pa.int16(), nullable=True),
        pa.field("control", pa.int8(), nullable=True),
        pa.field("value", pa.int8(), nullable=True),
        pa.field("program", pa.int8(), nullable=True),
        # Score-only columns (partitura)
        pa.field("voice", pa.int8(), nullable=True),
        pa.field("staff", pa.int8(), nullable=True),
        pa.field("part_id", pa.string(), nullable=True),
    ]
