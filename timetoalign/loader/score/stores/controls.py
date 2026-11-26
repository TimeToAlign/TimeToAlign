"""ControlEventData: Storage for control events (dynamics, tempo, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa
from typing_extensions import Self

from timetoalign.core import IntervalPolicy, NumberType, TimeUnit
from timetoalign.core.ids import IdMinter
from timetoalign.loader._ids import give_event_ids, minter_of
from timetoalign.storage.events import EventData
from timetoalign.storage.schema import make_rational_field

if TYPE_CHECKING:
    pass


class ControlEventData(EventData):
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

    @classmethod
    def empty(
        cls,
        unit: TimeUnit = TimeUnit.quarters,
        number_type: NumberType | None = None,
    ) -> Self:
        """Create empty ControlEventData."""
        return super().empty(unit, number_type)

    @classmethod
    def from_dicts(
        cls,
        rows: list[dict[str, Any]],
        unit: TimeUnit = TimeUnit.quarters,
        number_type: NumberType | None = None,
        *,
        interval_policy: IntervalPolicy = IntervalPolicy.warn,
        seed: int | None = None,
        minter: IdMinter | None = None,
    ) -> Self:
        """Create from dicts.

        A row without an ``id`` gets one minted from *minter* (a loader's),
        else from ``IdMinter(seed)``, else from the content minter of the
        event types (:func:`~timetoalign.storage.mint_event_ids`); a stated
        ``id`` is kept. Giving both *seed* and *minter* raises ``TypeError``.
        """
        if not rows:
            return cls.empty(unit, number_type)

        from timetoalign.storage.schema import make_table_metadata

        schema = cls.get_schema(unit)
        metadata = make_table_metadata(
            unit, unit.resolve_number_type(number_type), loader_class=cls.__name__
        )
        schema = schema.with_metadata(metadata)

        processed_rows = []
        for row in rows:
            processed = dict(row)

            # Ensure name field exists
            if "name" not in processed:
                processed["name"] = None

            # Map legacy temporal fields to base
            if "quarterbeats" in processed:
                processed["start"] = processed.pop("quarterbeats")
            if "duration_qb" in processed:
                processed["duration"] = processed.pop("duration_qb")
            # Remove unused
            processed.pop("quarterbeats_float", None)
            processed.pop("duration_qb_float", None)

            # Unified interval normalisation: converts coordinate fields
            # to struct format, fills missing end/duration, and infers
            # temporal_type.
            EventData._normalize_intervals_row(
                processed,
                policy=interval_policy,
                number_type=unit.resolve_number_type(number_type),
            )

            processed_rows.append(processed)

        give_event_ids(processed_rows, minter_of(seed, minter))
        table = pa.Table.from_pylist(processed_rows, schema=schema)
        return cls(table, unit, number_type)
