"""AnnotationEventData: Storage for text annotations and directions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa
from typing_extensions import Self

from timetoalign.core import IntervalPolicy, NumberType, TimeUnit
from timetoalign.core.ids import IdMinter, given_minter
from timetoalign.loader._ids import give_event_ids
from timetoalign.storage.events import EventData
from timetoalign.storage.mixins import HarmonyAccessMixin
from timetoalign.storage.schema import make_rational_field

if TYPE_CHECKING:
    pass


class AnnotationEventData(EventData, HarmonyAccessMixin):
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

    @classmethod
    def empty(
        cls,
        unit: TimeUnit = TimeUnit.quarters,
        number_type: NumberType | None = None,
    ) -> Self:
        """Create empty AnnotationEventData."""
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

        give_event_ids(processed_rows, given_minter(seed, minter))
        table = pa.Table.from_pylist(processed_rows, schema=schema)
        return cls(table, unit, number_type)
