"""NoteEventData: Storage for note/rest/chord events."""

from __future__ import annotations

from typing import Any, ClassVar

import pyarrow as pa
from typing_extensions import Self

from timetoalign.core import IntervalPolicy, NumberType, TimeUnit
from timetoalign.core.events import EnharmonicPitchField, SpecificPitchField
from timetoalign.core.fields import SemanticField
from timetoalign.core.ids import IdMinter
from timetoalign.loader._ids import give_event_ids, minter_of
from timetoalign.storage.events import EventData
from timetoalign.storage.schema import make_rational_field


class NoteEventData(EventData):
    """EventData for note, rest, and chord events.

    Rich temporal schema following TSV gold standard:
    - start: Continuous logical time (Fraction) - from 'quarterbeats'
    - duration: Duration (Fraction) - from 'duration_qb'
    - mc/mn: Measure context
    - mc_onset/mn_onset: Measure-relative offsets (Fraction)

    Pitch is **represented exactly once**.  A spelled score source
    faithfully supports :class:`~timetoalign.core.events.SpecificPitch`
    (step + alter + octave), so ``specific_pitch`` is the **sole default
    semantic pitch field** and :meth:`get_pitch_field` returns it.  The
    source MIDI pitch is kept only as the **non-default raw ``midi``
    int** column — redundant data the spelling already determines — which
    the EventData *affords* as an
    :class:`~timetoalign.core.events.EnharmonicPitch` view on request via
    :attr:`_afforded_fields` (so ``get_field(EnharmonicPitch)`` works
    without storing a second pitch struct, and a future check can confirm
    the raw number agrees with the value derived from ``specific_pitch``).

    Pitch / pitch-adjacent fields:
    - specific_pitch: {step, alter, octave, cents} (SpecificPitchField, default)
    - midi: source MIDI pitch (raw int64; affords EnharmonicPitch on request)
    - tpc: Tonal Pitch Class (fifths)
    - octave: Octave number
    """

    _extra_fields: ClassVar[list[pa.Field]] = [
        # Temporal - Measure context
        pa.field("mc", pa.int64(), nullable=True, metadata={"number_type": "int64"}),
        pa.field("mn", pa.string(), nullable=True),
        make_rational_field("mc_onset", nullable=True),
        make_rational_field("mn_onset", nullable=True),
        # Pitch — represented once.  SpecificPitch is the sole default
        # semantic pitch field (schema derived from the paired Field's
        # pydantic model, see core/events.py).
        pa.field("specific_pitch", SpecificPitchField.struct_schema(), nullable=True),
        # The source MIDI pitch, kept as a NON-DEFAULT raw int.  It is
        # redundant with ``specific_pitch`` (the spelling determines it),
        # so it is NOT a semantic pitch struct; it affords an
        # EnharmonicPitch view on request (see _afforded_fields).
        pa.field("midi", pa.int64(), nullable=True, metadata={"number_type": "int64"}),
        pa.field(
            "tpc",
            pa.int64(),
            nullable=True,
            metadata={"number_type": "int64", "unit": "fifths"},
        ),
        pa.field(
            "octave", pa.int64(), nullable=True, metadata={"number_type": "int64"}
        ),
        # Attributes
        # ms3 tie values: 1 start, 0 middle, -1 end; null where no tie is stated
        pa.field("tied", pa.int64(), nullable=True),
        pa.field("gracenote", pa.string(), nullable=True),
        pa.field("chord_id", pa.string(), nullable=True),  # the source's chord id
        # Context
        pa.field("voice", pa.int64(), nullable=True, metadata={"number_type": "int64"}),
        pa.field("staff", pa.int64(), nullable=True, metadata={"number_type": "int64"}),
        pa.field("part_id", pa.string(), nullable=True),
    ]

    # The raw ``midi`` integer affords an EnharmonicPitch view on request
    # (non-default — SpecificPitch remains the sole default pitch field).
    _afforded_fields: dict[str, type[SemanticField[Any]]] = {
        "midi": EnharmonicPitchField,
    }

    def __init__(
        self,
        table: pa.Table,
        unit: TimeUnit,
        number_type: NumberType | None = None,
        has_rests: bool = False,
    ) -> None:
        """Initialize NoteEventData.

        Args:
            table: PyArrow table.
            unit: Time unit.
            number_type: Number type.
            has_rests: Whether the source explicitly includes rests.
        """
        super().__init__(table, unit, number_type)
        self._has_rests = has_rests

    @property
    def has_rests(self) -> bool:
        """Return whether the store explicitly contains rests."""
        return self._has_rests

    @property
    def enharmonic_pitch_field(self) -> EnharmonicPitchField:
        """Return an ``EnharmonicPitchField`` view over the source MIDI pitch.

        The score stores pitch once (as ``specific_pitch``); the
        EnharmonicPitch view is *afforded* on request from the raw
        ``midi`` column (see :attr:`_afforded_fields`).

        Returns:
            An ``EnharmonicPitchField`` wrapping the afforded ``midi`` view.

        Raises:
            KeyError: If the table affords no EnharmonicPitch view.
        """
        result = self.get_field(EnharmonicPitchField)
        assert isinstance(result, EnharmonicPitchField)
        return result

    @property
    def specific_pitch_field(self) -> SpecificPitchField:
        """Extract the ``specific_pitch`` field as a ``SpecificPitchField``.

        Returns:
            A ``SpecificPitchField`` wrapping the ``specific_pitch`` field.

        Raises:
            KeyError: If the table has no ``specific_pitch`` field.
        """
        try:
            result = self.get_pitch_field(SpecificPitchField)
            if isinstance(result, SpecificPitchField):
                return result
        except KeyError:
            pass

        col = self._table.column("specific_pitch")
        pa_field = self._table.schema.field("specific_pitch")
        return SpecificPitchField.from_field((col, pa_field))

    @classmethod
    def empty(
        cls,
        unit: TimeUnit = TimeUnit.quarters,
        number_type: NumberType | None = None,
        has_rests: bool = False,
    ) -> Self:
        """Create empty NoteEventData."""
        store = super().empty(unit, number_type)
        store._has_rests = has_rests
        return store

    @classmethod
    def from_dicts(
        cls,
        rows: list[dict[str, Any]],
        unit: TimeUnit = TimeUnit.quarters,
        number_type: NumberType | None = None,
        has_rests: bool = False,
        *,
        interval_policy: IntervalPolicy = IntervalPolicy.warn,
        seed: int | None = None,
        minter: IdMinter | None = None,
    ) -> Self:
        """Create from dicts with has_rests metadata.

        Builds PyArrow table directly using NoteEventData schema.

        A row without an ``id`` gets one minted from *minter* (a loader's),
        else from ``IdMinter(seed)``, else from the content minter of the
        event types (:func:`~timetoalign.storage.mint_event_ids`); a stated
        ``id`` is kept. Giving both *seed* and *minter* raises ``TypeError``.
        """
        if not rows:
            return cls.empty(unit, number_type, has_rests)

        from timetoalign.storage.schema import make_table_metadata

        schema = cls.get_schema(unit)
        metadata = make_table_metadata(
            unit, unit.resolve_number_type(number_type), loader_class=cls.__name__
        )
        schema = schema.with_metadata(metadata)

        # Ensure all required fields exist with proper defaults
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
            # Remove unused/redundant fields
            for k in [
                "quarterbeats_float",
                "duration_qb_float",
                "nominal_duration",
                "scalar",
                "timesig",
                "instant",
            ]:
                processed.pop(k, None)

            # Unified interval normalisation: converts coordinate fields
            # to struct format, fills missing end/duration, and infers
            # temporal_type.  Delegates to the base EventData method.
            EventData._normalize_intervals_row(
                processed,
                policy=interval_policy,
                number_type=unit.resolve_number_type(number_type),
            )

            processed_rows.append(processed)

        give_event_ids(processed_rows, minter_of(seed, minter))
        table = pa.Table.from_pylist(processed_rows, schema=schema)
        store = cls(table, unit, number_type, has_rests)
        return store
