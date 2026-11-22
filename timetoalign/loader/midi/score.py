"""ScoreMidiLoader: Load score MIDI using partitura."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa

from timetoalign.core.backends import suppressed_backend_warnings

with suppressed_backend_warnings():
    import partitura as pt
    from partitura.score import Part, PartGroup, Score

from timetoalign.core import NumberType, TimeUnit

from .base import MidiLoader
from .constants import MidiEventType
from .events import MidiEventData, ScoreMidiEventData


class ScoreMidiLoader(MidiLoader):
    """Load score MIDI files using partitura or a mido fast path.

    This loader uses partitura's sophisticated MIDI parsing to extract structural
    information like voices, parts, and specific pitches (optional). It is ideal
    for quantized MIDI files representing scores.

    The default partitura parser emits :class:`ScoreMidiEventData`, the wider
    schema containing ``voice``, ``staff``, and ``part_id``. ``parser="mido"``
    parses only MIDI messages and emits the base :class:`MidiEventData` schema;
    it is appropriate when structural score information is not required.
    """

    _event_data_class: ClassVar[type[MidiEventData]] = ScoreMidiEventData

    def __init__(
        self,
        *,
        part_voice_assign_mode: int = 0,
        quantization_unit: int | None = None,
        estimate_voice_info: bool = False,
        estimate_key: bool = False,
        assign_note_ids: bool = True,
        parser: str = "partitura",
        unit: TimeUnit | None = None,
        number_type: NumberType | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ScoreMidiLoader.

        Args:
            part_voice_assign_mode: Partitura mode for assigning parts/voices (0-5).
            quantization_unit: Quantize times to multiples of this unit.
            estimate_voice_info: Use Chew & Wu algorithm for voice separation.
            estimate_key: Use Krumhansl algorithm for key estimation.
            assign_note_ids: Ensure unique note IDs.
            parser: MIDI parser, either ``"partitura"`` or ``"mido"``.
            unit: Time unit for coordinates.
            number_type: Number type for coordinates. Defaults to the
                one the unit itself uses.
            **kwargs: Additional arguments passed to parent Loader.
        """
        if parser not in {"partitura", "mido"}:
            raise ValueError("parser must be either 'partitura' or 'mido'")
        if parser == "mido" and (
            part_voice_assign_mode != 0
            or quantization_unit is not None
            or estimate_voice_info
            or estimate_key
        ):
            raise ValueError(
                "parser='mido' cannot use structural score options; "
                "part_voice_assign_mode, quantization_unit, estimate_voice_info, "
                "and estimate_key need partitura"
            )
        if parser == "mido":
            self._event_data_class = MidiEventData
        super().__init__(unit=unit, number_type=number_type, **kwargs)
        self._part_voice_assign_mode = part_voice_assign_mode
        self._quantization_unit = quantization_unit
        self._estimate_voice_info = estimate_voice_info
        self._estimate_key = estimate_key
        self._assign_note_ids = assign_note_ids
        self._parser = parser
        self._ticks_per_beat: int | None = None

    @property
    def ticks_per_beat(self) -> int | None:
        """Return the ticks per beat (PPQ) of the loaded file."""
        return self._ticks_per_beat

    def _load_source(
        self, source: Path
    ) -> tuple[dict[str, Any], dict[str, pa.ChunkedArray]]:
        """Load a MIDI file using the configured parser.

        Args:
            source: Path to the MIDI file.

        Returns:
            Tuple of metadata and vectorized event field arrays.
        """
        if self._parser == "mido":
            return self._parse_mido_source(
                source,
                parse_durations=True,
                include_controls=False,
                include_program_changes=False,
                on0_means_off=True,
            )

        # Load score using partitura
        # Note: partitura can return Score, Part, PartGroup, or list
        with suppressed_backend_warnings():
            score_data = pt.load_score_midi(
                source,
                part_voice_assign_mode=self._part_voice_assign_mode,
                quantization_unit=self._quantization_unit,
                estimate_voice_info=self._estimate_voice_info,
                estimate_key=self._estimate_key,
                assign_note_ids=self._assign_note_ids,
            )

        # Normalize to a list of parts
        parts: list[Part] = []
        if isinstance(score_data, Score):
            parts = score_data.parts
        elif isinstance(score_data, (Part, PartGroup)):
            # PartGroup handling is recursive in partitura, but we need flat list for note_array
            # Helper to flatten
            def flatten_parts(obj):
                if isinstance(obj, Part):
                    return [obj]
                elif isinstance(obj, PartGroup):
                    flat = []
                    for child in obj.children:
                        flat.extend(flatten_parts(child))
                    return flat
                return []

            parts = flatten_parts(score_data)
        elif isinstance(score_data, list):
            # List of Part/PartGroup
            parts = []
            for item in score_data:
                if isinstance(item, Part):
                    parts.append(item)
                # ... theoretically could be PartGroup in list, but load_score_midi usually returns Score or parts

        if not parts:
            # Empty score
            return {"format": "midi", "parser": "partitura", "parts": 0}, {}

        # Get PPQ from the first part (partitura stores this in quarter_map logic usually,
        # but for MIDI loading, it's often implicit or stored in the score object if available)
        # We'll try to get it from the first part's quarter_map if simple
        # NOTE: Partitura's MIDI loader doesn't always expose original PPQ directly on Part.
        # But for now we can leave it None or check if we can extract it.
        # Since we are using note_array, we will rely on its units.

        # Generate note array with all info
        # We use include_staff=True if available in partitura version
        note_array = pt.utils.music.note_array_from_part_list(
            parts,
            include_staff=True,
            include_divs_per_quarter=True,  # To get PPQ info
        )

        # Extract events
        events = []

        # Iterate over structured array
        for row in note_array:
            # row fields: onset_div, duration_div, pitch, voice, id, staff, divs_pq, ...

            # Update PPQ if we found it
            if "divs_pq" in row.dtype.names and self._ticks_per_beat is None:
                self._ticks_per_beat = int(row["divs_pq"])

            event = {
                "id": str(row["id"]),
                "temporal_type": "interval",
                "event_type": MidiEventType.NOTE,
                "start": int(row["onset_div"]),
                "end": int(row["onset_div"] + row["duration_div"]),
                "duration": int(row["duration_div"]),
                "pitch": int(row["pitch"]),
                # Partitura's score parse carries no velocity, so none is written;
                # parser="mido" reads the file's own values.
                "velocity": None,
                "voice": int(row["voice"]) if "voice" in row.dtype.names else None,
                "staff": int(row["staff"]) if "staff" in row.dtype.names else None,
                # Part ID is tricky in unified array, but partitura usually encodes it in ID
                # or we can't easily get it per row without complex mapping.
                # For now, we'll leave part_id null or extract from ID if encoded.
                "part_id": None,
                # Nullable mido fields
                "channel": None,
                "track": None,
                "control": None,
                "value": None,
                "program": None,
            }
            events.append(event)

        metadata = {
            "format": "midi",
            "parser": "partitura",
            "part_voice_assign_mode": self._part_voice_assign_mode,
            "parts": len(parts),
            "ticks_per_beat": self._ticks_per_beat,
        }

        if not events:
            return metadata, {}
        table = pa.Table.from_pylist(events)
        fields = {name: table.column(name) for name in table.column_names}
        return metadata, fields
