"""MidiLoader: Abstract base class for MIDI loaders."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import mido
import pyarrow as pa
import pyarrow.compute as pc

from timetoalign.core import TimeUnit
from timetoalign.loader import EventLoader
from timetoalign.loader.base import LoadSourceResult

from .constants import MidiEventType
from .events import MidiEventData

if TYPE_CHECKING:
    from timetoalign.loader.midi.store import MidiStore


class MidiLoader(EventLoader):
    """Abstract base class for MIDI loaders.

    Subclasses must implement:
    - _load_source(): Parse MIDI file into event rows
    - ticks_per_beat: Property returning PPQ
    """

    _default_unit: ClassVar[TimeUnit] = TimeUnit.ticks
    _event_data_class = MidiEventData

    @property
    @abstractmethod
    def ticks_per_beat(self) -> int | None:
        """Return the ticks per beat (PPQ) if available."""
        ...

    @property
    def events(self) -> MidiEventData:
        """The MidiEventData containing all loaded events."""
        return cast(MidiEventData, self._events)

    def _parse_mido_source(
        self,
        source: Path,
        *,
        parse_durations: bool,
        include_controls: bool,
        include_program_changes: bool,
        on0_means_off: bool,
    ) -> LoadSourceResult:
        """Parse a MIDI file's event stream with mido.

        Args:
            source: Path to the MIDI file.
            parse_durations: Whether to pair note-on and note-off messages.
            include_controls: Whether to emit control-change messages.
            include_program_changes: Whether to emit program-change messages.
            on0_means_off: Whether zero-velocity note-ons end active notes.

        Returns:
            Tuple of MIDI metadata and vectorized event field arrays.
        """
        try:
            mid = mido.MidiFile(source)
        except (OSError, EOFError, ValueError) as e:
            # mido raises diverse exceptions for invalid files
            raise ValueError(f"Invalid MIDI file {source}: {e}") from e

        self._ticks_per_beat = mid.ticks_per_beat

        events = []

        for i, track in enumerate(mid.tracks):
            absolute_time = 0
            # active_notes: (channel, pitch) -> event_dict
            active_notes: dict[tuple[int, int], dict[str, Any]] = {}

            for msg in track:
                absolute_time += msg.time

                # Handle Note Events
                if msg.type == "note_on" or msg.type == "note_off":
                    channel = msg.channel
                    note = msg.note
                    velocity = msg.velocity

                    is_note_on = msg.type == "note_on" and velocity > 0
                    is_note_off = msg.type == "note_off" or (
                        on0_means_off and msg.type == "note_on" and velocity == 0
                    )

                    if is_note_on:
                        # Start of a note
                        note_event = {
                            "id": f"n{i}_{absolute_time}_{note}_{channel}",
                            "temporal_type": "interval",
                            "event_type": MidiEventType.NOTE,
                            "start": absolute_time,
                            "instant": None,
                            "end": None,  # Will be filled by note_off
                            "duration": None,
                            "pitch": note,
                            "velocity": velocity,
                            "channel": channel,
                            "track": i,
                            "control": None,
                            "value": None,
                            "program": None,
                        }
                        if parse_durations:
                            active_notes[(channel, note)] = note_event
                        else:
                            # If not parsing durations, store as instant event (unusual but supported)
                            note_event["temporal_type"] = "instant"
                            note_event["instant"] = absolute_time
                            events.append(note_event)

                    elif is_note_off:
                        if parse_durations:
                            if (channel, note) in active_notes:
                                note_event = active_notes.pop((channel, note))
                                note_event["end"] = absolute_time
                                note_event["duration"] = (
                                    absolute_time - note_event["start"]
                                )
                                events.append(note_event)
                            else:
                                # Orphaned note_off - ignore or warn?
                                # For now, ignore to match legacy behavior
                                pass

                # Handle Control Changes
                elif msg.type == "control_change" and include_controls:
                    cc_event = {
                        "id": f"cc{i}_{absolute_time}_{msg.control}",
                        "temporal_type": "instant",
                        "event_type": MidiEventType.CONTROL_CHANGE,
                        "start": None,
                        "instant": absolute_time,
                        "end": None,
                        "duration": None,
                        "control": msg.control,
                        "value": msg.value,
                        "channel": msg.channel,
                        "track": i,
                        # Other fields null
                        "pitch": None,
                        "velocity": None,
                        "program": None,
                    }
                    events.append(cc_event)

                # Handle Program Changes
                elif msg.type == "program_change" and include_program_changes:
                    pc_event = {
                        "id": f"pc{i}_{absolute_time}",
                        "temporal_type": "instant",
                        "event_type": MidiEventType.PROGRAM_CHANGE,
                        "start": None,
                        "instant": absolute_time,
                        "end": None,
                        "duration": None,
                        "program": msg.program,
                        "channel": msg.channel,
                        "track": i,
                        # Other fields null
                        "pitch": None,
                        "velocity": None,
                        "control": None,
                        "value": None,
                    }
                    events.append(pc_event)

            # Check for unclosed notes at end of track
            if parse_durations and active_notes:
                warnings.warn(
                    f"{len(active_notes)} events on track {i} in {source.name} "
                    f"have not been ended by note_off. Duration set to 0."
                )
                for note_event in active_notes.values():
                    note_event["end"] = absolute_time
                    note_event["duration"] = absolute_time - note_event["start"]
                    events.append(note_event)

        metadata = {
            "format": "midi",
            "parser": "mido",
            "type": mid.type,
            "ticks_per_beat": mid.ticks_per_beat,
            "num_tracks": len(mid.tracks),
            "length_seconds": mid.length,
        }

        if not events:
            return metadata, {}
        table = pa.Table.from_pylist(events)
        fields = {name: table.column(name) for name in table.column_names}
        if "instant" in fields:
            starts = fields.get(
                "start",
                pa.chunked_array([pa.nulls(len(events), type=fields["instant"].type)]),
            )
            instants = fields["instant"]
            if pa.types.is_null(instants.type):
                fields["start"] = starts
            elif pa.types.is_null(starts.type):
                fields["start"] = instants
            else:
                fields["start"] = pc.coalesce(starts, instants)
            del fields["instant"]
        return metadata, fields

    @property
    def store(self) -> "MidiStore":
        """Return a MidiStore wrapping the loader's events.

        The single MidiEventData is split into notes and controls
        for consistent interface with ScoreStore.

        Returns:
            A MidiStore with notes and controls separated.
        """
        from timetoalign.loader.midi.store import MidiStore

        return MidiStore.from_data(self.events, metadata=self.metadata)
