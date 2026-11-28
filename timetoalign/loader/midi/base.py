"""MidiLoader: Abstract base class for MIDI loaders."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from timetoalign.core import TimeUnit
from timetoalign.loader import EventLoader
from timetoalign.loader._optional import missing_dependency
from timetoalign.loader._sources import Source, describe_source
from timetoalign.loader.base import LoadSourceResult

from .constants import MidiEventType
from .events import MidiEventData

if TYPE_CHECKING:
    import mido

    from timetoalign.loader.midi.store import MidiStore
    from timetoalign.timelines import Timeline


def _read_midi_file(source: Source) -> mido.MidiFile:
    """Parse the MIDI file *source* with mido, from its open file object.

    A path and a member of a ZIP archive are both opened in binary mode and
    parsed in place (mido parses a file it opens itself the same way). The
    ``filename`` mido records, which Partitura takes as the score's id, is
    the source as messages name it.

    Args:
        source: The MIDI file: a path or a ZIP archive member.

    Returns:
        The parsed file.

    Raises:
        ImportError: If mido is not installed (the caller names the extra).
        OSError, EOFError, ValueError: If mido cannot read the file.
    """
    import mido

    with source.open("rb") as handle:
        return mido.MidiFile(filename=describe_source(source), file=handle)


def _end_of_track(mid: mido.MidiFile) -> int:
    """The tick at which the last track of *mid* ends.

    A track ends at its last message, the ``end_of_track`` meta message a
    valid file closes it with: the sum of the track's delta times. The file
    ends where its latest track ends; a file without tracks ends at 0.
    """
    return max(
        (sum(message.time for message in track) for track in mid.tracks), default=0
    )


def _stated_meter_and_tempo(mid: mido.MidiFile) -> dict[str, list[tuple[int, ...]]]:
    """The tempo changes and time signatures *mid* states, by tick.

    Every ``set_tempo`` and ``time_signature`` meta message of any track is
    read at its absolute tick, the integers as the file states them:
    ``tempo_events`` holds ``(tick, microseconds_per_quarter)`` pairs,
    ``time_signatures`` ``(tick, numerator, denominator)`` triples. Both are
    sorted by tick; of two messages at one tick the later one replaces the
    earlier, "later" in reading order (tracks in file order, each track's
    messages in its order), as a sequencer reading the tracks in turn would
    leave the last one in force.

    Args:
        mid: The parsed MIDI file.

    Returns:
        ``{"tempo_events": [...], "time_signatures": [...]}``.
    """
    tempi: dict[int, tuple[int, ...]] = {}
    signatures: dict[int, tuple[int, ...]] = {}
    for track in mid.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type == "set_tempo":
                tempi[tick] = (tick, int(message.tempo))
            elif message.type == "time_signature":
                signatures[tick] = (
                    tick,
                    int(message.numerator),
                    int(message.denominator),
                )
    return {
        "tempo_events": [tempi[tick] for tick in sorted(tempi)],
        "time_signatures": [signatures[tick] for tick in sorted(signatures)],
    }


#: The facts about its time axis each MIDI source's metadata records.
_AXIS_FACTS = ("ticks_per_beat", "end_of_track", "tempo_events", "time_signatures")


class MidiLoader(EventLoader):
    """Abstract base class for MIDI loaders.

    Subclasses must implement:
    - _load_source(): Parse MIDI file into event rows
    - ticks_per_beat: Property returning PPQ

    Every parse records, in the source's metadata, the facts the file states
    about its time axis: ``ticks_per_beat`` (the header's resolution),
    ``end_of_track`` (:attr:`end_of_track`), ``tempo_events`` and
    ``time_signatures`` (see :func:`_stated_meter_and_tempo`). The timeline
    :meth:`create_timeline` builds stays on ticks and carries them as its
    conversion maps and its measure map (:meth:`MidiStore.create_timeline`).
    """

    _default_unit: ClassVar[TimeUnit] = TimeUnit.ticks
    _event_data_class = MidiEventData

    #: The end of track of the file read last, in ticks; None before a file is read.
    _end_of_track: int | None = None

    @property
    @abstractmethod
    def ticks_per_beat(self) -> int | None:
        """Return the ticks per beat (PPQ) if available."""
        ...

    @property
    def end_of_track(self) -> int | None:
        """The end of track of the MIDI file read last, in ticks.

        The tick of the ``end_of_track`` message closing the file's latest
        track: the extent the file states, which may lie after the last
        note's end (MIDI files often close a track one tick after it) and
        never before it. In quarter notes it is
        ``Fraction(end_of_track, ticks_per_beat)``, exactly. Each source's
        metadata records its own (``metadata["sources"][i]["end_of_track"]``).

        Returns:
            The tick, or None before a file is read.
        """
        return self._end_of_track

    @property
    def events(self) -> MidiEventData:
        """The MidiEventData containing all loaded events."""
        return cast(MidiEventData, self._events)

    def clear(self) -> None:
        """Clear all loaded sources and events, and the file read last's state.

        :attr:`ticks_per_beat` and :attr:`end_of_track` answer None again,
        as before a file is read.
        """
        super().clear()
        self._ticks_per_beat = None
        self._end_of_track = None

    def _parse_mido_source(
        self,
        source: Source,
        *,
        parse_durations: bool,
        include_controls: bool,
        include_program_changes: bool,
        on0_means_off: bool,
    ) -> LoadSourceResult:
        """Parse a MIDI file's event stream with mido.

        Args:
            source: The MIDI file: a path or a ZIP archive member.
            parse_durations: Whether to pair note-on and note-off messages.
            include_controls: Whether to emit control-change messages.
            include_program_changes: Whether to emit program-change messages.
            on0_means_off: Whether zero-velocity note-ons end active notes.

        Returns:
            Tuple of MIDI metadata and vectorized event field arrays.

        Raises:
            ImportError: If ``mido`` is not installed, naming the extra.
            ValueError: If the file is no valid MIDI file.
        """
        # The missing extra is named before any file is read.
        try:
            import mido  # noqa: F401
        except ModuleNotFoundError as error:
            raise ImportError(
                missing_dependency(type(self).__name__, error.name, "midi")
            ) from error
        try:
            mid = _read_midi_file(source)
        except (OSError, EOFError, ValueError) as e:
            # mido raises diverse exceptions for invalid files
            raise ValueError(f"Invalid MIDI file {describe_source(source)}: {e}") from e

        self._ticks_per_beat = mid.ticks_per_beat
        self._end_of_track = _end_of_track(mid)

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
            "end_of_track": self._end_of_track,
            **_stated_meter_and_tempo(mid),
            "num_tracks": len(mid.tracks),
            "length_seconds": mid.length,
        }

        if not events:
            return metadata, {}
        table = pa.Table.from_pylist(events)
        columns = {name: table.column(name) for name in table.column_names}
        if "instant" in columns:
            starts = columns.get(
                "start",
                pa.chunked_array([pa.nulls(len(events), type=columns["instant"].type)]),
            )
            instants = columns["instant"]
            if pa.types.is_null(instants.type):
                columns["start"] = starts
            elif pa.types.is_null(starts.type):
                columns["start"] = instants
            else:
                columns["start"] = pc.coalesce(starts, instants)
            del columns["instant"]
        fields: dict[str, np.ndarray | pa.Array | pa.ChunkedArray] = dict(columns)
        return metadata, fields

    def _stated_axis(self) -> dict[str, Any]:
        """The time-axis facts the loaded sources state, for one timeline.

        One source states its own. Several, whose events share one ticks
        axis, state a fact when every source states it alike; the end of
        track is the latest. A fact the sources state differently is left
        out, and one ``UserWarning`` names each: a resolution, tempo map
        or meter of one file is no fact of the others.

        Returns:
            The facts of :data:`_AXIS_FACTS` the sources agree on.

        Warns:
            UserWarning: If the sources state a fact differently.
        """
        sources = self._source_metadata
        if not sources:
            return {}
        axis: dict[str, Any] = {}
        ends = [meta["end_of_track"] for meta in sources if "end_of_track" in meta]
        if ends:
            axis["end_of_track"] = max(ends)
        for key in _AXIS_FACTS:
            if key == "end_of_track":
                continue
            values = [meta.get(key) for meta in sources]
            if all(value == values[0] for value in values):
                if values[0] is not None:
                    axis[key] = values[0]
                continue
            warnings.warn(
                f"{type(self).__name__} holds {len(sources)} sources that state "
                f"different {key}; the timeline carries none of them",
                UserWarning,
                stacklevel=3,
            )
        return axis

    @property
    def store(self) -> "MidiStore":
        """Return a MidiStore wrapping the loader's events.

        The single MidiEventData is split into notes and controls
        for consistent interface with ScoreStore. The store's metadata
        holds the loader's (:attr:`metadata`) and, at its top level, the
        time-axis facts the sources state (``ticks_per_beat``,
        ``end_of_track``, ``tempo_events``, ``time_signatures``), from
        which it builds the timeline's conversion maps and measure map.

        Returns:
            A MidiStore with notes and controls separated.

        Warns:
            UserWarning: If several sources state a fact of the axis
                differently (the store then holds none for it).
        """
        from timetoalign.loader.midi.store import MidiStore

        return MidiStore.from_data(
            self.events, metadata={**self.metadata, **self._stated_axis()}
        )

    def create_timeline(
        self,
        uid: str | None = None,
        store_filters: dict[str, dict[str, Any]] | None = None,
        include_stores: list[str] | None = None,
        exclude_stores: list[str] | None = None,
        flatten: bool = False,
        *,
        attach_cmaps: bool = True,
    ) -> Timeline:
        """Create the ticks timeline of the loaded file, with its stated axis.

        The timeline stays on ticks, the file's own unit, and reaches the
        units the file states through what it states
        (:meth:`MidiStore.create_timeline`): quarters at the file's
        resolution, seconds from its tempo map, and its bars as a measure
        map. Every ID it mints comes from the loader's minter.

        Args:
            uid: Unique ID for the parent timeline. Auto-generated if None.
            store_filters: Per-data filter kwargs to apply before timeline
                creation.
            include_stores: Only include these data (default: all non-empty).
            exclude_stores: Exclude these data from the timeline.
            flatten: If True, merge all events into a single parent timeline.
            attach_cmaps: If True (default), attach the conversion maps and
                the measure map the file states; if False, none.

        Returns:
            The ticks timeline.

        Warns:
            UserWarning: If several sources state a fact of the axis
                differently (see :attr:`store`).
        """
        return self.store.create_timeline(
            uid=uid,
            store_filters=store_filters,
            include_stores=include_stores,
            exclude_stores=exclude_stores,
            flatten=flatten,
            attach_cmaps=attach_cmaps,
            minter=self._id_minter,
        )
