"""MidiStore: Container for MIDI EventData split by event category.

This module provides a store class that separates MIDI events into
notes and control events, mirroring the ScoreStore structure.

Design principles:
- Consistent interface with ScoreStore
- Automatic splitting of MidiEventData by event_type
- Support for both performance and score MIDI
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from timetoalign.core.ids import CMAP_PREFIX, IdMinter
from timetoalign.loader.midi.events import MidiEventData
from timetoalign.storage.store import EventStore

if TYPE_CHECKING:
    from timetoalign.maps import ConversionMap

# Event types that belong to "notes" category
NOTE_EVENT_TYPES: frozenset[str] = frozenset({"Note"})

# Event types that belong to "controls" category
CONTROL_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "ControlChange",
        "ProgramChange",
        "PitchBend",
    }
)

# Store names in canonical order
STORE_NAMES: tuple[str, ...] = ("notes", "controls")


@dataclass
class MidiStore(EventStore):
    """Container for MIDI data organized by category.

    Splits a single MidiEventData into notes and control events
    for consistent store interface with ScoreStore. This enables
    uniform timeline creation across all loader types.

    Attributes:
        notes: MidiEventData containing Note events.
        controls: MidiEventData containing ControlChange, ProgramChange,
            PitchBend, and other control events.
        metadata: Source metadata (ticks_per_beat, format, etc.).

    Examples:
        >>> # Create from a loaded MidiEventData
        >>> store = MidiStore.from_data(loader.events)

        >>> # Access data
        >>> print(f"Notes: {len(store.notes)}")
        >>> print(f"Controls: {len(store.controls)}")

        >>> # Create timeline
        >>> timeline = store.create_timeline(uid="midi_score")
    """

    notes: MidiEventData
    controls: MidiEventData
    metadata: dict[str, Any] = field(default_factory=dict)

    # region Class Methods

    @classmethod
    def empty(cls) -> MidiStore:
        """Create an empty MidiStore with empty data.

        Returns:
            A MidiStore with empty notes and controls data.
        """
        from timetoalign.core import TimeUnit

        return cls(
            notes=MidiEventData.empty(TimeUnit.ticks),
            controls=MidiEventData.empty(TimeUnit.ticks),
            metadata={},
        )

    @classmethod
    def from_data(
        cls,
        data: MidiEventData,
        metadata: dict[str, Any] | None = None,
    ) -> MidiStore:
        """Create a MidiStore by splitting a single MidiEventData.

        Separates Note events from control events (ControlChange,
        ProgramChange, PitchBend) into separate data.

        Args:
            data: The source MidiEventData containing all events.
            metadata: Optional metadata to attach to the store.

        Returns:
            A MidiStore with notes and controls separated.

        Examples:
            >>> loader = PerformanceMidiLoader()
            >>> loader.load("piece.mid")
            >>> store = MidiStore.from_data(loader.events)
        """
        # Filter notes
        notes = data.filter(event_type="Note")

        # Filter controls: accumulate all control event types.  Use the
        # incoming data's concrete class so the wider score-MIDI schema
        # is preserved when present; ``MidiEventData.filter`` already
        # honours ``self.__class__`` for the notes branch above.
        concrete_cls = type(data)
        controls = concrete_cls.empty(data.unit, data.number_type)
        for event_type in CONTROL_EVENT_TYPES:
            filtered = data.filter(event_type=event_type)
            if len(filtered) > 0:
                controls = controls.concat(filtered)

        return cls(
            notes=notes,
            controls=controls,
            metadata=metadata or {},
        )

    # endregion

    # region Instance Methods

    def extend(self, other: MidiStore) -> None:
        """Extend this store with another store's data.

        Args:
            other: The MidiStore to merge into this one.
        """
        self.notes = self.notes.concat(other.notes)
        self.controls = self.controls.concat(other.controls)
        self.metadata.update(other.metadata)

    def summary(self) -> dict[str, Any]:
        """Extend the store summary with MIDI-level facts.

        Returns:
            Summary with table information and MIDI metadata facts.
        """
        result = super().summary()
        result["facts"].update(self.metadata)
        return result

    def get_cmaps(self, *, minter: IdMinter | None = None) -> dict[str, ConversionMap]:
        """Get ConversionMaps derivable from MIDI store metadata.

        Returns C-Maps based on available metadata:
        - "quarters": ticks -> quarters (if ticks_per_beat available)
        - "seconds": ticks -> seconds (if tempo events available)

        Args:
            minter: The minter of the ticks -> quarters map's ID (a
                loader's); without one, the map mints its own. The
                ticks -> seconds map mints its own.

        Returns:
            Dict mapping target unit name to ConversionMap.

        Examples:
            >>> store = midi_loader.load("performance.mid")
            >>> cmaps = store.get_cmaps()
            >>> quarters_map = cmaps.get("quarters")
            >>> if quarters_map:
            ...     quarters = quarters_map(960)  # Fraction(2, 1) at 480 PPQ
        """
        from timetoalign.maps import TableMap
        from timetoalign.maps.convenience import TicksToQuarters

        cmaps: dict[str, ConversionMap] = {}

        # Get PPQ from metadata
        ppq = self.metadata.get("ticks_per_beat")
        if ppq is None:
            return cmaps

        # Always available if we have PPQ: ticks -> quarters, exact
        cmaps["quarters"] = TicksToQuarters(
            ppq=ppq, uid=None if minter is None else minter.mint(CMAP_PREFIX)
        )

        # Check for tempo events to build ticks -> seconds map
        # Tempo events would be stored in controls as TempoChange events
        # with 'tempo_bpm' or similar field
        # For now, check if we have tempo data in metadata
        tempo_events = self._extract_tempo_events()
        if tempo_events:
            tick_positions = [t for t, _ in tempo_events]
            tempos_bpm = [bpm for _, bpm in tempo_events]
            cmaps["seconds"] = TableMap.from_tempo_changes(
                tick_positions=tick_positions,
                tempos_bpm=tempos_bpm,
                ticks_per_quarter=ppq,
                source_unit="ticks",
                target_unit="seconds",
            )

        return cmaps

    def _extract_tempo_events(self) -> list[tuple[int, float]]:
        """Extract tempo events from controls data or metadata.

        Returns:
            List of (tick_position, bpm) tuples, sorted by tick position.
            Empty list if no tempo information available.
        """
        # First check metadata for pre-extracted tempo events
        if "tempo_events" in self.metadata:
            return self.metadata["tempo_events"]

        # TODO: Extract from controls data if TempoChange events are stored there
        # For now, return empty list - tempo extraction will be added when
        # MIDI loaders are enhanced to capture set_tempo messages

        return []

    # endregion

    # region Magic Methods

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MidiStore(notes={len(self.notes)}, controls={len(self.controls)})"

    # endregion

    # region EventStore Protocol

    def __iter__(self) -> Iterator[MidiEventData]:
        """Iterate over data in canonical order.

        Yields:
            MidiEventData: notes, then controls.
        """
        yield self.notes
        yield self.controls

    def __len__(self) -> int:
        """Return number of data (always 2)."""
        return len(STORE_NAMES)

    def __getitem__(self, name: str) -> MidiEventData:
        """Get data by name.

        Args:
            name: Data name ("notes" or "controls").

        Returns:
            The MidiEventData for that category.

        Raises:
            KeyError: If name is not valid.
        """
        if name == "notes":
            return self.notes
        elif name == "controls":
            return self.controls
        raise KeyError(f"Unknown data: {name!r}. Valid: {STORE_NAMES}")

    def keys(self) -> tuple[str, ...]:
        """Return data names in canonical order.

        Returns:
            ("notes", "controls")
        """
        return STORE_NAMES

    def items(self) -> Iterator[tuple[str, MidiEventData]]:
        """Iterate over (name, data) pairs.

        Yields:
            Tuples of (name, MidiEventData) in canonical order.
        """
        yield ("notes", self.notes)
        yield ("controls", self.controls)

    def values(self) -> Iterator[MidiEventData]:
        """Iterate over data.

        Yields:
            MidiEventData in canonical order.
        """
        yield self.notes
        yield self.controls

    # endregion
