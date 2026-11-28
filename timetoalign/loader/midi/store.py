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
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from timetoalign.core.enums import ExtrapolationPolicy, InterpolationKind, TimeUnit
from timetoalign.core.ids import CMAP_PREFIX, IdMinter
from timetoalign.core.time import number_to_json
from timetoalign.loader.midi.events import MidiEventData
from timetoalign.storage.store import EventStore

if TYPE_CHECKING:
    from timetoalign.maps import ConversionMap, TableMap
    from timetoalign.timelines import MeasureMap, Timeline

#: The tempo of a Standard MIDI File before its first ``set_tempo``, in
#: microseconds per quarter (120 quarters a minute).
_SMF_DEFAULT_TEMPO = 500_000

#: The time signature of a Standard MIDI File before its first ``time_signature``.
_SMF_DEFAULT_TIME_SIGNATURE = "4/4"

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

    def create_timeline(
        self,
        uid: str | None = None,
        store_filters: dict[str, dict[str, Any]] | None = None,
        include_stores: list[str] | None = None,
        exclude_stores: list[str] | None = None,
        flatten: bool = False,
        attach_cmaps: bool = True,
        *,
        minter: IdMinter | None = None,
    ) -> Timeline:
        """Create the ticks timeline of the MIDI data, with the axis it states.

        The timeline stays on ticks. With *attach_cmaps*, it carries the
        conversion maps of :meth:`get_cmaps` and the bars of the file as its
        measure map (:meth:`_measure_map`), which the ticks axis reaches
        through a ticks → quarters ``LinearMap`` of its own: the timeline
        writes ``measure_maps`` and the ``ChainMap`` reading them in its
        manifest. Its length stays what its events give; the measure map
        may reach beyond it, to the end of track.

        Args:
            uid: Unique ID for the parent timeline. Auto-generated if None.
            store_filters: Per-data filter kwargs to apply before timeline
                creation.
            include_stores: Only include these data (default: all non-empty).
            exclude_stores: Exclude these data from the timeline.
            flatten: If True, merge all events into a single parent timeline.
            attach_cmaps: If True (default), attach the conversion maps and
                the measure map; if False, none.
            minter: The minter of every timeline, map and measure map ID the
                call mints (a loader's); without one, each mints its own.

        Returns:
            The ticks timeline.
        """
        from timetoalign.maps.convenience import TicksToQuarters
        from timetoalign.timelines.factory import create_timeline_from_bundle

        timeline = create_timeline_from_bundle(
            self,
            uid=uid,
            store_filters=store_filters,
            include_stores=include_stores,
            exclude_stores=exclude_stores,
            flatten=flatten,
            minter=minter,
        )
        if not attach_cmaps:
            return timeline
        for cmap in self.get_cmaps(minter=minter).values():
            timeline.add_conversion_map(cmap)
        measure_map = self._measure_map(minter=minter)
        if measure_map is not None:
            # A map of its own: one map id is held once in a manifest.
            to_quarters = TicksToQuarters(
                ppq=self.metadata["ticks_per_beat"],
                uid=None if minter is None else minter.mint(CMAP_PREFIX),
            )
            timeline.add_measure_map(measure_map, to_quarters=to_quarters)
        return timeline

    def get_cmaps(self, *, minter: IdMinter | None = None) -> dict[str, ConversionMap]:
        """Get the ConversionMaps the MIDI file states about its ticks axis.

        - ``"quarters"``: ticks → quarters, ``TicksToQuarters`` at the file's
          resolution (``ticks_per_beat``), exact.
        - ``"seconds"``: ticks → seconds from the tempo map, a linear
          ``TableMap`` (:meth:`_tempo_map`); it needs the resolution and the
          end of track (``end_of_track``).

        Args:
            minter: The minter of the maps' IDs (a loader's); without one,
                each map mints its own.

        Returns:
            Dict mapping target unit name to ConversionMap; empty without a
            resolution.

        Examples:
            >>> store = loader.store  # a MIDI loader holding a file
            >>> quarters_map = store.get_cmaps()["quarters"]
            >>> quarters_map(960)  # at 480 ticks per quarter
            Fraction(2, 1)
        """
        from timetoalign.maps.convenience import TicksToQuarters

        cmaps: dict[str, ConversionMap] = {}
        ppq = self.metadata.get("ticks_per_beat")
        if ppq is None:
            return cmaps
        cmaps["quarters"] = TicksToQuarters(
            ppq=ppq, uid=None if minter is None else minter.mint(CMAP_PREFIX)
        )
        tempo_map = self._tempo_map(minter=minter)
        if tempo_map is not None:
            cmaps["seconds"] = tempo_map
        return cmaps

    def _tempo_map(self, *, minter: IdMinter | None = None) -> TableMap | None:
        """The ticks → seconds map of the file's tempo map, exact.

        The knots are tick 0, every tick of ``tempo_events`` before the end
        of track, and the end of track; the seconds at each are the exact
        ``Fraction`` sum over the segments before it of
        Δticks · µs / (ticks_per_beat · 10⁶), each segment at the tempo of
        the last ``set_tempo`` at or before its start. Before the first
        ``set_tempo``, and in a file without one, the tempo is the
        Standard MIDI File default of 500,000 µs per quarter (120 quarters
        a minute). A ``set_tempo`` at the end of track governs no tick of
        the file and gives no knot. Between knots the map interpolates
        linearly; past the end of track (and before tick 0) it extrapolates
        linearly with the tempo of the segment it leaves, which is the tempo
        in force at the end of track.

        Args:
            minter: The minter of the map's ID; without one, the map mints
                its own.

        Returns:
            The map, or None without a resolution, an end of track after
            tick 0 (a file without extent spans no tempo segment) or
            ``tempo_events`` (which a file states even when empty; a store
            of several files whose tempo maps differ holds none).
        """
        from timetoalign.maps import TableMap

        ppq = self.metadata.get("ticks_per_beat")
        end = self.metadata.get("end_of_track")
        tempo_events = self.metadata.get("tempo_events")
        if ppq is None or end is None or end <= 0 or tempo_events is None:
            return None
        tempo = _SMF_DEFAULT_TEMPO
        knots: list[int] = [0]
        seconds: list[Fraction] = [Fraction(0)]
        for tick, microseconds in tempo_events:
            if tick >= end:
                break
            if tick > knots[-1]:
                seconds.append(
                    seconds[-1] + Fraction((tick - knots[-1]) * tempo, ppq * 10**6)
                )
                knots.append(tick)
            tempo = microseconds
        seconds.append(seconds[-1] + Fraction((end - knots[-1]) * tempo, ppq * 10**6))
        knots.append(end)
        return TableMap(
            x_values=knots,
            y_values=seconds,
            kind=InterpolationKind.linear,
            extrapolate=ExtrapolationPolicy.extrapolate,
            source_unit=TimeUnit.ticks,
            target_unit=TimeUnit.seconds,
            uid=None if minter is None else minter.mint(CMAP_PREFIX),
        )

    def _measure_map(self, *, minter: IdMinter | None = None) -> MeasureMap | None:
        """The file's bars, from its time signatures and its end of track.

        Bars run from tick 0, each of its signature's length (numerator · 4
        / denominator quarters). Every ``time_signature`` message starts a
        new bar, so a bar it interrupts keeps the length it actually has;
        bars follow until the end of track, which may cut the last one
        short, and a bar that would start at the end of track is not
        counted. Before the first ``time_signature``, and in a file without
        one, the signature is 4/4, the Standard MIDI File default. Each row
        states its count (from 1), its signature as the file writes it (6/8
        stays 6/8) and, where it is cut short, its actual length; an SMF
        states no printed measure number, so the row states none and the
        Measure Map default applies.

        Args:
            minter: The minter of the measure map's ID; without one, the
                map mints its own from its content.

        Returns:
            The measure map, or None without a resolution, an end of track
            after tick 0 or ``time_signatures`` (which a file states even
            when empty; a store of several files whose signatures differ
            holds none).
        """
        from timetoalign.core.ids import MEASURE_MAP_PREFIX
        from timetoalign.timelines import MeasureMap

        ppq = self.metadata.get("ticks_per_beat")
        end = self.metadata.get("end_of_track")
        signatures = self.metadata.get("time_signatures")
        if ppq is None or end is None or end <= 0 or signatures is None:
            return None
        changes = [
            (Fraction(tick, ppq), f"{numerator}/{denominator}")
            for tick, numerator, denominator in signatures
            if tick < end
        ]
        if not changes or changes[0][0] > 0:
            changes.insert(0, (Fraction(0), _SMF_DEFAULT_TIME_SIGNATURE))
        stop = Fraction(end, ppq)
        rows: list[dict[str, Any]] = []
        for index, (start, signature) in enumerate(changes):
            until = changes[index + 1][0] if index + 1 < len(changes) else stop
            numerator, denominator = (int(part) for part in signature.split("/"))
            nominal = Fraction(4 * numerator, denominator)
            position = start
            while position < until:
                row: dict[str, Any] = {"count": len(rows) + 1}
                if position == start:
                    row["time_signature"] = signature
                actual = min(nominal, until - position)
                if actual != nominal:
                    row["actual_length"] = number_to_json(actual)
                rows.append(row)
                position += actual
        return MeasureMap.from_rows(
            rows, id=None if minter is None else minter.mint(MEASURE_MAP_PREFIX)
        )

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
