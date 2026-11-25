"""PerformanceMidiLoader: Load performance MIDI using mido."""

from __future__ import annotations

from typing import Any

from timetoalign.core import NumberType, TimeUnit
from timetoalign.loader.base import LoadSourceResult, Source

from .base import MidiLoader


class PerformanceMidiLoader(MidiLoader):
    """Load performance MIDI files using mido.

    This loader parses raw MIDI messages, preserving the performance characteristics
    (timing, velocity, control changes) without attempting structural analysis.

    It pairs note_on/note_off events into Note intervals and stores control
    changes as Instant events.
    """

    _validate_vectorized = False

    def __init__(
        self,
        *,
        parse_durations: bool = True,
        on0_means_off: bool = True,
        include_controls: bool = True,
        include_program_changes: bool = True,
        unit: TimeUnit | None = None,
        number_type: NumberType | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize PerformanceMidiLoader.

        Args:
            parse_durations: Whether to pair note_on/off into intervals.
            on0_means_off: Treat note_on with velocity 0 as note_off.
            include_controls: Include Control Change events.
            include_program_changes: Include Program Change events.
            unit: Time unit for coordinates.
            number_type: Number type for coordinates. Defaults to the
                one the unit itself uses.
            **kwargs: Additional arguments passed to parent Loader.
        """
        super().__init__(unit=unit, number_type=number_type, **kwargs)
        self._parse_durations = parse_durations
        self._on0_means_off = on0_means_off
        self._include_controls = include_controls
        self._include_program_changes = include_program_changes
        self._ticks_per_beat: int | None = None

    @property
    def ticks_per_beat(self) -> int | None:
        """Return the ticks per beat (PPQ) of the loaded file."""
        return self._ticks_per_beat

    def _load_source(self, source: Source) -> LoadSourceResult:
        """Load a MIDI file using mido.

        Args:
            source: The MIDI file: a path or a ZIP archive member.

        Returns:
            Tuple of metadata and vectorized event field arrays.
        """
        return self._parse_mido_source(
            source,
            parse_durations=self._parse_durations,
            include_controls=self._include_controls,
            include_program_changes=self._include_program_changes,
            on0_means_off=self._on0_means_off,
        )
