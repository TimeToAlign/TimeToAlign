"""Named convenience classes for common unit conversions.

These classes wrap ScalarMap with preset source/target units and
provide a more ergonomic API for common conversions.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from ..core.enums import TimeUnit
from .linear import ScalarMap

# region Logical Domain: Ticks <-> Quarters


class TicksToQuarters(ScalarMap):
    """Convert MIDI ticks to quarter notes by dividing by PPQ.

    This is a simple linear map: quarters = ticks / ppq

    Attributes:
        ppq: Pulses (ticks) per quarter note.

    Examples:
        An exact input converts exactly — the map divides by a ``Fraction``,
        so a tick position never picks up a rounding error on the way to
        quarters:

        >>> t2q = TicksToQuarters(ppq=480)
        >>> t2q(480)
        Fraction(1, 1)
        >>> t2q(960)
        Fraction(2, 1)

        A float input is taken at face value and answers in float:

        >>> q2t = t2q.inverse()
        >>> q2t(2.0)
        960.0
    """

    _default_source_unit = TimeUnit.ticks
    _default_target_unit = TimeUnit.quarters

    def __init__(self, ppq: int = 480, **kwargs: Any) -> None:
        """Initialize TicksToQuarters.

        Args:
            ppq: Pulses per quarter note. Defaults to 480.
            **kwargs: Additional arguments passed to ScalarMap.
        """
        # Remove scalar from kwargs if present (from inverse() calls)
        kwargs.pop("scalar", None)
        super().__init__(scalar=Fraction(1, ppq), **kwargs)
        self._ppq = ppq

    @property
    def ppq(self) -> int:
        """Pulses per quarter note."""
        return self._ppq

    def inverse(self) -> QuartersToTicks:
        """Return the inverse map (QuartersToTicks)."""
        return QuartersToTicks(ppq=self._ppq)

    def to_dict(self, *, minimal: bool = False) -> dict[str, Any]:
        """Serialize to dictionary."""
        d = super().to_dict(minimal=minimal)
        d["ppq"] = self._ppq
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TicksToQuarters:
        """Deserialize from dictionary."""
        return cls(
            ppq=data.get("ppq", 480),
            uid=data.get("id"),
            name=data.get("name"),
        )

    def __repr__(self) -> str:
        return f"TicksToQuarters(ppq={self._ppq})"


class QuartersToTicks(ScalarMap):
    """Convert quarter notes to MIDI ticks by multiplying by PPQ.

    This is a simple linear map: ticks = quarters * ppq

    Attributes:
        ppq: Pulses (ticks) per quarter note.

    Examples:
        >>> q2t = QuartersToTicks(ppq=480)
        >>> q2t(1.0)
        480.0
        >>> q2t(2.5)
        1200.0

        The inverse takes an exact tick position back to an exact quarter:

        >>> t2q = q2t.inverse()
        >>> t2q(960)
        Fraction(2, 1)
    """

    _default_source_unit = TimeUnit.quarters
    _default_target_unit = TimeUnit.ticks

    def __init__(self, ppq: int = 480, **kwargs: Any) -> None:
        """Initialize QuartersToTicks.

        Args:
            ppq: Pulses per quarter note. Defaults to 480.
            **kwargs: Additional arguments passed to ScalarMap.
        """
        # Remove scalar from kwargs if present (from inverse() calls)
        kwargs.pop("scalar", None)
        super().__init__(scalar=ppq, **kwargs)
        self._ppq = ppq

    @property
    def ppq(self) -> int:
        """Pulses per quarter note."""
        return self._ppq

    def inverse(self) -> TicksToQuarters:
        """Return the inverse map (TicksToQuarters)."""
        return TicksToQuarters(ppq=self._ppq)

    def to_dict(self, *, minimal: bool = False) -> dict[str, Any]:
        """Serialize to dictionary."""
        d = super().to_dict(minimal=minimal)
        d["ppq"] = self._ppq
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuartersToTicks:
        """Deserialize from dictionary."""
        return cls(
            ppq=data.get("ppq", 480),
            uid=data.get("id"),
            name=data.get("name"),
        )

    def __repr__(self) -> str:
        return f"QuartersToTicks(ppq={self._ppq})"


# endregion

# region Physical Domain: Samples <-> Seconds


class SamplesToSeconds(ScalarMap):
    """Convert audio samples to seconds by dividing by sample rate.

    This is a simple linear map: seconds = samples / sample_rate

    Attributes:
        sample_rate: Audio sample rate in Hz.

    Examples:
        A sample index is exact, so the elapsed time comes back exact too:

        >>> s2s = SamplesToSeconds(sample_rate=44100)
        >>> s2s(44100)
        Fraction(1, 1)
        >>> s2s(88200)
        Fraction(2, 1)

        A float input answers in float:

        >>> sec2samp = s2s.inverse()
        >>> sec2samp(1.0)
        44100.0
    """

    _default_source_unit = TimeUnit.samples
    _default_target_unit = TimeUnit.seconds

    def __init__(self, sample_rate: int = 44100, **kwargs: Any) -> None:
        """Initialize SamplesToSeconds.

        Args:
            sample_rate: Audio sample rate in Hz. Defaults to 44100.
            **kwargs: Additional arguments passed to ScalarMap.
        """
        # Remove scalar from kwargs if present (from inverse() calls)
        kwargs.pop("scalar", None)
        super().__init__(scalar=Fraction(1, sample_rate), **kwargs)
        self._sample_rate = sample_rate

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        return self._sample_rate

    def inverse(self) -> SecondsToSamples:
        """Return the inverse map (SecondsToSamples)."""
        return SecondsToSamples(sample_rate=self._sample_rate)

    def to_dict(self, *, minimal: bool = False) -> dict[str, Any]:
        """Serialize to dictionary."""
        d = super().to_dict(minimal=minimal)
        d["sample_rate"] = self._sample_rate
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SamplesToSeconds:
        """Deserialize from dictionary."""
        return cls(
            sample_rate=data.get("sample_rate", 44100),
            uid=data.get("id"),
            name=data.get("name"),
        )

    def __repr__(self) -> str:
        return f"SamplesToSeconds(sample_rate={self._sample_rate})"


class SecondsToSamples(ScalarMap):
    """Convert seconds to audio samples by multiplying by sample rate.

    This is a simple linear map: samples = seconds * sample_rate

    Attributes:
        sample_rate: Audio sample rate in Hz.

    Examples:
        >>> s2s = SecondsToSamples(sample_rate=44100)
        >>> s2s(1.0)
        44100.0
        >>> s2s(2.5)
        110250.0

        The inverse takes an exact sample index back to an exact time:

        >>> samp2sec = s2s.inverse()
        >>> samp2sec(44100)
        Fraction(1, 1)
    """

    _default_source_unit = TimeUnit.seconds
    _default_target_unit = TimeUnit.samples

    def __init__(self, sample_rate: int = 44100, **kwargs: Any) -> None:
        """Initialize SecondsToSamples.

        Args:
            sample_rate: Audio sample rate in Hz. Defaults to 44100.
            **kwargs: Additional arguments passed to ScalarMap.
        """
        # Remove scalar from kwargs if present (from inverse() calls)
        kwargs.pop("scalar", None)
        super().__init__(scalar=sample_rate, **kwargs)
        self._sample_rate = sample_rate

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        return self._sample_rate

    def inverse(self) -> SamplesToSeconds:
        """Return the inverse map (SamplesToSeconds)."""
        return SamplesToSeconds(sample_rate=self._sample_rate)

    def to_dict(self, *, minimal: bool = False) -> dict[str, Any]:
        """Serialize to dictionary."""
        d = super().to_dict(minimal=minimal)
        d["sample_rate"] = self._sample_rate
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SecondsToSamples:
        """Deserialize from dictionary."""
        return cls(
            sample_rate=data.get("sample_rate", 44100),
            uid=data.get("id"),
            name=data.get("name"),
        )

    def __repr__(self) -> str:
        return f"SecondsToSamples(sample_rate={self._sample_rate})"


# endregion
