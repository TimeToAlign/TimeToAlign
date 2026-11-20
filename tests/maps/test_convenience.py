"""Tests for convenience map classes."""

from __future__ import annotations

import numpy as np

from timetoalign.core import TimeUnit
from timetoalign.maps import (
    QuartersToTicks,
    SamplesToSeconds,
    SecondsToSamples,
    TicksToQuarters,
)

# region TicksToQuarters


class TestTicksToQuarters:
    """Tests for TicksToQuarters conversion map."""

    def test_basic_conversion(self) -> None:
        """Test basic tick to quarter conversion."""
        t2q = TicksToQuarters(ppq=480)
        assert t2q(480) == 1.0
        assert t2q(960) == 2.0
        assert t2q(0) == 0.0
        assert t2q(240) == 0.5

    def test_default_ppq(self) -> None:
        """Test default PPQ value."""
        t2q = TicksToQuarters()
        assert t2q.ppq == 480

    def test_units(self) -> None:
        """Test source and target units."""
        t2q = TicksToQuarters(ppq=480)
        assert t2q.source_unit == TimeUnit.ticks
        assert t2q.target_unit == TimeUnit.quarters

    def test_array_conversion(self) -> None:
        """Test array input."""
        t2q = TicksToQuarters(ppq=480)
        ticks = np.array([0, 480, 960, 1440])
        quarters = t2q(ticks)
        np.testing.assert_array_equal(quarters, np.array([0.0, 1.0, 2.0, 3.0]))

    def test_inverse(self) -> None:
        """Test inverse returns QuartersToTicks-like map."""
        t2q = TicksToQuarters(ppq=480)
        q2t = t2q.inverse()
        assert q2t(1.0) == 480.0
        assert q2t(2.0) == 960.0


# endregion

# region QuartersToTicks


class TestQuartersToTicks:
    """Tests for QuartersToTicks conversion map."""

    def test_basic_conversion(self) -> None:
        """Test basic quarter to tick conversion."""
        q2t = QuartersToTicks(ppq=480)
        assert q2t(1.0) == 480.0
        assert q2t(2.0) == 960.0
        assert q2t(0.0) == 0.0
        assert q2t(0.5) == 240.0

    def test_units(self) -> None:
        """Test source and target units."""
        q2t = QuartersToTicks(ppq=480)
        assert q2t.source_unit == TimeUnit.quarters
        assert q2t.target_unit == TimeUnit.ticks

    def test_inverse(self) -> None:
        """Test inverse returns TicksToQuarters-like map."""
        q2t = QuartersToTicks(ppq=480)
        t2q = q2t.inverse()
        assert t2q(480) == 1.0
        assert t2q(960) == 2.0

    def test_roundtrip(self) -> None:
        """Test roundtrip conversion."""
        t2q = TicksToQuarters(ppq=480)
        q2t = QuartersToTicks(ppq=480)

        ticks = 720
        quarters = t2q(ticks)
        back = q2t(quarters)
        # 720 ticks / 480 ppq = 1.5 quarters; 1.5 * 480 = 720.0 exactly
        assert back == ticks


# endregion

# region SamplesToSeconds


class TestSamplesToSeconds:
    """Tests for SamplesToSeconds conversion map."""

    def test_basic_conversion(self) -> None:
        """Test basic sample to seconds conversion."""
        s2s = SamplesToSeconds(sample_rate=44100)
        assert s2s(44100) == 1.0
        assert s2s(88200) == 2.0
        assert s2s(0) == 0.0

    def test_default_sample_rate(self) -> None:
        """Test default sample rate."""
        s2s = SamplesToSeconds()
        assert s2s.sample_rate == 44100

    def test_units(self) -> None:
        """Test source and target units."""
        s2s = SamplesToSeconds(sample_rate=44100)
        assert s2s.source_unit == TimeUnit.samples
        assert s2s.target_unit == TimeUnit.seconds

    def test_48khz(self) -> None:
        """Test with 48kHz sample rate."""
        s2s = SamplesToSeconds(sample_rate=48000)
        assert s2s(48000) == 1.0
        assert s2s.sample_rate == 48000


# endregion

# region SecondsToSamples


class TestSecondsToSamples:
    """Tests for SecondsToSamples conversion map."""

    def test_basic_conversion(self) -> None:
        """Test basic seconds to sample conversion."""
        s2s = SecondsToSamples(sample_rate=44100)
        assert s2s(1.0) == 44100.0
        assert s2s(2.0) == 88200.0
        assert s2s(0.0) == 0.0

    def test_units(self) -> None:
        """Test source and target units."""
        s2s = SecondsToSamples(sample_rate=44100)
        assert s2s.source_unit == TimeUnit.seconds
        assert s2s.target_unit == TimeUnit.samples

    def test_roundtrip(self) -> None:
        """Test roundtrip conversion."""
        s2s = SamplesToSeconds(sample_rate=44100)
        s2samp = SecondsToSamples(sample_rate=44100)

        samples = 22050
        seconds = s2s(samples)
        back = s2samp(seconds)
        # 22050 samples / 44100 Hz = 0.5 seconds; 0.5 * 44100 = 22050.0 exactly
        assert back == samples


# endregion
