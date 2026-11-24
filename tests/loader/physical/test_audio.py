"""Tests for AudioLoader.

These tests verify that AudioLoader correctly:
1. Extracts metadata from audio files
2. Creates DiscretePhysicalTimelines in samples with a SamplesToSeconds C-map
3. Creates ContinuousPhysicalTimelines in seconds with a SecondsToSamples C-map
4. Handles various audio formats and backends
"""

from __future__ import annotations

import os
import struct
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from timetoalign.core import IdCoordinate, NumberType, TimeUnit
from timetoalign.loader.physical import AudioInfo, AudioLoader
from timetoalign.loader.physical import audio as audio_module
from timetoalign.maps import LinearMap, SamplesToSeconds, SecondsToSamples
from timetoalign.testdata import ensure_data
from timetoalign.timelines import (
    ContinuousPhysicalTimeline,
    DiscretePhysicalTimeline,
    Timeline,
)

BASS_KICK = ensure_data("audio") / "hard_techno" / "Bass Kick.mp3"

# region Test Fixtures


def create_wav_file(
    path: Path,
    n_samples: int = 44100,
    sample_rate: int = 44100,
    channels: int = 1,
    bits_per_sample: int = 16,
    format_tag: int = 1,
    fact: bool = False,
) -> Path:
    """Create a minimal WAV file for testing.

    Args:
        path: Path to write the WAV file.
        n_samples: Number of samples (frames).
        sample_rate: Sample rate in Hz.
        channels: Number of channels.
        bits_per_sample: Bit depth.
        format_tag: Audio format (1 = PCM, 3 = IEEE float).
        fact: Whether to write a ``fact`` chunk stating ``n_samples``.

    Returns:
        Path to the created file.
    """
    bytes_per_sample = bits_per_sample // 8
    data_size = n_samples * channels * bytes_per_sample
    fact_size = 12 if fact else 0

    # WAV file structure
    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + fact_size + data_size))  # File size - 8
        f.write(b"WAVE")

        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # Chunk size
        f.write(struct.pack("<H", format_tag))  # Audio format
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(
            struct.pack("<I", sample_rate * channels * bytes_per_sample)
        )  # Byte rate
        f.write(struct.pack("<H", channels * bytes_per_sample))  # Block align
        f.write(struct.pack("<H", bits_per_sample))

        if fact:
            f.write(b"fact")
            f.write(struct.pack("<I", 4))
            f.write(struct.pack("<I", n_samples))

        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        # Write silence (zeros)
        f.write(b"\x00" * data_size)

    return path


@pytest.fixture
def no_soundfile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report soundfile as not installed, so the loader's next backends answer."""
    monkeypatch.setitem(sys.modules, "soundfile", None)


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    """Create a test WAV file (1 second, 44.1kHz, mono, 16-bit)."""
    return create_wav_file(
        tmp_path / "test.wav",
        n_samples=44100,
        sample_rate=44100,
        channels=1,
        bits_per_sample=16,
    )


@pytest.fixture
def stereo_wav_file(tmp_path: Path) -> Path:
    """Create a stereo test WAV file (2 seconds, 48kHz, stereo, 24-bit)."""
    return create_wav_file(
        tmp_path / "stereo.wav",
        n_samples=96000,
        sample_rate=48000,
        channels=2,
        bits_per_sample=24,
    )


# endregion


# region AudioInfo Tests


class TestAudioInfo:
    """Tests for AudioInfo dataclass."""

    def test_basic_properties(self):
        """Test basic AudioInfo properties."""
        info = AudioInfo(
            n_samples=44100,
            sample_rate=44100,
            channels=1,
            duration_seconds=1.0,
            format="WAV",
        )

        assert info.n_samples == 44100
        assert info.sample_rate == 44100
        assert info.channels == 1
        assert info.duration_seconds == 1.0
        assert info.format == "WAV"

    def test_length_alias(self):
        """Test length_in_samples property is an alias for n_samples."""
        info = AudioInfo(
            n_samples=88200,
            sample_rate=44100,
            channels=2,
            duration_seconds=2.0,
            format="WAV",
        )

        assert info.length_in_samples == info.n_samples
        assert info.length_in_samples == 88200

    def test_mono_stereo_properties(self):
        """Test is_mono and is_stereo properties."""
        mono = AudioInfo(
            n_samples=1000,
            sample_rate=44100,
            channels=1,
            duration_seconds=0.02,
            format="WAV",
        )
        stereo = AudioInfo(
            n_samples=1000,
            sample_rate=44100,
            channels=2,
            duration_seconds=0.02,
            format="WAV",
        )
        surround = AudioInfo(
            n_samples=1000,
            sample_rate=44100,
            channels=6,
            duration_seconds=0.02,
            format="WAV",
        )

        assert mono.is_mono is True
        assert mono.is_stereo is False

        assert stereo.is_mono is False
        assert stereo.is_stereo is True

        assert surround.is_mono is False
        assert surround.is_stereo is False


# endregion


# region AudioLoader Basic Tests


class TestAudioLoaderBasic:
    """Basic tests for AudioLoader that don't require audio files."""

    def test_init(self):
        """Test AudioLoader initialization."""
        loader = AudioLoader()
        assert loader._audio_info is None
        assert loader._source_path is None

    def test_properties_before_load(self):
        """Test that properties raise before loading."""
        loader = AudioLoader()

        with pytest.raises(RuntimeError, match="No audio file loaded"):
            _ = loader.audio_info

        with pytest.raises(RuntimeError, match="No audio file loaded"):
            _ = loader.n_samples

    def test_file_not_found(self, tmp_path: Path):
        """Test error when file doesn't exist."""
        loader = AudioLoader()

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            loader.load(tmp_path / "nonexistent.wav")

    def test_repr_before_load(self):
        """Test repr before loading."""
        loader = AudioLoader()
        assert repr(loader) == "AudioLoader(not loaded)"


# endregion


# region AudioLoader WAV Tests


class TestAudioLoaderWAV:
    """Tests for AudioLoader with WAV files."""

    def test_load_wav(self, wav_file: Path):
        """Test loading a WAV file."""
        loader = AudioLoader()
        result = loader.load(wav_file)

        # Should return self for chaining
        assert result is loader

        # Check metadata
        assert loader.n_samples == 44100
        assert loader.sample_rate == 44100
        assert loader.channels == 1
        assert loader.duration_seconds == 1.0
        assert loader.format == "WAV"
        assert loader.source_path == wav_file

    def test_load_stereo_wav(self, stereo_wav_file: Path):
        """Test loading a stereo WAV file."""
        loader = AudioLoader().load(stereo_wav_file)

        assert loader.n_samples == 96000
        assert loader.sample_rate == 48000
        assert loader.channels == 2
        assert loader.duration_seconds == 2.0

    def test_from_file_convenience(self, wav_file: Path):
        """Test from_file class method."""
        loader = AudioLoader.from_file(wav_file)

        assert loader.n_samples == 44100
        assert loader.sample_rate == 44100

    def test_repr_after_load(self, wav_file: Path):
        """Test repr after loading."""
        loader = AudioLoader().load(wav_file)
        r = repr(loader)

        assert "samples=44100" in r
        assert "rate=44100Hz" in r
        assert "duration=1.00s" in r
        assert "format=WAV" in r

    def test_audio_info_access(self, wav_file: Path):
        """Test accessing the full AudioInfo object."""
        loader = AudioLoader().load(wav_file)
        info = loader.audio_info

        assert isinstance(info, AudioInfo)
        assert info.n_samples == 44100
        assert info.source_path == wav_file
        assert info.is_mono is True


# endregion


# region Timeline Creation Tests


class TestAudioLoaderTimeline:
    """Tests for creating timelines from AudioLoader."""

    def test_create_timeline_basic(self, wav_file: Path):
        """The timeline is in samples, its length the sample count."""
        loader = AudioLoader().load(wav_file)
        timeline = loader.create_timeline()

        assert type(timeline) is DiscretePhysicalTimeline
        assert timeline.unit == TimeUnit.samples
        assert timeline.number_type == NumberType.int
        assert timeline.length.value == 44100
        assert type(timeline.length.value) is int

    def test_create_timeline_with_uid(self, wav_file: Path):
        """Test timeline creation with custom uid."""
        loader = AudioLoader().load(wav_file)
        timeline = loader.create_timeline(uid="my_audio", name="My Audio File")

        assert timeline.id == "my_audio"
        assert timeline.name == "My Audio File"

    def test_create_timeline_default_uid_from_filename(self, wav_file: Path):
        """Test that uid defaults to filename stem."""
        loader = AudioLoader().load(wav_file)
        timeline = loader.create_timeline()

        assert timeline.id == "test"  # From "test.wav"
        assert timeline.name == "test.wav"

    def test_create_timeline_with_cmap(self, wav_file: Path):
        """The default attaches a SamplesToSeconds map at the file's rate."""
        loader = AudioLoader().load(wav_file)
        timeline = loader.create_timeline()

        cmap = timeline.get_conversion_map(TimeUnit.seconds)
        assert isinstance(cmap, SamplesToSeconds)
        assert cmap.sample_rate == 44100
        assert cmap(44100) == 1.0
        assert timeline.n_conversion_maps == 1

    def test_create_timeline_without_cmap(self, wav_file: Path):
        """Test timeline creation without C-map attachment."""
        loader = AudioLoader().load(wav_file)
        timeline = loader.create_timeline(attach_cmap=False)

        assert timeline.get_conversion_map(TimeUnit.seconds) is None
        assert timeline.n_conversion_maps == 0

    def test_create_continuous_timeline(self, stereo_wav_file: Path):
        """The continuous twin is in seconds, its length the duration."""
        loader = AudioLoader().load(stereo_wav_file)
        timeline = loader.create_continuous_timeline()

        assert type(timeline) is ContinuousPhysicalTimeline
        assert timeline.unit == TimeUnit.seconds
        assert timeline.number_type == NumberType.float
        assert timeline.length.value == 2.0
        assert timeline.id == "stereo"
        assert timeline.name == "stereo.wav"

    def test_create_continuous_timeline_with_cmap(self, stereo_wav_file: Path):
        """The default attaches a SecondsToSamples map at the file's rate."""
        loader = AudioLoader().load(stereo_wav_file)
        timeline = loader.create_continuous_timeline()

        cmap = timeline.get_conversion_map(TimeUnit.samples)
        assert isinstance(cmap, SecondsToSamples)
        assert cmap.sample_rate == 48000
        assert cmap(2.0) == 96000
        assert timeline.n_conversion_maps == 1

    def test_create_continuous_timeline_without_cmap(self, stereo_wav_file: Path):
        """``attach_cmap=False`` attaches nothing to the continuous twin."""
        loader = AudioLoader().load(stereo_wav_file)
        timeline = loader.create_continuous_timeline(
            uid="take", name="Take", attach_cmap=False
        )

        assert timeline.get_conversion_map(TimeUnit.samples) is None
        assert timeline.n_conversion_maps == 0
        assert timeline.id == "take"
        assert timeline.name == "Take"

    def test_create_timeline_metadata(self, wav_file: Path):
        """Test that timeline has metadata attached."""
        loader = AudioLoader().load(wav_file)
        timeline = loader.create_timeline()

        assert hasattr(timeline, "_metadata")
        assert timeline._metadata["sample_rate"] == 44100
        assert timeline._metadata["channels"] == 1
        assert timeline._metadata["format"] == "WAV"
        assert timeline._metadata["backend"] == "soundfile"

    def test_create_samples_to_seconds_map(self, wav_file: Path):
        """Test creating a standalone C-map."""
        loader = AudioLoader().load(wav_file)
        cmap = loader.create_samples_to_seconds_map()

        assert isinstance(cmap, SamplesToSeconds)
        assert cmap(44100) == 1.0
        assert cmap(22050) == 0.5


# endregion


# region Integration Tests


class TestAudioLoaderIntegration:
    """Integration tests for AudioLoader with the TTA framework."""

    def test_timeline_coordinate_conversion(self, wav_file: Path):
        """A timestamp on the samples axis carries its seconds through the C-map."""
        loader = AudioLoader().load(wav_file)
        timeline = loader.create_timeline()

        stamp = timeline.get_timestamp_at(22050)
        assert stamp.get_unit(TimeUnit.seconds) == IdCoordinate(
            0.5, TimeUnit.seconds, "test"
        )

    def test_both_timelines_span_the_same_file(self, stereo_wav_file: Path):
        """Each timeline's map takes its own end to the other's end."""
        loader = AudioLoader().load(stereo_wav_file)
        samples_tl = loader.create_timeline(uid="audio")
        seconds_tl = loader.create_continuous_timeline(uid="audio_seconds")

        assert samples_tl.length.value == 96000
        assert seconds_tl.length.value == 2.0
        to_seconds = samples_tl.get_conversion_map(TimeUnit.seconds)
        to_samples = seconds_tl.get_conversion_map(TimeUnit.samples)
        assert to_seconds(samples_tl.length.value) == seconds_tl.length.value
        assert to_samples(seconds_tl.length.value) == samples_tl.length.value


# endregion


# region Sample Count

# Counts whose float round trip at 44100 Hz truncates: int(n / 44100 * 44100) == n - 1.
TRUNCATED_COUNTS = [15, 44109, 1999995]


class TestAudioLoaderSampleCount:
    """The sample count is a count of frames, never a float round trip."""

    def test_short_wav_through_soundfile(self, tmp_path: Path):
        wav_path = create_wav_file(tmp_path / "fifteen.wav", n_samples=15)

        loader = AudioLoader().load(wav_path)
        assert loader.audio_info.backend == "soundfile"
        assert loader.n_samples == 15

    def test_short_wav_through_wave(self, tmp_path: Path, no_soundfile: None):
        wav_path = create_wav_file(tmp_path / "fifteen.wav", n_samples=15)

        loader = AudioLoader().load(wav_path)
        assert loader.audio_info.backend == "wave"
        assert loader.n_samples == 15
        assert loader.duration_seconds == 15 / 44100

    @pytest.mark.parametrize("n_samples", TRUNCATED_COUNTS)
    def test_truncated_counts_load_exactly(self, tmp_path: Path, n_samples: int):
        assert int(n_samples / 44100 * 44100) == n_samples - 1
        wav_path = create_wav_file(tmp_path / "sweep.wav", n_samples=n_samples)

        loader = AudioLoader().load(wav_path)
        assert loader.audio_info.backend == "soundfile"
        assert loader.n_samples == n_samples
        assert loader.create_timeline().length.value == n_samples

    @pytest.mark.parametrize("n_samples", TRUNCATED_COUNTS)
    def test_mutagen_estimate_rounds(self, tmp_path: Path, n_samples: int):
        """mutagen states a WAV's duration as frames / rate; rounding recovers them."""
        wav_path = create_wav_file(tmp_path / "sweep.wav", n_samples=n_samples)

        info = AudioLoader()._load_with_mutagen(wav_path)
        assert info.backend == "mutagen"
        assert info.duration_seconds == n_samples / 44100
        assert info.n_samples == n_samples


# endregion


# region MP4 Count

HARD_TECHNO = ensure_data("audio") / "hard_techno"


def _box(kind: bytes, *children: bytes, body: bytes = b"") -> bytes:
    """One ISO media box: size, type, *body*, then *children*."""
    payload = body + b"".join(children)
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


_FULL = bytes(4)  # version 0, no flags


def create_mp4_file(
    path: Path,
    media_duration: int,
    *,
    media_scale: int = 44100,
    movie_scale: int = 1000,
    edits: list[tuple[int, int]] | None = None,
    itunsmpb: str | None = None,
    sample_rate: int = 44100,
) -> Path:
    """Write the boxes of an MP4 sound track that state its length; no ``mdat``.

    Args:
        path: Path to write the file.
        media_duration: ``mdhd`` duration, in *media_scale* units.
        media_scale: ``mdhd`` timescale.
        movie_scale: ``mvhd`` timescale, the unit of an edit's segment duration.
        edits: ``elst`` entries as (segment duration, media time); None for no
            edit list.
        itunsmpb: The text of an ``iTunSMPB`` free-form tag, if any.
        sample_rate: The ``mp4a`` sample entry's rate.
    """
    mvhd = _box(
        b"mvhd", body=_FULL + struct.pack(">4I", 0, 0, movie_scale, 0) + bytes(80)
    )
    mdhd = _box(
        b"mdhd",
        body=_FULL + struct.pack(">4I2H", 0, 0, media_scale, media_duration, 0, 0),
    )
    hdlr = _box(b"hdlr", body=_FULL + struct.pack(">I4s", 0, b"soun") + bytes(13))
    mp4a = _box(
        b"mp4a",
        _box(b"free"),
        body=bytes(6)
        + struct.pack(">H", 1)
        + bytes(8)
        + struct.pack(">4HI", 2, 16, 0, 0, sample_rate << 16),
    )
    stsd = _box(b"stsd", mp4a, body=_FULL + struct.pack(">I", 1))
    mdia = _box(b"mdia", mdhd, hdlr, _box(b"minf", _box(b"stbl", stsd)))
    trak = [mdia]
    if edits is not None:
        entries = b"".join(struct.pack(">IihH", seg, time, 1, 0) for seg, time in edits)
        elst = _box(b"elst", body=_FULL + struct.pack(">I", len(edits)) + entries)
        trak.insert(0, _box(b"edts", elst))
    moov = [mvhd, _box(b"trak", *trak)]
    if itunsmpb is not None:
        freeform = _box(
            b"----",
            _box(b"mean", body=_FULL + b"com.apple.iTunes"),
            _box(b"name", body=_FULL + b"iTunSMPB"),
            _box(b"data", body=struct.pack(">II", 1, 0) + itunsmpb.encode()),
        )
        meta_hdlr = _box(
            b"hdlr", body=_FULL + struct.pack(">I4s", 0, b"mdir") + bytes(13)
        )
        meta = _box(b"meta", meta_hdlr, _box(b"ilst", freeform), body=_FULL)
        moov.append(_box(b"udta", meta))
    ftyp = _box(b"ftyp", body=b"M4A " + bytes(4) + b"M4A isom")
    path.write_bytes(ftyp + _box(b"moov", *moov))
    return path


class TestAudioLoaderMP4Count:
    """An MP4's count is the one its boxes state for the audio as played."""

    @pytest.mark.parametrize(
        "media_duration,edits,itunsmpb,n_samples,count_from",
        [
            (7370134, [(167100, 1024)], None, 7369110, "elst"),
            (5120, [(50, 1024)], None, 2205, "elst"),
            (5120, [(51, 1024)], None, 2250, "elst"),
            (5120, [(200, 1024)], None, 4096, "elst"),
            (
                5120,
                None,
                " 00000000 00000840 000003F0 00000000000007D0",
                2000,
                "iTunSMPB",
            ),
            (5120, None, None, 5120, "mdhd"),
            (5120, [(10, -1), (100, 0)], None, 5120, "mdhd"),
        ],
        ids=[
            "priming-to-end",
            "edit-ends-on-a-sample",
            "edit-ends-between-samples",
            "edit-past-the-media",
            "itunsmpb",
            "media-only",
            "two-edits",
        ],
    )
    def test_stated_count(
        self,
        tmp_path: Path,
        media_duration: int,
        edits: list[tuple[int, int]] | None,
        itunsmpb: str | None,
        n_samples: int,
        count_from: str,
    ):
        path = create_mp4_file(
            tmp_path / "track.m4a", media_duration, edits=edits, itunsmpb=itunsmpb
        )

        loader = AudioLoader().load(path)
        assert loader.audio_info.backend == "mutagen"
        assert loader.audio_info.extra["sample_count_from"] == count_from
        assert loader.n_samples == n_samples
        assert loader.duration_seconds == n_samples / 44100
        assert loader.create_timeline().length.value == n_samples

    def test_no_whole_number_of_samples(self, tmp_path: Path):
        path = create_mp4_file(tmp_path / "track.m4a", 1001, media_scale=1000)

        loader = AudioLoader().load(path)
        assert loader.audio_info.extra["sample_count_from"] == "duration"
        assert loader.n_samples == 44144
        assert loader.duration_seconds == 1.001

    @pytest.mark.parametrize(
        "name,n_samples", [("Bye Bye.m4a", 12977142), ("Ao Céu.m4a", 12318718)]
    )
    def test_specimen(self, name: str, n_samples: int):
        loader = AudioLoader().load(HARD_TECHNO / name)

        assert loader.audio_info.backend == "mutagen"
        assert loader.audio_info.extra["sample_count_from"] == "elst"
        assert loader.n_samples == n_samples
        assert loader.sample_rate == 44100
        assert loader.duration_seconds == n_samples / 44100

    def test_estimate_is_named(self, tmp_path: Path):
        wav_path = create_wav_file(tmp_path / "fifteen.wav", n_samples=15)

        info = AudioLoader()._load_with_mutagen(wav_path)
        assert info.extra["sample_count_from"] == "duration"


# endregion


# region Backend Chain


class TestAudioLoaderBackendChain:
    """Every backend refuses what it cannot read, and the error names each reason."""

    def test_zero_rate_names_every_backend(self, tmp_path: Path):
        """A header stating rate 0 is unreadable, not a division by zero."""
        soundfile = pytest.importorskip("soundfile")
        wav_path = create_wav_file(tmp_path / "zero.wav", n_samples=15, sample_rate=0)
        with pytest.raises(soundfile.SoundFileError) as declined:
            soundfile.info(str(wav_path))

        with pytest.raises(ValueError) as excinfo:
            AudioLoader().load(wav_path)
        assert str(excinfo.value) == (
            f"Cannot read audio file '{wav_path}':\n"
            f"  soundfile: {declined.value}\n"
            "  wave: the file states sample rate 0\n"
            "  riff: the file states sample rate 0\n"
            "  mutagen: the file states sample rate 0"
        )

    def test_missing_package_names_the_extra(self, tmp_path: Path, no_soundfile: None):
        wav_path = create_wav_file(tmp_path / "zero.wav", n_samples=15, sample_rate=0)

        with pytest.raises(ValueError) as excinfo:
            AudioLoader().load(wav_path)
        assert str(excinfo.value) == (
            f"Cannot read audio file '{wav_path}':\n"
            "  soundfile: not installed\n"
            "  wave: the file states sample rate 0\n"
            "  riff: the file states sample rate 0\n"
            "  mutagen: the file states sample rate 0\n"
            'Install the missing packages with pip install "timetoalign[audio]".'
        )

    def test_zero_channels(self, tmp_path: Path, no_soundfile: None):
        wav_path = create_wav_file(tmp_path / "silent.wav", n_samples=15, channels=0)

        with pytest.raises(ValueError) as excinfo:
            AudioLoader().load(wav_path)
        assert str(excinfo.value) == (
            f"Cannot read audio file '{wav_path}':\n"
            "  soundfile: not installed\n"
            "  wave: bad # of channels\n"
            "  riff: the file states 0 channels\n"
            "  mutagen: the file states 0 channels\n"
            'Install the missing packages with pip install "timetoalign[audio]".'
        )

    @pytest.mark.parametrize("fact", [True, False], ids=["fact", "no_fact"])
    def test_float_wav_through_riff(
        self, tmp_path: Path, no_soundfile: None, fact: bool
    ):
        """The RIFF parser reads the IEEE-float WAV that ``wave`` refuses."""
        wav_path = create_wav_file(
            tmp_path / "float.wav",
            n_samples=15,
            channels=2,
            bits_per_sample=32,
            format_tag=3,
            fact=fact,
        )
        loader = AudioLoader()
        with pytest.raises(audio_module._CannotRead, match="^unknown format: 3$"):
            loader._load_with_wave(wav_path)

        loader.load(wav_path)
        assert loader.audio_info.backend == "riff"
        assert loader.n_samples == 15
        assert loader.channels == 2
        assert loader.duration_seconds == 15 / 44100

    def test_other_exceptions_propagate(
        self, wav_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A reader's defect surfaces as itself, not as an unreadable file."""

        def broken_reader(self: AudioLoader, path: Path) -> AudioInfo:
            raise TypeError("broken reader")

        monkeypatch.setattr(AudioLoader, "_load_with_soundfile", broken_reader)

        with pytest.raises(TypeError, match="^broken reader$"):
            AudioLoader().load(wav_file)


# endregion


# region Specimen


class TestAudioLoaderSpecimen:
    """Both timelines of a real MP3, read by the soundfile backend."""

    @pytest.fixture(scope="class")
    def loader(self) -> AudioLoader:
        return AudioLoader().load(BASS_KICK)

    def test_audio_info(self, loader: AudioLoader):
        assert loader.audio_info.backend == "soundfile"
        assert loader.n_samples == 10188800
        assert loader.sample_rate == 44100
        assert loader.channels == 2
        assert loader.duration_seconds == 231.03854875283446

    def test_mutagen_count_is_an_estimate(self, no_soundfile: None):
        """Known issue: mutagen's whole-frame duration overcounts the decoder."""
        loader = AudioLoader().load(BASS_KICK)

        assert loader.audio_info.backend == "mutagen"
        assert loader.n_samples == 10190592
        assert loader.duration_seconds == 231.0791836734694
        assert loader.create_timeline()._metadata["backend"] == "mutagen"
        assert loader.create_continuous_timeline()._metadata["backend"] == "mutagen"

    def test_samples_timeline(self, loader: AudioLoader):
        timeline = loader.create_timeline()

        assert type(timeline) is DiscretePhysicalTimeline
        assert timeline.unit == TimeUnit.samples
        assert timeline.number_type == NumberType.int
        assert timeline.length.value == 10188800
        assert timeline.id == "Bass Kick"
        assert timeline.name == "Bass Kick.mp3"
        end = timeline.get_conversion_map(TimeUnit.seconds)(10188800)
        assert end == 231.03854875283446
        assert type(end) is float

    def test_seconds_timeline(self, loader: AudioLoader):
        timeline = loader.create_continuous_timeline()

        assert type(timeline) is ContinuousPhysicalTimeline
        assert timeline.unit == TimeUnit.seconds
        assert timeline.number_type == NumberType.float
        assert timeline.length.value == 231.03854875283446
        end = timeline.get_conversion_map(TimeUnit.samples)(231.03854875283446)
        assert end == 10188800
        assert type(end) is int

    def test_no_cmap(self, loader: AudioLoader):
        samples_tl = loader.create_timeline(attach_cmap=False)
        seconds_tl = loader.create_continuous_timeline(attach_cmap=False)

        assert samples_tl.n_conversion_maps == 0
        assert seconds_tl.n_conversion_maps == 0

    @pytest.mark.parametrize(
        ("create", "cls", "unit", "length", "target", "scalar", "converted"),
        [
            (
                "create_timeline",
                DiscretePhysicalTimeline,
                TimeUnit.samples,
                10188800,
                TimeUnit.seconds,
                Fraction(1, 44100),
                231.03854875283446,
            ),
            (
                "create_continuous_timeline",
                ContinuousPhysicalTimeline,
                TimeUnit.seconds,
                231.03854875283446,
                TimeUnit.samples,
                44100,
                10188800,
            ),
        ],
        ids=["samples", "seconds"],
    )
    def test_manifest_round_trip(
        self,
        loader: AudioLoader,
        create: str,
        cls: type,
        unit: TimeUnit,
        length: int | float,
        target: TimeUnit,
        scalar: int | Fraction,
        converted: int | float,
    ):
        timeline = getattr(loader, create)()
        restored = Timeline.from_dict(timeline.to_dict())

        assert type(restored) is cls
        assert restored.unit == unit
        assert restored.length.value == length
        assert restored.n_conversion_maps == 1
        cmap = restored.get_conversion_map(target)
        assert isinstance(cmap, LinearMap)
        assert cmap.source_unit == unit
        assert cmap.target_unit == target
        assert cmap.scalar == scalar
        assert cmap(length) == converted
        assert type(cmap(length)) is type(converted)
        assert restored.to_dict() == timeline.to_dict()


# endregion


# region Edge Cases


class TestAudioLoaderEdgeCases:
    """Edge case tests for AudioLoader."""

    def test_very_short_audio(self, tmp_path: Path):
        """Test loading a very short audio file (10 samples)."""
        wav_path = create_wav_file(
            tmp_path / "short.wav",
            n_samples=10,
            sample_rate=44100,
            channels=1,
        )

        loader = AudioLoader().load(wav_path)
        assert loader.n_samples == 10
        assert loader.duration_seconds == 10 / 44100

    def test_high_sample_rate(self, tmp_path: Path):
        """Test loading audio with high sample rate (192kHz)."""
        wav_path = create_wav_file(
            tmp_path / "highres.wav",
            n_samples=192000,
            sample_rate=192000,
            channels=2,
            bits_per_sample=24,
        )

        loader = AudioLoader().load(wav_path)
        assert loader.sample_rate == 192000
        assert loader.duration_seconds == 1.0

    def test_method_chaining(self, wav_file: Path):
        """Test that load() returns self for method chaining."""
        timeline = AudioLoader().load(wav_file).create_timeline()
        assert isinstance(timeline, DiscretePhysicalTimeline)


# endregion


# region Backend Probe Silence


@pytest.fixture
def mislabelled_mp3(tmp_path: Path) -> Path:
    """A file named ``.mp3`` that holds no MPEG frames at all.

    70,000 zero bytes: past the 65,536-byte window libsndfile's MP3 decoder
    searches for a frame header before reporting its failure on file
    descriptor 2.
    """
    path = tmp_path / "mislabelled.mp3"
    path.write_bytes(b"\x00" * 70_000)
    return path


class TestAudioLoaderProbeSilence:
    """Probing a file no backend can read prints nothing."""

    def test_unreadable_file_probe_is_silent(self, mislabelled_mp3: Path, capfd):
        """A failing probe reports through the exception, not the terminal."""
        with pytest.raises(ValueError):
            AudioLoader().load(mislabelled_mp3)

        captured = capfd.readouterr()
        assert captured.err == ""
        assert captured.out == ""

    def test_stderr_survives_the_probe(self, mislabelled_mp3: Path, capfd):
        """The muted descriptor is restored, failure or not."""
        with pytest.raises(ValueError):
            AudioLoader().load(mislabelled_mp3)
        os.write(2, b"after the probe\n")

        assert capfd.readouterr().err == "after the probe\n"

    def test_readable_file_load_is_silent(self, wav_file: Path, capfd):
        """Muting the probe does not swallow the file it succeeds on."""
        loader = AudioLoader().load(wav_file)

        assert loader.n_samples == 44100
        assert capfd.readouterr().err == ""


# endregion
