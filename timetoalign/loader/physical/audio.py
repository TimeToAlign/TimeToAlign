"""AudioLoader for loading audio file metadata into TimeToAlign!

This module provides a manifest-style loader that extracts metadata from audio
files (WAV, FLAC, OGG, MP3, etc.) without loading the actual sample data.
The resulting information creates a DiscretePhysicalTimeline in samples with a
SamplesToSeconds conversion map, or, on request, a ContinuousPhysicalTimeline in
seconds with a SecondsToSamples conversion map.

Design Philosophy:
    AudioLoader follows the same pattern as IIIFManifestLoader: it extracts
    dimensions and metadata from files without loading the heavy content.
    This allows users to:
    1. Create timelines representing audio files
    2. Attach conversion maps for coordinate transformations
    3. Add events from other loaders (annotations, beat markers, etc.)

Backend Support:
    The loader tries its backends in order, exact sample counts first:
    soundfile (libsndfile's frame count, WAV, FLAC, OGG, MP3 and more), then for
    WAV files Python's built-in wave module and a RIFF header parser, then
    mutagen (MP3, M4A and more), which states only a duration and so yields an
    estimated count.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal, TypeVar

from timetoalign.loader._optional import optional_module
from timetoalign.loader.base import Loader

if TYPE_CHECKING:
    from timetoalign.core import NumberType, TimeUnit
    from timetoalign.maps import SamplesToSeconds
    from timetoalign.timelines import (
        ContinuousPhysicalTimeline,
        DiscretePhysicalTimeline,
        PhysicalTimeline,
    )

module_logger = logging.getLogger(__name__)

_PhysicalTimelineT = TypeVar("_PhysicalTimelineT", bound="PhysicalTimeline")

_WAV_SUFFIXES = frozenset({".wav", ".wave"})


# region Backend Detection


class _CannotRead(Exception):
    """A backend is not installed or cannot read the file; the next one tries."""


class _NotInstalled(_CannotRead):
    """A backend's package is not installed."""


def _require_rate_and_channels(sample_rate: int, channels: int) -> None:
    """Refuse a header whose sample rate or channel count describes no audio.

    Raises:
        _CannotRead: If the sample rate or the channel count is not positive.
    """
    if sample_rate <= 0:
        raise _CannotRead(f"the file states sample rate {sample_rate}")
    if channels <= 0:
        raise _CannotRead(f"the file states {channels} channels")


def _get_wave():
    """Built-in wave module as fallback for WAV files."""
    import wave

    return wave


@contextmanager
def _muted_native_stderr() -> Iterator[None]:
    """Mute writes to file descriptor 2 for the duration of the block.

    The decoder libraries behind libsndfile report unreadable input by
    writing to file descriptor 2 directly, bypassing Python's ``sys.stderr``
    and therefore both :mod:`warnings` and :mod:`logging`. A probe that fails
    is a normal step of the backend chain — the next backend then reads the
    file — so those messages are noise, and a caller has no way to silence
    them. Muting the descriptor around the probe is the only place that can.

    Python-level stderr is flushed first so no buffered message is swallowed,
    and the descriptor is restored even if the probe raises. If the
    descriptor cannot be duplicated (a platform without ``os.dup``, or a
    closed stream), the block runs unmuted rather than failing.
    """
    try:
        saved_fd = os.dup(2)
    except (AttributeError, OSError):
        yield
        return
    try:
        sys.stderr.flush()
    except (AttributeError, ValueError):
        pass
    try:
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


# endregion


# region AudioInfo


@dataclass
class AudioInfo:
    """Metadata for an audio file.

    This dataclass holds all relevant information about an audio file
    that is needed to create a physical timeline.

    Attributes:
        n_samples: Total number of samples in the file (frames). Exact from the
            soundfile, wave and riff backends; estimated from the stated
            duration on the mutagen backend.
        sample_rate: Sample rate in Hz (e.g., 44100, 48000).
        channels: Number of audio channels (1=mono, 2=stereo).
        duration_seconds: Duration in seconds: ``n_samples / sample_rate`` where
            the count is exact, the duration the file states on the mutagen
            backend.
        format: Audio format/codec (e.g., "WAV", "FLAC", "MP3").
        subtype: Audio subtype/encoding (e.g., "PCM_16", "PCM_24", "VORBIS").
        bits_per_sample: Bit depth (e.g., 16, 24, 32). May be None for lossy formats.
        source_path: Path to the source file.
        extra: Additional format-specific metadata.
        backend: The backend that read the file: ``"soundfile"``, ``"wave"``,
            ``"riff"`` or ``"mutagen"``; None for metadata not read by the loader.
    """

    n_samples: int
    sample_rate: int
    channels: int
    duration_seconds: float
    format: str
    subtype: str | None = None
    bits_per_sample: int | None = None
    source_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    backend: Literal["soundfile", "wave", "riff", "mutagen"] | None = None

    @property
    def length_in_samples(self) -> int:
        """Alias for n_samples, matching timeline terminology."""
        return self.n_samples

    @property
    def is_mono(self) -> bool:
        """Whether the audio is mono (single channel)."""
        return self.channels == 1

    @property
    def is_stereo(self) -> bool:
        """Whether the audio is stereo (two channels)."""
        return self.channels == 2


# endregion


# region AudioLoader


class AudioLoader(Loader[AudioInfo]):
    """Load audio file metadata for creating physical timelines.

    AudioLoader extracts metadata from audio files without loading the actual
    sample data. This is efficient for creating physical timelines that
    represent audio files in the TimeToAlign! framework.

    The loader automatically:
    - Tries its backends in order, exact counts first: soundfile, then for WAV
      files the built-in wave module and a RIFF header parser, then mutagen
    - Extracts sample count, sample rate, channels, and format info
    - Creates a samples timeline with a SamplesToSeconds C-map
      (``create_timeline``) or a seconds timeline with a SecondsToSamples
      C-map (``create_continuous_timeline``)

    Supported formats depend on installed backends:
    - soundfile: WAV, FLAC, OGG, AIFF, MP3 (libsndfile >= 1.1), and many more
    - wave (builtin) and the RIFF header parser: WAV only
    - mutagen: MP3, M4A, FLAC, OGG, and more

    Known issue:
        mutagen states only a duration, so its sample count is the estimate
        ``round(duration * sample_rate)``. For an MP3 the duration counts whole
        MPEG frames, encoder delay and padding included, or, without a VBR
        header, is guessed from the file size and bitrate, so the estimate can
        miss the decoded count in either direction. It is used only when
        soundfile is not installed or cannot read the file;
        ``audio_info.backend`` names the backend that answered, and the
        timeline's metadata carries it.

        The RIFF header parser derives the block size from the bit depth and
        channel count rather than reading the header's block align, so without
        a ``fact`` chunk it miscounts samples not packed in whole bytes.

        Where mutagen's stream info has no ``channels`` attribute the loader
        assumes 2; a stated 0, mutagen's default for several formats, is
        refused.

        A WAV truncated below the size its ``data`` chunk states gets a
        backend-dependent count: soundfile counts the frames present, the other
        backends answer the header's count.

    Examples:
        >>> loader = AudioLoader()
        >>> loader.load("recording.wav")
        >>> loader.n_samples
        7938000
        >>> loader.sample_rate
        44100
        >>> loader.duration_seconds
        180.0

        >>> # Create a timeline
        >>> timeline = loader.create_timeline(uid="my_audio")
        >>> timeline.unit
        <TimeUnit.samples: 'samples'>
        >>> timeline.length
        Coordinate(7938000, samples)

    Attributes:
        audio_info: Parsed audio metadata (after loading).
    """

    def __init__(self) -> None:
        """Initialize the loader."""
        super().__init__()
        self._audio_info: AudioInfo | None = None
        self._source_path: Path | None = None
        self._logger = module_logger.getChild("AudioLoader")

    def _load_source(self, source: Path) -> AudioInfo:
        """Load metadata from one audio file.

        The backends are tried in order, exact sample counts first: soundfile,
        then for a WAV file the wave module and the RIFF header parser, then
        mutagen. A backend that is not installed or cannot read the file hands
        it on to the next; any other exception is a defect and propagates.
        When no backend reads the file, the error names every backend's reason.

        Args:
            source: Path to the audio file.

        Returns:
            Parsed audio metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is not supported or cannot be read.
        """
        if not source.exists():
            raise FileNotFoundError(f"Audio file not found: {source}")

        readers = [("soundfile", self._load_with_soundfile)]
        if source.suffix.lower() in _WAV_SUFFIXES:
            readers += [
                ("wave", self._load_with_wave),
                ("riff", self._load_with_riff_parser),
            ]
        readers.append(("mutagen", self._load_with_mutagen))

        reasons = []
        missing_package = False
        for backend, read in readers:
            try:
                info = read(source)
            except _CannotRead as e:
                self._logger.debug(f"{backend} cannot read {source}: {e}")
                reasons.append(f"  {backend}: {e}")
                missing_package = missing_package or isinstance(e, _NotInstalled)
                continue
            self._logger.debug(
                f"Read audio metadata from {source} using {info.backend}: "
                f"{info.n_samples} samples @ {info.sample_rate} Hz"
            )
            return info

        message = f"Cannot read audio file '{source}':\n" + "\n".join(reasons)
        if missing_package:
            message += (
                '\nInstall the missing packages with pip install "timetoalign[audio]".'
            )
        raise ValueError(message)

    def _accept_source(
        self,
        path: Path,
        source_meta: dict[str, Any],
        payload: AudioInfo,
    ) -> None:
        """Retain parsed audio information from the shared lifecycle."""
        super()._accept_source(path, source_meta, payload)
        self._audio_info = payload
        self._source_path = path

    def _load_with_soundfile(self, path: Path) -> AudioInfo:
        """Load metadata using soundfile (libsndfile backend).

        ``frames`` is the count libsndfile's decoder produces, so the sample
        count is exact and ``duration_seconds`` is ``n_samples / sample_rate``.

        The probe runs with file-descriptor-level stderr muted: libsndfile's
        decoders print their own diagnostics there when the input is not what
        its extension claims, and such a file is handed on to the next backend
        rather than being an error the user must see.

        Raises:
            _CannotRead: If soundfile is not installed or cannot read the file.
        """
        sf = optional_module("soundfile")
        if sf is None:
            raise _NotInstalled("not installed")
        try:
            with _muted_native_stderr():
                info = sf.info(str(path))
        except sf.SoundFileError as e:
            raise _CannotRead(e) from e
        _require_rate_and_channels(info.samplerate, info.channels)

        return AudioInfo(
            n_samples=info.frames,
            sample_rate=info.samplerate,
            channels=info.channels,
            duration_seconds=info.frames / info.samplerate,
            format=info.format,
            subtype=info.subtype,
            bits_per_sample=self._subtype_to_bits(info.subtype),
            source_path=path,
            extra={"sections": info.sections},
            backend="soundfile",
        )

    def _load_with_mutagen(self, path: Path) -> AudioInfo:
        """Load metadata using mutagen (for MP3, M4A, etc.).

        mutagen states a duration, not a frame count, so the sample count is the
        estimate ``round(duration * sample_rate)`` and ``duration_seconds`` is
        the stated duration. For an MP3 the duration counts whole MPEG frames,
        encoder delay and padding included, or, without a VBR header, is
        guessed from the file size and bitrate, so the estimate can miss the
        decoded count in either direction.

        Raises:
            _CannotRead: If mutagen is not installed, cannot parse the file,
                states no sample rate or duration for it, or states a sample
                rate or channel count that is not positive.
        """
        mutagen = optional_module("mutagen")
        if mutagen is None:
            raise _NotInstalled("not installed")
        try:
            audio = mutagen.File(str(path))
        except mutagen.MutagenError as e:
            raise _CannotRead(e) from e
        if audio is None:
            raise _CannotRead("mutagen does not recognise the format")

        # Get audio info
        info = audio.info

        # Sample rate
        sample_rate = getattr(info, "sample_rate", None)
        if sample_rate is None:
            raise _CannotRead("mutagen states no sample rate")

        # Duration and sample count
        duration = getattr(info, "length", None)
        if duration is None:
            raise _CannotRead("mutagen states no duration")

        # The duration is a double, so the nearest count, never a truncation
        n_samples = round(duration * sample_rate)

        # Channels
        channels = getattr(info, "channels", 2)
        _require_rate_and_channels(sample_rate, channels)

        # Bits per sample (if available)
        bits = getattr(info, "bits_per_sample", None)

        # Determine format from file extension
        ext = path.suffix.lower()
        format_map = {
            ".mp3": "MP3",
            ".m4a": "M4A",
            ".aac": "AAC",
            ".flac": "FLAC",
            ".ogg": "OGG",
            ".opus": "OPUS",
        }
        fmt = format_map.get(ext, ext.upper().lstrip("."))

        return AudioInfo(
            n_samples=n_samples,
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=duration,
            format=fmt,
            subtype=None,
            bits_per_sample=bits,
            source_path=path,
            extra={
                "bitrate": getattr(info, "bitrate", None),
                "codec": getattr(info, "codec", None),
            },
            backend="mutagen",
        )

    def _load_with_wave(self, path: Path) -> AudioInfo:
        """Load metadata using Python's built-in wave module (PCM WAV only).

        Raises:
            _CannotRead: If the wave module cannot read the file, or the header
                states a sample rate or channel count that is not positive.
        """
        wave = _get_wave()

        try:
            with wave.open(str(path), "rb") as wf:
                n_samples = wf.getnframes()
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                bits_per_sample = wf.getsampwidth() * 8
        except (wave.Error, EOFError) as e:
            raise _CannotRead(e) from e
        _require_rate_and_channels(sample_rate, channels)

        return AudioInfo(
            n_samples=n_samples,
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=n_samples / sample_rate,
            format="WAV",
            subtype=f"PCM_{bits_per_sample}",
            bits_per_sample=bits_per_sample,
            source_path=path,
            backend="wave",
        )

    def _load_with_riff_parser(self, path: Path) -> AudioInfo:
        """Load metadata by manually parsing the RIFF/WAV header.

        This handles WAV formats that Python's built-in wave module cannot read,
        including IEEE float (format tag 3) commonly used by audio feature
        extraction tools (e.g., Essentia, RepoVizz).

        Only reads the header chunks (fmt, fact, data) — no sample data is loaded.

        Args:
            path: Path to the WAV file.

        Returns:
            AudioInfo with metadata extracted from the RIFF header.

        Raises:
            _CannotRead: If the file is not a valid RIFF/WAV file, or its header
                states a sample rate or channel count that is not positive.
        """
        import struct

        format_names = {
            1: "PCM",
            3: "IEEE_FLOAT",
            6: "A_LAW",
            7: "MU_LAW",
            0xFFFE: "EXTENSIBLE",
        }

        try:
            with open(path, "rb") as f:
                # RIFF header
                riff_id = f.read(4)
                if riff_id != b"RIFF":
                    raise _CannotRead("not a RIFF file")
                f.read(4)  # file size (unused)
                wave_id = f.read(4)
                if wave_id != b"WAVE":
                    raise _CannotRead("not a WAVE file")

                # Parse chunks
                has_fmt = False
                channels = 0
                sample_rate = 0
                bits_per_sample = 0
                format_tag = 0
                n_samples: int | None = None
                data_size = 0

                while True:
                    chunk_header = f.read(8)
                    if len(chunk_header) < 8:
                        break
                    chunk_id = chunk_header[:4]
                    chunk_size = struct.unpack("<I", chunk_header[4:])[0]

                    if chunk_id == b"fmt ":
                        has_fmt = True
                        fmt_data = f.read(chunk_size)
                        format_tag = struct.unpack("<H", fmt_data[:2])[0]
                        channels = struct.unpack("<H", fmt_data[2:4])[0]
                        sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
                        bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
                    elif chunk_id == b"fact":
                        fact_data = f.read(chunk_size)
                        n_samples = struct.unpack("<I", fact_data[:4])[0]
                    elif chunk_id == b"data":
                        data_size = chunk_size
                        break  # data chunk is last relevant chunk
                    else:
                        # Skip unknown chunks
                        f.seek(chunk_size, 1)
        except struct.error as e:
            raise _CannotRead(f"truncated RIFF chunk: {e}") from e

        if not has_fmt:
            raise _CannotRead("no fmt chunk")
        _require_rate_and_channels(sample_rate, channels)

        # Compute n_samples from data chunk if fact chunk was absent
        if n_samples is None:
            bytes_per_sample = max(bits_per_sample // 8, 1)
            block_align = bytes_per_sample * max(channels, 1)
            n_samples = data_size // block_align if block_align > 0 else 0

        duration_seconds = n_samples / sample_rate
        fmt_name = format_names.get(format_tag, f"UNKNOWN_{format_tag}")
        subtype = f"{fmt_name}_{bits_per_sample}" if bits_per_sample else fmt_name

        return AudioInfo(
            n_samples=n_samples,
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=duration_seconds,
            format="WAV",
            subtype=subtype,
            bits_per_sample=bits_per_sample,
            source_path=path,
            extra={"format_tag": format_tag},
            backend="riff",
        )

    def _subtype_to_bits(self, subtype: str | None) -> int | None:
        """Convert soundfile subtype to bits per sample."""
        if subtype is None:
            return None

        bit_map = {
            "PCM_16": 16,
            "PCM_24": 24,
            "PCM_32": 32,
            "PCM_S8": 8,
            "PCM_U8": 8,
            "FLOAT": 32,
            "DOUBLE": 64,
        }
        return bit_map.get(subtype)

    # endregion

    # region Properties

    @property
    def audio_info(self) -> AudioInfo:
        """Return parsed audio metadata.

        Raises:
            RuntimeError: If no audio file has been loaded.
        """
        if self._audio_info is None:
            raise RuntimeError("No audio file loaded. Call load() first.")
        return self._audio_info

    @property
    def n_samples(self) -> int:
        """Total number of samples (frames) in the audio file."""
        return self.audio_info.n_samples

    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz."""
        return self.audio_info.sample_rate

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        return self.audio_info.channels

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.audio_info.duration_seconds

    @property
    def format(self) -> str:
        """Audio format (e.g., 'WAV', 'FLAC', 'MP3')."""
        return self.audio_info.format

    @property
    def source_path(self) -> Path | None:
        """Path to the loaded audio file."""
        return self._source_path

    # endregion

    # region Timeline Creation

    def create_timeline(
        self,
        uid: str | None = None,
        name: str | None = None,
        attach_cmap: bool = True,
    ) -> "DiscretePhysicalTimeline":
        """Create a DiscretePhysicalTimeline in samples from the loaded audio.

        The timeline is created with:
        - unit=TimeUnit.samples
        - length=n_samples
        - integer coordinates
        - a SamplesToSeconds C-map, unless ``attach_cmap`` is False

        Args:
            uid: Unique identifier for the timeline. If None, uses filename.
            name: Human-readable name. If None, uses filename.
            attach_cmap: If True, attach a SamplesToSeconds conversion map at
                the file's sample rate.

        Returns:
            A DiscretePhysicalTimeline representing the audio file.

        Raises:
            RuntimeError: If no audio file has been loaded.

        Examples:
            >>> loader = AudioLoader().load("song.wav")
            >>> timeline = loader.create_timeline()
            >>> timeline.unit
            <TimeUnit.samples: 'samples'>
        """
        from timetoalign.core import NumberType, TimeUnit
        from timetoalign.timelines import DiscretePhysicalTimeline

        timeline = self._create_audio_timeline(
            DiscretePhysicalTimeline,
            length=self.audio_info.n_samples,
            unit=TimeUnit.samples,
            number_type=NumberType.int,
            uid=uid,
            name=name,
        )
        if attach_cmap:
            timeline.add_conversion_map(self.create_samples_to_seconds_map())
        return timeline

    def create_continuous_timeline(
        self,
        uid: str | None = None,
        name: str | None = None,
        attach_cmap: bool = True,
    ) -> "ContinuousPhysicalTimeline":
        """Create a ContinuousPhysicalTimeline in seconds from the loaded audio.

        The timeline is created with:
        - unit=TimeUnit.seconds
        - length=duration_seconds
        - float coordinates
        - a SecondsToSamples C-map, unless ``attach_cmap`` is False

        Args:
            uid: Unique identifier for the timeline. If None, uses filename.
            name: Human-readable name. If None, uses filename.
            attach_cmap: If True, attach a SecondsToSamples conversion map at
                the file's sample rate.

        Returns:
            A ContinuousPhysicalTimeline representing the audio file.

        Raises:
            RuntimeError: If no audio file has been loaded.

        Examples:
            >>> loader = AudioLoader().load("song.wav")
            >>> timeline = loader.create_continuous_timeline()
            >>> timeline.unit
            <TimeUnit.seconds: 'seconds'>
        """
        from timetoalign.core import NumberType, TimeUnit
        from timetoalign.timelines import ContinuousPhysicalTimeline

        timeline = self._create_audio_timeline(
            ContinuousPhysicalTimeline,
            length=self.audio_info.duration_seconds,
            unit=TimeUnit.seconds,
            number_type=NumberType.float,
            uid=uid,
            name=name,
        )
        if attach_cmap:
            timeline.add_conversion_map(self.create_samples_to_seconds_map().inverse())
        return timeline

    def create_samples_to_seconds_map(self) -> "SamplesToSeconds":
        """Create a SamplesToSeconds conversion map for this audio.

        Returns:
            A SamplesToSeconds C-map configured with this audio's sample rate.

        Raises:
            RuntimeError: If no audio file has been loaded.
        """
        from timetoalign.maps import SamplesToSeconds

        return SamplesToSeconds(sample_rate=self.audio_info.sample_rate)

    def _create_audio_timeline(
        self,
        timeline_cls: type[_PhysicalTimelineT],
        *,
        length: int | float,
        unit: TimeUnit,
        number_type: NumberType,
        uid: str | None,
        name: str | None,
    ) -> _PhysicalTimelineT:
        """Create a timeline of ``timeline_cls`` named after and describing the file.

        The timeline is locked: the file states its length, so nothing added
        later lengthens it unless ``allow_expansion=True`` is stated.

        Args:
            timeline_cls: The physical timeline class to instantiate.
            length: Length of the timeline in ``unit``.
            unit: Unit of the timeline's axis.
            number_type: Number type of the timeline's axis.
            uid: Unique identifier for the timeline. If None, uses filename.
            name: Human-readable name. If None, uses filename.

        Returns:
            The timeline, carrying the file's metadata.
        """
        info = self.audio_info
        if uid is None and info.source_path is not None:
            uid = info.source_path.stem
        if name is None and info.source_path is not None:
            name = info.source_path.name

        timeline = timeline_cls(
            uid=uid,
            name=name,
            length=length,
            unit=unit,
            number_type=number_type,
            locked=True,
        )
        timeline._metadata = {
            "source_path": str(info.source_path) if info.source_path else None,
            "sample_rate": info.sample_rate,
            "channels": info.channels,
            "duration_seconds": info.duration_seconds,
            "format": info.format,
            "subtype": info.subtype,
            "bits_per_sample": info.bits_per_sample,
            "backend": info.backend,
        }
        return timeline

    # endregion

    # region Convenience Methods

    @classmethod
    def from_file(cls, path: Path | str) -> "AudioLoader":
        """Load an audio file and return the loader (convenience constructor).

        Args:
            path: Path to the audio file.

        Returns:
            An AudioLoader with the file already loaded.

        Examples:
            >>> loader = AudioLoader.from_file("song.wav")
            >>> print(loader.duration_seconds)
        """
        loader = cls()
        loader.load(path)
        return loader

    # endregion

    def __repr__(self) -> str:
        if self._audio_info is None:
            return "AudioLoader(not loaded)"
        return (
            f"AudioLoader("
            f"samples={self._audio_info.n_samples}, "
            f"rate={self._audio_info.sample_rate}Hz, "
            f"duration={self._audio_info.duration_seconds:.2f}s, "
            f"format={self._audio_info.format})"
        )


# endregion
