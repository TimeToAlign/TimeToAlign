"""EepNotesLoader: Loader for EEP (Expressive Ensemble Performance) .notes files.

This module provides a :class:`CsvLoader` subclass for the ``.notes`` format
used by the EEP dataset to store per-instrument note alignment data.

File Format:
    Whitespace-separated, no header, three columns per line::

        onset_seconds  offset_seconds  pitch
        1.0 1.1 Eb5
        1.1 1.216644 F5
        31.541633 31.833288 rest
        170.266644 172.400998 G3,D4,B4,F5

    - **onset/offset**: float seconds
    - **pitch**: note name (e.g., ``Eb5``, ``C#4``), ``rest``, or
      comma-separated chord (e.g., ``G3,D4,B4,F5``); the loader respells
      the accidentals as ``♯``/``♭`` (``E♭5``, ``C♯4``)

Typical Usage:
    Four ``.notes`` files per recording (one per instrument: vln1, vln2, vla, cello).
    The instrument/staff assignment is inferred from the filename suffix.

This loader demonstrates how easy it is to create a custom loader by subclassing
:class:`CsvLoader` with just a few class-variable overrides and a
:meth:`_read_dataframe` override for the headerless format.

Reference:
    ``dashboard/processing/notebooks/repovizz_parsing.py``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

import pandas as pd

from timetoalign.core import TimeUnit
from timetoalign.core.events import canonical_pitch_names
from timetoalign.loader.tabular.csv import CsvLoader

module_logger = logging.getLogger(__name__)

# Mapping from filename suffix to staff number (standard string quartet order)
INSTRUMENT_TO_STAFF: dict[str, int] = {
    "vln1": 1,
    "vln2": 2,
    "vla": 3,
    "cello": 4,
}


class EepNotesLoader(CsvLoader):
    """Loader for EEP ``.notes`` alignment files.

    Subclasses :class:`CsvLoader` with configuration for the whitespace-separated,
    headerless ``.notes`` format. Each line contains onset (seconds), offset (seconds),
    and pitch (note name, "rest", or comma-separated chord).

    The loader:
    - Assigns column names ``start``, ``end``, ``pitch``
    - Infers the ``staff`` number from the filename (e.g., ``*_vln1.notes`` -> staff 1)
    - Respells every note name's accidentals as ``♯``/``♭`` (``Eb5`` ->
      ``E♭5``), the spelling of the library's score loaders, so names
      compare equal across loaders
    - Preserves "rest" entries and comma-separated chords

    Chord explosion (splitting ``G3,D4,B4,F5`` into separate rows) is left to
    post-processing, typically during note matching.

    Examples:
        >>> loader = EepNotesLoader.from_file("EEP_Normal_align_vln1.notes")
        >>> len(loader)
        1266
        >>> loader.events.summary()

        >>> # Load all 4 instruments for a recording
        >>> loader = EepNotesLoader()
        >>> loader.load(*sorted(eep_dir.glob("*_align_*.notes")))
        >>> len(loader)
        4026
    """

    # Whitespace-separated, no header
    delimiter: ClassVar[str] = r"\s+"
    header_row: ClassVar[int] = -1  # Sentinel for "no header"

    # Column mapping: onset -> start, offset -> end, pitch -> extra
    start_column = "start"
    end_column = "end"
    id_column: ClassVar[str | None] = None  # Auto-generate
    name_column: ClassVar[str | None] = "pitch"
    event_type_column: ClassVar[str | None] = None
    default_event_type: ClassVar[str] = "Note"

    # Coordinate configuration -- the representation follows the unit.
    _default_unit: ClassVar[TimeUnit] = TimeUnit.seconds

    # column_specs: pitch is already captured via name_column; pitch
    # is preserved as a string column (chord/rest tokens like
    # "G3,D4,B4,F5"), and staff is the integer derived from the
    # filename suffix in _read_dataframe.
    column_specs = {"pitch": str, "staff": int}

    def _read_dataframe(self, source: Path) -> pd.DataFrame:
        """Read a headerless .notes file with staff inference from filename.

        Args:
            source: Path to the .notes file.

        Returns:
            DataFrame with columns: start, end, pitch (accidentals
            respelled ``♯``/``♭``), staff.
        """
        df = pd.read_csv(
            source,
            sep=r"\s+",
            header=None,
            names=["start", "end", "pitch"],
            encoding=self.encoding,
            engine="python",
        )
        df["pitch"] = canonical_pitch_names(df["pitch"], sep=",")

        # Infer staff from filename suffix (e.g., "*_align_vln1.notes" -> 1)
        stem = source.stem  # e.g., "StringQuartetEEP_I_Normal_align_vln1"
        staff = 0  # default: unknown
        for instrument, staff_num in INSTRUMENT_TO_STAFF.items():
            if stem.endswith(instrument):
                staff = staff_num
                break

        df["staff"] = staff
        return df
