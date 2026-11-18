"""Cross-Validation Test Suite for Score Loaders.

This module validates that all score loaders produce consistent results by
cross-comparing their output on the same musical work (Chopin Op. 10 No. 3).

## Validation Rationale

The cross-validation is trustworthy because:

1. **TSV as Gold Standard**: ms3's TSV files are derived from MuseScore (.mscx)
   files using established parsing. They represent "ground truth" for temporal
   positions, durations, and pitch spelling.

2. **Independent Implementations**: Each loader uses a different parsing library:
   - TSV: ms3 (direct tabular load)
   - Partitura: partitura (specialized MIR library)
   - Music21: music21 (general-purpose musicology library)

   Agreement across 3 independent implementations strongly suggests correctness.

3. **Overlapping Columns**: We compare fields that all loaders extract:
   - quarterbeats: High trust (computed from divs/ticks)
   - duration_qb: High trust (computed from note duration)
   - midi: Very high trust (the raw MIDI number; a spelled score keeps
     it as a non-default raw int that affords EnharmonicPitch)
   - mc: Medium trust (derived from measure structure)

NOTE: Tests for quarterbeats, duration_qb, and mc cross-validation were removed
because the unified schema now uses 'start' and 'duration' struct columns instead
of the original per-field columns. Only note count and MIDI pitch (exact match)
cross-validation remain.
"""

from pathlib import Path

import pandas as pd
import pytest

from timetoalign.core.events import EnharmonicPitch, MidiPitch
from timetoalign.loader.score.ms3 import Ms3Loader
from timetoalign.loader.score.music21 import Music21Loader
from timetoalign.loader.score.partitura import PartituraLoader
from timetoalign.loader.score.store import ScoreStore
from timetoalign.testdata import ensure_data

DATA_DIR = ensure_data("vienna_1x22")
MS3_DIR = DATA_DIR / "ms3"
CHOPIN_XML = DATA_DIR / "Chopin_op10_no3.musicxml"
CHOPIN_TSV = MS3_DIR / "chopin_op10_no3.notes.tsv"


def extract_notes_df(store: ScoreStore, loader_name: str) -> pd.DataFrame:
    """Extract notes as sorted DataFrame with flattened struct columns."""
    df = store.notes.to_dataframe()

    # Filter to Notes only (exclude Rests for fair comparison)
    df = df[df["event_type"] == "Note"].reset_index(drop=True)

    # Flatten rational structs to floats for comparison
    for col in ["quarterbeats", "duration_qb", "mc_onset", "mn_onset"]:
        if col in df.columns:
            df[f"{col}_float"] = df[col].apply(
                lambda x: (
                    x["numerator"] / x["denominator"]
                    if isinstance(x, dict) and x.get("denominator")
                    else None
                )
            )

    # The source MIDI number lives in the raw ``midi`` int column
    # (represent once; EnharmonicPitch is afforded over it).
    if "midi" in df.columns:
        df["midi_number"] = df["midi"].apply(lambda x: int(x) if pd.notna(x) else None)

    # Sort deterministically by (quarterbeats, pitch)
    sort_cols = ["quarterbeats_float", "midi_number"]
    df = df.sort_values(by=[c for c in sort_cols if c in df.columns]).reset_index(
        drop=True
    )

    return df


@pytest.fixture
def gold_df():
    """Load TSV gold standard."""
    loader = Ms3Loader().load(CHOPIN_TSV)
    return extract_notes_df(loader.store, "TSV")


@pytest.fixture
def partitura_df():
    """Load Partitura MusicXML."""
    loader = PartituraLoader().load(CHOPIN_XML)
    return extract_notes_df(loader.store, "Partitura")


@pytest.fixture
def music21_df():
    """Load Music21 MusicXML."""
    loader = Music21Loader().load(CHOPIN_XML)
    return extract_notes_df(loader.store, "Music21")


class TestNoteCountConsistency:
    """Verify all loaders return consistent note counts."""

    def test_note_counts_exact_match(self, gold_df, partitura_df, music21_df):
        """All loaders should return exactly 498 notes."""
        assert len(gold_df) == 498, f"TSV gold has {len(gold_df)} notes, expected 498"
        assert (
            len(partitura_df) == 498
        ), f"Partitura has {len(partitura_df)} notes, expected 498"
        assert (
            len(music21_df) == 498
        ), f"Music21 has {len(music21_df)} notes, expected 498"


class TestMidiPitchExact:
    """MIDI pitch must match exactly (no tolerance).

    Rationale: the MIDI number is unambiguous - a note is either the right pitch
    or wrong. There is no room for interpretation or rounding. This is our
    highest-confidence validation field.
    """

    def test_partitura_pitch(self, gold_df, partitura_df):
        """Partitura MIDI pitch matches TSV gold exactly."""
        mismatches = gold_df["midi_number"] != partitura_df["midi_number"]

        if mismatches.any():
            first_idx = mismatches.idxmax()
            pytest.fail(
                f"MIDI pitch mismatch at index {first_idx}: "
                f"Gold={gold_df['midi_number'].iloc[first_idx]}, "
                f"Partitura={partitura_df['midi_number'].iloc[first_idx]}"
            )

    def test_music21_pitch(self, gold_df, music21_df):
        """Music21 MIDI pitch matches TSV gold exactly."""
        mismatches = gold_df["midi_number"] != music21_df["midi_number"]

        if mismatches.any():
            first_idx = mismatches.idxmax()
            pytest.fail(
                f"MIDI pitch mismatch at index {first_idx}: "
                f"Gold={gold_df['midi_number'].iloc[first_idx]}, "
                f"Music21={music21_df['midi_number'].iloc[first_idx]}"
            )

    @pytest.mark.parametrize(
        ("loader_class", "source"),
        [
            (Ms3Loader, CHOPIN_TSV),
            (PartituraLoader, CHOPIN_XML),
            (Music21Loader, CHOPIN_XML),
        ],
    )
    def test_loader_affords_both_midi_number_views(
        self, loader_class, source: Path
    ) -> None:
        """Every score loader exposes EP and MP over identical MIDI numbers."""
        events = loader_class.from_file(source).get_events()
        enharmonic = events.get_field(EnharmonicPitch)
        midi = events.get_field(MidiPitch)

        assert len(enharmonic) == len(midi) == 498
        assert [enharmonic[i].midi_number for i in range(498)] == [
            midi[i].midi_number for i in range(498)
        ]
        if loader_class is Ms3Loader:
            assert enharmonic[0].midi_number == midi[0].midi_number == 59
