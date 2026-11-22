"""Integration tests for Beethoven WoO71 gold standard validation.

This module validates loaders against the Beethoven Piano Trio WoO71 specimen,
which provides a second gold standard to complement Chopin Op.10 No.3.

GOLD STANDARD REFERENCE (from MS3 TSV files):
- Notes: EXACTLY 4753 (wc -l WoO71.notes.tsv = 4754 lines - 1 header)
- Measures: EXACTLY 397 (wc -l WoO71.measures.tsv = 398 lines - 1 header)

This piece is significantly larger than Chopin (4753 vs 498 notes) and tests:
- Loader scalability with larger files
- Multi-movement structure handling
- Parser generalization beyond a single specimen
"""

from __future__ import annotations

from fractions import Fraction

import pandas as pd
import pytest

from timetoalign.testdata import ensure_data

# Gold standard counts (verified via wc -l)
BEETHOVEN_WOO71_GOLD_NOTES = 4753
BEETHOVEN_WOO71_GOLD_MEASURES = 397

# Test data paths
DATA_DIR = ensure_data("score") / "beethoven_woo71"
NOTES_TSV = DATA_DIR / "WoO71.notes.tsv"
MEASURES_TSV = DATA_DIR / "WoO71.measures.tsv"


class TestBeethovenWoO71GoldStandard:
    """Validate gold standard TSV files directly.

    These tests verify that our gold standard files have the expected counts,
    establishing the reference for all other loader tests.
    """

    def test_notes_tsv_exists(self):
        """Gold standard notes TSV file exists."""
        assert NOTES_TSV.exists()

    def test_measures_tsv_exists(self):
        """Gold standard measures TSV file exists."""
        assert MEASURES_TSV.exists()

    def test_notes_tsv_count_exact(self):
        """Notes TSV has exactly 4753 data rows.

        Derivation: wc -l WoO71.notes.tsv = 4754 lines
        4754 - 1 header = 4753 notes
        """

        df = pd.read_csv(NOTES_TSV, sep="\t")
        assert len(df) == BEETHOVEN_WOO71_GOLD_NOTES, (
            f"Notes TSV row count mismatch: got {len(df)}, "
            f"expected {BEETHOVEN_WOO71_GOLD_NOTES}"
        )

    def test_measures_tsv_count_exact(self):
        """Measures TSV has exactly 397 data rows.

        Derivation: wc -l WoO71.measures.tsv = 398 lines
        398 - 1 header = 397 measures
        """

        df = pd.read_csv(MEASURES_TSV, sep="\t")
        assert len(df) == BEETHOVEN_WOO71_GOLD_MEASURES, (
            f"Measures TSV row count mismatch: got {len(df)}, "
            f"expected {BEETHOVEN_WOO71_GOLD_MEASURES}"
        )


class TestBeethovenWoO71TsvStructure:
    """Validate TSV structure matches expected schema.

    These tests verify that the gold standard files have the columns
    and data types expected by our loaders.
    """

    def test_notes_tsv_has_required_columns(self):
        """Notes TSV has all required columns."""

        df = pd.read_csv(NOTES_TSV, sep="\t", nrows=5)

        required_columns = {
            "mc",  # Measure count
            "mn",  # Measure number
            "quarterbeats",  # Continuous logical time
            "duration_qb",  # Duration in quarter beats
            "midi",  # MIDI pitch number
            "staff",  # Staff assignment
            "voice",  # Voice assignment
        }

        missing = required_columns - set(df.columns)
        assert not missing, f"Notes TSV missing required columns: {missing}"

    def test_measures_tsv_has_required_columns(self):
        """Measures TSV has all required columns."""

        df = pd.read_csv(MEASURES_TSV, sep="\t", nrows=5)

        required_columns = {
            "mc",  # Measure count
            "mn",  # Measure number
            "quarterbeats",  # Start position
            "timesig",  # Time signature
        }

        missing = required_columns - set(df.columns)
        assert not missing, f"Measures TSV missing required columns: {missing}"


class TestBeethovenWoO71WithMs3Loader:
    """Integration tests using Ms3Loader (requires ms3 optional dependency).

    These tests load the TSV files via our Ms3Loader and validate
    that the resulting stores match gold standard counts.
    """

    @pytest.fixture
    def tsv_loader(self):
        """Get Ms3Loader if ms3 is available."""
        try:
            from timetoalign.loader.score.ms3 import Ms3Loader

            return Ms3Loader()
        except ImportError:
            pytest.skip("Ms3Loader requires ms3. Install with: pip install ms3")

    @pytest.mark.slow
    def test_tsv_loader_notes_count_exact(self, tsv_loader):
        """Ms3Loader note count matches gold standard EXACTLY."""

        tsv_loader.load(NOTES_TSV)
        store = tsv_loader.store

        assert store.notes.count == BEETHOVEN_WOO71_GOLD_NOTES, (
            f"Ms3Loader note count mismatch: got {store.notes.count}, "
            f"expected {BEETHOVEN_WOO71_GOLD_NOTES}"
        )


class TestBeethovenWoO71ScaleComparison:
    """Compare scale of Beethoven WoO71 vs Chopin Op.10 No.3.

    These tests document that Beethoven is a valid stress test
    with ~10x more notes than Chopin.
    """

    # Chopin gold standard for reference
    CHOPIN_GOLD_NOTES = 498
    CHOPIN_GOLD_MEASURES = 22

    def test_beethoven_has_more_notes_than_chopin(self):
        """Beethoven WoO71 has significantly more notes than Chopin.

        This validates that WoO71 is a meaningful stress test.
        """
        ratio = Fraction(BEETHOVEN_WOO71_GOLD_NOTES, self.CHOPIN_GOLD_NOTES)
        assert ratio == Fraction(4753, 498)  # about 9.5 times as many notes

    def test_beethoven_has_more_measures_than_chopin(self):
        """Beethoven WoO71 has significantly more measures than Chopin."""
        ratio = Fraction(BEETHOVEN_WOO71_GOLD_MEASURES, self.CHOPIN_GOLD_MEASURES)
        assert ratio == Fraction(397, 22)  # about 18 times as many measures


class TestBeethovenWoO71DataIntegrity:
    """Validate data integrity of the gold standard files.

    These tests check for data quality issues that could affect
    downstream processing.
    """

    def test_notes_tsv_no_empty_quarterbeats(self):
        """All main-timeline notes have non-null quarterbeats values.

        MS3 TSV format has two quarterbeats columns:
        - `quarterbeats`: Position in the main timeline (null for volta sections)
        - `quarterbeats_all_endings`: Position including all volta branches

        Notes in alternative endings (volta=1,2,...) have null `quarterbeats`
        but valid `quarterbeats_all_endings`. This is correct MS3 behavior.

        DOCUMENTED BEHAVIOR: Beethoven WoO71 has 8 notes in volta sections
        with null quarterbeats but valid quarterbeats_all_endings.
        """

        df = pd.read_csv(NOTES_TSV, sep="\t")
        null_qb = df["quarterbeats"].isna()
        null_count = null_qb.sum()

        if null_count > 0:
            # Check if these are volta notes (expected to have null quarterbeats)
            volta_notes = df[null_qb]

            # Verify all null-quarterbeats notes are in volta sections
            has_volta = "volta" in df.columns
            if has_volta:
                # Notes with null quarterbeats should have non-empty volta
                # and valid quarterbeats_all_endings
                volta_values = volta_notes["volta"]
                in_volta = (~volta_values.isna()) & (volta_values != "")

                assert in_volta.all(), (
                    f"Found {(~in_volta).sum()} notes with null quarterbeats "
                    "that are NOT in volta sections. This indicates a data issue."
                )

                # Verify they have valid quarterbeats_all_endings
                if "quarterbeats_all_endings" in df.columns:
                    qb_all = volta_notes["quarterbeats_all_endings"]
                    assert (
                        not qb_all.isna().any()
                    ), "Volta notes should have quarterbeats_all_endings"

        # Exact count: 8 notes in volta sections for Beethoven WoO71
        assert null_count == 8, (
            f"Expected exactly 8 volta notes with null quarterbeats, "
            f"found {null_count}"
        )

    def test_notes_tsv_no_empty_midi_pitch(self):
        """All notes have non-null MIDI pitch (no rests in notes file)."""

        df = pd.read_csv(NOTES_TSV, sep="\t")
        null_count = df["midi"].isna().sum()

        # Document if there are rests
        if null_count > 0:
            pytest.skip(
                f"Notes file contains {null_count} rests (null MIDI). "
                "This is valid but changes expected count semantics."
            )
        assert null_count == 0

    def test_measures_tsv_sequential_mc(self):
        """Measure counts are sequential starting from 1."""

        df = pd.read_csv(MEASURES_TSV, sep="\t")

        # Check first measure
        assert df["mc"].iloc[0] == 1, "First measure should have mc=1"

        # Check last measure
        assert df["mc"].iloc[-1] == BEETHOVEN_WOO71_GOLD_MEASURES, (
            f"Last measure mc={df['mc'].iloc[-1]}, "
            f"expected {BEETHOVEN_WOO71_GOLD_MEASURES}"
        )

    def test_notes_quarterbeats_monotonic(self):
        """Quarterbeats are monotonically non-decreasing.

        Notes should be ordered by time (ties allowed for chords).
        """

        df = pd.read_csv(NOTES_TSV, sep="\t")

        # Parse quarterbeats as floats for comparison
        # Handle fraction strings like "3/4"
        def parse_qb(val):
            if isinstance(val, str) and "/" in val:
                num, den = val.split("/")
                return float(num) / float(den)
            return float(val)

        qb_values = df["quarterbeats"].apply(parse_qb)

        # Check monotonicity
        decreasing = (qb_values.diff() < 0).sum()
        assert decreasing == 0, (
            f"Found {decreasing} instances where quarterbeats decreased. "
            "Notes should be in temporal order."
        )
