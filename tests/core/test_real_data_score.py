"""Real-world data validation for the score scalar/Field surface.

This module validates paired ``XField`` implementations against published
musical corpora rather than synthetic test data.

Schema invariants pinned here (represent pitch exactly once):
* A spelled score faithfully supports ``SpecificPitch``, so
  ``specific_pitch`` ({step, alter, octave, cents}) is the sole default
  semantic pitch field on ``NoteEventData``.
* The source MIDI number is kept only as a non-default raw ``midi``
  int64 column, which affords an ``EnharmonicPitch`` view on request
  (``notes.get_field(EnharmonicPitch)``, whose ``.name`` is ``"midi"``).
  No redundant ``midi_pitch`` struct column is stored.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timetoalign.core.events import (
    DcmlHarmony,
    EnharmonicPitch,
    EnharmonicPitchField,
)
from timetoalign.loader.score.ms3 import Ms3Loader
from timetoalign.testdata import ensure_data

# ---------------------------------------------------------------------------
# Paths to test specimens
# ---------------------------------------------------------------------------

VIENNA_DIR, SCORE_DIR = ensure_data("vienna_1x22", "score")
MS3_DIR = VIENNA_DIR / "ms3"

CHOPIN_NOTES_TSV = MS3_DIR / "chopin_op10_no3.notes.tsv"
CHOPIN_MEASURES_TSV = MS3_DIR / "chopin_op10_no3.measures.tsv"
CHOPIN_XML = VIENNA_DIR / "Chopin_op10_no3.musicxml"

BEETHOVEN_DIR = SCORE_DIR / "beethoven_op18-4iv_multimodal" / "ABC"
BEETHOVEN_HARMONIES_TSV = BEETHOVEN_DIR / "n04op18-4_04.harmonies.tsv"


# ---------------------------------------------------------------------------
# Chopin Op.10/3 -- Notes (EnharmonicPitchField from real TSV)
# ---------------------------------------------------------------------------


class TestChopinEnharmonicPitchFieldFromTSV:
    """Validate EnharmonicPitchField extraction from the Chopin Op.10/3 notes TSV."""

    @pytest.fixture
    def chopin_store(self):
        loader = Ms3Loader().load(CHOPIN_NOTES_TSV)
        return loader.store

    def test_note_count_exact(self, chopin_store) -> None:
        notes = chopin_store.notes
        df = notes.to_dataframe()
        note_rows = df[df["event_type"] == "Note"]
        assert len(note_rows) == 498, f"Expected 498 notes, got {len(note_rows)}"

    def test_first_note_midi_number(self, chopin_store) -> None:
        """First note (B3) carries the source MIDI number 59 in the raw ``midi`` int."""
        notes = chopin_store.notes
        # Represent-once: no redundant midi_pitch struct column.
        assert "midi_pitch" not in notes.table.column_names
        df = notes.to_dataframe()
        note_rows = df[df["event_type"] == "Note"].reset_index(drop=True)
        first_midi = note_rows.iloc[0]["midi"]
        assert int(first_midi) == 59, f"Expected midi=59 (B3), got {first_midi}"

    def test_paired_field_construction_from_store(self, chopin_store) -> None:
        # EnharmonicPitch is afforded over the raw ``midi`` int column.
        ef = chopin_store.notes.get_field(EnharmonicPitch)
        assert isinstance(ef, EnharmonicPitchField)
        assert ef.name == "midi"
        # One row per row of the notes TSV: 498, every one a note (no rests).
        assert len(ef) == 498

    def test_paired_field_first_element(self, chopin_store) -> None:
        """The afforded EnharmonicPitchField[0] is EnharmonicPitch(B3)."""
        ef = chopin_store.notes.get_field(EnharmonicPitch)
        first = ef[0]
        assert first is not None
        assert isinstance(first, EnharmonicPitch)
        assert first.midi_number == 59
        assert first.pitch_class == 11


# ---------------------------------------------------------------------------
# Chopin Op.10/3 -- Measures
# ---------------------------------------------------------------------------


class TestChopinMeasuresFromTSV:
    @pytest.fixture
    def chopin_measure_store(self):
        loader = Ms3Loader().load(CHOPIN_MEASURES_TSV)
        return loader.store

    def test_measure_count_exact(self, chopin_measure_store) -> None:
        measures = chopin_measure_store.measures
        assert len(measures) == 22, f"Expected 22 measures, got {len(measures)}"

    def test_first_measure_mc(self, chopin_measure_store) -> None:
        measures = chopin_measure_store.measures
        df = measures.to_dataframe()
        assert df.iloc[0]["mc"] == 1

    def test_first_measure_mn(self, chopin_measure_store) -> None:
        measures = chopin_measure_store.measures
        df = measures.to_dataframe()
        assert df.iloc[0]["mn"] == "1"

    def test_first_measure_time_signature(self, chopin_measure_store) -> None:
        measures = chopin_measure_store.measures
        df = measures.to_dataframe()
        assert df.iloc[0]["timesig"] == "2/4"
        assert df.iloc[0]["timesig_num"] == 2
        assert df.iloc[0]["timesig_den"] == 4

    def test_last_measure_mc(self, chopin_measure_store) -> None:
        measures = chopin_measure_store.measures
        df = measures.to_dataframe()
        assert df.iloc[-1]["mc"] == 22

    def test_all_measures_have_2_4_timesig(self, chopin_measure_store) -> None:
        measures = chopin_measure_store.measures
        df = measures.to_dataframe()
        timesigs = df["timesig"].unique().tolist()
        assert timesigs == ["2/4"], f"Expected only 2/4, got {timesigs}"


# ---------------------------------------------------------------------------
# Beethoven Op.18/4 -- Harmonies (DcmlHarmony.from_row roundtrip)
# ---------------------------------------------------------------------------


class TestBeethovenHarmonies:
    """Validate DCML harmony import via the import shape (DcmlStorageSchema).

    The DCML harmonies.tsv shape (figbass, bass_note, form, ...) maps to
    the canonical ``DcmlHarmony`` model via ``DcmlHarmony.from_row``.
    """

    @pytest.fixture
    def dcml_rows(self) -> list[DcmlHarmony]:
        df = pd.read_csv(BEETHOVEN_HARMONIES_TSV, sep="\t")
        rows: list[DcmlHarmony] = []
        for _, row in df.iterrows():
            d = {
                "label": (
                    str(row.get("label", "")) if pd.notna(row.get("label")) else None
                ),
                "globalkey": (
                    str(row.get("globalkey", ""))
                    if pd.notna(row.get("globalkey"))
                    else None
                ),
                "localkey": (
                    str(row.get("localkey", ""))
                    if pd.notna(row.get("localkey"))
                    else None
                ),
                "numeral": (
                    str(row.get("numeral", ""))
                    if pd.notna(row.get("numeral"))
                    else None
                ),
                "form": str(row.get("form", "")) if pd.notna(row.get("form")) else None,
                "figbass": (
                    str(row.get("figbass", ""))
                    if pd.notna(row.get("figbass"))
                    else None
                ),
                "chord_type": (
                    str(row.get("chord_type", ""))
                    if pd.notna(row.get("chord_type"))
                    else None
                ),
                "root": int(row["root"]) if pd.notna(row.get("root")) else None,
                "bass_note": (
                    int(row["bass_note"]) if pd.notna(row.get("bass_note")) else None
                ),
            }
            h = DcmlHarmony.from_row(d)
            if h is not None:
                rows.append(h)
        return rows

    def test_first_label(self, dcml_rows: list[DcmlHarmony]) -> None:
        assert (
            dcml_rows[0].label == "c.i"
        ), f"Expected 'c.i', got {dcml_rows[0].label!r}"

    def test_first_globalkey(self, dcml_rows: list[DcmlHarmony]) -> None:
        assert dcml_rows[0].globalkey == "c"

    def test_first_numeral(self, dcml_rows: list[DcmlHarmony]) -> None:
        assert dcml_rows[0].numeral == "i"

    def test_second_harmony_v65(self, dcml_rows: list[DcmlHarmony]) -> None:
        assert dcml_rows[1].label == "V65"

    def test_second_harmony_chord_type(self, dcml_rows: list[DcmlHarmony]) -> None:
        assert dcml_rows[1].chord_type == "Mm7"

    def test_harmony_count(self, dcml_rows: list[DcmlHarmony]) -> None:
        # One scalar per row of the harmonies TSV.
        assert len(dcml_rows) == 386


# ---------------------------------------------------------------------------
# Parquet round-trip with real data
# ---------------------------------------------------------------------------


class TestEnharmonicPitchFieldParquetRoundtrip:
    def test_round_trip(self, tmp_path: Path) -> None:
        parquet_path = tmp_path / "chopin_pitches.parquet"

        loader = Ms3Loader().load(CHOPIN_NOTES_TSV)
        # The EnharmonicPitch view is afforded over the raw ``midi`` int
        # column (represent once).  Materialise it, then round-trip the
        # afforded {midi_number} struct through Parquet.
        ef = loader.store.notes.get_field(EnharmonicPitch)
        assert isinstance(ef, EnharmonicPitchField)

        ep_col = ef.to_pyarrow()
        ep_field = ef.to_field()
        table = pa.table({ep_field.name: ep_col}, schema=pa.schema([ep_field]))
        pq.write_table(table, str(parquet_path))
        table_back = pq.read_table(str(parquet_path))

        col_back = table_back.column(ep_field.name)
        field_back = table_back.schema.field(ep_field.name)
        ef2 = EnharmonicPitchField.from_field((col_back, field_back))

        first = ef2[0]
        assert first is not None
        assert first.midi_number == 59
        assert first.pitch_class == 11

        assert len(ef2) == len(ef)

        for i in range(0, len(ef), 50):
            orig = ef[i]
            restored = ef2[i]
            if orig is None:
                assert restored is None, f"Mismatch at index {i}: expected None"
            else:
                assert restored is not None, f"Mismatch at index {i}: expected non-None"
                assert orig.midi_number == restored.midi_number, (
                    f"midi_number mismatch at index {i}: "
                    f"{orig.midi_number} != {restored.midi_number}"
                )


# ---------------------------------------------------------------------------
# Cross-loader pitch consistency (optional Music21)
# ---------------------------------------------------------------------------


class TestCrossLoaderPitchConsistency:
    def test_cross_loader_pitch_consistency(self) -> None:
        """Load Chopin via Ms3Loader AND Music21Loader, compare midi pitch arrays."""
        pytest.importorskip("music21")

        from timetoalign.loader.score.music21 import Music21Loader

        tsv_loader = Ms3Loader().load(CHOPIN_NOTES_TSV)
        tsv_df = tsv_loader.store.notes.to_dataframe()
        tsv_notes = tsv_df[tsv_df["event_type"] == "Note"].reset_index(drop=True)

        m21_loader = Music21Loader().load(CHOPIN_XML)
        m21_df = m21_loader.store.notes.to_dataframe()
        m21_notes = m21_df[m21_df["event_type"] == "Note"].reset_index(drop=True)

        assert len(tsv_notes) == 498
        assert len(m21_notes) == 498

        def extract_sorted_midi(df):
            df = df.copy()
            # The source MIDI number lives in the raw ``midi`` int column
            # (represent once; EnharmonicPitch is afforded over it).
            df["midi_number"] = df["midi"].apply(
                lambda x: int(x) if pd.notna(x) else None
            )
            if "start" in df.columns:
                df["start_val"] = df["start"].apply(
                    lambda x: x["value"] if isinstance(x, dict) else None
                )
                df = df.sort_values(by=["start_val", "midi_number"]).reset_index(
                    drop=True
                )
            return df["midi_number"].tolist()

        tsv_pitches = extract_sorted_midi(tsv_notes)
        m21_pitches = extract_sorted_midi(m21_notes)

        mismatches = [
            (i, t, m)
            for i, (t, m) in enumerate(zip(tsv_pitches, m21_pitches))
            if t != m
        ]
        assert len(mismatches) == 0, (
            f"MIDI pitch mismatches at {len(mismatches)} positions. "
            f"First: index={mismatches[0][0]}, TSV={mismatches[0][1]}, M21={mismatches[0][2]}"
        )
