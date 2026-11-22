"""ZERO TOLERANCE correctness tests for vectorized TabularLoader.

This module validates that the vectorized loader produces EXACT results
matching the gold standard data from specimens.

ZERO TOLERANCE POLICY:
- Exact counts required, no approximations
- Exact coordinate values, no tolerances unless mathematically justified
- Every mismatch must be investigated and fixed

Test specimens:
- beethoven_woo71/WoO71.notes.tsv: 4753 notes (ms3 format)
- beethoven_woo71/WoO71.measures.tsv: 397 measures (ms3 format)
- rachmaninoff_concerto2/.../notes.tsv: 14315 notes (ms3 format)
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from timetoalign.core import NumberType, TimeUnit
from timetoalign.loader.tabular import TsvLoader
from timetoalign.testdata import ensure_data


class Ms3TsvFixture(TsvLoader):
    """Generic TsvLoader configuration for ms3 coordinate gold vectors."""

    id_column = None
    name_column = "name"
    start_column = "quarterbeats_all_endings"
    _fallback_start_column = "quarterbeats"
    end_column = None
    duration_column = "duration"
    event_type_column = None
    default_event_type = "Note"
    _default_unit = TimeUnit.quarters
    coordinate_type = NumberType.fraction


# region Test Data Paths


@pytest.fixture
def specimens_base() -> Path:
    """Get base path for specimens (the ``score`` test-data corpus)."""
    return ensure_data("score")


@pytest.fixture
def beethoven_notes(specimens_base: Path) -> Path:
    """Get path to Beethoven WoO71 notes TSV."""
    path = specimens_base / "beethoven_woo71" / "WoO71.notes.tsv"
    return path


@pytest.fixture
def beethoven_measures(specimens_base: Path) -> Path:
    """Get path to Beethoven WoO71 measures TSV."""
    path = specimens_base / "beethoven_woo71" / "WoO71.measures.tsv"
    return path


@pytest.fixture
def rachmaninoff_notes(specimens_base: Path) -> Path:
    """Get path to Rachmaninoff Concerto 2 notes TSV."""
    path = (
        specimens_base
        / "rachmaninoff_concerto2"
        / "score"
        / "Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff.notes.tsv"
    )
    return path


# endregion


# region ZERO TOLERANCE Event Count Tests


class TestZeroToleranceEventCounts:
    """Validate exact event counts from gold standard specimens.

    ZERO TOLERANCE: Exact counts required, no ranges or approximations.
    """

    def test_beethoven_woo71_notes_exact_count(self, beethoven_notes: Path) -> None:
        """Validate exact note count for Beethoven WoO71.

        Gold standard: WoO71.notes.tsv from ms3 parser
        Expected: 4753 note events (4754 lines - 1 header)
        """
        loader = Ms3TsvFixture()
        loader.load(beethoven_notes)

        # ZERO TOLERANCE: Exact count required
        assert len(loader.events) == 4753, (
            f"Expected 4753 notes, got {len(loader.events)}. "
            "This is a GOLD STANDARD mismatch - investigate!"
        )

    def test_beethoven_woo71_measures_exact_count(
        self, beethoven_measures: Path
    ) -> None:
        """Validate exact measure count for Beethoven WoO71.

        Gold standard: WoO71.measures.tsv from ms3 parser
        Expected: 397 measure events (398 lines - 1 header)
        """
        loader = Ms3TsvFixture()
        loader.load(beethoven_measures)

        # ZERO TOLERANCE: Exact count required
        assert len(loader.events) == 397, (
            f"Expected 397 measures, got {len(loader.events)}. "
            "This is a GOLD STANDARD mismatch - investigate!"
        )

    def test_rachmaninoff_notes_exact_count(self, rachmaninoff_notes: Path) -> None:
        """Validate exact note count for Rachmaninoff Concerto 2.

        Gold standard: Rachmaninoff notes.tsv from ms3 parser
        Expected: 14315 note events (14316 lines - 1 header)
        """
        loader = Ms3TsvFixture()
        loader.load(rachmaninoff_notes)

        # ZERO TOLERANCE: Exact count required
        assert len(loader.events) == 14315, (
            f"Expected 14315 notes, got {len(loader.events)}. "
            "This is a GOLD STANDARD mismatch - investigate!"
        )


# endregion


# region ZERO TOLERANCE Coordinate Tests


class TestZeroToleranceCoordinates:
    """Validate exact coordinate values from gold standard specimens.

    ZERO TOLERANCE: Exact values required for fraction coordinates.
    """

    def test_beethoven_first_note_coordinate(self, beethoven_notes: Path) -> None:
        """Validate first note starts at quarterbeat 0.

        Gold standard: First data row has quarterbeats=0
        """
        loader = Ms3TsvFixture()
        loader.load(beethoven_notes)

        # Get first event's start coordinate
        table = loader.events.table
        first_start = table["start"][0]

        # ZERO TOLERANCE: Exact value required
        value = first_start["value"].as_py()
        assert value == 0.0, f"First note should start at 0, got {value}"

        # Check fraction components
        num = first_start["numerator"].as_py()
        den = first_start["denominator"].as_py()

        assert num == 0, f"First note numerator should be 0, got {num}"
        assert den == 1, f"First note denominator should be 1, got {den}"

    def test_beethoven_coordinate_range(self, beethoven_notes: Path) -> None:
        """Validate coordinate range matches source data.

        Gold standard: quarterbeats range from 0 to ~875 (last note start)
        """
        loader = Ms3TsvFixture()
        loader.load(beethoven_notes)

        coord_range = loader.events.coordinate_range()
        assert coord_range is not None

        # ZERO TOLERANCE: Start must be 0
        assert (
            coord_range[0] == 0.0
        ), f"Min coordinate should be 0, got {coord_range[0]}"

        # ZERO TOLERANCE: End coordinate is last onset + duration = 877.75
        # (gold standard: WoO71.notes.tsv, last event onset + duration)
        assert (
            coord_range[1] == 877.75
        ), f"Max coordinate should be 877.75, got {coord_range[1]}"

    def test_fraction_roundtrip_preservation(self, beethoven_notes: Path) -> None:
        """Validate fractions are preserved exactly (no floating point loss).

        Gold standard: ms3 uses exact fractions like 1/2, 3/4, 7/8
        The loader should preserve numerator/denominator exactly.
        """
        loader = Ms3TsvFixture()
        loader.load(beethoven_notes)

        table = loader.events.table
        start_col = table["start"]

        # Sample some events and verify fraction round-trip
        errors = []
        sample_indices = [0, 1, 4, 9, 100, 500, 1000, 2000, 4000]

        for idx in sample_indices:
            if idx >= len(start_col):
                break

            coord = start_col[idx]
            value = coord["value"].as_py()
            num = coord["numerator"].as_py()
            den = coord["denominator"].as_py()

            if num is not None and den is not None:
                # Reconstruct fraction and compare
                frac = Fraction(num, den)
                reconstructed = float(frac)

                # IEEE 754: float(Fraction(num, den)) may differ from the stored
                # value at the least-significant bit because the original float
                # was rounded during serialisation and Fraction→float performs
                # an independent division.  1e-15 ≈ 4.5 × machine epsilon for
                # float64 (2.22e-16), covering at most one ULP of rounding.
                if abs(reconstructed - value) > 1e-15:
                    errors.append(
                        f"idx={idx}: {num}/{den}={reconstructed} != value={value}"
                    )

        assert not errors, "Fraction round-trip failures:\n" + "\n".join(errors)


# endregion


# region ZERO TOLERANCE Temporal Type Tests


class TestZeroToleranceTemporalTypes:
    """Validate temporal type inference is correct.

    ZERO TOLERANCE: Temporal types must match the data exactly.
    """

    def test_beethoven_notes_mostly_intervals(self, beethoven_notes: Path) -> None:
        """Validate most notes are intervals (have duration).

        Gold standard: ms3 notes.tsv includes duration_qb column.
        Notes with valid duration should be intervals.
        """
        loader = Ms3TsvFixture()
        loader.load(beethoven_notes)

        types = loader.count_events_by_temporal_type()

        # All notes should be intervals (have duration)
        # The Ms3TsvFixture uses quarterbeats_all_endings column, so all notes
        # including those in volta brackets have valid start coordinates.
        interval_count = types.get("interval", 0)
        instant_count = types.get("instant", 0)

        # All 4753 notes of the TSV have a duration, so all are intervals
        assert (interval_count, instant_count) == (4753, 0)
        assert interval_count == len(loader.events), (
            f"Expected all {len(loader.events)} to be intervals, "
            f"got {interval_count} intervals and {instant_count} instants"
        )

    def test_beethoven_all_note_event_type(self, beethoven_notes: Path) -> None:
        """Validate all events have event_type='Note'.

        Gold standard: Ms3TsvFixture sets default_event_type='Note'.
        """
        loader = Ms3TsvFixture()
        loader.load(beethoven_notes)

        types = loader.count_events_by_type()

        # ZERO TOLERANCE: All events must be 'Note'
        assert types == {
            "Note": 4753
        }, f"Expected all 4753 events to be 'Note', got {types}"


# endregion


# region ZERO TOLERANCE Unit Tests


class TestZeroToleranceUnits:
    """Validate unit metadata is correct."""

    def test_ms3_loader_uses_quarters(self, beethoven_notes: Path) -> None:
        """Validate Ms3TsvFixture uses TimeUnit.quarters.

        Gold standard: ms3 quarterbeats are in quarter note units.
        """
        loader = Ms3TsvFixture()
        loader.load(beethoven_notes)

        # ZERO TOLERANCE: Unit must be quarters
        assert (
            loader.unit == TimeUnit.quarters
        ), f"Expected TimeUnit.quarters, got {loader.unit}"

    def test_ms3_loader_uses_fraction_type(self, beethoven_notes: Path) -> None:
        """Validate Ms3TsvFixture uses NumberType.fraction.

        Gold standard: ms3 quarterbeats are fractions (1/2, 3/4, etc.)
        """
        loader = Ms3TsvFixture()
        loader.load(beethoven_notes)

        # ZERO TOLERANCE: Number type must be fraction
        assert (
            loader.number_type == NumberType.fraction
        ), f"Expected NumberType.fraction, got {loader.number_type}"


# endregion


# region Cross-File Validation


class TestCrossFileValidation:
    """Validate consistency across related files."""

    def test_notes_and_measures_coordinate_compatibility(
        self, beethoven_notes: Path, beethoven_measures: Path
    ) -> None:
        """Validate notes and measures use same coordinate system.

        Gold standard: Both files from same ms3 export use quarterbeats.
        """
        notes_loader = Ms3TsvFixture()
        notes_loader.load(beethoven_notes)

        measures_loader = Ms3TsvFixture()
        measures_loader.load(beethoven_measures)

        # Both should use same unit
        assert (
            notes_loader.unit == measures_loader.unit
        ), f"Notes use {notes_loader.unit}, measures use {measures_loader.unit}"

        # Both should start at 0
        notes_range = notes_loader.events.coordinate_range()
        measures_range = measures_loader.events.coordinate_range()

        assert notes_range is not None
        assert measures_range is not None

        assert notes_range[0] == 0.0, f"Notes should start at 0, got {notes_range[0]}"
        assert (
            measures_range[0] == 0.0
        ), f"Measures should start at 0, got {measures_range[0]}"


# endregion
