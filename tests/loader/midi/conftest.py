"""Fixtures for MIDI loader tests."""

from pathlib import Path

import pytest

from timetoalign.testdata import ensure_data

DATA_DIR, VIENNA_DIR = ensure_data("midi", "vienna_1x22")


@pytest.fixture
def performance_midi_dir() -> Path:
    """Path to performance MIDI directory."""
    return DATA_DIR / "performance"


@pytest.fixture
def score_midi_dir() -> Path:
    """Path to score MIDI directory."""
    return DATA_DIR / "score"


@pytest.fixture
def supra_raw_path(performance_midi_dir: Path) -> Path:
    """Path to supra_raw.mid."""
    return performance_midi_dir / "supra_raw.mid"


@pytest.fixture
def chopin_perf_path() -> Path:
    """Path to Chopin_op10_no3_p01.mid."""
    return VIENNA_DIR / "Chopin_op10_no3_p01.mid"


@pytest.fixture
def beethoven_score_path(score_midi_dir: Path) -> Path:
    """Path to beethoven_op18.mid."""
    return score_midi_dir / "beethoven_op18.mid"
