"""EepNotesLoader writes pitch names with ``♯`` and ``♭``.

Validation logic: ``tests/loader/physical/README.md``, "test_eep_notes.py".
"""

from __future__ import annotations

import pytest

from timetoalign.loader.physical.eep_notes import EepNotesLoader
from timetoalign.testdata import ensure_data

NOTES = "1.0 1.1 Eb5\n1.1 1.2 rest\n1.2 1.3 G3,D4,Bb4,F#5\n1.3 1.4 C#4\n1.4 1.5 Fb3\n"


def test_names_are_respelled_at_load(tmp_path):
    path = tmp_path / "Piece_align_vln1.notes"
    path.write_text(NOTES, encoding="utf-8")

    frame = EepNotesLoader.from_file(path).events.to_dataframe()

    expected = ["E♭5", "rest", "G3,D4,B♭4,F♯5", "C♯4", "F♭3"]
    assert frame["pitch"].tolist() == expected
    assert frame["name"].tolist() == expected
    assert frame["staff"].tolist() == [1] * 5


@pytest.mark.slow
def test_the_recordings_names_hold_only_sharp_and_flat_signs():
    directory = (
        ensure_data("score")
        / "beethoven_op18-4iv_multimodal"
        / "StringQuartetEEP_I_Normal"
    )
    loader = EepNotesLoader()
    loader.load(*sorted(directory.glob("*_align_*.notes")))
    pitch = loader.events.to_dataframe()["pitch"]

    assert len(pitch) == 4026
    assert int(pitch.str.count("♭").sum()) == 837
    assert int(pitch.str.count("♯").sum()) == 118
    assert int(pitch.str.count("#").sum()) == 0
    assert int(pitch.str.count("[A-Ga-g]b").sum()) == 0
