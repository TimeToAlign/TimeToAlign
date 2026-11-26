"""Tests for scoped warning hygiene around third-party backends."""

from __future__ import annotations

import warnings
from pathlib import Path

from timetoalign.core import resolve_id
from timetoalign.core.backends import suppressed_backend_warnings
from timetoalign.loader.midi import ScoreMidiLoader
from timetoalign.loader.score import PartituraLoader
from timetoalign.testdata import ensure_data


def _partitura_warning_messages(recorded: list[warnings.WarningMessage]) -> list[str]:
    """Return messages whose source file belongs to the Partitura package."""
    return [
        str(item.message)
        for item in recorded
        if "partitura" in Path(item.filename).parts
    ]


def test_partitura_score_load_emits_no_backend_warnings() -> None:
    """Loading MusicXML does not expose Partitura's internal warnings."""
    score_path = ensure_data("vienna_1x22") / "Chopin_op10_no3.musicxml"

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        PartituraLoader().load(score_path)

    assert _partitura_warning_messages(recorded) == []


def test_partitura_midi_load_emits_no_backend_warnings() -> None:
    """Loading score MIDI does not expose Partitura's progress warnings."""
    midi_path = ensure_data("midi") / "score" / "beethoven_mtd.mid"

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        ScoreMidiLoader().load(midi_path)

    assert _partitura_warning_messages(recorded) == []


def test_backend_load_restores_warning_filters() -> None:
    """A backend load leaves the caller's warning filters unchanged."""
    midi_path = ensure_data("midi") / "score" / "beethoven_mtd.mid"
    filters_before = list(warnings.filters)

    ScoreMidiLoader().load(midi_path)

    assert list(warnings.filters) == filters_before


def test_timetoalign_warning_remains_visible_in_backend_context() -> None:
    """The backend context does not suppress a TimeToAlign! warning."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with suppressed_backend_warnings():
            result = resolve_id("timeline-.", ["timeline-1", "timeline-2"])

    assert result == "timeline-1"
    assert [item.category for item in recorded] == [UserWarning]
    assert [str(item.message) for item in recorded] == [
        "Pattern 'timeline-.' matches 2 IDs: ['timeline-1', 'timeline-2']. "
        "Returning first match: 'timeline-1'"
    ]
