"""Sounding onsets of an ms3 notes table for note matching.

Validation logic: ``tests/alignment/README.md``, "Note Onsets for Matching".
"""

from __future__ import annotations

import pandas as pd
import pytest

from timetoalign.alignment.matching import prepare_abc_notes_for_matching

IDS = ["a", "b", "c", "d", "e", "f", "g"]
# untied, a chain 1 0 0 -1, a two-note tie 1 -1
TIES = [None, 1, 0, 0, -1, 1, -1]
COORDS = ["0", "1", "2", "3", "4", "707/2", "355"]


def _frame(tie_dtype: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": IDS,
            "name": ["C4", "D4", "D4", "D4", "D4", "E4", "E4"],
            "staff": [1] * 7,
            "tied": pd.array(TIES, dtype=tie_dtype),
            "quarterbeats_playthrough": COORDS,
        }
    )


@pytest.mark.parametrize("tie_dtype", ["float64", "Int64"])
def test_untied_notes_and_tie_starts_are_the_onsets(tie_dtype):
    prepared = prepare_abc_notes_for_matching(_frame(tie_dtype))

    assert prepared["id"].tolist() == ["a", "b", "f"]
    assert prepared.index.tolist() == [0, 1, 2]
    assert prepared["quarterbeats_playthrough"].tolist() == [0.0, 1.0, 353.5]
    assert prepared["pitch"].tolist() == ["C4", "D4", "E4"]
    assert "name" not in prepared.columns


@pytest.mark.parametrize("blank", ["", " "], ids=["empty", "whitespace"])
def test_blank_text_tie_cells_are_no_tie(blank):
    frame = _frame("float64")
    frame["tied"] = pd.array([blank, "1", "0", "0", "-1", "1", "-1"], dtype="str")

    prepared = prepare_abc_notes_for_matching(frame)

    assert prepared["id"].tolist() == ["a", "b", "f"]
    assert prepared.index.tolist() == [0, 1, 2]


def test_a_tie_value_outside_ms3s_is_refused():
    frame = _frame("float64").iloc[[1, 2]].copy()
    frame["tied"] = [1.0, 2.0]

    with pytest.raises(ValueError) as raised:
        prepare_abc_notes_for_matching(frame)

    assert str(raised.value) == (
        "Column 'tied' holds tie values other than 1, 0, -1: [2.0]"
    )
