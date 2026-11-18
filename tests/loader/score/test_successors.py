"""The ``next`` column that ``ScoreLoader`` derives from the measure columns.

Validation logic: ``tests/loader/score/README.md``, section "Successors derived from the measure
columns".
"""

from __future__ import annotations

import csv
from functools import cache
from pathlib import Path

import pytest

from timetoalign.core.enums import FlowMode
from timetoalign.loader.score import Music21Loader, PartituraLoader, VerovioLoader
from timetoalign.loader.score.base import ScoreLoader

from .conftest import SCORE_DIR

COUPERIN_DIR = SCORE_DIR / "couperin_concerts"

_NOTE = (
    "<note><pitch><step>C</step><octave>4</octave></pitch>"
    "<duration>4</duration><type>whole</type></note>"
)
_ATTRIBUTES = (
    "<attributes><divisions>1</divisions>"
    "<time><beats>4</beats><beat-type>4</beat-type></time>"
    "<clef><sign>G</sign><line>2</line></clef></attributes>"
)
_BARLINES = {
    1: ('<barline location="left"><repeat direction="forward"/></barline>', ""),
    4: ('<barline location="left"><ending number="1" type="start"/></barline>', ""),
    5: (
        "",
        '<barline location="right"><bar-style>light-heavy</bar-style>'
        '<ending number="1" type="stop"/><repeat direction="backward"/></barline>',
    ),
    6: (
        '<barline location="left"><ending number="2" type="start"/></barline>',
        '<barline location="right"><ending number="2" type="discontinue"/></barline>',
    ),
}


def _two_measure_ending(path: Path) -> Path:
    """Write seven 4/4 measures: repeat from m1, first ending m4-m5, second ending m6."""
    measures = []
    for number in range(1, 8):
        left, right = _BARLINES.get(number, ("", ""))
        attributes = _ATTRIBUTES if number == 1 else ""
        measures.append(
            f'<measure number="{number}">{attributes}{left}{_NOTE}{right}</measure>'
        )
    path.write_text(
        '<?xml version="1.0"?><score-partwise version="3.1"><part-list>'
        '<score-part id="P1"><part-name>P</part-name></score-part></part-list>'
        f'<part id="P1">{"".join(measures)}</part></score-partwise>',
        encoding="utf-8",
    )
    return path


class TestTwoMeasureEnding:
    """An ending of two measures is one run of volta measures."""

    @pytest.fixture
    def source(self, tmp_path: Path) -> Path:
        return _two_measure_ending(tmp_path / "ending.musicxml")

    @pytest.mark.parametrize("loader_cls", [Music21Loader, PartituraLoader])
    def test_successors_and_default_flow(
        self, source: Path, loader_cls: type[ScoreLoader]
    ) -> None:
        loader = loader_cls().load(source)
        measures = list(loader.store.measures)

        assert [row["volta"] for row in measures] == [None, None, None, 1, 1, 2, None]
        assert [row["next"] for row in measures] == [
            None,
            None,
            "4, 6",
            None,
            "1",
            None,
            None,
        ]
        flow = loader.create_flow_controller().compute_flow(FlowMode.default)
        assert flow.to_mc_sequence() == [1, 2, 3, 4, 5, 1, 2, 3, 6, 7]

    def test_music21_and_partitura_agree(self, source: Path) -> None:
        music21 = Music21Loader().load(source).store.measures
        partitura = PartituraLoader().load(source).store.measures

        assert music21.column_values("next") == partitura.column_values("next")


SOURCES = {
    "musicxml": PartituraLoader,
    "mei": VerovioLoader,
}

# (score, source, MC) -> (the derived set, why it differs from the TSV's)
DIFFERENCES = {
    ("c05n05_musete", "mei", 31): (
        {17, 1},
        "the MEI states a daCapo where the TSV has a D.S. to the segno at MC 6",
    ),
    ("c05n05_musete", "mei", 16): (
        {1, 17, -1},
        "the TSV continues at MC 32 after the D.S. al Fine; the source states no continuation",
    ),
    ("c05n05_musete", "musicxml", 16): (
        {1, 17, -1},
        "the TSV continues at MC 32 after the D.S. al Fine; the rule derives no continuation",
    ),
    ("c11n08_Rondeau", "mei", 9): (
        {1, 10},
        "the TSV continues at MCs 19 and 28 after each D.C. al segno; the source states no "
        "continuation",
    ),
    ("c11n08_Rondeau", "musicxml", 9): (
        {1, 10},
        "the TSV continues at MCs 19 and 28 after each D.C. al segno; the rule derives no "
        "continuation",
    ),
}


def _as_sets(mcs: list[int], entries: list[str | None]) -> dict[int, frozenset[int]]:
    """Read ``next`` cells as sets of MCs; null is the following MC, -1 after the last."""
    following = [*mcs[1:], -1]
    return {
        mc: (
            frozenset({after})
            if entry is None
            else frozenset(
                int(part) for part in entry.strip("()").split(",") if part.strip()
            )
        )
        for mc, after, entry in zip(mcs, following, entries)
    }


@cache
def _ms3_successors(score: str) -> dict[int, frozenset[int]]:
    with open(
        COUPERIN_DIR / f"{score}.measures.tsv", newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    return _as_sets(
        [int(row["mc"]) for row in rows], [row["next"] or None for row in rows]
    )


@cache
def _derived_successors(score: str, extension: str) -> dict[int, frozenset[int]]:
    if extension == "mei":
        pytest.importorskip("verovio")
    loader = SOURCES[extension]().load(COUPERIN_DIR / f"{score}.{extension}")
    measures = loader.store.measures
    return _as_sets(measures.column_values("mc"), measures.column_values("next"))


@pytest.mark.parametrize("extension", sorted(SOURCES))
@pytest.mark.parametrize("score", ["c05n05_musete", "c11n08_Rondeau"])
def test_derived_successors_agree_with_ms3(score: str, extension: str) -> None:
    def kept(successors: dict[int, frozenset[int]]) -> dict[int, frozenset[int]]:
        return {
            mc: following
            for mc, following in successors.items()
            if (score, extension, mc) not in DIFFERENCES
        }

    assert kept(_derived_successors(score, extension)) == kept(_ms3_successors(score))


@pytest.mark.parametrize(("score", "extension", "mc"), list(DIFFERENCES))
def test_documented_differences_are_exact(score: str, extension: str, mc: int) -> None:
    assert (
        _derived_successors(score, extension)[mc]
        == DIFFERENCES[score, extension, mc][0]
    )


@pytest.mark.parametrize(
    ("score", "extension", "mc"),
    [
        pytest.param(*key, marks=pytest.mark.xfail(strict=True, reason=reason))
        for key, (_, reason) in DIFFERENCES.items()
    ],
)
def test_documented_differences_from_ms3(score: str, extension: str, mc: int) -> None:
    assert _derived_successors(score, extension)[mc] == _ms3_successors(score)[mc]
