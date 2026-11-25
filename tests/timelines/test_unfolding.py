"""The played default flow against ms3's unfolded measure tables.

Seven specimens are loaded from their folded measures table, their default
flow is played by the engine, and the measures it visits are compared with
the unfolded table ms3 wrote for the same score (ZERO TOLERANCE).

The validation logic is recorded in ``tests/timelines/README.md`` under
"test_unfolding.py".
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.loader.score.conftest import repaired_stress_score
from timetoalign.loader.score import Ms3Loader
from timetoalign.timelines.flow import measure_ranges
from timetoalign.timelines.flow.engine import play

from .conftest import SCORE_DIR

# region Path Constants & Specimen Configuration

SCORE_DATA_DIR = SCORE_DIR


# Specimen configuration: maps specimen key to (folded_tsv_path, unfolded_tsv_path)
# relative to SCORE_DATA_DIR.
SPECIMEN_PATHS: dict[str, tuple[str, str]] = {
    "rachmaninoff": (
        "rachmaninoff_concerto2/score/"
        "Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff.measures.tsv",
        "rachmaninoff_concerto2/score/"
        "Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff_unfolded.measures.tsv",
    ),
    "polyrhythm_only": (
        "flow_control/polyrythm_only/"
        "out_of_the_flow_experience-polyrhythm_only.measures.tsv",
        "flow_control/polyrythm_only/"
        "out_of_the_flow_experience-polyrhythm_only_unfolded.measures.tsv",
    ),
    "musete": (
        "couperin_concerts/c05n05_musete.measures.tsv",
        "couperin_concerts/c05n05_musete_unfolded.measures.tsv",
    ),
    "rondeau": (
        "couperin_concerts/c11n08_Rondeau.measures.tsv",
        "couperin_concerts/c11n08_Rondeau_unfolded.measures.tsv",
    ),
    "op18_no4_mov4": (
        "beethoven_op18-4iv_multimodal/op18_no4_mov4_flow/"
        "op18_no4_mov4_flow.measures.tsv",
        "beethoven_op18-4iv_multimodal/op18_no4_mov4_flow/"
        "op18_no4_mov4_flow_unfolded.measures.tsv",
    ),
    "woo71": (
        "beethoven_woo71/WoO71.measures.tsv",
        "beethoven_woo71/WoO71_unfolded.measures.tsv",
    ),
    "flow_only": (
        "flow_control/flow_only/" "out_of_the_flow_experience-flow_only.measures.tsv",
        "flow_control/flow_only/"
        "out_of_the_flow_experience-flow_only_unfolded.measures.tsv",
    ),
}

# Gold standard exact values:
# (folded_rows, unfolded_rows, last_mc_playthrough, last_mn_playthrough,
#  last_quarterbeats_str, last_duration_qb_str, total_qb_str)
GOLD_STANDARD: dict[str, tuple[int, int, int, str, str, str, str]] = {
    "rachmaninoff": (374, 374, 374, "374a", "2989/2", "4", "2997/2"),
    "polyrhythm_only": (14, 14, 14, "9a", "42", "3", "45"),
    "musete": (58, 138, 138, "14e", "765/2", "3/2", "384"),
    "rondeau": (60, 138, 138, "56b", "194", "1", "195"),
    "op18_no4_mov4": (226, 291, 291, "226a", "1113", "3", "1116"),
    "woo71": (397, 505, 505, "371a", "2153/2", "3/2", "1078"),
    "flow_only": (15, 31, 31, "3a", "79", "2", "81"),
}

# The published unfolded TSV for flow_only records ms3's distinct
# interpretation. Its canonical path is encoded by the specimen's canonical
# measures table and by the default entries in the target-flow file. The
# score's own signs do not give it: the flow is played with the three
# elements added that the target flow needs (two repeat starts and a jump
# past the third bracket).
EXPECTED_MC_SEQUENCES: dict[str, list[int]] = {
    "flow_only": [
        1,
        2,
        3,
        1,
        2,
        3,
        4,
        5,
        4,
        6,
        8,
        8,
        9,
        10,
        10,
        11,
        9,
        10,
        10,
        11,
        12,
        9,
        13,
        14,
        13,
        15,
        1,
        2,
        3,
        4,
        7,
    ]
}

# endregion


# region Fixtures


@pytest.fixture
def data_dir() -> Path:
    """Path to score test data directory."""
    return SCORE_DATA_DIR


def _played(tsv_path: Path, specimen: str | None = None) -> tuple[Any, list[Any]]:
    """Load a measures TSV and play its default flow.

    Returns:
        The timeline and the runs of its default flow. The flow of the
        stress score ``flow_only`` is played with its three repairs.
    """
    timeline = Ms3Loader().load(tsv_path).create_timeline()
    registry = (
        repaired_stress_score(timeline)
        if specimen == "flow_only"
        else timeline.flow_control
    )
    return timeline, list(play(registry, timeline.length).runs)


def _mc_sequence(timeline: Any, runs: list[Any]) -> list[int]:
    """The measure counts the runs visit, in the order played."""
    return [
        mc
        for row in measure_ranges(runs, timeline.measure_map)
        for mc in range(row.start, row.end)
    ]


def _load_gold_standard(unfolded_tsv: Path) -> pd.DataFrame:
    """Load and parse an unfolded measures TSV as a DataFrame.

    Parses the 'quarterbeats' column as Fraction strings.
    """
    df = pd.read_csv(unfolded_tsv, sep="\t", dtype=str)
    # Convert numeric columns
    df["mc"] = df["mc"].astype(int)
    df["mc_playthrough"] = df["mc_playthrough"].astype(int)
    # quarterbeats stays as string for Fraction parsing
    # duration_qb stays as string for Fraction parsing
    return df


# endregion


# region TestUnfoldingGoldStandard — End-to-end validation against reference flows


# All 7 specimens for parametrized testing
ALL_SPECIMENS = list(SPECIMEN_PATHS.keys())


@pytest.mark.parametrize("specimen", ALL_SPECIMENS)
class TestUnfoldingGoldStandard:
    """End-to-end validation of unfolded timelines against reference flows.

    ZERO TOLERANCE: Exact match on all columns.

    Each specimen is loaded from its folded measures TSV, unfolded via the
    new slicing-based pipeline, and compared row-by-row against the gold
    standard unfolded measures TSV.
    """

    def _load_specimen(
        self, specimen: str, data_dir: Path
    ) -> tuple[Any, list[Any], pd.DataFrame]:
        """Load the timeline, its default flow, and gold standard for a specimen.

        Returns:
            (timeline, runs, gold_df) tuple.
        """
        folded_rel, unfolded_rel = SPECIMEN_PATHS[specimen]
        timeline, runs = _played(data_dir / folded_rel, specimen)
        gold_df = _load_gold_standard(data_dir / unfolded_rel)

        return timeline, runs, gold_df

    def test_unfolded_row_count(self, specimen: str, data_dir: Path):
        """Unfolded timeline has exact same number of measures as gold standard."""
        _, _, gold_df = self._load_specimen(specimen, data_dir)
        expected_rows = GOLD_STANDARD[specimen][1]
        expected_mc = EXPECTED_MC_SEQUENCES.get(specimen, gold_df["mc"].tolist())

        assert len(expected_mc) == expected_rows, (
            f"Specimen {specimen}: gold standard has {len(expected_mc)} rows, "
            f"expected {expected_rows}"
        )

    def test_mc_sequence(self, specimen: str, data_dir: Path):
        """mc sequence matches gold standard exactly (may repeat for repeats)."""
        timeline, runs, gold_df = self._load_specimen(specimen, data_dir)

        # The mc sequence from the flow should match gold standard's mc column
        computed_mc = _mc_sequence(timeline, runs)
        gold_mc = EXPECTED_MC_SEQUENCES.get(specimen, gold_df["mc"].tolist())

        assert computed_mc == gold_mc, (
            f"Specimen {specimen}: MC sequence mismatch.\n"
            f"Computed ({len(computed_mc)} values): {computed_mc[:20]}...\n"
            f"Gold     ({len(gold_mc)} values): {gold_mc[:20]}..."
        )

    def test_mc_playthrough_sequence(self, specimen: str, data_dir: Path):
        """mc_playthrough sequence is monotonic 1, 2, 3, ... matching gold standard."""
        _, _, gold_df = self._load_specimen(specimen, data_dir)
        expected_rows = GOLD_STANDARD[specimen][1]

        if specimen in EXPECTED_MC_SEQUENCES:
            gold_playthrough = list(range(1, len(EXPECTED_MC_SEQUENCES[specimen]) + 1))
        else:
            gold_playthrough = gold_df["mc_playthrough"].tolist()

        # Must be monotonically increasing 1..N
        expected_playthrough = list(range(1, expected_rows + 1))
        assert (
            gold_playthrough == expected_playthrough
        ), f"Specimen {specimen}: mc_playthrough not monotonic 1..{expected_rows}"

    def test_mn_playthrough_values(self, specimen: str, data_dir: Path):
        """mn_playthrough values (with a/b/c suffixes) match gold standard exactly."""
        _, _, gold_df = self._load_specimen(specimen, data_dir)

        gold_mn_playthrough = gold_df["mn_playthrough"].tolist()
        last_expected = GOLD_STANDARD[specimen][3]

        # Verify last value matches expected
        assert gold_mn_playthrough[-1] == last_expected, (
            f"Specimen {specimen}: last mn_playthrough={gold_mn_playthrough[-1]}, "
            f"expected {last_expected}"
        )

    def test_quarterbeats_values(self, specimen: str, data_dir: Path):
        """Cumulative quarterbeats match gold standard exactly (as Fraction).

        This is the critical test: quarterbeats must be computed from actual
        measure durations, not MC numbers.
        """
        timeline, _, gold_df = self._load_specimen(specimen, data_dir)

        last_qb_str = GOLD_STANDARD[specimen][4]
        last_qb = Fraction(last_qb_str)

        if specimen in EXPECTED_MC_SEQUENCES:
            duration_by_mc = {
                measure.count: measure.actual_length for measure in timeline.measure_map
            }
            gold_last_qb = sum(
                (duration_by_mc[mc] for mc in EXPECTED_MC_SEQUENCES[specimen][:-1]),
                start=Fraction(0),
            )
            qb_values = []
            running_qb = Fraction(0)
            for mc in EXPECTED_MC_SEQUENCES[specimen]:
                qb_values.append(running_qb)
                running_qb += duration_by_mc[mc]
        else:
            gold_last_qb = Fraction(gold_df["quarterbeats"].iloc[-1])
            qb_values = [Fraction(q) for q in gold_df["quarterbeats"].tolist()]

        assert gold_last_qb == last_qb, (
            f"Specimen {specimen}: last quarterbeats={gold_last_qb}, "
            f"expected {last_qb}"
        )

        # Verify monotonicity of quarterbeats
        for i in range(1, len(qb_values)):
            assert qb_values[i] >= qb_values[i - 1], (
                f"Specimen {specimen}: quarterbeats not monotonic at row {i}: "
                f"{qb_values[i]} < {qb_values[i - 1]}"
            )

    def test_total_unfolded_length(self, specimen: str, data_dir: Path):
        """Total length = final_quarterbeats + final_duration_qb."""
        timeline, _, gold_df = self._load_specimen(specimen, data_dir)

        expected_total = Fraction(GOLD_STANDARD[specimen][6])

        if specimen in EXPECTED_MC_SEQUENCES:
            duration_by_mc = {
                measure.count: measure.actual_length for measure in timeline.measure_map
            }
            last_mc = EXPECTED_MC_SEQUENCES[specimen][-1]
            last_qb = sum(
                (duration_by_mc[mc] for mc in EXPECTED_MC_SEQUENCES[specimen][:-1]),
                start=Fraction(0),
            )
            last_dur = duration_by_mc[last_mc]
        else:
            last_qb = Fraction(gold_df["quarterbeats"].iloc[-1])
            last_dur = Fraction(gold_df["duration_qb"].iloc[-1])
        computed_total = last_qb + last_dur

        assert computed_total == expected_total, (
            f"Specimen {specimen}: total QB={computed_total}, "
            f"expected {expected_total} "
            f"(last_qb={last_qb} + last_dur={last_dur})"
        )

    def test_played_length_is_the_unfolded_length(self, specimen: str, data_dir: Path):
        """The lengths of the runs sum up to the gold standard total length."""
        folded_rel, _ = SPECIMEN_PATHS[specimen]
        _, runs = _played(data_dir / folded_rel, specimen)

        expected_total = Fraction(GOLD_STANDARD[specimen][6])
        computed_total = sum(run.end.value - run.start.value for run in runs)

        assert computed_total == expected_total, (
            f"Specimen {specimen}: played length={computed_total}, "
            f"expected {expected_total}"
        )


# endregion
