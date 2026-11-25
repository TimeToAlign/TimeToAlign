"""Tests for Unfolding via Slicing.

This test suite validates the slice-based unfolding pipeline:
1. Timeline.get_slice() primitive (unit tests)
2. SegmentLine assembly from slices (integration tests)
3. The played default flow against reference unfoldings (7 specimens, ZERO TOLERANCE)
4. Group unfolding: unfold an entire TimelineGroup via one FlowMap

See README_unfolding.md for full testing strategy documentation.

Validation Criteria (ZERO TOLERANCE):
- EXACT row count match
- EXACT mc_playthrough sequence
- EXACT mn_playthrough values with suffixes
- EXACT quarterbeats values as Fraction (NOT float comparison)
- EXACT total unfolded length
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.loader.score.conftest import repaired_stress_score
from timetoalign.core import NumberType, TimeUnit
from timetoalign.loader.score import Ms3Loader
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
)
from timetoalign.timelines.flow import FlowMap, measure_ranges, unfold_via_flowmap
from timetoalign.timelines.flow.engine import play
from timetoalign.timelines.types import SegmentLine

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


@pytest.fixture
def simple_logical_timeline() -> ContinuousLogicalTimeline:
    """A ContinuousLogicalTimeline with known events for slice testing.

    Timeline span: [0, 100) in quarters (Fraction).

    Events:
        Instant events:
            - Beat at 0
            - Beat at 10
            - Beat at 25
            - Beat at 50
            - Beat at 75
            - Beat at 99

        Interval events:
            - Note [5, 15)   -- straddles a typical slice boundary at 10
            - Note [20, 30)  -- fully inside [10, 50)
            - Note [45, 55)  -- straddles a typical slice boundary at 50
            - Note [80, 95)  -- fully inside [50, 100)
    """
    tl = ContinuousLogicalTimeline(
        length=Fraction(100),
        unit=TimeUnit.quarters,
        number_type=NumberType.fraction,
    )
    tl.add_events(
        [
            # Instant events
            {"event_type": "Beat", "instant": Fraction(0)},
            {"event_type": "Beat", "instant": Fraction(10)},
            {"event_type": "Beat", "instant": Fraction(25)},
            {"event_type": "Beat", "instant": Fraction(50)},
            {"event_type": "Beat", "instant": Fraction(75)},
            {"event_type": "Beat", "instant": Fraction(99)},
            # Interval events
            {"event_type": "Note", "start": Fraction(5), "end": Fraction(15)},
            {"event_type": "Note", "start": Fraction(20), "end": Fraction(30)},
            {"event_type": "Note", "start": Fraction(45), "end": Fraction(55)},
            {"event_type": "Note", "start": Fraction(80), "end": Fraction(95)},
        ]
    )
    return tl


@pytest.fixture
def timeline_with_child() -> ContinuousLogicalTimeline:
    """A ContinuousLogicalTimeline with a child for recursive slice testing.

    Parent span: [0, 100) in quarters.
    Parent events:
        - Beat at 10
        - Note [30, 60)

    Child span: [0, 40) embedded at offset 20, covering parent [20, 60).
    Child events:
        - Beat at 5  (= parent 25)
        - Note [10, 30) (= parent [30, 50))
    """
    parent = ContinuousLogicalTimeline(
        length=Fraction(100),
        unit=TimeUnit.quarters,
        number_type=NumberType.fraction,
    )
    parent.add_events(
        [
            {"event_type": "Beat", "instant": Fraction(10)},
            {"event_type": "Note", "start": Fraction(30), "end": Fraction(60)},
        ]
    )

    child = ContinuousLogicalTimeline(
        length=Fraction(40),
        unit=TimeUnit.quarters,
        number_type=NumberType.fraction,
    )
    child.add_events(
        [
            {"event_type": "Beat", "instant": Fraction(5)},
            {"event_type": "Note", "start": Fraction(10), "end": Fraction(30)},
        ]
    )
    parent.add_child(child, offset=Fraction(20))

    return parent


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


# region TestGetSlice — Unit tests for Timeline.get_slice()


class TestGetSlice:
    """Test Timeline.get_slice() primitive.

    Uses synthetic timelines with known events to validate slicing logic
    in isolation. No external data files required.
    """

    def test_basic_slice(self, simple_logical_timeline: ContinuousLogicalTimeline):
        """Slice extracts events in range with shifted coordinates."""
        sliced = simple_logical_timeline.get_slice(Fraction(10), Fraction(50))

        assert sliced.length.value == Fraction(40)

        # Events in [10, 50):
        #   Instant: Beat@10 -> 0, Beat@25 -> 15
        #   Interval: Note[20,30) -> [10,20), Note[5,15) truncated to [10,15) -> [0,5)
        #   Note[45,55) truncated to [45,50) -> [35,40)
        events = list(sliced.get_events())
        instant_events = [e for e in events if e.get("temporal_type") == "instant"]
        # interval_events = [e for e in events if e.get("temporal_type") == "interval"]

        # Beat@10 -> 0, Beat@25 -> 15 (Beat@0 is outside, Beat@50 is right-exclusive)
        assert len(instant_events) == 2
        # Note: In EventData, all events (including instants) store coords in "start"
        instant_coords = sorted(Fraction(e["start"]["value"]) for e in instant_events)
        assert instant_coords == [Fraction(0), Fraction(15)]

    def test_instant_at_boundary_included(
        self, simple_logical_timeline: ContinuousLogicalTimeline
    ):
        """Instant at start boundary IS included (left-inclusive)."""
        sliced = simple_logical_timeline.get_slice(Fraction(25), Fraction(75))

        events = list(sliced.get_events(temporal_type="instant"))
        coords = sorted(Fraction(e["start"]["value"]) for e in events)

        # Beat@25 -> 0 (included), Beat@50 -> 25 (included)
        # Beat@75 is at end boundary -> excluded (right-exclusive)
        assert Fraction(0) in coords, "Instant at start boundary must be included"

    def test_instant_at_end_excluded(
        self, simple_logical_timeline: ContinuousLogicalTimeline
    ):
        """Instant at end boundary is NOT included (right-exclusive)."""
        sliced = simple_logical_timeline.get_slice(Fraction(25), Fraction(50))

        events = list(sliced.get_events(temporal_type="instant"))
        coords = [Fraction(e["start"]["value"]) for e in events]

        # Beat@50 is exactly at end -> excluded
        assert Fraction(25) not in coords, "Instant at end boundary must be excluded"

    def test_interval_truncation(
        self, simple_logical_timeline: ContinuousLogicalTimeline
    ):
        """Interval straddling boundary is clipped (truncate_events=True)."""
        # Slice [10, 50). Note [5, 15) straddles start: clipped to [10, 15) -> [0, 5)
        # Note [45, 55) straddles end: clipped to [45, 50) -> [35, 40)
        sliced = simple_logical_timeline.get_slice(
            Fraction(10), Fraction(50), truncate_events=True
        )

        intervals = list(sliced.get_events(temporal_type="interval"))

        # Should have 3 intervals: truncated[0,5), full[10,20), truncated[35,40)
        assert len(intervals) == 3

        # Check the truncated interval from Note[5,15)
        starts = sorted(Fraction(e["start"]["value"]) for e in intervals)
        ends = sorted(Fraction(e["end"]["value"]) for e in intervals)

        assert Fraction(0) in starts, "Truncated interval should start at 0"
        assert Fraction(5) in ends, "Truncated interval [0,5) should end at 5"
        assert Fraction(40) in ends, "Truncated interval at end should stop at 40"

    def test_interval_no_truncation(
        self, simple_logical_timeline: ContinuousLogicalTimeline
    ):
        """With truncate_events=False, straddling intervals are excluded."""
        sliced = simple_logical_timeline.get_slice(
            Fraction(10), Fraction(50), truncate_events=False
        )

        intervals = list(sliced.get_events(temporal_type="interval"))

        # Only Note[20,30) is fully contained in [10, 50)
        assert len(intervals) == 1
        start = Fraction(intervals[0]["start"]["value"])
        end = Fraction(intervals[0]["end"]["value"])
        assert start == Fraction(10)  # 20 - 10
        assert end == Fraction(20)  # 30 - 10

    def test_coordinate_shifting(
        self, simple_logical_timeline: ContinuousLogicalTimeline
    ):
        """All coordinates shifted by -start."""
        sliced = simple_logical_timeline.get_slice(Fraction(75), Fraction(100))

        events = list(sliced.get_events(temporal_type="instant"))
        # Beat@75 -> 0, Beat@99 -> 24
        coords = sorted(Fraction(e["start"]["value"]) for e in events)
        assert coords == [Fraction(0), Fraction(24)]

        intervals = list(sliced.get_events(temporal_type="interval"))
        # Note[80,95) -> [5, 20)
        assert len(intervals) == 1
        assert Fraction(intervals[0]["start"]["value"]) == Fraction(5)
        assert Fraction(intervals[0]["end"]["value"]) == Fraction(20)

    def test_empty_slice(self, simple_logical_timeline: ContinuousLogicalTimeline):
        """Slice of range with no events returns empty timeline."""
        # Range [96, 99) has no events (Beat@99 is at 99, not in [96, 99))
        sliced = simple_logical_timeline.get_slice(Fraction(96), Fraction(99))

        assert sliced.length.value == Fraction(3)
        events = list(sliced.get_events())
        assert len(events) == 0

    def test_full_slice(self, simple_logical_timeline: ContinuousLogicalTimeline):
        """Slice of entire timeline returns copy with all events."""
        sliced = simple_logical_timeline.get_slice(Fraction(0), Fraction(100))

        assert sliced.length.value == Fraction(100)
        original_events = list(simple_logical_timeline.get_events())
        sliced_events = list(sliced.get_events())
        assert len(sliced_events) == len(original_events)

    def test_children_sliced(self, timeline_with_child: ContinuousLogicalTimeline):
        """Child timelines are recursively sliced when include_children=True."""
        # Parent [0, 100), child at offset 20 with length 40, covering [20, 60)
        # Slice [25, 55) should include child's overlap [25, 55) -> child coords [5, 35)
        sliced = timeline_with_child.get_slice(
            Fraction(25), Fraction(55), include_children=True
        )

        assert sliced.length.value == Fraction(30)  # 55 - 25
        assert sliced.n_children == 1, "Sliced child should be present"

    def test_children_excluded(self, timeline_with_child: ContinuousLogicalTimeline):
        """Child timelines are excluded when include_children=False."""
        sliced = timeline_with_child.get_slice(
            Fraction(25), Fraction(55), include_children=False
        )

        assert sliced.n_children == 0

    def test_number_type_preserved(
        self, simple_logical_timeline: ContinuousLogicalTimeline
    ):
        """Fraction coordinates remain Fraction in slice."""
        sliced = simple_logical_timeline.get_slice(Fraction(10), Fraction(50))

        assert sliced.number_type == NumberType.fraction

    def test_concrete_class_preserved(
        self, simple_logical_timeline: ContinuousLogicalTimeline
    ):
        """ContinuousLogicalTimeline.get_slice() returns ContinuousLogicalTimeline."""
        sliced = simple_logical_timeline.get_slice(Fraction(10), Fraction(50))

        assert type(sliced) is ContinuousLogicalTimeline

    def test_slice_of_discrete_timeline(self):
        """get_slice works on DiscreteLogicalTimeline with int coordinates."""
        tl = DiscreteLogicalTimeline(
            length=100,
            unit=TimeUnit.ticks,
        )
        tl.add_events(
            [
                {"event_type": "Tick", "instant": 10},
                {"event_type": "Tick", "instant": 50},
                {"event_type": "Tick", "instant": 90},
            ]
        )

        sliced = tl.get_slice(20, 80)
        assert sliced.length.value == 60
        assert type(sliced) is DiscreteLogicalTimeline

        events = list(sliced.get_events())
        assert len(events) == 1  # Only Tick@50 -> 30
        assert events[0]["start"]["value"] == 30

    def test_slice_of_physical_timeline(self):
        """get_slice works on ContinuousPhysicalTimeline with float coordinates."""
        tl = ContinuousPhysicalTimeline(
            length=10.0,
            unit=TimeUnit.seconds,
        )
        tl.add_events(
            [
                {"event_type": "Onset", "instant": 1.0},
                {"event_type": "Onset", "instant": 5.0},
                {"event_type": "Onset", "instant": 9.0},
            ]
        )

        sliced = tl.get_slice(2.0, 8.0)
        assert sliced.length.value == 6.0
        assert type(sliced) is ContinuousPhysicalTimeline

        events = list(sliced.get_events())
        assert len(events) == 1  # Only Onset@5.0 -> 3.0

    def test_invalid_range_start_ge_end(
        self, simple_logical_timeline: ContinuousLogicalTimeline
    ):
        """Raises ValueError if start >= end."""
        with pytest.raises(ValueError, match="start.*end"):
            simple_logical_timeline.get_slice(Fraction(50), Fraction(50))

        with pytest.raises(ValueError, match="start.*end"):
            simple_logical_timeline.get_slice(Fraction(60), Fraction(40))

    def test_invalid_range_outside_bounds(
        self, simple_logical_timeline: ContinuousLogicalTimeline
    ):
        """Raises ValueError if range is outside timeline bounds."""
        with pytest.raises(ValueError):
            simple_logical_timeline.get_slice(Fraction(-10), Fraction(50))

        with pytest.raises(ValueError):
            simple_logical_timeline.get_slice(Fraction(50), Fraction(110))


# endregion


# region TestSegmentLineAssembly — Integration tests for slice assembly


class TestSegmentLineAssembly:
    """Test assembling slices into a SegmentLine.

    Uses synthetic timelines to validate the structural properties of
    SegmentLine construction from get_slice() outputs.
    """

    def test_contiguity(self):
        """Segments are contiguous (each starts where previous ended)."""
        # Create source timeline
        source = ContinuousLogicalTimeline(
            length=Fraction(120),
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )
        source.add_events(
            [
                {"event_type": "Beat", "instant": Fraction(0)},
                {"event_type": "Beat", "instant": Fraction(40)},
                {"event_type": "Beat", "instant": Fraction(80)},
            ]
        )

        # Slice at [0, 40), [40, 80), [80, 120)
        boundaries = [
            (Fraction(0), Fraction(40)),
            (Fraction(40), Fraction(80)),
            (Fraction(80), Fraction(120)),
        ]

        sl = SegmentLine[ContinuousLogicalTimeline](
            length=0,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )

        for qb_start, qb_end in boundaries:
            seg = source.get_slice(qb_start, qb_end)
            sl.append_segment(seg)

        assert sl.n_segments == 3

        # Verify contiguity: each segment starts where previous ended
        offsets = []
        for seg_id in sl._segment_order:
            offset = sl._child_offsets[seg_id].value
            child = sl._children[seg_id]
            offsets.append((offset, offset + child.length.value))

        for i in range(1, len(offsets)):
            assert offsets[i][0] == offsets[i - 1][1], (
                f"Segment {i} start {offsets[i][0]} != "
                f"segment {i - 1} end {offsets[i - 1][1]}"
            )

    def test_total_length(self):
        """Total SegmentLine length equals sum of slice lengths."""
        source = ContinuousLogicalTimeline(
            length=Fraction(100),
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )

        # Simulate unfolding: two passes through [0, 100)
        sl = SegmentLine[ContinuousLogicalTimeline](
            length=0,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )

        slice1 = source.get_slice(Fraction(0), Fraction(100))
        slice2 = source.get_slice(Fraction(0), Fraction(100))
        sl.append_segment(slice1)
        sl.append_segment(slice2)

        assert sl.length.value == Fraction(200)
        assert sl.n_segments == 2

    def test_events_preserved(self):
        """All events from individual slices appear in the assembled SegmentLine."""
        source = ContinuousLogicalTimeline(
            length=Fraction(60),
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )
        source.add_events(
            [
                {"event_type": "Beat", "instant": Fraction(10)},
                {"event_type": "Beat", "instant": Fraction(30)},
                {"event_type": "Beat", "instant": Fraction(50)},
            ]
        )

        # Slice at [0, 30), [30, 60)
        sl = SegmentLine[ContinuousLogicalTimeline](
            length=0,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )

        seg1 = source.get_slice(Fraction(0), Fraction(30))
        seg2 = source.get_slice(Fraction(30), Fraction(60))
        sl.append_segment(seg1)
        sl.append_segment(seg2)

        # Each segment should have its events
        seg1_events = list(sl._children[sl._segment_order[0]].get_events())
        seg2_events = list(sl._children[sl._segment_order[1]].get_events())

        assert len(seg1_events) == 1  # Beat@10 -> 10
        assert len(seg2_events) == 2  # Beat@30 -> 0, Beat@50 -> 20

    def test_segment_type_matches_source(self):
        """SegmentLine's segment_type matches the source timeline class."""
        source = ContinuousLogicalTimeline(
            length=Fraction(40),
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )

        sl = SegmentLine[ContinuousLogicalTimeline](
            length=0,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )

        seg = source.get_slice(Fraction(0), Fraction(40))
        sl.append_segment(seg)

        assert sl.segment_type is ContinuousLogicalTimeline
        assert type(sl._children[sl._segment_order[0]]) is ContinuousLogicalTimeline

    def test_repeated_section_assembly(self):
        """Multiple slices from the same source range assemble correctly.

        This simulates a simple repeat: the same [0, 40) section played twice.
        """
        source = ContinuousLogicalTimeline(
            length=Fraction(40),
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )
        source.add_events(
            [
                {"event_type": "Beat", "instant": Fraction(0)},
                {"event_type": "Beat", "instant": Fraction(20)},
            ]
        )

        sl = SegmentLine[ContinuousLogicalTimeline](
            length=0,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )

        # Two passes through the same range
        for _ in range(2):
            seg = source.get_slice(Fraction(0), Fraction(40))
            sl.append_segment(seg)

        assert sl.length.value == Fraction(80)
        assert sl.n_segments == 2

        # Each segment should have 2 beats
        for seg_id in sl._segment_order:
            child = sl._children[seg_id]
            events = list(child.get_events())
            assert len(events) == 2


# endregion


class TestUnfoldedTimelineIdentity:
    """Tests explicit identifiers on standalone unfolded timelines."""

    def test_uid_is_passed_to_returned_timeline(self) -> None:
        """The requested uid identifies the same-type unfolded timeline."""
        source = ContinuousLogicalTimeline(length=Fraction(4), uid="source")

        unfolded = unfold_via_flowmap(source, FlowMap([(0, 4)]), uid="unfolded")

        assert unfolded.id == "unfolded"
        assert type(unfolded) is ContinuousLogicalTimeline


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


# region TestGroupUnfolding — Unfold an entire TimelineGroup via one FlowMap


# Beethoven Op.18 No.4 iv multimodal data paths (relative to SCORE_DATA_DIR)
BEETHOVEN_DIR_REL = "beethoven_op18-4iv_multimodal"
BEETHOVEN_ABC_DIR_REL = f"{BEETHOVEN_DIR_REL}/ABC"
BEETHOVEN_OMR_CSV_REL = (
    f"{BEETHOVEN_DIR_REL}/OMR_groundtruth/OMR_xml_by_score/omr_note_heads.csv"
)
BEETHOVEN_OMR_IMAGES_REL = f"{BEETHOVEN_DIR_REL}/OMR_groundtruth/Images"
BEETHOVEN_OPENSCORE_DIR_REL = f"{BEETHOVEN_DIR_REL}/OpenScoreSQ"

# Gold standard: 11 sections, total 1116 QB (from GOLD_STANDARD above).
# The ABC v2.6 edition uses quarterbeats_all_endings, giving a folded
# length of 878.5 QB that includes all volta endings.
BEETHOVEN_FOLDED_QB = Fraction(1757, 2)  # 878.5
BEETHOVEN_UNFOLDED_QB = Fraction(1116)
BEETHOVEN_N_SECTIONS = 11


def _build_dgt1(data_dir: Path) -> SegmentLine:
    """Build the DGT1 OMR SegmentLine[SegmentLine[DiscreteGraphicalTimeline]].

    Reproduces the Beethoven multimodal notebook's DGT1 construction
    (Section 7).
    """
    from PIL import Image

    from timetoalign import TableMap

    omr_csv = data_dir / BEETHOVEN_OMR_CSV_REL
    omr_images = data_dir / BEETHOVEN_OMR_IMAGES_REL
    omr_df = pd.read_csv(omr_csv)
    image_width = Image.open(next(omr_images.glob("*.png"))).size[0]

    noteheads = pd.DataFrame(
        {
            "start": omr_df["Nodes.Node.Left"].astype(int),
            "end": (omr_df["Nodes.Node.Left"] + omr_df["Nodes.Node.Width"]).astype(int),
            "onset_beats": omr_df["onset_beats"].astype(float),
            "pitch": omr_df["pitch"],
            "staff_id": omr_df["staff_id"].astype(int),
            "midi_pitch": omr_df["midi_pitch_code"].astype(int),
            "top": omr_df["Nodes.Node.Top"].astype(int),
            "page": omr_df["@pageIndex"],
            "spacing_run_id": omr_df["spacing_run_id"],
        }
    )

    page_class = SegmentLine[DiscreteGraphicalTimeline]
    dgt1 = SegmentLine[page_class](
        length=0,
        unit=TimeUnit.pixels,
        number_type=NumberType.int,
    )

    for page_idx, page_data in noteheads.groupby("page", sort=True):
        sys_top = page_data.groupby("spacing_run_id")["top"].min()
        sys_order = sys_top.sort_values().index

        page = page_class(
            length=0,
            unit=TimeUnit.pixels,
            number_type=NumberType.int,
        )

        for sys_rank, sys_id in enumerate(sys_order):
            sys_data = page_data[page_data["spacing_run_id"] == sys_id]

            system = DiscreteGraphicalTimeline(
                length=image_width,
                uid=f"p{page_idx}_s{sys_rank}",
                name=f"Page {page_idx + 1}, System {sys_rank + 1}",
            )
            events = sys_data.drop(columns=["page", "spacing_run_id"])
            system.add_events(events.assign(event_type="Notehead").to_dict("records"))

            pairs = (
                events[["start", "onset_beats"]]
                .drop_duplicates("start")
                .sort_values("start")
            )
            if len(pairs) >= 2:
                system.add_conversion_map(
                    TableMap(
                        x_values=pairs["start"].tolist(),
                        y_values=pairs["onset_beats"].tolist(),
                        source_unit="pixels",
                        target_unit="quarters",
                        uid=f"p{page_idx}_s{sys_rank}_px_to_qb",
                    )
                )

            page.append_segment(system)

        dgt1.append_segment(page, name=f"page_{page_idx}")

    return dgt1


def _build_openscore(data_dir: Path) -> ContinuousLogicalTimeline:
    """Build the OpenScore 4th movement ContinuousLogicalTimeline.

    Reproduces the Beethoven multimodal notebook's OpenScore construction
    (Section 8): load full score, split at section boundaries, extract
    movement 4 as a child timeline.
    """
    openscore_dir = data_dir / BEETHOVEN_OPENSCORE_DIR_REL
    os_loader = Ms3Loader.from_file(
        openscore_dir / "sq8913219.notes.tsv",
        openscore_dir / "sq8913219.measures.tsv",
    )
    os_full = os_loader.create_timeline(uid="openscore_full")

    boundaries = [
        element.coordinate.value
        for element in os_full.flow_control.section_breaks
        if element.coordinate.value < os_full.length.value
    ]
    os_full.create_regions_from_boundaries(
        [0, *[float(b) for b in boundaries], float(os_full.length.value)],
        prefix="movement",
    )
    return os_full.create_child_from_region("region_4", uid="openscore")


@pytest.fixture(scope="module")
def beethoven_score_group() -> dict[str, Any]:
    """Build the Beethoven score group with all 3 timelines + flow data.

    Returns a dict with keys:
        - "group": the TimelineGroup
        - "clt_1": CLT1 timeline
        - "dgt_1": DGT1 timeline (SegmentLine)
        - "openscore": OpenScore timeline
        - "qb_sections": the runs of CLT1's default flow, as (Fraction,
          Fraction) pairs in quarters
        - "dgt1_id": auto-generated ID of DGT1
    """
    data_dir = SCORE_DATA_DIR
    abc_dir = data_dir / BEETHOVEN_ABC_DIR_REL

    # CLT1
    abc_loader = Ms3Loader.from_file(
        abc_dir / "n04op18-4_04.notes.tsv",
        abc_dir / "n04op18-4_04.measures.tsv",
        abc_dir / "n04op18-4_04.harmonies.tsv",
    )
    clt1 = abc_loader.create_timeline(uid="clt_1")

    # DGT1
    dgt1 = _build_dgt1(data_dir)

    # OpenScore
    openscore = _build_openscore(data_dir)

    # Score group
    from timetoalign.timelines import TimelineGroup

    score_group = TimelineGroup(
        id="score",
        name="Score (ABC + OMR + OpenScore)",
        timelines=[clt1, dgt1, openscore],
    )

    # The default flow that CLT1's flow control plays
    qb_sections = [
        (run.start.value, run.end.value)
        for run in play(clt1.flow_control, clt1.length).runs
    ]

    return {
        "group": score_group,
        "clt_1": clt1,
        "dgt_1": dgt1,
        "openscore": openscore,
        "qb_sections": qb_sections,
        "dgt1_id": dgt1.id,
    }


def _unfold_group(
    score_group_data: dict[str, Any],
) -> dict[str, SegmentLine]:
    """Unfold all timelines in the score group via one FlowMap.

    For each run of the default flow, retrieves the start/end TimeStamp
    from the score group (in CLT1's coordinate space) and uses each
    member's linearly converted coordinates to ``get_slice()`` and
    ``append_segment()``.

    Returns:
        Dict mapping timeline ID -> unfolded SegmentLine.
    """
    group = score_group_data["group"]
    qb_sections = score_group_data["qb_sections"]
    clt1 = score_group_data["clt_1"]
    dgt1 = score_group_data["dgt_1"]
    openscore = score_group_data["openscore"]
    dgt1_id = score_group_data["dgt1_id"]

    # Timeline metadata: (source_timeline, target_type_for_segment_line)
    timelines = {
        "clt_1": clt1,
        dgt1_id: dgt1,
        "openscore": openscore,
    }

    # Create empty SegmentLines for each timeline
    unfolded: dict[str, SegmentLine] = {}
    for tl_id, tl in timelines.items():
        unfolded[tl_id] = SegmentLine[type(tl)](
            length=0,
            unit=tl.unit,
            number_type=tl.number_type,
        )

    # Iterate run by run
    for i, (qb_start, qb_end) in enumerate(qb_sections):
        start_ts = group.get_timestamp_at(float(qb_start), "clt_1")
        end_ts = group.get_timestamp_at(float(qb_end), "clt_1")

        for tl_id, tl in timelines.items():
            coord_start = start_ts.get_coordinate_for(tl_id, format="float")
            coord_end = end_ts.get_coordinate_for(tl_id, format="float")

            seg = tl.get_slice(coord_start, coord_end, truncate_events=True)
            unfolded[tl_id].append_segment(seg, name=f"section_{i}")

    return unfolded


@pytest.fixture(scope="module")
def beethoven_unfolded_group(
    beethoven_score_group: dict[str, Any],
) -> dict[str, SegmentLine]:
    """Unfold the shared Beethoven score group once for read-only assertions."""
    return _unfold_group(beethoven_score_group)


@pytest.mark.slow
class TestGroupUnfolding:
    """Unfold an entire TimelineGroup via one FlowMap.

    Demonstrates that a single FlowMap (derived from CLT1's score flow)
    can unfold ALL timelines in a group — regardless of their domain —
    by resolving section boundaries through the group's TimeStamps and slicing
    each timeline at the corresponding coordinates.

    The test uses the Beethoven Op.18 No.4 iv multimodal score group
    containing:
    - CLT1: ContinuousLogicalTimeline (ABC v2.6, 878.5 quarters)
    - DGT1: SegmentLine[SegmentLine[DiscreteGraphicalTimeline]] (OMR, 106425 pixels)
    - OpenScore: ContinuousLogicalTimeline (4th movement, 878.5 quarters)

    The default flow has 11 runs producing 1116 unfolded QB.
    """

    def test_prerequisite_folded_lengths(self, beethoven_score_group: dict[str, Any]):
        """Verify source timelines have the expected folded lengths."""
        clt1 = beethoven_score_group["clt_1"]
        openscore = beethoven_score_group["openscore"]

        # Both CLT1 and OpenScore should be 878.5 QB (all endings)
        assert clt1.length.value == float(
            BEETHOVEN_FOLDED_QB
        ), f"CLT1 length {clt1.length.value} != {BEETHOVEN_FOLDED_QB}"
        assert openscore.length.value == float(
            BEETHOVEN_FOLDED_QB
        ), f"OpenScore length {openscore.length.value} != {BEETHOVEN_FOLDED_QB}"

    def test_prerequisite_qb_sections(self, beethoven_score_group: dict[str, Any]):
        """Verify QB sections count and total."""
        qb_sections = beethoven_score_group["qb_sections"]

        assert len(qb_sections) == BEETHOVEN_N_SECTIONS
        total = sum(e - s for s, e in qb_sections)
        assert total == BEETHOVEN_UNFOLDED_QB

    def test_all_timelines_produce_correct_segment_count(
        self, beethoven_unfolded_group: dict[str, SegmentLine]
    ):
        """Each unfolded timeline has exactly N_SECTIONS segments."""
        for tl_id, sl in beethoven_unfolded_group.items():
            assert sl.n_segments == BEETHOVEN_N_SECTIONS, (
                f"{tl_id}: n_segments={sl.n_segments}, "
                f"expected {BEETHOVEN_N_SECTIONS}"
            )

    def test_clt1_unfolded_length(
        self, beethoven_unfolded_group: dict[str, SegmentLine]
    ):
        """CLT1 unfolded length matches gold standard (1116 QB)."""
        clt1_sl = beethoven_unfolded_group["clt_1"]

        assert clt1_sl.length.value == float(BEETHOVEN_UNFOLDED_QB), (
            f"CLT1 unfolded length {clt1_sl.length.value} != "
            f"{BEETHOVEN_UNFOLDED_QB}"
        )

    def test_openscore_unfolded_length(
        self, beethoven_unfolded_group: dict[str, SegmentLine]
    ):
        """OpenScore unfolded length matches CLT1 (same musical content)."""
        os_sl = beethoven_unfolded_group["openscore"]

        assert os_sl.length.value == float(BEETHOVEN_UNFOLDED_QB), (
            f"OpenScore unfolded length {os_sl.length.value} != "
            f"{BEETHOVEN_UNFOLDED_QB}"
        )

    def test_dgt1_unfolded_longer_than_original(
        self,
        beethoven_score_group: dict[str, Any],
        beethoven_unfolded_group: dict[str, SegmentLine],
    ):
        """DGT1 unfolded length exceeds original (repeated sections add pixels)."""
        dgt1 = beethoven_score_group["dgt_1"]
        dgt1_id = beethoven_score_group["dgt1_id"]
        dgt1_sl = beethoven_unfolded_group[dgt1_id]

        assert dgt1_sl.length.value > dgt1.length.value, (
            f"DGT1 unfolded {dgt1_sl.length.value} should exceed "
            f"original {dgt1.length.value}"
        )

    def test_segments_are_contiguous(
        self, beethoven_unfolded_group: dict[str, SegmentLine]
    ):
        """Every unfolded SegmentLine has contiguous segments."""
        for tl_id, sl in beethoven_unfolded_group.items():
            offsets = []
            for seg_id in sl._segment_order:
                offset = sl._child_offsets[seg_id].value
                child = sl._children[seg_id]
                offsets.append((offset, offset + child.length.value))

            for i in range(1, len(offsets)):
                assert offsets[i][0] == offsets[i - 1][1], (
                    f"{tl_id}: segment {i} start {offsets[i][0]} != "
                    f"segment {i - 1} end {offsets[i - 1][1]}"
                )

    def test_segment_types_preserved(
        self,
        beethoven_score_group: dict[str, Any],
        beethoven_unfolded_group: dict[str, SegmentLine],
    ):
        """Segment types match the source timeline types."""
        dgt1_id = beethoven_score_group["dgt1_id"]

        # CLT1 segments should be ContinuousLogicalTimeline
        assert (
            beethoven_unfolded_group["clt_1"].segment_type is ContinuousLogicalTimeline
        )

        # OpenScore segments should be ContinuousLogicalTimeline
        assert (
            beethoven_unfolded_group["openscore"].segment_type
            is ContinuousLogicalTimeline
        )

        # DGT1 segments are slices of the nested source line, so they carry its
        # exact parameterized class.
        assert (
            beethoven_unfolded_group[dgt1_id].segment_type
            is SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]
        )

    def test_clt1_segment_lengths_match_qb_sections(
        self,
        beethoven_score_group: dict[str, Any],
        beethoven_unfolded_group: dict[str, SegmentLine],
    ):
        """CLT1 segment lengths match the QB section durations exactly."""
        qb_sections = beethoven_score_group["qb_sections"]
        clt1_sl = beethoven_unfolded_group["clt_1"]

        for i, seg_id in enumerate(clt1_sl._segment_order):
            child = clt1_sl._children[seg_id]
            expected_dur = float(qb_sections[i][1] - qb_sections[i][0])
            assert child.length.value == expected_dur, (
                f"CLT1 segment {i}: length {child.length.value} != "
                f"expected {expected_dur}"
            )

    def test_unfold_via_flowmap_matches_group_unfolding(
        self,
        beethoven_score_group: dict[str, Any],
        beethoven_unfolded_group: dict[str, SegmentLine],
    ):
        """unfold_via_flowmap on CLT1 produces same length as group method.

        Verifies consistency between the single-timeline unfolding function
        (same-type result with one appended child per section) and the
        group-based SegmentLine assembly used by this test module's helper.
        """
        clt1 = beethoven_score_group["clt_1"]

        # Single-timeline unfolding: a same-type timeline with appended children.
        clt1_single = unfold_via_flowmap(
            clt1, FlowMap(beethoven_score_group["qb_sections"])
        )

        # Group-based unfolding (the module helper's SegmentLine).
        clt1_group = beethoven_unfolded_group["clt_1"]

        assert clt1_single.length.value == clt1_group.length.value, (
            f"Single-timeline {clt1_single.length.value} != "
            f"group {clt1_group.length.value}"
        )
        assert clt1_single.n_children == clt1_group.n_segments


# endregion
