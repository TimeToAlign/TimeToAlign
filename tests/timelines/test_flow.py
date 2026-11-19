"""Tests for Flow API.

This test suite validates the Flow API against ms3 gold standard unfolded
TSV files. Tests are organized by complexity, starting with the simplest
case (no flow control) and progressing to complex D.S./D.C. structures.

Test Specimens (Complexity Order):
1. rachmaninoff_concerto2 - 374 MCs, no flow control (baseline)
2. flow_control/polyrhythm_only - 14 MCs, line breaks only
3. couperin/c05n05_musete - 58 MCs -> 138 unfolded, D.S. al Fine
4. couperin/c11n08_Rondeau - Rondeau form
5. beethoven_op18_flow - Repeats + Voltas
6. beethoven_woo71 - Complex split bars
7. flow_control/flow_only - All edge cases

Validation Criteria (ZERO TOLERANCE):
- EXACT mc_playthrough sequence
- EXACT mn_playthrough values with suffixes
- EXACT quarterbeats values
- EXACT row count
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from timetoalign.core import TimeUnit
from timetoalign.core.enums import FlowControlElement, FlowMode, IncompletePosition
from timetoalign.loader.score import Ms3Loader
from timetoalign.storage.events import EventData
from timetoalign.testdata import ensure_data
from timetoalign.timelines import Flow, FlowMap, MeasureUnit, ScoreFlowController
from timetoalign.timelines.flow import (
    AtomicSection,
    PlaythroughSection,
    load_valid_flows,
)

from .conftest import SCORE_DIR

# region Test Helpers


class _MockMeasureData(EventData):
    """Lightweight mock for MeasureData, wrapping a PyArrow table.

    Used in tests that construct ScoreFlowController from synthetic measure data
    without going through a full loader pipeline. Subclassing EventData means it
    exposes the same public surface (``schema``, ``column_values()``) that
    ScoreFlowController relies on.
    """

    def __init__(self, tbl):
        super().__init__(tbl, unit=TimeUnit.quarters)


# endregion


# region Fixtures


@pytest.fixture
def data_dir() -> Path:
    """Path to test data directory."""
    return SCORE_DIR


@pytest.fixture
def target_flows_dir() -> Path:
    """Path to target_flows directory."""
    return ensure_data("target_flows")


@pytest.fixture
def rachmaninoff_measures_tsv(data_dir: Path) -> Path:
    """Path to Rachmaninoff folded measures TSV."""
    return (
        data_dir
        / "rachmaninoff_concerto2"
        / "score"
        / "Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff.measures.tsv"
    )


@pytest.fixture
def rachmaninoff_unfolded_tsv(data_dir: Path) -> Path:
    """Path to Rachmaninoff unfolded measures TSV."""
    return (
        data_dir
        / "rachmaninoff_concerto2"
        / "score"
        / "Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff_unfolded.measures.tsv"
    )


@pytest.fixture
def musete_measures_tsv(data_dir: Path) -> Path:
    """Path to Couperin Musete folded measures TSV."""
    return data_dir / "couperin_concerts" / "c05n05_musete.measures.tsv"


@pytest.fixture
def musete_unfolded_tsv(data_dir: Path) -> Path:
    """Path to Couperin Musete unfolded measures TSV."""
    return data_dir / "couperin_concerts" / "c05n05_musete_unfolded.measures.tsv"


@pytest.fixture
def rondeau_measures_tsv(data_dir: Path) -> Path:
    """Path to Couperin Rondeau folded measures TSV."""
    return data_dir / "couperin_concerts" / "c11n08_Rondeau.measures.tsv"


@pytest.fixture
def rondeau_unfolded_tsv(data_dir: Path) -> Path:
    """Path to Couperin Rondeau unfolded measures TSV."""
    return data_dir / "couperin_concerts" / "c11n08_Rondeau_unfolded.measures.tsv"


# endregion

# region Unit Tests: MeasureUnit


class TestMeasureUnit:
    """Test MeasureUnit dataclass."""

    def test_creation(self) -> None:
        """MeasureUnit can be created with all fields."""
        unit = MeasureUnit(
            mc=1,
            mn="1",
            duration_qb=Fraction(4),
            next=(2,),
            volta=None,
            timesig="4/4",
            start_repeat=False,
            end_repeat=False,
        )
        assert unit.mc == 1
        assert unit.mn == "1"
        assert unit.duration_qb == Fraction(4)
        assert unit.next == (2,)
        assert unit.volta is None
        assert unit.timesig == "4/4"
        assert unit.start_repeat is False
        assert unit.end_repeat is False

    def test_frozen(self) -> None:
        """MeasureUnit is immutable."""
        unit = MeasureUnit(
            mc=1,
            mn="1",
            duration_qb=Fraction(4),
            next=(2,),
        )
        with pytest.raises(AttributeError):
            unit.mc = 2  # type: ignore

    def test_to_dict(self) -> None:
        """MeasureUnit can be converted to dict."""
        unit = MeasureUnit(
            mc=5,
            mn="5",
            duration_qb=Fraction(4),
            next=(6, 10),
            volta=1,
        )
        d = unit.to_dict()
        assert d["mc"] == 5
        assert d["mn"] == "5"
        assert d["duration_qb"] == 4.0  # Converted to float for serialization
        assert d["next"] == [6, 10]
        assert d["volta"] == 1

    def test_flowcontrol_fields(self) -> None:
        """MeasureUnit supports all FlowControlElement fields."""
        unit = MeasureUnit(
            mc=5,
            mn="5",
            duration_qb=Fraction(4),
            next=(6, 10),
            volta=2,
            timesig="4/4",
            timesig_duration_qb=Fraction(4),
            start_repeat=True,
            end_repeat=False,
            jump_from=True,
            jump_to=False,
            segno="segno",
            coda=None,
            fine=False,
            section_break=False,
            flow_control_types=("repeat_start", "segno", "jump_from"),
        )
        assert unit.jump_from is True
        assert unit.jump_to is False
        assert unit.segno == "segno"
        assert unit.coda is None
        assert unit.fine is False
        assert unit.section_break is False
        assert unit.timesig_duration_qb == Fraction(4)
        assert unit.flow_control_types == ("repeat_start", "segno", "jump_from")

    def test_to_dict_with_flowcontrol(self) -> None:
        """MeasureUnit.to_dict() includes FlowControlElement fields."""
        unit = MeasureUnit(
            mc=5,
            mn="5",
            duration_qb=Fraction(4),
            next=(6,),
            segno="segno",
            fine=True,
            flow_control_types=("segno", "fine"),
        )
        d = unit.to_dict()
        assert d["segno"] == "segno"
        assert d["fine"] is True
        assert d["flow_control_types"] == "segno;fine"

    def test_from_dict(self) -> None:
        """MeasureUnit can be created from dict."""
        d = {
            "mc": 5,
            "mn": "5",
            "duration_qb": 4.0,
            "next": [6, 10],
            "volta": 2,
            "timesig": "4/4",
            "start_repeat": True,
            "segno": "segno",
            "flow_control_types": "repeat_start;segno;jump_from",
        }
        unit = MeasureUnit.from_dict(d)
        assert unit.mc == 5
        assert unit.mn == "5"
        assert unit.duration_qb == Fraction(4)
        assert unit.next == (6, 10)
        assert unit.volta == 2
        assert unit.start_repeat is True
        assert unit.segno == "segno"
        assert unit.flow_control_types == ("repeat_start", "segno", "jump_from")

    def test_from_dict_roundtrip(self) -> None:
        """MeasureUnit round-trip (to_dict -> from_dict) preserves data."""
        original = MeasureUnit(
            mc=5,
            mn="5",
            duration_qb=Fraction(4),
            next=(6, 10),
            volta=2,
            timesig="4/4",
            timesig_duration_qb=Fraction(4),
            start_repeat=True,
            segno="segno",
            flow_control_types=("repeat_start", "segno", "jump_from"),
        )
        d = original.to_dict()
        restored = MeasureUnit.from_dict(d)

        assert restored.mc == original.mc
        assert restored.mn == original.mn
        assert restored.next == original.next
        assert restored.volta == original.volta
        assert restored.start_repeat == original.start_repeat
        assert restored.segno == original.segno
        assert restored.flow_control_types == original.flow_control_types

    def test_repr(self) -> None:
        """MeasureUnit has useful repr."""
        unit = MeasureUnit(mc=1, mn="1", duration_qb=Fraction(4), next=(2,))
        r = repr(unit)
        assert "MC 1" in r
        assert "next=(2,)" in r


# endregion

# region Unit Tests: Typed MeasureUnit Subclasses


class TestTypedMeasures:
    """Test typed MeasureUnit subclasses.

    These classes represent the Typing step of the two-step typing/grouping
    algorithm:
    - IncompleteMeasure: duration < expected
    - CompleteMeasure: duration == expected
    - OverlengthMeasure: duration > expected
    """

    def test_incomplete_measure_creation(self) -> None:
        """IncompleteMeasure can be created with position field."""
        from timetoalign.timelines import IncompleteMeasure

        unit = IncompleteMeasure(
            mc=1,
            mn="0",  # Anacrusis typically has mn="0"
            duration_qb=Fraction(1),  # 1 quarterbeat
            next=(2,),
            timesig="4/4",
            timesig_duration_qb=Fraction(4),  # Expected 4 quarterbeats
            position=IncompletePosition.anacrusis,
        )
        assert unit.mc == 1
        assert unit.duration_qb == Fraction(1)
        assert unit.timesig_duration_qb == Fraction(4)
        assert unit.position == IncompletePosition.anacrusis

    def test_incomplete_measure_is_measureunit(self) -> None:
        """IncompleteMeasure is a MeasureUnit subclass."""
        from timetoalign.core.enums import IncompletePosition
        from timetoalign.timelines import IncompleteMeasure

        unit = IncompleteMeasure(
            mc=1,
            mn="0",
            duration_qb=Fraction(1),
            next=(2,),
            position=IncompletePosition.anacrusis,
        )
        assert isinstance(unit, MeasureUnit)

    def test_complete_measure_creation(self) -> None:
        """CompleteMeasure can be created."""
        from timetoalign.timelines import CompleteMeasure

        unit = CompleteMeasure(
            mc=2,
            mn="1",
            duration_qb=Fraction(4),
            next=(3,),
            timesig="4/4",
            timesig_duration_qb=Fraction(4),
        )
        assert unit.mc == 2
        assert unit.duration_qb == Fraction(4)
        assert unit.timesig_duration_qb == Fraction(4)

    def test_complete_measure_is_measureunit(self) -> None:
        """CompleteMeasure is a MeasureUnit subclass."""
        from timetoalign.timelines import CompleteMeasure

        unit = CompleteMeasure(
            mc=2,
            mn="1",
            duration_qb=Fraction(4),
            next=(3,),
        )
        assert isinstance(unit, MeasureUnit)

    def test_overlength_measure_creation(self) -> None:
        """OverlengthMeasure can be created."""
        from timetoalign.timelines import OverlengthMeasure

        unit = OverlengthMeasure(
            mc=10,
            mn="9",
            duration_qb=Fraction(9),  # 9/4 in a 4/4 measure
            next=(11,),
            timesig="4/4",
            timesig_duration_qb=Fraction(4),
        )
        assert unit.mc == 10
        assert unit.duration_qb == Fraction(9)
        assert unit.timesig_duration_qb == Fraction(4)

    def test_overlength_measure_is_measureunit(self) -> None:
        """OverlengthMeasure is a MeasureUnit subclass."""
        from timetoalign.timelines import OverlengthMeasure

        unit = OverlengthMeasure(
            mc=10,
            mn="9",
            duration_qb=Fraction(9),
            next=(11,),
        )
        assert isinstance(unit, MeasureUnit)

    def test_typed_measures_preserve_flowcontrol(self) -> None:
        """Typed measures inherit all FlowControlElement fields."""
        from timetoalign.core.enums import IncompletePosition
        from timetoalign.timelines import IncompleteMeasure

        unit = IncompleteMeasure(
            mc=1,
            mn="0",
            duration_qb=Fraction(1),
            next=(2,),
            start_repeat=True,
            segno="segno",
            jump_to=True,
            flow_control_types=("repeat_start", "segno", "jump_to"),
            position=IncompletePosition.anacrusis,
        )
        assert unit.start_repeat is True
        assert unit.segno == "segno"
        assert unit.jump_to is True
        assert unit.flow_control_types == ("repeat_start", "segno", "jump_to")

    def test_incomplete_position_enum_values(self) -> None:
        """IncompletePosition enum has expected values."""
        from timetoalign.core.enums import IncompletePosition

        assert IncompletePosition.anacrusis.value == "anacrusis"
        assert IncompletePosition.final.value == "final"
        assert IncompletePosition.split_first.value == "split_first"
        assert IncompletePosition.split_second.value == "split_second"
        assert IncompletePosition.unknown.value == "unknown"


# endregion

# region Unit Tests: AtomicSection with typed_measures


class TestAtomicSectionTypedMeasures:
    """Test AtomicSection with typed_measures field."""

    def test_atomic_section_typed_measures_field(self) -> None:
        """AtomicSection has typed_measures field."""
        from timetoalign.timelines import CompleteMeasure

        typed = (
            CompleteMeasure(mc=1, mn="1", duration_qb=Fraction(4), next=(2,)),
            CompleteMeasure(mc=2, mn="2", duration_qb=Fraction(4), next=(3,)),
        )
        sec = AtomicSection(
            id="A",
            mc_start=1,
            mc_end=3,
            typed_measures=typed,
        )
        assert sec.typed_measures is not None
        assert len(sec.typed_measures) == 2
        assert sec.typed_measures[0].mc == 1
        assert sec.typed_measures[1].mc == 2

    def test_atomic_section_typed_measures_none_by_default(self) -> None:
        """AtomicSection typed_measures is None by default."""
        sec = AtomicSection(id="A", mc_start=1, mc_end=5)
        assert sec.typed_measures is None

    def test_atomic_section_to_dict_includes_typed_counts(self) -> None:
        """AtomicSection to_dict includes typed measure counts."""
        from timetoalign.core.enums import IncompletePosition
        from timetoalign.timelines import CompleteMeasure, IncompleteMeasure

        typed = (
            IncompleteMeasure(
                mc=1,
                mn="0",
                duration_qb=Fraction(1),
                next=(2,),
                position=IncompletePosition.anacrusis,
            ),
            CompleteMeasure(mc=2, mn="1", duration_qb=Fraction(4), next=(3,)),
            CompleteMeasure(mc=3, mn="2", duration_qb=Fraction(4), next=(4,)),
        )
        sec = AtomicSection(
            id="A",
            mc_start=1,
            mc_end=4,
            typed_measures=typed,
        )
        d = sec.to_dict()
        assert d["typed_measures_count"] == 3
        assert d["incomplete_count"] == 1
        assert d["overlength_count"] == 0


class TestPlaythroughSectionTypedMeasures:
    """Test PlaythroughSection with typed_measures field."""

    def test_playthrough_section_typed_measures_field(self) -> None:
        """PlaythroughSection has typed_measures field."""
        from timetoalign.timelines import CompleteMeasure

        typed = (
            CompleteMeasure(mc=1, mn="1", duration_qb=Fraction(4), next=(2,)),
            CompleteMeasure(mc=2, mn="2", duration_qb=Fraction(4), next=(3,)),
        )
        sec = PlaythroughSection(
            mc_start=1,
            mc_end=3,
            atomic_section_ids=("A",),
            typed_measures=typed,
        )
        assert sec.typed_measures is not None
        assert len(sec.typed_measures) == 2

    def test_playthrough_section_typed_measures_none_by_default(self) -> None:
        """PlaythroughSection typed_measures is None by default."""
        sec = PlaythroughSection(mc_start=1, mc_end=5, atomic_section_ids=("A",))
        assert sec.typed_measures is None


# endregion

# region Unit Tests: MeasureGroup


class TestMeasureGroup:
    """Test MeasureGroup base class and subclasses.

    MeasureGroup represents the Grouping step of the two-step typing/grouping
    algorithm:
    - VoltaGroup: measures under same volta bracket
    - SplitMeasure: IncompleteMeasures that together complete
    - IncompleteGroup: isolated IncompleteMeasures
    - CompleteMeasureGroup: adjacent CompleteMeasures
    - OverlengthGroup: OverlengthMeasures
    """

    def test_measure_group_base_class(self) -> None:
        """MeasureGroup can be created with members."""
        from timetoalign.timelines import CompleteMeasure, MeasureGroup

        m1 = CompleteMeasure(mc=1, mn="1", duration_qb=Fraction(4), next=(2,))
        m2 = CompleteMeasure(mc=2, mn="2", duration_qb=Fraction(4), next=(3,))
        group = MeasureGroup(members=(m1, m2))

        assert len(group) == 2
        assert group.mc_start == 1
        assert group.mc_end == 3  # Right-open
        assert group.mc_range == (1, 3)
        assert group.total_duration_qb == Fraction(8)

    def test_measure_group_frozen(self) -> None:
        """MeasureGroup is immutable."""
        from timetoalign.timelines import CompleteMeasure, MeasureGroup

        m1 = CompleteMeasure(mc=1, mn="1", duration_qb=Fraction(4), next=(2,))
        group = MeasureGroup(members=(m1,))
        with pytest.raises(AttributeError):
            group.members = ()  # type: ignore

    def test_measure_group_empty_raises(self) -> None:
        """MeasureGroup raises on empty members."""
        from timetoalign.timelines import MeasureGroup

        with pytest.raises(ValueError, match="must have at least one member"):
            MeasureGroup(members=())

    def test_volta_group_creation(self) -> None:
        """VoltaGroup can be created with volta_number."""
        from timetoalign.timelines import CompleteMeasure, VoltaGroup

        m1 = CompleteMeasure(mc=5, mn="5", duration_qb=Fraction(4), next=(6,), volta=1)
        m2 = CompleteMeasure(mc=6, mn="6", duration_qb=Fraction(4), next=(7,), volta=1)
        group = VoltaGroup(members=(m1, m2), volta_number=1)

        assert group.volta_number == 1
        assert len(group) == 2
        assert "volta=1" in repr(group)

    def test_volta_group_invalid_number_raises(self) -> None:
        """VoltaGroup raises on invalid volta_number."""
        from timetoalign.timelines import CompleteMeasure, VoltaGroup

        m = CompleteMeasure(mc=1, mn="1", duration_qb=Fraction(4), next=(2,))
        with pytest.raises(ValueError, match="volta_number must be >= 1"):
            VoltaGroup(members=(m,), volta_number=0)

    def test_split_measure_creation(self) -> None:
        """SplitMeasure groups IncompleteMeasures that together complete."""
        from timetoalign.core.enums import IncompletePosition
        from timetoalign.timelines import IncompleteMeasure, SplitMeasure

        anacrusis = IncompleteMeasure(
            mc=1,
            mn="0",
            duration_qb=Fraction(1),  # 1/4
            next=(2,),
            timesig="4/4",
            timesig_duration_qb=Fraction(4),
            position=IncompletePosition.anacrusis,
        )
        final = IncompleteMeasure(
            mc=58,
            mn="57",
            duration_qb=Fraction(3),  # 3/4
            next=(-1,),
            timesig="4/4",
            timesig_duration_qb=Fraction(4),
            position=IncompletePosition.final,
        )
        split = SplitMeasure(members=(anacrusis, final))

        assert len(split) == 2
        assert split.total_duration_qb == Fraction(4)  # 1/4 + 3/4 = 4/4
        assert "SplitMeasure" in repr(split)

    def test_split_measure_non_incomplete_raises(self) -> None:
        """SplitMeasure raises if member is not IncompleteMeasure."""
        from timetoalign.timelines import CompleteMeasure, SplitMeasure

        m = CompleteMeasure(mc=1, mn="1", duration_qb=Fraction(4), next=(2,))
        with pytest.raises(TypeError, match="must be IncompleteMeasure"):
            SplitMeasure(members=(m,))

    def test_incomplete_group_creation(self) -> None:
        """IncompleteGroup wraps isolated IncompleteMeasures."""
        from timetoalign.core.enums import IncompletePosition
        from timetoalign.timelines import IncompleteGroup, IncompleteMeasure

        anacrusis = IncompleteMeasure(
            mc=1,
            mn="0",
            duration_qb=Fraction(1),
            next=(2,),
            position=IncompletePosition.anacrusis,
        )
        group = IncompleteGroup(members=(anacrusis,))

        assert len(group) == 1
        assert "anacrusis" in repr(group)

    def test_incomplete_group_non_incomplete_raises(self) -> None:
        """IncompleteGroup raises if member is not IncompleteMeasure."""
        from timetoalign.timelines import CompleteMeasure, IncompleteGroup

        m = CompleteMeasure(mc=1, mn="1", duration_qb=Fraction(4), next=(2,))
        with pytest.raises(TypeError, match="must be IncompleteMeasure"):
            IncompleteGroup(members=(m,))

    def test_complete_measure_group_creation(self) -> None:
        """CompleteMeasureGroup groups adjacent CompleteMeasures."""
        from timetoalign.timelines import CompleteMeasure, CompleteMeasureGroup

        m1 = CompleteMeasure(mc=2, mn="1", duration_qb=Fraction(4), next=(3,))
        m2 = CompleteMeasure(mc=3, mn="2", duration_qb=Fraction(4), next=(4,))
        m3 = CompleteMeasure(mc=4, mn="3", duration_qb=Fraction(4), next=(5,))
        group = CompleteMeasureGroup(members=(m1, m2, m3))

        assert len(group) == 3
        assert group.mc_start == 2
        assert group.mc_end == 5  # Right-open
        assert group.total_duration_qb == Fraction(12)
        assert "CompleteMeasureGroup" in repr(group)

    def test_complete_measure_group_non_complete_raises(self) -> None:
        """CompleteMeasureGroup raises if member is not CompleteMeasure."""
        from timetoalign.core.enums import IncompletePosition
        from timetoalign.timelines import CompleteMeasureGroup, IncompleteMeasure

        m = IncompleteMeasure(
            mc=1,
            mn="0",
            duration_qb=Fraction(1),
            next=(2,),
            position=IncompletePosition.anacrusis,
        )
        with pytest.raises(TypeError, match="must be CompleteMeasure"):
            CompleteMeasureGroup(members=(m,))

    def test_overlength_group_creation(self) -> None:
        """OverlengthGroup wraps OverlengthMeasures."""
        from timetoalign.timelines import OverlengthGroup, OverlengthMeasure

        m = OverlengthMeasure(
            mc=100,
            mn="100",
            duration_qb=Fraction(10),  # Longer than expected
            next=(101,),
            timesig="4/4",
            timesig_duration_qb=Fraction(4),
        )
        group = OverlengthGroup(members=(m,))

        assert len(group) == 1
        assert "OverlengthGroup" in repr(group)


class TestBuildGroups:
    """Test ScoreFlowController._build_groups() algorithm."""

    def test_groups_populated_in_atomic_section(self) -> None:
        """AtomicSection.groups is populated by ScoreFlowController."""
        import pyarrow as pa

        from timetoalign.timelines import ScoreFlowController

        # Create test data with anacrusis + complete measures
        table = pa.table(
            {
                "mc": [1, 2, 3],
                "mn": ["0", "1", "2"],
                "duration": [{"value": 1.0}, {"value": 4.0}, {"value": 4.0}],
                "start": [{"value": 0}, {"value": 1}, {"value": 5}],
                "next": [[2], [3], [-1]],
                "timesig": ["4/4", "4/4", "4/4"],
                "volta": [None, None, None],
            }
        )

        md = _MockMeasureData(table)
        controller = ScoreFlowController(md)

        sections = controller.get_sections()
        assert len(sections) == 1

        sec = sections[0]
        assert sec.groups is not None
        assert len(sec.groups) == 2  # IncompleteGroup + CompleteMeasureGroup

    def test_groups_complete_coverage(self) -> None:
        """Every typed_measure belongs to exactly one group."""
        import pyarrow as pa

        from timetoalign.timelines import ScoreFlowController

        table = pa.table(
            {
                "mc": [1, 2, 3, 4, 5],
                "mn": ["0", "1", "2", "3", "4"],
                "duration": [
                    {"value": 1.0},
                    {"value": 4.0},
                    {"value": 4.0},
                    {"value": 4.0},
                    {"value": 3.0},
                ],
                "start": [
                    {"value": 0},
                    {"value": 1},
                    {"value": 5},
                    {"value": 9},
                    {"value": 13},
                ],
                "next": [[2], [3], [4], [5], [-1]],
                "timesig": ["4/4", "4/4", "4/4", "4/4", "4/4"],
                "volta": [None, None, None, None, None],
            }
        )

        md = _MockMeasureData(table)
        controller = ScoreFlowController(md)

        sections = controller.get_sections()
        sec = sections[0]

        # Count total measures in groups
        grouped_mcs = set()
        for g in sec.groups or []:
            for m in g.members:
                assert m.mc not in grouped_mcs, f"MC {m.mc} appears in multiple groups"
                grouped_mcs.add(m.mc)

        # Verify all typed_measures are covered
        typed_mcs = {m.mc for m in sec.typed_measures or []}
        assert grouped_mcs == typed_mcs, "Not all measures covered by groups"

    def test_volta_groups_detected(self) -> None:
        """VoltaGroups are created for measures with same volta number."""
        import pyarrow as pa

        from timetoalign.timelines import ScoreFlowController, VoltaGroup

        table = pa.table(
            {
                "mc": [1, 2, 3, 4],
                "mn": ["1", "2", "2", "3"],
                "duration": [
                    {"value": 4.0},
                    {"value": 4.0},
                    {"value": 4.0},
                    {"value": 4.0},
                ],
                "start": [{"value": 0}, {"value": 4}, {"value": 8}, {"value": 12}],
                "next": [[2, 3], [4], [4], [-1]],  # Volta at MC 2 and 3
                "timesig": ["4/4", "4/4", "4/4", "4/4"],
                "volta": [None, 1, 2, None],  # MC 2 is volta 1, MC 3 is volta 2
            }
        )

        md = _MockMeasureData(table)
        controller = ScoreFlowController(md)

        # Each volta should be in its own section due to breaks
        sections = controller.get_sections()

        # Find VoltaGroups
        volta_groups = []
        for sec in sections:
            if sec.groups:
                for g in sec.groups:
                    if isinstance(g, VoltaGroup):
                        volta_groups.append(g)

        # WoO71 has exactly 2 VoltaGroups
        assert len(volta_groups) == 2

    def test_split_measure_same_mn_detected(self) -> None:
        """SplitMeasures are detected when IncompleteMeasures share same mn."""
        import pyarrow as pa

        from timetoalign.timelines import ScoreFlowController, SplitMeasure

        # WoO71-style split measures: MC 11 and MC 12 both have mn="10"
        table = pa.table(
            {
                "mc": [10, 11, 12, 13],
                "mn": ["9", "10", "10", "11"],  # MC 11 and 12 share mn="10"
                "duration": [
                    {"value": 2.0},
                    {"value": 1.0},
                    {"value": 1.0},
                    {"value": 2.0},
                ],
                "start": [{"value": 0}, {"value": 2}, {"value": 3}, {"value": 4}],
                "next": [[11], [12], [13], [-1]],
                "timesig": ["2/4", "2/4", "2/4", "2/4"],  # 2/4 = 2 quarterbeats
                "volta": [None, None, None, None],
            }
        )

        md = _MockMeasureData(table)
        controller = ScoreFlowController(md)

        sections = controller.get_sections()

        # Find SplitMeasures
        split_measures = []
        for sec in sections:
            if sec.groups:
                for g in sec.groups:
                    if isinstance(g, SplitMeasure):
                        split_measures.append(g)

        # Should detect the split measure (MC 11 + MC 12 = 2 quarterbeats = 2/4)
        assert len(split_measures) == 1
        assert len(split_measures[0].members) == 2
        assert split_measures[0].total_duration_qb == Fraction(2)


# endregion

# region Flow invariants


def _volta_with_following_section_table():
    """Synthetic measures: a repeat with prima/seconda volta, then a non-volta
    section, with a section break two measures after the seconda volta.

    This is a well-formed volta score. The seconda volta (MC 5, volta 2)
    continues sequentially into the music that follows the bracket, absorbing
    it up to the next genuine boundary (the section break at MC 7 places a
    boundary at MC 8), exactly as a seconda volta does in real scores. The
    only volta sections are the prima (-> repeat-start, non-volta) and the
    seconda (-> the continuation, non-volta), so no volta section flows to
    another volta section.
    """
    import pyarrow as pa

    return pa.table(
        {
            "mc": [1, 2, 3, 4, 5, 6, 7, 8],
            "mn": ["1", "2", "3", "3", "4", "5", "6", "7"],
            "duration": [{"value": 4.0}] * 8,
            "start": [{"value": float(4 * i)} for i in range(8)],
            "next": [[2], [3], [4, 5], [1], [6], [7], [8], [-1]],
            "timesig": ["4/4"] * 8,
            "volta": [None, None, None, 1, 2, None, None, None],
            "section_break": [
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
            ],
        }
    )


class TestFlowInvariants:
    """``ScoreFlowController.check_invariants`` — volta-follows-volta rule."""

    def test_clean_score_yields_no_diagnostics(self) -> None:
        """A well-formed volta score reports no invariant violations."""
        controller = ScoreFlowController(
            _MockMeasureData(_volta_with_following_section_table())
        )
        assert controller.check_invariants() == []

    def test_malformed_score_reports_volta_follows_volta(self) -> None:
        """A volta section whose ``to`` edge lands on another volta section is
        reported."""
        import pyarrow as pa

        from timetoalign.timelines import FlowDiagnostic

        # Prima volta (MC 4, volta 1) mis-resolves its next forward to the
        # seconda volta (MC 5) instead of back to the repeat-start (MC 1).
        table = pa.table(
            {
                "mc": [1, 2, 3, 4, 5, 6, 7],
                "mn": ["1", "2", "3", "3", "4", "5", "6"],
                "duration": [{"value": 4.0}] * 7,
                "start": [{"value": float(4 * i)} for i in range(7)],
                "next": [[2], [3], [4, 5], [5], [6], [7], [-1]],
                "timesig": ["4/4"] * 7,
                "volta": [None, None, None, 1, 2, None, None],
                "section_break": [False] * 7,
            }
        )
        controller = ScoreFlowController(_MockMeasureData(table))

        sections = controller.get_sections()
        by_id = {sec.id: sec for sec in sections}
        # The mis-resolution makes the prima-volta section A1 point at the
        # seconda-volta section A2.
        assert by_id["A1"].to == ("A2",)
        assert controller._measure_lookup[by_id["A1"].mc_start]["volta"] == 1
        assert controller._measure_lookup[by_id["A2"].mc_start]["volta"] == 2

        diagnostics = controller.check_invariants()
        assert len(diagnostics) == 1
        diag = diagnostics[0]
        assert isinstance(diag, FlowDiagnostic)
        assert diag.kind == "volta_follows_volta"
        assert diag.section_id == "A1"
        assert diag.mc == 4
        assert "'A1'" in diag.message
        assert "'A2'" in diag.message


# endregion

# region Guarded default-flow traversal


def _guarded_flow_controller(n: int, **columns) -> ScoreFlowController:
    """Build a controller from a synthetic measure table.

    Fills the mandatory measure fields (``mc``/``mn``/``duration``/``start``/
    ``timesig``) for ``n`` uniform 4/4 measures and overlays any flow-control
    columns supplied by name, mirroring the real column-to-``MeasureUnit``
    pipeline via the public constructor.
    """
    import pyarrow as pa

    table = {
        "mc": list(range(1, n + 1)),
        "mn": [str(i) for i in range(1, n + 1)],
        "duration": [{"value": 4.0}] * n,
        "start": [{"value": float(4 * i)} for i in range(n)],
        "timesig": ["4/4"] * n,
    }
    table.update(columns)
    return ScoreFlowController(_MockMeasureData(pa.table(table)))


def _mc_playthrough(flow: Flow) -> list[int]:
    """Flatten a computed flow to its MC visitation sequence."""
    sequence: list[int] = []
    for section in flow.sections:
        sequence.extend(range(section.mc_start, section.mc_end))
    return sequence


class TestGuardedFlowTraversal:
    """The default flow is a guarded transition system.

    Every fixture is a synthetic measure table built through the public
    ``ScoreFlowController`` constructor; assertions are exact. See
    ``README.md`` → "Guarded default-flow traversal" for the topologies.
    """

    def test_repeat_back_edge_outranks_unarmed_coda_jump(self) -> None:
        """An owed repeat plays before an armed al-coda exit fires.

        The inner repeat-end at MC 4 is also the armed "to coda" trigger (via
        the section-end convention, no explicit ``coda`` marker). The repeat
        back-edge outranks the coda jump, so the post-D.S. pass replays the
        inner ``2, 3, 4`` before exiting to ``codab``. The old ordering fired
        the coda jump first and dropped that repetition
        (``[1, 2, 3, 4, 2, 3, 4, 5, 2, 3, 4, 6]``).
        """
        controller = _guarded_flow_controller(
            6,
            next=[[2], [3], [4], [5], [2], [-1]],
            segno=[None, "segno", None, None, None, None],
            coda=[None, None, None, None, None, "codab"],
            markers=[None, "segno", None, None, None, "codab"],
            start_repeat=[False, True, False, False, False, False],
            end_repeat=[False, False, False, True, False, False],
            jump_bwd=[None, None, None, None, "segno", None],
            jump_fwd=[None, None, None, None, "codab", None],
            play_until=[None, None, None, None, "coda", None],
            section_break=[False, False, False, True, False, False],
        )

        sequence = _mc_playthrough(controller.compute_flow(FlowMode.default))

        assert sequence == [1, 2, 3, 4, 2, 3, 4, 5, 2, 3, 4, 2, 3, 4, 6]
        # The inner section [2, 4) is entered on both the natural and the
        # post-D.S. pass, so measures 2/3/4 each appear four times.
        assert sequence.count(2) == 4
        assert sequence.count(4) == 4

    def test_unterminating_flow_reports_cycle_instead_of_truncating(self) -> None:
        """A bare D.C. with no Fine is reported as a cycle, not spun.

        The lasso check stops the traversal at ``[1, 2]`` and appends a single
        ``flow_cycle`` diagnostic naming the closing D.S./D.C. edge. The old
        traversal produced 60 rows (``2 * 30``) and no diagnostic.
        """
        from timetoalign.timelines import FlowDiagnostic

        controller = _guarded_flow_controller(
            2,
            next=[[2], [1]],
            jump_bwd=[None, "start"],
        )

        flow = controller.compute_flow(FlowMode.default)
        sequence = _mc_playthrough(flow)
        assert sequence == [1, 2]

        diagnostics = controller.flow_diagnostics(FlowMode.default)
        assert len(diagnostics) == 1
        diag = diagnostics[0]
        assert isinstance(diag, FlowDiagnostic)
        assert diag.kind == "flow_cycle"
        assert diag.section_id == "A"
        assert diag.mc == 1
        assert "MC 2" in diag.message
        assert "MC 1" in diag.message

    def test_doppia_coda_selects_target_by_pass(self) -> None:
        """Forward targets stay scalar across pass-indexed arming.

        A comma belongs to the literal destination name in the first fixture.
        The second fixture arms the same real single-target idiom twice and
        resolves ``codab`` on both passes before the lasso closes.
        """
        literal_controller = _guarded_flow_controller(
            6,
            next=[[2], [3], [4], [5], [2], [-1]],
            segno=[None, "segno", None, None, None, None],
            coda=[None, None, "coda", None, None, "codab,varcoda"],
            markers=[None, "segno", "coda", None, None, "codab,varcoda"],
            jump_bwd=[None, None, None, None, "segno", None],
            jump_fwd=[None, None, None, None, "codab,varcoda", None],
            play_until=[None, None, None, None, "coda", None],
        )

        literal_sequence = _mc_playthrough(
            literal_controller.compute_flow(FlowMode.default)
        )
        assert literal_sequence == [1, 2, 3, 4, 5, 2, 3, 6]
        assert literal_controller._coda_targets(
            "codab,varcoda", {"codab,varcoda": 6}
        ) == (6,)

        repeated_controller = _guarded_flow_controller(
            5,
            next=[[2], [3], [4], [5], [2]],
            segno=[None, "segno", None, None, None],
            coda=[None, None, "coda", "codab", None],
            markers=[None, "segno", "coda", "codab", None],
            jump_bwd=[None, None, None, None, "segno"],
            jump_fwd=[None, None, None, None, "codab"],
            play_until=[None, None, None, None, "coda"],
        )

        repeated_sequence = _mc_playthrough(
            repeated_controller.compute_flow(FlowMode.default)
        )
        assert repeated_sequence == [1, 2, 3, 4, 5, 2, 3, 4, 5]
        assert repeated_sequence.count(4) == 2
        assert repeated_sequence.count(5) == 2
        assert [d.kind for d in repeated_controller.flow_diagnostics()] == [
            "flow_cycle"
        ]

    def test_repeat_end_with_single_candidate_auto_resolves(self) -> None:
        """One open jump-to instant resolves without a diagnostic.

        The only candidate upstream of the repeat-end at MC 3 is the
        conventional left-open piece-start (MC 1).
        """
        controller = _guarded_flow_controller(
            3,
            next=[[2], [3], [1]],
            end_repeat=[False, False, True],
        )

        targets, diagnostics = controller._repeat_end_resolution()

        assert targets == {3: 1}
        assert diagnostics == []

    def test_repeat_end_with_multiple_candidates_reports_ambiguity(self) -> None:
        """Several explicit starts resolve nearest-first with a diagnostic.

        Both repeat starts are open at MC 4. The inner start at MC 3 wins,
        while the diagnostic records the outer start at MC 2.
        """
        from timetoalign.timelines import FlowDiagnostic

        controller = _guarded_flow_controller(
            4,
            next=[[2], [3], [4], [3]],
            start_repeat=[False, True, True, False],
            end_repeat=[False, False, False, True],
        )

        targets, diagnostics = controller._repeat_end_resolution()

        assert targets == {4: 3}
        assert len(diagnostics) == 1
        diag = diagnostics[0]
        assert isinstance(diag, FlowDiagnostic)
        assert diag.kind == "ambiguous_repeat_end"
        assert diag.mc == 4
        assert "MC 3" in diag.message
        assert "MC 2" in diag.message

    def test_repeat_end_with_no_candidate_is_dangling(self) -> None:
        """A repeat-end with no scope or backward edge emits no jump.

        The explicit MC 2 start closes at MC 3. MC 5 has no open explicit
        start and its encoded successor terminates, so a
        ``dangling_repeat_end`` diagnostic names it.
        """
        from timetoalign.timelines import FlowDiagnostic

        controller = _guarded_flow_controller(
            5,
            next=[[2], [3], [2], [5], [-1]],
            start_repeat=[False, True, False, False, False],
            end_repeat=[False, False, True, False, True],
        )

        targets, diagnostics = controller._repeat_end_resolution()

        assert targets == {3: 2}
        assert len(diagnostics) == 1
        diag = diagnostics[0]
        assert isinstance(diag, FlowDiagnostic)
        assert diag.kind == "dangling_repeat_end"
        assert diag.mc == 5

        # No repeat-end jump is emitted for the dangling MC 5.
        repeat_jumps = [
            jump
            for jump in controller.get_jumps()
            if jump.control_type == FlowControlElement.repeat_end
        ]
        assert all(
            jump.from_position != controller._qb_at_mc_end(5) for jump in repeat_jumps
        )


class TestRegistryTraversalAgreement:
    """Canonical registry repeat jumps match executable default-flow edges."""

    @pytest.mark.parametrize(
        ("relative_path", "expected"),
        [
            pytest.param(
                "couperin_concerts/c05n05_musete.measures.tsv",
                {16: 1, 31: 17, 58: 32},
                id="musete",
            ),
            pytest.param(
                "couperin_concerts/c11n08_Rondeau.measures.tsv",
                {9: 1, 18: 10, 27: 19, 60: 28},
                id="rondeau",
            ),
            pytest.param(
                (
                    "beethoven_op18-4iv_multimodal/op18_no4_mov4_flow/"
                    "op18_no4_mov4_flow.measures.tsv"
                ),
                {9: 1, 18: 10, 27: 19, 44: 28, 84: 78, 93: 85, 102: 95},
                id="op18",
            ),
        ],
    )
    def test_registry_repeat_map_matches_default_traversal(
        self,
        data_dir: Path,
        relative_path: str,
        expected: dict[int, int],
    ) -> None:
        """The exact registry map is the first repeat edge executed."""
        loader = Ms3Loader()
        loader.load(data_dir / relative_path)
        controller = ScoreFlowController(loader.store.measures)

        start_mc_by_qb = {
            controller._qb_at_mc(unit.mc): unit.mc for unit in controller.iter_units()
        }
        end_mc_by_qb = {
            controller._qb_at_mc_end(unit.mc): unit.mc
            for unit in controller.iter_units()
        }
        registry = controller.get_flow_control_registry()
        repeat_map = {
            end_mc_by_qb[jump.from_position]: start_mc_by_qb[jump.to_position]
            for jump in registry.jumps
            if jump.control_type == FlowControlElement.repeat_end
        }

        assert repeat_map == expected

        sequence = controller.compute_flow(FlowMode.default).to_mc_sequence()
        first_repeat_edges = {
            end_mc: sequence[sequence.index(end_mc) + 1] for end_mc in expected
        }
        assert first_repeat_edges == expected


# endregion

# region Unit Tests: AtomicSection


class TestAtomicSection:
    """Test AtomicSection dataclass."""

    def test_creation(self) -> None:
        """AtomicSection can be created with all fields."""
        # Right-open: mc_end=5 means MCs 1,2,3,4 (4 MCs total)
        sec = AtomicSection(
            id="A",
            mc_start=1,
            mc_end=5,  # Right-open: includes MCs 1-4
            to=("A", "B"),
            section_type="leap_end",
        )
        assert sec.id == "A"
        assert sec.mc_start == 1
        assert sec.mc_end == 5
        assert sec.to == ("A", "B")
        assert sec.section_type == "leap_end"

    def test_mc_range(self) -> None:
        """AtomicSection mc_range property."""
        # Right-open: mc_end=9 means MCs 5,6,7,8 (4 MCs total)
        sec = AtomicSection(id="B", mc_start=5, mc_end=9)
        assert sec.mc_range == (5, 9)
        assert sec.mc_count == 4  # 9 - 5 = 4

    def test_frozen(self) -> None:
        """AtomicSection is immutable."""
        sec = AtomicSection(id="A", mc_start=1, mc_end=5)
        with pytest.raises(AttributeError):
            sec.id = "B"  # type: ignore

    def test_invalid_range_raises(self) -> None:
        """AtomicSection raises on invalid mc range."""
        with pytest.raises(ValueError, match="mc_end.*cannot be before mc_start"):
            AtomicSection(id="A", mc_start=10, mc_end=5)

    def test_invalid_section_type_raises(self) -> None:
        """AtomicSection raises on invalid section_type."""
        with pytest.raises(ValueError, match="invalid section_type"):
            AtomicSection(id="A", mc_start=1, mc_end=5, section_type="invalid")

    def test_to_dict(self) -> None:
        """AtomicSection can be converted to dict."""
        # Right-open: mc_end=5 means MCs 1,2,3,4
        sec = AtomicSection(id="A", mc_start=1, mc_end=5, to=("B",))
        d = sec.to_dict()
        assert d["id"] == "A"
        assert d["mc_start"] == 1
        assert d["mc_end"] == 5
        assert d["to"] == ["B"]


# endregion

# region Unit Tests: PlaythroughSection


class TestPlaythroughSection:
    """Test PlaythroughSection dataclass."""

    def test_creation(self) -> None:
        """PlaythroughSection can be created with all fields."""
        seg = PlaythroughSection(
            mc_start=1,
            mc_end=8,
            atomic_section_ids=("A", "B"),
        )
        assert seg.mc_start == 1
        assert seg.mc_end == 8
        assert seg.atomic_section_ids == ("A", "B")

    def test_mc_range(self) -> None:
        """PlaythroughSection mc_range property."""
        # Right-open: mc_end=13 means MCs 5,6,7,8,9,10,11,12 (8 MCs total)
        seg = PlaythroughSection(mc_start=5, mc_end=13)
        assert seg.mc_range == (5, 13)
        assert seg.mc_count == 8  # 13 - 5 = 8

    def test_frozen(self) -> None:
        """PlaythroughSection is immutable."""
        # Right-open: mc_end=9 means MCs 1-8
        seg = PlaythroughSection(mc_start=1, mc_end=9)
        with pytest.raises(AttributeError):
            seg.mc_start = 5  # type: ignore

    def test_invalid_range_raises(self) -> None:
        """PlaythroughSection raises on invalid mc range."""
        with pytest.raises(ValueError, match="mc_end.*cannot be before mc_start"):
            PlaythroughSection(mc_start=10, mc_end=5)

    def test_to_dict(self) -> None:
        """PlaythroughSection can be converted to dict."""
        # Right-open: mc_end=9 means MCs 1-8
        seg = PlaythroughSection(mc_start=1, mc_end=9, atomic_section_ids=("A", "B"))
        d = seg.to_dict()
        assert d["mc_start"] == 1
        assert d["mc_end"] == 9
        assert d["atomic_sections"] == "A;B"

    def test_to_mc_sequence(self) -> None:
        """PlaythroughSection to_mc_sequence expands range."""
        # Right-open: mc_end=9 means MCs 5,6,7,8 (4 MCs)
        seg = PlaythroughSection(mc_start=5, mc_end=9)
        assert seg.to_mc_sequence() == [5, 6, 7, 8]


# endregion

# region Unit Tests: Flow


class TestFlow:
    """Test Flow class."""

    def test_empty_flow(self) -> None:
        """Empty flow has correct properties."""
        flow = Flow(sections=[], mode=FlowMode.default, folded_length=0)
        assert flow.unfolded_length == 0
        assert flow.total_quarterbeats == Fraction(0)
        assert flow.has_repeats is False
        assert flow.to_mc_sequence() == []

    def test_simple_flow(self) -> None:
        """Simple flow with sequential measures (section-based)."""
        # Right-open: mc_end=4 means MCs 1,2,3
        sections = [
            PlaythroughSection(mc_start=1, mc_end=4, atomic_section_ids=("A",)),
        ]
        flow = Flow.from_sections(sections, FlowMode.default, folded_length=3)

        assert flow.unfolded_length == 3
        assert flow.folded_length == 3
        assert flow.has_repeats is False
        # total_quarterbeats requires controller (returns 0 for detached flows)
        assert flow.total_quarterbeats == Fraction(0)
        assert flow.to_mc_sequence() == [1, 2, 3]

    def test_flow_with_repeats(self) -> None:
        """Flow with repeated measures has correct properties."""
        # Right-open: mc_end=3 means MCs 1,2
        sections = [
            PlaythroughSection(mc_start=1, mc_end=3, atomic_section_ids=("A",)),
            PlaythroughSection(mc_start=1, mc_end=3, atomic_section_ids=("A",)),
        ]
        flow = Flow.from_sections(sections, FlowMode.default, folded_length=2)

        assert flow.unfolded_length == 4
        assert flow.folded_length == 2
        assert flow.has_repeats is True
        assert flow.to_mc_sequence() == [1, 2, 1, 2]

    def test_to_dataframe(self) -> None:
        """Flow can be converted to DataFrame (section-based)."""
        # Right-open: mc_end=3 means MCs 1,2
        sections = [
            PlaythroughSection(mc_start=1, mc_end=3, atomic_section_ids=("A",)),
            PlaythroughSection(mc_start=3, mc_end=5, atomic_section_ids=("B",)),
        ]
        flow = Flow.from_sections(sections, FlowMode.default)

        df = flow.to_dataframe()
        assert len(df) == 2
        assert list(df.columns) == ["mc_start", "mc_end", "atomic_sections"]
        assert df["mc_start"].tolist() == [1, 3]
        assert df["mc_end"].tolist() == [3, 5]


class TestFlowSegmentBased:
    """Test Flow segment-based API."""

    def test_from_sections(self) -> None:
        """Flow can be created from segments."""
        # Right-open: mc_end=5 means MCs 1-4, mc_end=9 means MCs 5-8
        segments = [
            PlaythroughSection(mc_start=1, mc_end=5, atomic_section_ids=("A",)),
            PlaythroughSection(mc_start=5, mc_end=9, atomic_section_ids=("B",)),
        ]
        flow = Flow.from_sections(segments, FlowMode.default)

        assert len(flow.sections) == 2
        assert flow.mode == FlowMode.default
        assert flow.sections[0].mc_start == 1
        assert flow.sections[1].mc_end == 9

    def test_from_records(self) -> None:
        """Flow can be created from records."""
        # Right-open: mc_end=5 means MCs 1-4, mc_end=9 means MCs 5-8
        records = [
            {"mc_start": 1, "mc_end": 5, "atomic_sections": "A"},
            {"mc_start": 5, "mc_end": 9, "atomic_sections": "B"},
        ]
        flow = Flow.from_records(records, FlowMode.default)

        assert len(flow.sections) == 2
        assert flow.sections[0].atomic_section_ids == ("A",)
        assert flow.sections[1].atomic_section_ids == ("B",)

    def test_from_records_semicolon_separated(self) -> None:
        """Flow.from_records handles semicolon-separated atomic_sections."""
        # Right-open: mc_end=9 means MCs 1-8
        records = [
            {"mc_start": 1, "mc_end": 9, "atomic_sections": "A;B"},
        ]
        flow = Flow.from_records(records, FlowMode.default)

        assert flow.sections[0].atomic_section_ids == ("A", "B")

    def test_to_records(self) -> None:
        """Flow can be exported to records."""
        # Right-open: mc_end=5 means MCs 1-4, mc_end=9 means MCs 5-8
        segments = [
            PlaythroughSection(mc_start=1, mc_end=5, atomic_section_ids=("A",)),
            PlaythroughSection(mc_start=5, mc_end=9, atomic_section_ids=("B",)),
        ]
        flow = Flow.from_sections(segments, FlowMode.default)

        records = flow.to_records()
        assert len(records) == 2
        assert records[0]["mc_start"] == 1
        assert records[0]["atomic_sections"] == "A"
        assert records[1]["mc_end"] == 9

    def test_to_csv_rows(self) -> None:
        """Flow can be exported to CSV rows format."""
        # Right-open: mc_end=5 means MCs 1-4
        segments = [
            PlaythroughSection(mc_start=1, mc_end=5, atomic_section_ids=("A",)),
        ]
        flow = Flow.from_sections(segments, FlowMode.default)

        rows = flow.to_csv_rows("test.tsv", "test v1.0")
        assert len(rows) == 1
        assert rows[0]["flow_mode"] == "default"
        assert rows[0]["source_file"] == "test.tsv"
        assert rows[0]["software_version"] == "test v1.0"
        assert rows[0]["mc_start"] == 1
        assert rows[0]["mc_end"] == 5

    def test_is_equivalent_true(self) -> None:
        """Flow.is_equivalent returns True for matching flows."""
        # Right-open: mc_end=5 means MCs 1-4, mc_end=9 means MCs 5-8
        flow1 = Flow.from_records(
            [
                {"mc_start": 1, "mc_end": 5, "atomic_sections": "A"},
                {"mc_start": 5, "mc_end": 9, "atomic_sections": "B"},
            ],
            FlowMode.default,
        )
        flow2 = Flow.from_records(
            [
                {"mc_start": 1, "mc_end": 5, "atomic_sections": "X"},  # Different ID
                {"mc_start": 5, "mc_end": 9, "atomic_sections": "Y"},  # Different ID
            ],
            FlowMode.partitura_minimal,  # Different mode
        )
        # is_equivalent only compares MC ranges, not IDs or modes
        assert flow1.is_equivalent(flow2)

    def test_is_equivalent_false_different_length(self) -> None:
        """Flow.is_equivalent returns False for different segment counts."""
        # Right-open: mc_end=9 means MCs 1-8
        flow1 = Flow.from_records(
            [{"mc_start": 1, "mc_end": 9, "atomic_sections": "A"}],
            FlowMode.default,
        )
        flow2 = Flow.from_records(
            [
                {"mc_start": 1, "mc_end": 5, "atomic_sections": "A"},
                {"mc_start": 5, "mc_end": 9, "atomic_sections": "B"},
            ],
            FlowMode.default,
        )
        assert not flow1.is_equivalent(flow2)

    def test_is_equivalent_false_different_ranges(self) -> None:
        """Flow.is_equivalent returns False for different MC ranges."""
        # Right-open: mc_end=5 means MCs 1-4, mc_end=6 means MCs 1-5
        flow1 = Flow.from_records(
            [{"mc_start": 1, "mc_end": 5, "atomic_sections": "A"}],
            FlowMode.default,
        )
        flow2 = Flow.from_records(
            [{"mc_start": 1, "mc_end": 6, "atomic_sections": "A"}],  # Different end
            FlowMode.default,
        )
        assert not flow1.is_equivalent(flow2)

    def test_to_mc_sequence_from_sections(self) -> None:
        """Flow.to_mc_sequence works with segment-only flows."""
        # Right-open: mc_end=5 means MCs 1-4, mc_end=9 means MCs 5-8
        segments = [
            PlaythroughSection(mc_start=1, mc_end=5, atomic_section_ids=("A",)),
            PlaythroughSection(mc_start=5, mc_end=9, atomic_section_ids=("B",)),
        ]
        flow = Flow.from_sections(segments, FlowMode.default)

        assert flow.to_mc_sequence() == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_to_atomic_sequence(self) -> None:
        """Flow.to_atomic_sequence returns flattened atomic IDs."""
        segments = [
            PlaythroughSection(mc_start=1, mc_end=17, atomic_section_ids=("A", "B")),
            PlaythroughSection(
                mc_start=1, mc_end=32, atomic_section_ids=("A", "B", "C")
            ),
            PlaythroughSection(mc_start=17, mc_end=32, atomic_section_ids=("C",)),
            PlaythroughSection(mc_start=6, mc_end=17, atomic_section_ids=("B",)),
        ]
        flow = Flow.from_sections(segments, FlowMode.default)

        expected = ["A", "B", "A", "B", "C", "C", "B"]
        assert flow.to_atomic_sequence() == expected

    def test_to_atomic_sequence_empty(self) -> None:
        """Flow.to_atomic_sequence handles empty flows."""
        flow = Flow(sections=[], mode=FlowMode.default)
        assert flow.to_atomic_sequence() == []

    def test_diff_flows(self) -> None:
        """Flow.diff_flows shows differences between flows."""
        flow1 = Flow.from_sections(
            [
                PlaythroughSection(mc_start=1, mc_end=5, atomic_section_ids=("A",)),
                PlaythroughSection(mc_start=5, mc_end=9, atomic_section_ids=("B",)),
                PlaythroughSection(mc_start=1, mc_end=5, atomic_section_ids=("A",)),
            ],
            FlowMode.default,
        )
        flow2 = Flow.from_sections(
            [
                PlaythroughSection(mc_start=1, mc_end=5, atomic_section_ids=("A",)),
                PlaythroughSection(mc_start=1, mc_end=5, atomic_section_ids=("A",)),
                PlaythroughSection(mc_start=5, mc_end=9, atomic_section_ids=("B",)),
            ],
            FlowMode.default,
        )
        # diff_flows should show the swapped A and B
        diff = flow1.diff_flows(flow2)
        assert "---" in diff  # Has unified diff header
        assert "+++" in diff

    def test_unfolded_length_from_sections(self) -> None:
        """Flow.unfolded_length works with segment-only flows."""
        # Right-open: mc_end=5 means MCs 1-4 (4 MCs each)
        segments = [
            PlaythroughSection(mc_start=1, mc_end=5, atomic_section_ids=("A",)),
            PlaythroughSection(
                mc_start=1, mc_end=5, atomic_section_ids=("A",)
            ),  # Repeat
        ]
        flow = Flow.from_sections(segments, FlowMode.default)

        # 4 + 4 = 8 total MC visitations
        assert flow.unfolded_length == 8


class TestFlowCSVLoading:
    """Test Flow loading from .flow.csv files."""

    def test_from_csv(self, target_flows_dir: Path) -> None:
        """Flow.from_csv loads a specific mode from CSV."""
        csv_path = target_flows_dir / "c05n05_musete.flow.csv"

        flow = Flow.from_csv(csv_path, FlowMode.atomic)

        assert flow.mode == FlowMode.atomic
        assert len(flow.sections) == 4  # A, B, C, D
        assert flow.sections[0].mc_start == 1
        # Right-open: mc_end=6 means MCs 1-5 (5 measures)
        assert flow.sections[0].mc_end == 6

    def test_from_csv_invalid_mode_raises(self, target_flows_dir: Path) -> None:
        """Flow.from_csv raises for non-existent mode."""
        csv_path = target_flows_dir / "c05n05_musete.flow.csv"

        # MUSIC21 mode is not in the CSV, so this should raise
        with pytest.raises(ValueError, match="No entries for flow_mode"):
            Flow.from_csv(csv_path, FlowMode.music21)

    def test_load_valid_flows(self, target_flows_dir: Path) -> None:
        """load_valid_flows loads all modes from CSV."""
        csv_path = target_flows_dir / "c05n05_musete.flow.csv"

        flows = load_valid_flows(csv_path)

        assert len(flows) == 3  # ATOMIC, DEFAULT, SINGLE_PASS
        assert FlowMode.atomic in flows
        assert len(flows[FlowMode.atomic].sections) == 4


# endregion

# region Unit Tests: FlowMode


class TestFlowMode:
    """Test FlowMode enum."""

    def test_all_modes_exist(self) -> None:
        """All expected flow modes exist."""
        assert FlowMode.atomic.value == "atomic"
        assert FlowMode.default.value == "default"
        assert FlowMode.ms3.value == "ms3"
        assert FlowMode.partitura_minimal.value == "partitura_minimal"
        assert FlowMode.partitura_maximal.value == "partitura_maximal"
        assert FlowMode.music21.value == "music21"
        assert FlowMode.printed.value == "printed"
        assert FlowMode.single.value == "single"
        assert FlowMode.custom.value == "custom"


# endregion

# region Unit Tests: ScoreFlowController


class TestScoreFlowController:
    """Test ScoreFlowController computation logic."""

    def test_occurrence_to_suffix(self) -> None:
        """Test suffix generation: 1->a, 2->b, 27->aa."""
        from timetoalign.loader.score.stores import MeasureData

        # Create minimal MeasureData
        measures = MeasureData.empty()
        controller = ScoreFlowController(measures)

        assert controller._occurrence_to_suffix(1) == "a"
        assert controller._occurrence_to_suffix(2) == "b"
        assert controller._occurrence_to_suffix(26) == "z"
        assert controller._occurrence_to_suffix(27) == "aa"
        assert controller._occurrence_to_suffix(28) == "ab"

    def test_get_sections(self, rachmaninoff_measures_tsv: Path) -> None:
        """ScoreFlowController.get_sections() returns atomic sections by default."""

        loader = Ms3Loader()
        loader.load(rachmaninoff_measures_tsv)

        controller = ScoreFlowController(loader.store.measures)
        sections = controller.get_sections()  # mode=None -> atomic sections

        # Rachmaninoff has no flow control, should be 1 segment
        assert len(sections) == 1
        assert sections[0].id == "A"
        assert sections[0].mc_start == 1

    def test_get_sections_with_mode(self, rachmaninoff_measures_tsv: Path) -> None:
        """ScoreFlowController.get_sections(mode) returns playthrough sections."""

        loader = Ms3Loader()
        loader.load(rachmaninoff_measures_tsv)

        controller = ScoreFlowController(loader.store.measures)
        sections = controller.get_sections(FlowMode.default)

        # Should return PlaythroughSection objects
        assert len(sections) == 1
        assert hasattr(sections[0], "atomic_section_ids")

    def test_from_atomic_sections(self) -> None:
        """ScoreFlowController can be created from atomic sections directly."""
        # Right-open: mc_end=5 means MCs 1-4, mc_end=9 means MCs 5-8
        sections = [
            AtomicSection(id="A", mc_start=1, mc_end=5, to=("B",)),
            AtomicSection(id="B", mc_start=5, mc_end=9, to=()),
        ]
        controller = ScoreFlowController.from_atomic_sections(sections)

        assert controller.get_sections() == sections

    def test_flow_has_segments(self, rachmaninoff_measures_tsv: Path) -> None:
        """Computed flow includes segments."""

        loader = Ms3Loader()
        loader.load(rachmaninoff_measures_tsv)

        controller = ScoreFlowController(loader.store.measures)
        flow = controller.compute_flow(FlowMode.default)

        # Flow should have sections and correct unfolded length
        assert flow.unfolded_length == 374
        assert len(flow.sections) == 1
        # Controller ref should be attached
        assert flow.controller is not None


# endregion

# region Integration Tests: Example 1 - Rachmaninoff (No Flow Control)


class TestExample1Rachmaninoff:
    """Test Example 1: Rachmaninoff Concerto 2 (baseline, no repeats).

    This is the simplest case: 374 measures, all sequential, no flow control.
    The unfolded sequence should be identical to the folded sequence.

    Gold Standard:
    - Folded: 374 MCs
    - Unfolded: 374 measures (ratio 1.0)
    """

    def test_load_measures(self, rachmaninoff_measures_tsv: Path) -> None:
        """Can load Rachmaninoff measures."""

        loader = Ms3Loader()
        loader.load(rachmaninoff_measures_tsv)
        assert len(loader.store.measures) == 374

    def test_compute_flow(self, rachmaninoff_measures_tsv: Path) -> None:
        """Flow computation for sequential score."""

        loader = Ms3Loader()
        loader.load(rachmaninoff_measures_tsv)

        controller = ScoreFlowController(loader.store.measures)
        flow = controller.compute_flow(FlowMode.default)

        assert flow.folded_length == 374
        assert flow.unfolded_length == 374
        assert flow.has_repeats is False

    def test_flow_matches_unfolded_gold_standard(
        self, rachmaninoff_measures_tsv: Path, rachmaninoff_unfolded_tsv: Path
    ) -> None:
        """Flow output matches ms3 unfolded TSV exactly."""

        # Load folded and compute flow
        loader = Ms3Loader()
        loader.load(rachmaninoff_measures_tsv)
        controller = ScoreFlowController(loader.store.measures)
        flow = controller.compute_flow(FlowMode.default)

        # Load unfolded gold standard
        import pandas as pd

        gold = pd.read_csv(rachmaninoff_unfolded_tsv, sep="\t")

        # EXACT row count
        assert flow.unfolded_length == len(
            gold
        ), f"Row count mismatch: {flow.unfolded_length} != {len(gold)}"

        # EXACT MC sequence
        computed_mc_sequence = flow.to_mc_sequence()
        gold_mc_sequence = gold["mc"].tolist()
        assert computed_mc_sequence == gold_mc_sequence, "MC sequence mismatch"


# endregion

# region Integration Tests: Example 3 - Couperin Musete (D.S. al Fine)


class TestExample3CouperinMusete:
    """Test Example 3: Couperin Musete (D.S. al Fine structure).

    This tests a moderately complex flow with:
    - Anacrusis (MN=0)
    - Segno marker
    - D.S. al Fine jumps
    - Multiple couplets

    Gold Standard:
    - Folded: 58 MCs
    - Unfolded: 138 measures (ratio ~2.4)
    """

    def test_load_measures(self, musete_measures_tsv: Path) -> None:
        """Can load Musete measures."""

        loader = Ms3Loader()
        loader.load(musete_measures_tsv)
        # Exact count from TSV file (59 rows including header = 58 data rows)
        assert len(loader.store.measures) == 58

    def test_has_flow_control(self, musete_measures_tsv: Path) -> None:
        """Musete has flow control markers."""

        loader = Ms3Loader()
        loader.load(musete_measures_tsv)
        summary = loader.store.measures.get_flow_control_summary()

        # Should have repeats based on 'next' field branching
        assert summary["total_measures"] == 58

    def test_compute_flow_unfolded_count(self, musete_measures_tsv: Path) -> None:
        """Flow computation produces correct unfolded count."""

        loader = Ms3Loader()
        loader.load(musete_measures_tsv)

        controller = ScoreFlowController(loader.store.measures)
        flow = controller.compute_flow(FlowMode.default)

        assert flow.folded_length == 58
        # The unfolded count is 138 (from gold standard)
        assert flow.unfolded_length == 138

    def test_flow_matches_unfolded_gold_standard(
        self, musete_measures_tsv: Path, musete_unfolded_tsv: Path
    ) -> None:
        """Flow output matches ms3 unfolded TSV exactly."""

        # Load folded and compute flow
        loader = Ms3Loader()
        loader.load(musete_measures_tsv)
        controller = ScoreFlowController(loader.store.measures)
        flow = controller.compute_flow(FlowMode.default)

        # Load unfolded gold standard
        import pandas as pd

        gold = pd.read_csv(musete_unfolded_tsv, sep="\t")

        # EXACT row count
        assert flow.unfolded_length == len(
            gold
        ), f"Row count mismatch: {flow.unfolded_length} != {len(gold)}"


# endregion

# region Integration Tests: FlowMap


class TestFlowMap:
    """Test FlowMap coordinate transformation."""

    def test_create_flow_map(self, rachmaninoff_measures_tsv: Path) -> None:
        """Can create FlowMap from ScoreFlowController."""

        loader = Ms3Loader()
        loader.load(rachmaninoff_measures_tsv)

        controller = ScoreFlowController(loader.store.measures)
        flow_map = controller.create_flow_map()

        assert isinstance(flow_map, FlowMap)
        assert flow_map.flow.folded_length == 374


# endregion

# region Printed Mode Tests


class TestPrintedMode:
    """Test PRINTED flow mode (no unfolding)."""

    def test_printed_flow_equals_folded(self, rachmaninoff_measures_tsv: Path) -> None:
        """PRINTED mode returns same count as folded."""

        loader = Ms3Loader()
        loader.load(rachmaninoff_measures_tsv)

        controller = ScoreFlowController(loader.store.measures)
        flow = controller.compute_flow(FlowMode.printed)

        assert flow.mode == FlowMode.printed
        assert flow.folded_length == 374
        assert flow.unfolded_length == 374
        assert flow.has_repeats is False


# endregion
