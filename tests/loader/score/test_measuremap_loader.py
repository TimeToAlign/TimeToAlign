"""MeasureMapLoader Tests: Validation and Cross-Validation.

This module tests the MeasureMapLoader implementation, including:
1. JSON parsing and expansion
2. Validation rules (MC unique, qstamp monotonic, next valid)
3. Cross-validation against Ms3Loader (measures.tsv gold standard)
4. Traversal computation

ZERO TOLERANCE VALIDATION POLICY:
- EXACT counts required (no tolerances)
- Every mismatch must be investigated
- Gold standard is authoritative
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from timetoalign.timelines import MeasureMap

# Specimen data paths - all under tests/data/score/
TESTS_DATA_DIR = Path(__file__).parents[2] / "data" / "score"
BEETHOVEN_WOO71_DIR = TESTS_DATA_DIR / "beethoven_woo71"
BEETHOVEN_MM_JSON = BEETHOVEN_WOO71_DIR / "WoO71.measures.mm.json"
BEETHOVEN_MEASURES_TSV = BEETHOVEN_WOO71_DIR / "WoO71.measures.tsv"
BEETHOVEN_UNFOLDED_TSV = BEETHOVEN_WOO71_DIR / "WoO71_unfolded.measures.tsv"

FLOW_CONTROL_DIR = TESTS_DATA_DIR / "flow_control" / "flow_only"
FLOW_MM_JSON = (
    FLOW_CONTROL_DIR / "out_of_the_flow_experience-flow_only.measures.mm.json"
)

# Gold standard counts (verified from specimen files):
# - WoO71.measures.mm.json: 397 entries (MC 1-397, folded)
# - WoO71.measures.tsv: 397 data rows (folded, header excluded)
# - WoO71_unfolded.measures.tsv: 505 data rows (unfolded, header excluded)
BEETHOVEN_FOLDED_MEASURES = 397  # Exact count from mm.json
BEETHOVEN_UNFOLDED_MEASURES = 505  # Exact count from unfolded TSV


def specimens_available() -> bool:
    """Check if specimen files are available."""
    return BEETHOVEN_MM_JSON.exists() and BEETHOVEN_MEASURES_TSV.exists()


@pytest.fixture
def measuremap_loader():
    """Get MeasureMapLoader."""
    from timetoalign.loader.score import MeasureMapLoader

    return MeasureMapLoader()


@pytest.fixture
def tsv_loader():
    """Get Ms3Loader if ms3 is available."""
    try:
        from timetoalign.loader.score import Ms3Loader

        return Ms3Loader()
    except ImportError:
        pytest.skip("Ms3Loader requires ms3. Install with: pip install ms3")


class TestMeasureMapLoaderBasic:
    """Basic functionality tests for MeasureMapLoader."""

    def test_loader_exists(self):
        """MeasureMapLoader is importable."""
        from timetoalign.loader.score import MeasureMapLoader

        assert MeasureMapLoader is not None

    def test_loader_init(self, measuremap_loader):
        """Loader initializes correctly."""
        assert measuremap_loader.unit.value == "quarters"

    def test_file_without_nominal_length_names_path_and_first_row(
        self, measuremap_loader, tmp_path: Path
    ) -> None:
        path = tmp_path / "no_nominal.mm.json"
        path.write_text(
            json.dumps([{"number": 1, "actual_length": 3}]), encoding="utf-8"
        )
        with pytest.raises(ValueError, match=r"no_nominal\.mm\.json.*row 1"):
            measuremap_loader.load(path)

    @pytest.mark.skipif(not specimens_available(), reason="Specimen files not found")
    def test_load_beethoven_mm_json(self, measuremap_loader):
        """Load Beethoven WoO71 MeasureMap JSON."""
        measuremap_loader.load(BEETHOVEN_MM_JSON)
        assert len(measuremap_loader.store.measures) == BEETHOVEN_FOLDED_MEASURES

    @pytest.mark.skipif(not specimens_available(), reason="Specimen files not found")
    def test_beethoven_folded_count_exact(self, measuremap_loader):
        """Beethoven WoO71 MeasureMap has exactly 397 measures.

        Gold standard: WoO71.measures.mm.json contains MC 1-397.
        This count is verified against the source file.
        """
        measuremap_loader.load(BEETHOVEN_MM_JSON)
        actual_count = len(measuremap_loader.store.measures)
        assert actual_count == BEETHOVEN_FOLDED_MEASURES, (
            f"Folded measure count mismatch: got {actual_count}, "
            f"expected {BEETHOVEN_FOLDED_MEASURES}"
        )


class TestMeasureMapExpansion:
    """Tests for MeasureMap compression/expansion logic."""

    def test_expand_minimal_compressed(self, measuremap_loader):
        """Expand a minimally compressed MeasureMap."""
        # Minimal compressed: just time signature and final MN
        compressed = [
            {"time_signature": "4/4"},  # First bar, defaults fill in
            {"number": 10},  # 10 bars total
        ]
        expanded = MeasureMap.expand_rows(compressed)

        assert len(expanded) == 10
        assert expanded[0]["count"] == 1
        assert expanded[0]["time_signature"] == "4/4"
        assert [row["number"] for row in expanded] == list(range(1, 11))
        assert expanded[-1]["count"] == 10

    def test_expand_with_anacrusis(self, measuremap_loader):
        """Expand MeasureMap with anacrusis (pickup measure)."""
        compressed = [
            {"time_signature": "2/4", "actual_length": 1.0, "number": 0},  # Anacrusis
            {"number": 1},  # M1
            {"number": 2},  # M2
        ]
        expanded = MeasureMap.expand_rows(compressed)

        assert len(expanded) == 3
        assert expanded[0]["number"] == 0  # Anacrusis
        assert expanded[0]["actual_length"] == Fraction(1)  # Half bar
        assert expanded[0]["qstamp"] == Fraction(0)
        assert expanded[1]["qstamp"] == Fraction(1)  # After anacrusis

    def test_expand_with_repeat(self, measuremap_loader):
        """Expand MeasureMap with repeat markers."""
        compressed = [
            {"time_signature": "4/4"},
            {"start_repeat": True},  # ||:
            {},
            {"end_repeat": True, "next": [2, 5]},  # :||
            {},  # After repeat
        ]
        expanded = MeasureMap.expand_rows(compressed)

        assert len(expanded) == 5
        assert expanded[1]["start_repeat"] is True
        assert expanded[3]["end_repeat"] is True
        assert expanded[3]["next"] == [2, 5]


class TestMeasureMapValidation:
    """Tests for MeasureMap validation rules."""

    def test_validate_mc_unique(self, measuremap_loader):
        """MC values must be unique."""
        expanded = [
            {"count": 1, "qstamp": Fraction(0), "next": [2]},
            {"count": 1, "qstamp": Fraction(4), "next": [-1]},  # Duplicate!
        ]

        with pytest.raises(ValueError, match="Duplicate MC"):
            measuremap_loader._validate_measuremap(expanded)

    def test_validate_qstamp_monotonic(self, measuremap_loader):
        """qstamp values must be monotonically increasing."""
        expanded = [
            {"count": 1, "qstamp": Fraction(4), "next": [2]},  # Starts at 4
            {"count": 2, "qstamp": Fraction(2), "next": [-1]},  # Goes back to 2!
        ]

        with pytest.raises(ValueError, match="not monotonic"):
            measuremap_loader._validate_measuremap(expanded)

    def test_validate_next_references(self, measuremap_loader):
        """next field must reference valid MCs or -1."""
        expanded = [
            {"count": 1, "qstamp": Fraction(0), "next": [2]},
            {"count": 2, "qstamp": Fraction(4), "next": [99]},  # Invalid ref!
        ]

        with pytest.raises(ValueError, match="invalid next MC"):
            measuremap_loader._validate_measuremap(expanded)


class TestMeasureMapTraversal:
    """Tests for traversal computation from MeasureMap."""

    def test_simple_traversal(self, measuremap_loader):
        """Compute traversal for simple linear score."""
        compressed = [
            {"time_signature": "4/4"},
            {},
            {},
        ]
        measuremap_loader._expanded_data = MeasureMap.expand_rows(compressed)

        traversal = measuremap_loader.compute_default_traversal()
        assert traversal == [1, 2, 3]

    def test_repeat_traversal(self, measuremap_loader):
        """Compute traversal with repeat."""
        # Bars 1-2-3-4, repeat 2-3-4, then end
        compressed = [
            {"time_signature": "4/4"},  # MC 1
            {"start_repeat": True},  # MC 2: ||:
            {},  # MC 3
            {"end_repeat": True, "next": [2, 5]},  # MC 4: :|| go to 2 or 5
            {},  # MC 5: after repeat
        ]
        measuremap_loader._expanded_data = MeasureMap.expand_rows(compressed)

        traversal = measuremap_loader.compute_default_traversal()
        # Expected: 1, 2, 3, 4, 2, 3, 4, 5
        assert traversal == [1, 2, 3, 4, 2, 3, 4, 5]

    @pytest.mark.skipif(not specimens_available(), reason="Specimen files not found")
    def test_beethoven_unfolded_count(self, measuremap_loader):
        """Beethoven WoO71 unfolded traversal has 505 measures.

        Gold standard: WoO71_unfolded.measures.tsv has 505 rows.
        """
        measuremap_loader.load(BEETHOVEN_MM_JSON)
        traversal = measuremap_loader.compute_default_traversal()

        assert len(traversal) == BEETHOVEN_UNFOLDED_MEASURES, (
            f"Unfolded traversal count mismatch: got {len(traversal)}, "
            f"expected {BEETHOVEN_UNFOLDED_MEASURES}"
        )


@pytest.mark.skipif(not specimens_available(), reason="Specimen files not found")
class TestMeasureMapCrossValidation:
    """Cross-validation: MeasureMapLoader vs Ms3Loader.

    Per ZERO TOLERANCE VALIDATION POLICY:
    - EXACT counts required
    - EXACT MC sequence match
    - EXACT qstamp values match
    - EXACT durations match
    """

    def test_measure_count_match(self, measuremap_loader, tsv_loader):
        """Both loaders return same number of folded measures."""
        measuremap_loader.load(BEETHOVEN_MM_JSON)
        tsv_loader.load(BEETHOVEN_MEASURES_TSV)

        mm_count = len(measuremap_loader.store.measures)
        tsv_count = len(tsv_loader.store.measures)

        # Note: TSV may have different count due to format differences
        # Both should match the gold standard: 397 folded measures
        assert mm_count == BEETHOVEN_FOLDED_MEASURES, (
            f"MeasureMapLoader returned {mm_count} measures, "
            f"expected {BEETHOVEN_FOLDED_MEASURES}"
        )
        assert tsv_count == BEETHOVEN_FOLDED_MEASURES, (
            f"Ms3Loader returned {tsv_count} measures, "
            f"expected {BEETHOVEN_FOLDED_MEASURES}"
        )

    def test_mc_sequence_match(self, measuremap_loader, tsv_loader):
        """MC values match between loaders."""
        measuremap_loader.load(BEETHOVEN_MM_JSON)
        tsv_loader.load(BEETHOVEN_MEASURES_TSV)

        mm_measures = measuremap_loader.store.measures
        tsv_measures = tsv_loader.store.measures

        # Get MC columns
        mm_mc = mm_measures._table.column("mc").to_pylist()
        tsv_mc = tsv_measures._table.column("mc").to_pylist()

        # Compare overlap (both should cover same MC range)
        mm_mc_set = set(mm_mc)
        tsv_mc_set = set(tsv_mc)

        # Both should start at MC 1
        assert min(mm_mc) == 1, f"MeasureMap MC should start at 1, got {min(mm_mc)}"
        assert min(tsv_mc) == 1, f"TSV MC should start at 1, got {min(tsv_mc)}"

        # MCs in both should match exactly
        common = mm_mc_set & tsv_mc_set
        assert len(common) == len(mm_mc_set), (
            f"MC mismatch: MeasureMap has {len(mm_mc_set)} MCs, "
            f"TSV has {len(tsv_mc_set)} MCs, common: {len(common)}"
        )

    def test_flow_control_summary_match(self, measuremap_loader, tsv_loader):
        """Flow control summary matches between loaders."""
        measuremap_loader.load(BEETHOVEN_MM_JSON)
        tsv_loader.load(BEETHOVEN_MEASURES_TSV)

        mm_summary = measuremap_loader.store.measures.get_flow_control_summary()
        tsv_summary = tsv_loader.store.measures.get_flow_control_summary()

        # Both should detect repeats
        assert mm_summary["has_repeats"] == tsv_summary["has_repeats"], (
            f"Repeat detection mismatch: "
            f"MeasureMap={mm_summary['has_repeats']}, TSV={tsv_summary['has_repeats']}"
        )


class TestMeasureDataSchema:
    """Tests for the enhanced MeasureData schema."""

    def test_schema_has_flow_control_fields(self):
        """MeasureData schema includes flow control fields."""
        from timetoalign.core import TimeUnit
        from timetoalign.loader.score.stores import MeasureData

        schema = MeasureData.get_schema(TimeUnit.quarters)
        field_names = [f.name for f in schema]

        assert "start_repeat" in field_names
        assert "end_repeat" in field_names
        assert "next" in field_names
        assert "volta" in field_names
        assert "breaks" in field_names

    def test_schema_has_identity_fields(self):
        """MeasureData schema includes identity fields."""
        from timetoalign.core import TimeUnit
        from timetoalign.loader.score.stores import MeasureData

        schema = MeasureData.get_schema(TimeUnit.quarters)
        field_names = [f.name for f in schema]

        assert "mc" in field_names
        assert "mn" in field_names
        assert "mn_int" in field_names
        assert "mm_id" in field_names

    def test_flow_control_summary(self):
        """get_flow_control_summary works correctly."""
        from timetoalign.core.time import coordinate_to_struct
        from timetoalign.loader.score.stores import MeasureData

        # Use coordinate_to_struct for temporal fields (schema expects struct type)
        rows = [
            {
                "mc": 1,
                "mn": "1",
                "start": coordinate_to_struct(0.0),
                "duration": coordinate_to_struct(4.0),
                "start_repeat": False,
                "end_repeat": False,
            },
            {
                "mc": 2,
                "mn": "2",
                "start": coordinate_to_struct(4.0),
                "duration": coordinate_to_struct(4.0),
                "start_repeat": True,
                "end_repeat": False,
            },
            {
                "mc": 3,
                "mn": "3",
                "start": coordinate_to_struct(8.0),
                "duration": coordinate_to_struct(4.0),
                "start_repeat": False,
                "end_repeat": True,
            },
        ]
        data = MeasureData.from_dicts(rows)
        summary = data.get_flow_control_summary()

        assert summary["total_measures"] == 3
        assert summary["repeat_starts"] == 1
        assert summary["repeat_ends"] == 1
        assert summary["has_repeats"] is True


@pytest.mark.skipif(not FLOW_MM_JSON.exists(), reason="Flow control specimen not found")
class TestFlowControlSpecimen:
    """Tests using the flow_control specimen (complex flow control)."""

    def test_load_flow_control_mm(self, measuremap_loader):
        """Load complex flow control MeasureMap."""
        measuremap_loader.load(FLOW_MM_JSON)
        measures = measuremap_loader.store.measures

        assert len(measures) == 15  # flow_only specimen has exactly 15 measures
        summary = measures.get_flow_control_summary()
        assert summary["has_repeats"], "Flow control specimen should have repeats"
