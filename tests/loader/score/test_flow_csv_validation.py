"""Flow CSV Ground Truth Validation Tests.

This module validates .flow.csv structural properties and specific loader
behaviours that are NOT covered by the main score parsing matrix
(test_score_parsing_matrix.py).

Unique tests in this module:
- RIGHT-OPEN mc_end contiguity checks across all specimens
- Exact atomic segment boundary validation
- Partitura live segment boundary verification against CSV

Tests that overlap with test_score_parsing_matrix.py have been removed.
See that file for: CSV existence/mode checks, folded counts and loader
measure counts; and test_flow_gold.py for the flows the sources play.

ZERO TOLERANCE VALIDATION POLICY:
- EXACT counts required (no tolerances)
- mc_end is RIGHT-OPEN (aligns with TimeToAlign's TimeInterval definition)
- Every mismatch must be investigated
"""

from __future__ import annotations

import pytest

from .conftest import (
    TARGET_FLOWS_DIR,
    FlowEntry,
    find_source_file,
    parse_flow_csv,
)

# ============================================================================
# Test: Validate RIGHT-OPEN mc_end convention
# ============================================================================


class TestFlowCSVRightOpenConvention:
    """Validate that .flow.csv mc_end values follow RIGHT-OPEN convention."""

    @pytest.mark.parametrize(
        "csv_name",
        [
            "c05n05_musete.flow.csv",
            "c11n08_Rondeau.flow.csv",
            "out_of_the_flow_experience-polyrhythm_only.flow.csv",
            "out_of_the_flow_experience-flow_only.flow.csv",
            "Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff.flow.csv",
            "op18_no4_mov4_flow.flow.csv",
            "WoO71.flow.csv",
        ],
    )
    def test_mc_end_is_right_open(self, csv_name):
        """Verify mc_end values follow RIGHT-OPEN convention (contiguous segments)."""
        csv_path = TARGET_FLOWS_DIR / csv_name
        if not csv_path.exists():
            pytest.skip(f"Flow CSV not found: {csv_path}")

        entries = parse_flow_csv(csv_path)

        # Group entries by flow_mode
        by_mode: dict[str, list[FlowEntry]] = {}
        for entry in entries:
            if entry.flow_mode not in by_mode:
                by_mode[entry.flow_mode] = []
            by_mode[entry.flow_mode].append(entry)

        # For atomic, check that segments are contiguous (right-open)
        if "atomic" in by_mode:
            atomic_entries = sorted(by_mode["atomic"], key=lambda e: int(e.mc_start))
            for i in range(len(atomic_entries) - 1):
                current = atomic_entries[i]
                next_entry = atomic_entries[i + 1]
                # With RIGHT-OPEN mc_end, next segment should start at current mc_end
                assert int(current.mc_end) == int(next_entry.mc_start), (
                    f"Non-contiguous segments in {csv_name} atomic: "
                    f"segment ending at MC {current.mc_end} should equal "
                    f"next segment starting at MC {next_entry.mc_start}"
                )


# ============================================================================
# Test: Validate atomic segments
# ============================================================================


class TestAtomicSegmentValidation:
    """Validate atomic entries against expected segment boundaries."""

    @pytest.mark.parametrize(
        "csv_name,expected_segments",
        [
            # Right-open convention: mc_end is exclusive
            # A: MCs 1-5 (5 MCs), B: MCs 6-16 (11 MCs), C: MCs 17-31 (15 MCs), D: MCs 32-58 (27 MCs)
            (
                "c05n05_musete.flow.csv",
                {"A": (1, 6), "B": (6, 17), "C": (17, 32), "D": (32, 59)},
            ),
            # A: MCs 1-14 (14 MCs)
            ("out_of_the_flow_experience-polyrhythm_only.flow.csv", {"A": (1, 15)}),
            # A: MCs 1-374 (374 MCs)
            (
                "Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff.flow.csv",
                {"A": (1, 375)},
            ),
        ],
    )
    def test_atomic_segments(self, csv_name, expected_segments):
        """Verify atomic entries match expected segment boundaries (right-open)."""
        csv_path = TARGET_FLOWS_DIR / csv_name
        if not csv_path.exists():
            pytest.skip(f"Flow CSV not found: {csv_path}")

        entries = parse_flow_csv(csv_path)
        atomic_entries = [e for e in entries if e.flow_mode == "atomic"]

        if not atomic_entries:
            pytest.skip(f"No atomic entries in {csv_name}")

        # Build actual segments from CSV
        actual_segments = {}
        for entry in atomic_entries:
            seg_id = entry.atomic_segments
            actual_segments[seg_id] = (entry.mc_start, entry.mc_end)

        # Compare
        assert actual_segments == expected_segments, (
            f"Segment mismatch in {csv_name}:\n"
            f"  Expected: {expected_segments}\n"
            f"  Actual: {actual_segments}"
        )


# ============================================================================
# Test: Validate segment MC ranges against actual partitura output
# ============================================================================


@pytest.mark.parametrize(
    "csv_name",
    [
        "c05n05_musete.flow.csv",
        "out_of_the_flow_experience-polyrhythm_only.flow.csv",
        pytest.param(
            "Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff.flow.csv",
            marks=pytest.mark.slow,
        ),
    ],
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestPartituraSegmentValidation:
    """Validate partitura_minimal segments against live partitura output.

    Note: This test validates that partitura's segments match our atomic segments.
    If partitura_minimal entries exist in the CSV, it means partitura diverges from atomic.
    """

    def test_partitura_segment_boundaries(self, csv_name):
        """Verify CSV segment boundaries match actual partitura.add_segments() output."""
        try:
            import partitura as pt
        except ImportError:
            pytest.skip("partitura not installed")

        csv_path = TARGET_FLOWS_DIR / csv_name
        if not csv_path.exists():
            pytest.skip(f"Flow CSV not found: {csv_path}")

        entries = parse_flow_csv(csv_path)
        # Use partitura_minimal if present (divergent case), otherwise use atomic
        minimal_entries = [e for e in entries if e.flow_mode == "partitura_minimal"]
        if not minimal_entries:
            minimal_entries = [e for e in entries if e.flow_mode == "atomic"]

        if not minimal_entries:
            pytest.skip(f"No atomic or partitura_minimal entries in {csv_name}")

        # Find MusicXML file for specimen (partitura requires XML, not TSV)
        specimen_name = csv_name.replace(".flow.csv", "")

        # Build MusicXML filename from specimen name
        # e.g., "c05n05_musete.flow.csv" -> "c05n05_musete.musicxml"
        musicxml_filename = specimen_name + ".musicxml"
        source_path = find_source_file(musicxml_filename, specimen_name)

        # Also try .xml extension
        if source_path is None or not source_path.exists():
            xml_filename = specimen_name + ".xml"
            source_path = find_source_file(xml_filename, specimen_name)

        if source_path is None or not source_path.exists():
            pytest.skip(f"MusicXML not found for {specimen_name}")

        # Verify it's actually an XML file (not TSV)
        if source_path.suffix.lower() not in (".musicxml", ".xml"):
            pytest.skip(f"Source file is not MusicXML: {source_path}")

        # Load with partitura
        score = pt.load_musicxml(source_path)
        part = score[0]

        # Get measures
        measures = list(part.iter_all(pt.score.Measure))

        # Get segments
        pt.score.add_segments(part)
        segments = pt.score.get_segments(part)

        # Build expected segments from CSV
        csv_segments = {}
        for entry in minimal_entries:
            csv_segments[entry.atomic_segments] = (entry.mc_start, entry.mc_end)

        # Build actual segments from partitura
        actual_segments = {}
        for seg_id, seg in segments.items():
            start_t = seg.start.t if hasattr(seg.start, "t") else seg.start
            end_t = seg.end.t if hasattr(seg.end, "t") else seg.end

            # Map t values to MC numbers
            start_mc = None
            end_mc = None
            for i, m in enumerate(measures, start=1):
                # Start MC: first measure where segment start falls within
                if start_mc is None and m.start.t <= start_t < m.end.t:
                    start_mc = i
                # End MC: last measure where segment end falls at or before measure end
                if m.start.t < end_t <= m.end.t:
                    end_mc = i

            # Convert partitura's inclusive end to RIGHT-OPEN convention (mc_end exclusive)
            actual_segments[seg_id] = (start_mc, end_mc + 1 if end_mc else None)

        # Compare
        assert csv_segments == actual_segments, (
            f"Segment boundary mismatch in {csv_name}:\n"
            f"  CSV segments: {csv_segments}\n"
            f"  Partitura segments: {actual_segments}"
        )
