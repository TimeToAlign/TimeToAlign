"""Shared fixtures and utilities for score loader tests.

This module consolidates:
- FlowEntry NamedTuple and parse_flow_csv() (canonical parser with header validation)
- SpecimenConfig dataclass and SPECIMENS dict (test specimen configuration)
- Common path constants (TARGET_FLOWS_DIR, SCORE_DATA_DIR), resolved via ensure_data
- Shared helper functions (find_source_file, get_loader_for_source_file)
- Flows in the form of a .flow.csv file (gold_flows, played_ranges,
  atomic_ranges) and the stress score's three repairs (repaired_stress_score)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

from timetoalign.core import Coordinate
from timetoalign.testdata import ensure_data
from timetoalign.timelines.flow import MeasureRange, measure_ranges, read_flow_csv
from timetoalign.timelines.flow.engine import FlowResult, atomic_sections, play
from timetoalign.timelines.flowcontrol import (
    FlowControlRegistry,
    Jump,
    JumpDestination,
    JumpTarget,
)

ensure_data("score", "target_flows", "vienna_1x22", "midi", "supra")
SCORE_DIR = ensure_data("score")
FLOW_CONTROL_DIR = SCORE_DIR / "flow_control"
VIENNA_DIR = ensure_data("vienna_1x22")
PARANGONAR_DIR = ensure_data("parangonar")

# region Path Constants

TARGET_FLOWS_DIR = ensure_data("target_flows")
SCORE_DATA_DIR = SCORE_DIR

# endregion


# region FlowEntry Parsing


class FlowEntry(NamedTuple):
    """A single entry from a .flow.csv file."""

    flow_mode: str
    source_file: str
    software_version: str
    mc_start: int | str  # int or "ERROR"
    mc_end: int | str  # int or error code
    atomic_segments: str


def parse_flow_csv(csv_path: Path) -> list[FlowEntry]:
    """Parse a .flow.csv file into FlowEntry objects.

    Args:
        csv_path: Path to the .flow.csv file

    Returns:
        List of FlowEntry objects (skips comment lines and ERROR entries)

    Note:
        The parser requires the first 6 columns in order:
        flow_mode, source_file, software_version, mc_start, mc_end, atomic_segments

        Optional extra columns (e.g., "comment") are allowed and ignored.
    """
    entries = []
    required_columns = [
        "flow_mode",
        "source_file",
        "software_version",
        "mc_start",
        "mc_end",
        "atomic_segments",
    ]
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        # Validate that required columns are present in correct order
        if len(header) < 6:
            raise ValueError(
                f"Expected at least 6 columns, got {len(header)}: {header}"
            )
        assert (
            header[:6] == required_columns
        ), f"First 6 columns must be {required_columns}, got {header[:6]}"

        for row in reader:
            # Skip empty rows and comment lines
            if not row or row[0].startswith("#"):
                continue

            # Ensure row has at least 6 columns
            if len(row) < 6:
                continue

            # Unpack required columns (ignore any extra columns like "comment")
            flow_mode, source_file, software_version, mc_start, mc_end, segments = row[
                :6
            ]

            # Skip ERROR entries
            if mc_start == "ERROR":
                continue

            entries.append(
                FlowEntry(
                    flow_mode=flow_mode,
                    source_file=source_file,
                    software_version=software_version,
                    mc_start=int(mc_start),
                    mc_end=int(mc_end),
                    atomic_segments=segments,
                )
            )

    return entries


def get_flow_modes(csv_path: Path) -> set[str]:
    """Get all unique flow modes in a .flow.csv file."""
    entries = parse_flow_csv(csv_path)
    return {e.flow_mode for e in entries}


# endregion


# region Specimen Configuration


@dataclass
class SpecimenConfig:
    """Configuration for a test specimen."""

    name: str
    flow_csv: str
    tsv_dir: Path
    tsv_file: str
    musicxml_file: str | None
    mei_file: str | None
    mm_json_file: str | None
    folded_measures: int
    unfolded_measures: int
    has_flow_control: bool
    expected_atomic_sections: int
    notes: str = ""


# Specimen configurations - verified from test data
SPECIMENS = {
    "rachmaninoff": SpecimenConfig(
        name="Rachmaninoff Piano Concerto No. 2",
        flow_csv="Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff.flow.csv",
        tsv_dir=SCORE_DATA_DIR / "rachmaninoff_concerto2" / "score",
        tsv_file="Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff.measures.tsv",
        musicxml_file="Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff.musicxml",
        mei_file="Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff.mei",
        mm_json_file="Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff.measures.mm.json",
        folded_measures=374,
        unfolded_measures=374,
        has_flow_control=False,
        expected_atomic_sections=1,
        notes="No flow control - baseline specimen",
    ),
    "polyrhythm_only": SpecimenConfig(
        name="Out of the Flow Experience - Polyrhythm Only",
        flow_csv="out_of_the_flow_experience-polyrhythm_only.flow.csv",
        tsv_dir=SCORE_DATA_DIR / "flow_control" / "polyrythm_only",
        tsv_file="out_of_the_flow_experience-polyrhythm_only.measures.tsv",
        musicxml_file=(
            str(SCORE_DATA_DIR / "flow_control")
            + "/out_of_the_flow_experience-polyrhythm_only.musicxml"
        ),
        mei_file=(
            str(SCORE_DATA_DIR / "flow_control")
            + "/out_of_the_flow_experience-polyrhythm_only.mei"
        ),
        mm_json_file="out_of_the_flow_experience-polyrhythm_only.measures.mm.json",
        folded_measures=14,
        unfolded_measures=14,
        has_flow_control=False,
        expected_atomic_sections=1,
        notes="Line breaks only, no repeats - simple baseline",
    ),
    "c05n05_musete": SpecimenConfig(
        name="Couperin Concert 5, No. 5 - Musete",
        flow_csv="c05n05_musete.flow.csv",
        tsv_dir=SCORE_DATA_DIR / "couperin_concerts",
        tsv_file="c05n05_musete.measures.tsv",
        musicxml_file="c05n05_musete.musicxml",
        mei_file="c05n05_musete.mei",
        mm_json_file="c05n05_musete.measures.mm.json",
        folded_measures=58,
        unfolded_measures=138,
        has_flow_control=True,
        expected_atomic_sections=4,
        notes="D.S. al Fine - music21 DIVERGES (ignores D.S./Fine, produces 116 MCs)",
    ),
    "c11n08_Rondeau": SpecimenConfig(
        name="Couperin Concert 11, No. 8 - Rondeau",
        flow_csv="c11n08_Rondeau.flow.csv",
        tsv_dir=SCORE_DATA_DIR / "couperin_concerts",
        tsv_file="c11n08_Rondeau.measures.tsv",
        musicxml_file="c11n08_Rondeau.musicxml",
        mei_file="c11n08_Rondeau.mei",
        mm_json_file="c11n08_Rondeau.measures.mm.json",
        folded_measures=60,
        unfolded_measures=138,
        has_flow_control=True,
        expected_atomic_sections=4,
        notes="Rondeau form with D.S. - music21 DIVERGES",
    ),
    "op18_no4_mov4_flow": SpecimenConfig(
        name="Beethoven Op. 18 No. 4, Mov. 4 - Flow",
        flow_csv="op18_no4_mov4_flow.flow.csv",
        tsv_dir=SCORE_DATA_DIR / "beethoven_op18-4iv_multimodal" / "op18_no4_mov4_flow",
        tsv_file="op18_no4_mov4_flow.measures.tsv",
        musicxml_file="op18_no4_mov4_flow.musicxml",
        mei_file="op18_no4_mov4_flow.mei",
        mm_json_file="op18_no4_mov4_flow.measures.mm.json",
        folded_measures=226,  # 227 lines - 1 header = 226 data rows
        unfolded_measures=291,
        has_flow_control=True,
        expected_atomic_sections=13,
        notes="Repeats + Voltas - ALL PARSERS AGREE",
    ),
    "flow_only": SpecimenConfig(
        name="Out of the Flow Experience - Flow Only",
        flow_csv="out_of_the_flow_experience-flow_only.flow.csv",
        tsv_dir=SCORE_DATA_DIR / "flow_control" / "flow_only",
        tsv_file="out_of_the_flow_experience-flow_only.measures.tsv",
        musicxml_file="out_of_the_flow_experience-flow_only.musicxml",
        mei_file=(
            str(SCORE_DATA_DIR / "flow_control")
            + "/out_of_the_flow_experience-flow_only.mei"
        ),
        mm_json_file="out_of_the_flow_experience-flow_only.measures.mm.json",
        folded_measures=15,
        unfolded_measures=30,  # ms3 count (canonical=31 but mm.json follows ms3)
        has_flow_control=True,
        expected_atomic_sections=13,
        notes="D.S./D.C. + Voltas - ms3 DIVERGES from canonical, music21 FAILS",
    ),
    "WoO71": SpecimenConfig(
        name="Beethoven WoO71",
        flow_csv="WoO71.flow.csv",
        tsv_dir=SCORE_DATA_DIR / "beethoven_woo71",
        tsv_file="WoO71.measures.tsv",
        musicxml_file="WoO71.musicxml",
        mei_file="WoO71.mei",
        mm_json_file="WoO71.measures.mm.json",
        folded_measures=397,
        unfolded_measures=505,
        has_flow_control=True,
        expected_atomic_sections=26,
        notes="Complex split bars, needs audit",
    ),
}


# endregion


# region Helper Functions


def get_specimen_path(spec: SpecimenConfig, file_type: str) -> Path | None:
    """Get path to a specimen file of a given type.

    Args:
        spec: SpecimenConfig for the specimen
        file_type: One of 'tsv', 'musicxml', 'mei', 'mm_json'

    Returns:
        Path to the file, or None if not configured
    """
    if file_type == "tsv":
        return spec.tsv_dir / spec.tsv_file
    if file_type == "musicxml":
        if spec.musicxml_file is None:
            return None
        if "/" in spec.musicxml_file:
            return Path(spec.musicxml_file)
        return spec.tsv_dir / spec.musicxml_file
    if file_type == "mei":
        if spec.mei_file is None:
            return None
        if "/" in spec.mei_file:
            return Path(spec.mei_file)
        return spec.tsv_dir / spec.mei_file
    if file_type == "mm_json":
        if spec.mm_json_file is None:
            return None
        return spec.tsv_dir / spec.mm_json_file
    return None


# Maximum file size for MusicXML/MEI files to avoid test timeouts.
# PartituraLoader/Music21Loader can take 90+ seconds on large (2MB+) files.
MAX_MUSICXML_SIZE_BYTES = 500_000  # 500KB


def musicxml_too_large(path: Path | None) -> bool:
    """Check if MusicXML/MEI file is too large for reasonable test time.

    PartituraLoader/Music21Loader processing time scales poorly with file size.
    A 2MB MusicXML can take 90+ seconds to process, causing test timeouts.
    """
    if path is None or not path.exists():
        return False
    return path.stat().st_size > MAX_MUSICXML_SIZE_BYTES


def find_source_file(source_filename: str, specimen_name: str) -> Path | None:
    """Find the source file in the score data directory.

    Args:
        source_filename: The filename from the flow.csv
        specimen_name: The specimen name (derived from flow.csv filename)

    Returns:
        Path to the source file, or None if not found
    """
    # Map specimen names to directories
    specimen_dirs = {
        "c05n05_musete": SCORE_DATA_DIR / "couperin_concerts",
        "c11n08_Rondeau": SCORE_DATA_DIR / "couperin_concerts",
        "out_of_the_flow_experience-polyrhythm_only": SCORE_DATA_DIR / "flow_control",
        "out_of_the_flow_experience-flow_only": SCORE_DATA_DIR
        / "flow_control"
        / "flow_only",
        "Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff": SCORE_DATA_DIR
        / "rachmaninoff",
        "op18_no4_mov4_flow": SCORE_DATA_DIR
        / "beethoven_op18-4iv_multimodal"
        / "op18_no4_mov4_flow",
        "WoO71": SCORE_DATA_DIR / "beethoven_woo71",
    }

    # Also check subdirectories for unfolded TSV files
    specimen_subdirs = {
        "c05n05_musete": SCORE_DATA_DIR / "couperin_concerts",
        "c11n08_Rondeau": SCORE_DATA_DIR / "couperin_concerts",
        "out_of_the_flow_experience-polyrhythm_only": SCORE_DATA_DIR
        / "flow_control"
        / "polyrythm_only",
        "out_of_the_flow_experience-flow_only": SCORE_DATA_DIR
        / "flow_control"
        / "flow_only",
        "Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff": SCORE_DATA_DIR
        / "rachmaninoff",
        "op18_no4_mov4_flow": SCORE_DATA_DIR
        / "beethoven_op18-4iv_multimodal"
        / "op18_no4_mov4_flow",
        "WoO71": SCORE_DATA_DIR / "beethoven_woo71",
    }

    # Try main directory first
    if specimen_name in specimen_dirs:
        candidate = specimen_dirs[specimen_name] / source_filename
        if candidate.exists():
            return candidate

    # Try subdirectory
    if specimen_name in specimen_subdirs:
        candidate = specimen_subdirs[specimen_name] / source_filename
        if candidate.exists():
            return candidate

    # Try recursive search as fallback
    for path in SCORE_DATA_DIR.rglob(source_filename):
        return path

    return None


def get_loader_for_source_file(source_filename: str):
    """Get the appropriate loader class based on source file extension.

    Loader is determined by file extension, NOT by flow_mode:
    - .tsv: Ms3Loader
    - .mm.json, .json: MeasureMapLoader
    - .musicxml, .xml: PartituraLoader (or Music21Loader as fallback)
    - .mei: PartituraLoader (or Music21Loader as fallback)

    Args:
        source_filename: The source filename from the flow.csv

    Returns:
        Loader class or None if not available
    """
    ext = source_filename.lower()

    if ext.endswith(".tsv"):
        try:
            from timetoalign.loader.score import Ms3Loader

            return Ms3Loader
        except ImportError:
            return None

    if ext.endswith(".mm.json") or ext.endswith(".json"):
        from timetoalign.loader.score import MeasureMapLoader

        return MeasureMapLoader

    if ext.endswith(".musicxml") or ext.endswith(".xml"):
        try:
            from timetoalign.loader.score import PartituraLoader

            return PartituraLoader
        except ImportError:
            pass
        try:
            from timetoalign.loader.score import Music21Loader

            return Music21Loader
        except ImportError:
            return None

    if ext.endswith(".mei"):
        try:
            from timetoalign.loader.score import PartituraLoader

            return PartituraLoader
        except ImportError:
            pass
        try:
            from timetoalign.loader.score import Music21Loader

            return Music21Loader
        except ImportError:
            return None

    return None


# endregion


# region Flows in the form of a .flow.csv file


def gold_flows(specimen_name: str) -> dict[str, tuple[MeasureRange, ...]]:
    """The rows of a specimen's ``.flow.csv`` file, by ``flow_mode``."""
    return read_flow_csv(TARGET_FLOWS_DIR / SPECIMENS[specimen_name].flow_csv)


def played_ranges(
    timeline: Any,
    convention: str = "default",
    *,
    registry: FlowControlRegistry | None = None,
) -> tuple[tuple[MeasureRange, ...], FlowResult]:
    """Play a timeline's flow control and state the flow as its gold file would.

    The runs become ranges of measure counts, each listing the atomic
    sections it covers and divided at the section breaks.
    """
    held = timeline.flow_control if registry is None else registry
    result = play(held, timeline.length, convention=convention)
    ranges = measure_ranges(
        result.runs,
        timeline.measure_map,
        sections=atomic_sections(held, timeline.length),
        divide_at=[element.coordinate for element in held.section_breaks],
    )
    return ranges, result


def atomic_ranges(
    timeline: Any, *, registry: FlowControlRegistry | None = None
) -> tuple[MeasureRange, ...]:
    """The atomic sections of a timeline's flow control in measure counts."""
    held = timeline.flow_control if registry is None else registry
    sections = atomic_sections(held, timeline.length)
    return measure_ranges(sections, timeline.measure_map, sections=sections)


def repaired_stress_score(timeline: Any) -> FlowControlRegistry:
    """The stress score's flow control with three elements added by hand.

    Two repeat starts, at the starts of measure counts 8 and 13, which the
    repeat ends of counts 8 and 14 name; and a Jump at the end of count 6
    (the second of three brackets) to the start of count 8. The three carry
    the provenance method ``manual``. An element is never changed in place:
    the Jumps of the two repeat ends are written anew with their place.
    """
    start = {measure.count: measure.qstamp for measure in timeline.measure_map}
    end = {
        measure.count: measure.qstamp + measure.actual_length
        for measure in timeline.measure_map
    }
    manual = {"method": "manual"}

    def at(value: Any) -> Coordinate:
        return Coordinate(value, "quarters")

    added = [
        JumpTarget(
            at(start[8]), id="start_8", manifestation="repeat_start", provenance=manual
        ),
        JumpTarget(
            at(start[13]),
            id="start_13",
            manifestation="repeat_start",
            provenance=manual,
        ),
        Jump(
            at(end[6]), [JumpDestination(to="start_8")], id="on_to_8", provenance=manual
        ),
    ]
    named = {end[8]: "start_8", end[14]: "start_13"}
    kept = []
    for element in timeline.flow_control.elements:
        to = named.get(element.coordinate.value)
        if isinstance(element, Jump) and to is not None:
            assert [destination.to_dict() for destination in element.targets] == [
                {"repeat_count": 1, "manifestation": "repeat_end"}
            ]
            element = replace(element, targets=(replace(element.targets[0], to=to),))
        kept.append(element)
    registry = FlowControlRegistry(unit="quarters")
    registry.add_all([*kept, *added])
    return registry


# endregion
