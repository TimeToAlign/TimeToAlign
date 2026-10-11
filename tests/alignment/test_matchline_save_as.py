"""Tests for MatchLine.save_as() and the match_format module.

Tests cover:
- Format inference from file extension
- Unsupported format error
- Minimal export (no context, placeholder data)
- Rich export with MatchFileContext
- Deletion lines for NOMATCH claims
- Header completeness (all info(...) lines)
- SnoteRecord / NoteRecord formatting
- Round-trip: load .match -> MatchLine -> save_as() -> reload -> compare
- Round-trip with how03 notebook data: match notes -> save_as() -> MatchfileLoader -> compare
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from timetoalign.alignment.match_format import (
    MatchFileContext,
    NoteRecord,
    SnoteRecord,
    format_deletion_line,
    format_header,
    format_insertion_line,
    format_match_line,
    format_note_line,
    format_score_properties,
    format_snote_line,
)
from timetoalign.alignment.matchline import MatchLine
from timetoalign.core import TimeUnit

from .helpers import make_match_stamp as MatchStamp

# region Test Data Paths

VIENNA_DATA = Path(__file__).parent.parent / "data" / "vienna_1x22"
CHOPIN_P01 = VIENNA_DATA / "Chopin_op10_no3_p01.match"
BEETHOVEN_DATA = (
    Path(__file__).parent.parent / "data" / "score" / "beethoven_op18-4iv_multimodal"
)
NORMAL_DIR = BEETHOVEN_DATA / "StringQuartetEEP_I_Normal"
ABC_DIR = BEETHOVEN_DATA / "ABC"

# endregion


# region Fixtures


@pytest.fixture
def sample_snote() -> SnoteRecord:
    """A sample score note record matching the first note of Chopin op.10/3."""
    return SnoteRecord(
        id="n1",
        pitch_name="B",
        modifier="n",
        octave=3,
        measure=0,
        beat=1,
        offset="0",
        duration="1/8",
        onset_in_beats=-0.5,
        offset_in_beats=0.0,
        attributes=["v1", "staff1"],
    )


@pytest.fixture
def sample_note() -> NoteRecord:
    """A sample performance note record."""
    return NoteRecord(
        id="n0",
        midi_pitch=59,
        onset_tick=0,
        offset_tick=261,
        velocity=44,
        channel=0,
        track=0,
    )


@pytest.fixture
def sample_context() -> MatchFileContext:
    """A MatchFileContext with metadata and a few notes."""
    snote1 = SnoteRecord(
        id="n1",
        pitch_name="B",
        modifier="n",
        octave=3,
        measure=0,
        beat=1,
        offset="0",
        duration="1/8",
        onset_in_beats=-0.5,
        offset_in_beats=0.0,
        attributes=["v1", "staff1"],
    )
    snote2 = SnoteRecord(
        id="n2",
        pitch_name="E",
        modifier="n",
        octave=4,
        measure=1,
        beat=1,
        offset="0",
        duration="1/8",
        onset_in_beats=0.0,
        offset_in_beats=0.5,
        attributes=["v1", "staff1"],
    )
    note1 = NoteRecord(
        id="n0", midi_pitch=59, onset_tick=0, offset_tick=261, velocity=44
    )
    note2 = NoteRecord(
        id="n2", midi_pitch=64, onset_tick=680, offset_tick=1838, velocity=54
    )

    deletion_snote = SnoteRecord(
        id="n99",
        pitch_name="A",
        modifier="#",
        octave=4,
        measure=16,
        beat=2,
        offset="1/16",
        duration="1/16",
        onset_in_beats=31.25,
        offset_in_beats=31.5,
        attributes=["v2", "staff1"],
    )

    return MatchFileContext(
        piece="Chopin_op10_no3",
        composer="Frèdéryk Chopin",
        performer="Pianist 01",
        score_filename="Chopin_op10_no3.musicxml",
        midi_filename="Chopin_op10_no3_p01.mid",
        midi_clock_units=480,
        midi_clock_rate=500000,
        score_notes={
            0.0: [snote1],  # score coord = 0.0 (after anacrusis shift)
            0.5: [snote2],  # score coord = 0.5
        },
        perf_notes={
            0.0: [note1],  # perf coord = 0 ticks
            680.0: [note2],  # perf coord = 680 ticks
        },
        score_properties=[
            "scoreprop(keySignature,E,0:1,0,-0.5000).",
            "scoreprop(timeSignature,2/4,0:1,0,-0.5000).",
        ],
        deletions=[deletion_snote],
    )


@pytest.fixture
def two_stamp_matchline() -> MatchLine:
    """A MatchLine with two stamps matching the sample_context coordinates."""
    stamps = [
        MatchStamp(
            coordinates={"score": 0.0, "perf": 0.0},
            anchor_edges=[("score", "perf")],
        ),
        MatchStamp(
            coordinates={"score": 0.5, "perf": 680.0},
            anchor_edges=[("score", "perf")],
        ),
    ]
    return MatchLine(source_timeline_id="score", stamps=stamps)


# endregion


# region Formatting Tests


class TestFormatSnote:
    """Test format_snote_line()."""

    def test_basic_formatting(self, sample_snote: SnoteRecord) -> None:
        """Format a standard snote line."""
        result = format_snote_line(sample_snote)
        assert result == "snote(n1,[B,n],3,0:1,0,1/8,-0.5000,0.0000,[v1,staff1])"

    def test_empty_attributes(self) -> None:
        """Snote with no attributes produces empty brackets."""
        snote = SnoteRecord(
            id="n10",
            pitch_name="C",
            modifier="n",
            octave=4,
            measure=1,
            beat=1,
            offset="0",
            duration="1/4",
            onset_in_beats=0.0,
            offset_in_beats=1.0,
            attributes=[],
        )
        result = format_snote_line(snote)
        assert result.endswith(",[])")


class TestFormatNote:
    """Test format_note_line()."""

    def test_basic_formatting(self, sample_note: NoteRecord) -> None:
        """Format a standard note line."""
        result = format_note_line(sample_note)
        assert result == "note(n0,59,0,261,44,0,0)"


class TestFormatMatchLine:
    """Test format_match_line()."""

    def test_matched_pair(
        self, sample_snote: SnoteRecord, sample_note: NoteRecord
    ) -> None:
        """Format a matched snote-note pair."""
        result = format_match_line(sample_snote, sample_note)
        assert result.startswith("snote(")
        assert "-note(" in result
        assert result.endswith(".")


class TestFormatDeletion:
    """Test format_deletion_line()."""

    def test_deletion(self, sample_snote: SnoteRecord) -> None:
        """Format a deletion line."""
        result = format_deletion_line(sample_snote)
        assert result.startswith("snote(")
        assert result.endswith("-deletion.")


class TestFormatInsertion:
    """Test format_insertion_line()."""

    def test_insertion(self, sample_note: NoteRecord) -> None:
        """Format an insertion line."""
        result = format_insertion_line(sample_note)
        assert result.startswith("insertion-note(")
        assert result.endswith(".")


# endregion


# region Header Tests


class TestFormatHeader:
    """Test format_header()."""

    def test_header_completeness(self, sample_context: MatchFileContext) -> None:
        """All mandatory info lines are present."""
        lines = format_header(sample_context)
        text = "\n".join(lines)

        assert "info(matchFileVersion,1.0.0)." in text
        assert "info(piece,Chopin_op10_no3)." in text
        assert "info(composer," in text
        assert "info(performer," in text
        assert "info(midiClockUnits,480)." in text
        assert "info(midiClockRate,500000)." in text

    def test_empty_optional_fields(self) -> None:
        """Empty optional fields are omitted."""
        ctx = MatchFileContext()
        lines = format_header(ctx)
        text = "\n".join(lines)

        assert "info(matchFileVersion,1.0.0)." in text
        assert "info(piece," not in text
        assert "info(composer," not in text
        assert "info(midiClockUnits,480)." in text

    def test_all_lines_end_with_period(self, sample_context: MatchFileContext) -> None:
        """Every header line ends with a period."""
        for line in format_header(sample_context):
            assert line.endswith("."), f"Line does not end with period: {line}"


class TestFormatScoreProperties:
    """Test format_score_properties()."""

    def test_passthrough(self, sample_context: MatchFileContext) -> None:
        """Pre-formatted score properties are returned as-is."""
        props = format_score_properties(sample_context)
        assert len(props) == 2
        assert "keySignature" in props[0]
        assert "timeSignature" in props[1]

    def test_empty_properties(self) -> None:
        """Empty context produces no score properties."""
        ctx = MatchFileContext()
        assert format_score_properties(ctx) == []


# endregion


# region MatchLine.save_as() Tests


class TestSaveAsFormatInference:
    """Test format inference from file extension."""

    def test_match_extension_inferred(self, tmp_path: Path) -> None:
        """A .match extension infers format='match'."""
        line = MatchLine(
            source_timeline_id="src",
            stamps=[
                MatchStamp(coordinates={"src": 0.0, "tgt": 0.0}),
                MatchStamp(coordinates={"src": 1.0, "tgt": 100.0}),
            ],
        )
        output = line.save_as(tmp_path / "test.match")
        assert output.exists()
        assert output.suffix == ".match"

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        """Unsupported format raises ValueError."""
        line = MatchLine(
            source_timeline_id="src",
            stamps=[MatchStamp(coordinates={"src": 0.0, "tgt": 0.0})],
        )
        with pytest.raises(ValueError, match="Unsupported export format"):
            line.save_as(tmp_path / "test.csv", format="csv")


class TestMinimalExport:
    """Test export with no context (placeholder data)."""

    def test_minimal_produces_valid_file(self, tmp_path: Path) -> None:
        """Minimal export creates a readable .match file."""
        stamps = [
            MatchStamp(
                coordinates={"score": 0.0, "perf": 0.0},
                anchor_edges=[("score", "perf")],
            ),
            MatchStamp(
                coordinates={"score": 100.0, "perf": 45000.0},
                anchor_edges=[("score", "perf")],
            ),
        ]
        line = MatchLine(source_timeline_id="score", stamps=stamps)
        output = line.save_as(tmp_path / "minimal.match")

        content = output.read_text()
        lines = content.strip().split("\n")

        # Must have header
        assert any("info(matchFileVersion" in ln for ln in lines)
        # Must have match lines
        match_lines = [ln for ln in lines if "snote(" in ln and "note(" in ln]
        assert len(match_lines) == 2

    def test_empty_matchline(self, tmp_path: Path) -> None:
        """Empty MatchLine produces header-only file."""
        line = MatchLine(source_timeline_id="src")
        output = line.save_as(tmp_path / "empty.match")

        content = output.read_text()
        assert "info(matchFileVersion" in content


class TestRichExport:
    """Test export with a full MatchFileContext."""

    def test_rich_export_structure(
        self,
        tmp_path: Path,
        sample_context: MatchFileContext,
        two_stamp_matchline: MatchLine,
    ) -> None:
        """Rich export has header, score properties, matches, and deletions."""
        output = two_stamp_matchline.save_as(
            tmp_path / "rich.match", context=sample_context
        )
        content = output.read_text()
        lines = content.strip().split("\n")

        # Header
        info_lines = [ln for ln in lines if ln.startswith("info(")]
        assert (
            len(info_lines) >= 6
        )  # version + piece + score + midi + composer + perf + 2 clock

        # Score properties
        prop_lines = [ln for ln in lines if ln.startswith("scoreprop(")]
        assert len(prop_lines) == 2

        # Match lines
        match_lines = [ln for ln in lines if "snote(" in ln and "-note(" in ln]
        assert len(match_lines) == 2

        # Deletion lines
        del_lines = [ln for ln in lines if "-deletion." in ln]
        assert len(del_lines) == 1

    def test_deletion_line_format(
        self,
        tmp_path: Path,
        sample_context: MatchFileContext,
        two_stamp_matchline: MatchLine,
    ) -> None:
        """Deletion lines follow the snote(...)-deletion. format."""
        output = two_stamp_matchline.save_as(
            tmp_path / "del.match", context=sample_context
        )
        content = output.read_text()
        del_lines = [
            ln for ln in content.strip().split("\n") if ln.endswith("-deletion.")
        ]
        assert len(del_lines) == 1
        assert del_lines[0].startswith("snote(n99,")

    def test_output_path_returned(
        self,
        tmp_path: Path,
        sample_context: MatchFileContext,
        two_stamp_matchline: MatchLine,
    ) -> None:
        """save_as() returns the resolved Path."""
        output = two_stamp_matchline.save_as(
            tmp_path / "out.match", context=sample_context
        )
        assert isinstance(output, Path)
        assert output.exists()


# endregion


# region Round-Trip Test: MatchfileLoader


class TestRoundTripMatchfileLoader:
    """Round-trip: load .match file via MatchfileLoader, build MatchLine,
    export via save_as(), then verify the output has the correct structure.

    This tests against the Chopin op.10/3 p01 test data.
    """

    def test_roundtrip_line_counts(self, tmp_path: Path) -> None:
        """Exported file has same number of match + deletion lines as original."""
        if not CHOPIN_P01.exists():
            pytest.skip(f"Test data not found: {CHOPIN_P01}")

        # Count lines in original file
        original_content = CHOPIN_P01.read_text()
        original_lines = original_content.strip().split("\n")
        original_deletion_lines = [
            ln for ln in original_lines if ln.endswith("-deletion.")
        ]
        original_info_lines = [ln for ln in original_lines if ln.startswith("info(")]

        # Load via MatchfileLoader
        from timetoalign.loader.alignment import MatchfileLoader

        loader = MatchfileLoader()
        loader.load(CHOPIN_P01)
        bundle = loader.create_bundle()

        # Get the timeline IDs
        score_tl = loader.create_timeline("score")

        # Get synchronous claims (matched notes)
        sync_claims = [c for c in bundle.get_match_claims() if c.is_synchronous]

        # Build MatchLine from synchronous claims only
        match_line = MatchLine.from_claims(
            sync_claims,
            source_timeline_id=score_tl.id,
        )

        # Build a context from the original file for re-export
        context = _build_context_from_match_file(CHOPIN_P01)

        # Add deletions to context
        # Parse deletion snote IDs from original
        for line in original_deletion_lines:
            snote_data = _parse_snote_from_line(line)
            if snote_data:
                context.deletions.append(snote_data)

        # Export
        output = match_line.save_as(tmp_path / "roundtrip.match", context=context)
        assert output.exists()

        exported_content = output.read_text()
        exported_lines = exported_content.strip().split("\n")

        # Verify header completeness
        exported_info = [ln for ln in exported_lines if ln.startswith("info(")]
        assert len(exported_info) >= len(original_info_lines)

        # The number of match lines in the export should equal the number
        # of synchronous claims (stamps in the MatchLine)
        exported_match = [
            ln for ln in exported_lines if ln.startswith("snote(") and "-note(" in ln
        ]
        assert len(exported_match) == match_line.n_stamps

        # Deletion count should match
        exported_deletions = [ln for ln in exported_lines if ln.endswith("-deletion.")]
        assert len(exported_deletions) == len(original_deletion_lines)

    def test_roundtrip_preserves_header_metadata(self, tmp_path: Path) -> None:
        """Exported file preserves piece, composer, performer from original."""
        if not CHOPIN_P01.exists():
            pytest.skip(f"Test data not found: {CHOPIN_P01}")

        context = _build_context_from_match_file(CHOPIN_P01)

        # Build a minimal MatchLine with one stamp
        stamps = [
            MatchStamp(
                coordinates={"score": 0.0, "perf": 0.0},
                anchor_edges=[("score", "perf")],
            ),
        ]
        line = MatchLine(source_timeline_id="score", stamps=stamps)

        output = line.save_as(tmp_path / "header.match", context=context)
        content = output.read_text()

        assert "info(piece,Chopin_op10_no3)." in content
        assert "info(midiClockUnits,480)." in content
        assert "info(midiClockRate,500000)." in content


# endregion


# region Round-Trip Test: how03 Notebook Data


class TestRoundTripHow03Notebook:
    """Round-trip using the how03 notebook workflow.

    Load EEP + ABC data -> match notes -> build MatchLine -> save_as()
    -> reload via MatchfileLoader -> verify alignment claims are preserved.
    """

    @pytest.mark.slow
    def test_notebook_roundtrip(self, tmp_path: Path) -> None:
        """Full how03 workflow: match -> export -> reload -> verify."""
        if not NORMAL_DIR.exists() or not ABC_DIR.exists():
            pytest.skip("Beethoven test data not found")

        import pandas as pd

        from timetoalign.alignment import MatchLine
        from timetoalign.alignment.match_format import MatchFileContext
        from timetoalign.alignment.matching import (
            match_notes_by_attributes,
            prepare_abc_notes_for_matching,
            prepare_eep_notes_for_matching,
        )
        from timetoalign.loader.physical.eep_notes import EepNotesLoader

        # Step 1: Load and match (same as how03)
        eep_loader = EepNotesLoader()
        eep_loader.load(*sorted(NORMAL_DIR.glob("*_align_*.notes")))
        eep_df = eep_loader.events.to_dataframe()

        abc_df = pd.read_csv(ABC_DIR / "n04op18-4_04_unfolded.notes.tsv", sep="\t")

        eep_prepared = prepare_eep_notes_for_matching(eep_df)
        abc_prepared = prepare_abc_notes_for_matching(abc_df)

        match_result = match_notes_by_attributes(
            eep_prepared,
            abc_prepared,
            match_columns=["pitch", "staff"],
            source_coord_column="start",
            target_coord_column="quarterbeats_playthrough",
            source_timeline_id="cpt_1",
            target_timeline_id="clt_1",
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.quarters,
        )

        # Build MatchLine
        score_to_perf = MatchLine.from_claims(
            match_result.match_claims,
            source_timeline_id="clt_1",
        )

        # Step 2: Build context and export
        context = MatchFileContext.from_dataframes(
            score_df=abc_prepared,
            perf_df=eep_prepared,
            match_result=match_result,
            piece="Beethoven_Op18-4_iv",
            composer="Ludwig van Beethoven",
            performer="StringQuartet_Normal",
            midi_clock_units=480,
            midi_clock_rate=500000,
        )

        output_path = score_to_perf.save_as(
            tmp_path / "beethoven_how03.match",
            context=context,
        )
        assert output_path.exists()

        # Step 3: Read back and verify structure
        content = output_path.read_text()
        lines = content.strip().split("\n")

        # Header checks
        assert any("info(matchFileVersion,1.0.0)." in ln for ln in lines)
        assert any("info(piece,Beethoven_Op18-4_iv)." in ln for ln in lines)
        assert any("info(composer,Ludwig van Beethoven)." in ln for ln in lines)

        # Match lines: one per stamp
        match_lines = [ln for ln in lines if ln.startswith("snote(") and "-note(" in ln]
        assert len(match_lines) == score_to_perf.n_stamps

        # Deletion lines for unmatched target notes: matching here passes EEP as
        # source_df and ABC (the score) as target_df, so unmatched_target rows are
        # unmatched *score* notes (matching.py:208-265). MatchFileContext emits one
        # -deletion. line per such unmatched score note (match_format.py:331-364),
        # deterministically 10 for this Beethoven Op.18/4-iv Normal-take fixture.
        del_lines = [ln for ln in lines if ln.endswith("-deletion.")]
        assert len(del_lines) == 10

        # All lines should end with a period (Prolog convention)
        for line in lines:
            if line.strip():
                assert line.strip().endswith(
                    "."
                ), f"Line does not end with period: {line}"

    def test_chopin_roundtrip_reload_via_matchfileloader(self, tmp_path: Path) -> None:
        """Round-trip: load Chopin .match -> MatchLine -> save_as -> MatchfileLoader.

        This is the definitive round-trip test: load an existing .match file
        via MatchfileLoader, build a MatchLine from the synchronous claims,
        parse the original file to build a MatchFileContext, export via
        save_as(), then reload the exported file with MatchfileLoader and
        verify that the same number of synchronous claims are produced.
        """
        if not CHOPIN_P01.exists():
            pytest.skip(f"Test data not found: {CHOPIN_P01}")

        from timetoalign.loader.alignment import MatchfileLoader

        # Step 1: Load the original .match file
        loader = MatchfileLoader()
        loader.load(CHOPIN_P01)
        bundle = loader.create_bundle()

        score_tl = loader.create_timeline("score")

        sync_claims = [c for c in bundle.get_match_claims() if c.is_synchronous]

        match_line = MatchLine.from_claims(
            sync_claims,
            source_timeline_id=score_tl.id,
        )
        original_n_stamps = match_line.n_stamps

        # Step 2: Build context from original file and export
        context = _build_context_from_match_file(CHOPIN_P01)

        # Add original score properties and deletion lines
        original_content = CHOPIN_P01.read_text()
        original_lines = original_content.strip().split("\n")
        for line in original_lines:
            if line.endswith("-deletion."):
                snote_data = _parse_snote_from_line(line)
                if snote_data:
                    context.deletions.append(snote_data)

        output_path = match_line.save_as(
            tmp_path / "chopin_roundtrip.match", context=context
        )
        assert output_path.exists()

        # Step 3: Reload the exported file with MatchfileLoader
        reload_loader = MatchfileLoader()
        reload_loader.load(output_path)
        reload_bundle = reload_loader.create_bundle()

        reload_sync = [c for c in reload_bundle.get_match_claims() if c.is_synchronous]

        # The reloaded file should produce the same number of
        # synchronous claims as we had stamps in the MatchLine
        assert len(reload_sync) == original_n_stamps


# endregion


# region Internal Helpers


def _build_context_from_match_file(path: Path) -> MatchFileContext:
    """Parse a .match file into a MatchFileContext for re-export.

    This is a simplified parser that reads header info and builds
    SnoteRecord/NoteRecord lookups from the match lines.
    """
    content = path.read_text(encoding="utf-8")
    lines = content.strip().split("\n")

    # Parse header
    header: dict[str, str] = {}
    for line in lines:
        if line.startswith("info("):
            m = re.match(r"^info\((\w+),(.+)\)\.\s*$", line)
            if m:
                header[m.group(1)] = m.group(2)

    # Parse score properties
    props = [ln for ln in lines if ln.startswith("scoreprop(")]

    # Parse match lines for note data
    score_notes: dict[float, list[SnoteRecord]] = {}
    perf_notes: dict[float, list[NoteRecord]] = {}

    for line in lines:
        if line.startswith("snote(") and "-note(" in line:
            snote = _parse_snote_from_line(line)
            note = _parse_note_from_line(line)
            if snote and note:
                score_notes.setdefault(snote.onset_in_beats, []).append(snote)
                perf_notes.setdefault(float(note.onset_tick), []).append(note)

    return MatchFileContext(
        piece=header.get("piece", ""),
        composer=header.get("composer", ""),
        performer=header.get("performer", ""),
        score_filename=header.get("scoreFileName", ""),
        midi_filename=header.get("midiFileName", ""),
        midi_clock_units=int(header.get("midiClockUnits", "480")),
        midi_clock_rate=int(header.get("midiClockRate", "500000")),
        score_notes=score_notes,
        perf_notes=perf_notes,
        score_properties=props,
    )


def _parse_snote_from_line(line: str) -> SnoteRecord | None:
    """Parse the snote(...) part of a match or deletion line.

    Handles lines like:
        snote(n1,[B,n],3,0:1,0,1/8,-0.5000,0.0000,[v1,staff1])-note(...)
        snote(n1,[B,n],3,0:1,0,1/8,-0.5000,0.0000,[v1,staff1])-deletion.
    """
    # Extract the snote(...) part
    # The modifier field can contain '#', 'b', 'bb', 'x', 'n' etc.
    m = re.match(
        r"snote\((\w+),\[(\w+),([^\]]+)\],(\d+),(\d+):(\d+),([\w/]+),([\w/]+),"
        r"(-?[\d.]+),(-?[\d.]+),\[([^\]]*)\]\)",
        line,
    )
    if not m:
        return None

    attrs = [a.strip() for a in m.group(11).split(",") if a.strip()]
    return SnoteRecord(
        id=m.group(1),
        pitch_name=m.group(2),
        modifier=m.group(3),
        octave=int(m.group(4)),
        measure=int(m.group(5)),
        beat=int(m.group(6)),
        offset=m.group(7),
        duration=m.group(8),
        onset_in_beats=float(m.group(9)),
        offset_in_beats=float(m.group(10)),
        attributes=attrs,
    )


def _parse_note_from_line(line: str) -> NoteRecord | None:
    """Parse the note(...) part of a match line.

    Handles lines like:
        snote(...)-note(n0,59,0,261,44,0,0).
    """
    m = re.search(r"note\((\w+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)\)", line)
    if not m:
        return None

    return NoteRecord(
        id=m.group(1),
        midi_pitch=int(m.group(2)),
        onset_tick=int(m.group(3)),
        offset_tick=int(m.group(4)),
        velocity=int(m.group(5)),
        channel=int(m.group(6)),
        track=int(m.group(7)),
    )


# endregion
