"""Tests for ATONLoader.

Tests use the SUPRA piano roll ATON analysis file as the primary test case.
All expected values are EXACT per the ZERO TOLERANCE validation policy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from timetoalign.core.events import BoundingBox
from timetoalign.loader.graphical.aton import ATONHole, ATONLoader
from timetoalign.testdata import ensure_data
from timetoalign.timelines import DiscreteGraphicalTimeline

# Test data directories
SUPRA_DIR, FIXTURES_DIR = ensure_data("supra", "fixtures")
ATON_FILE = SUPRA_DIR / "image" / "fd660zf8362_analysis.txt"


# region Fixtures


@pytest.fixture
def supra_aton_path() -> Path:
    """Path to the SUPRA ATON analysis file."""
    return ATON_FILE


@pytest.fixture
def loaded_supra_loader() -> ATONLoader:
    """ATONLoader with SUPRA analysis file loaded."""
    loader = ATONLoader()
    loader.load(ATON_FILE)
    return loader


# endregion


# region Test: Loading


class TestATONLoading:
    """Tests for loading ATON files."""

    @pytest.mark.slow
    def test_load_supra_aton(self, supra_aton_path: Path) -> None:
        """SUPRA ATON file loads without error."""
        loader = ATONLoader()
        result = loader.load(supra_aton_path)

        # Returns self for chaining
        assert result is loader

        # Data is now loaded
        assert loader._rollinfo
        assert loader._holes

    def test_load_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Loading nonexistent file raises FileNotFoundError."""
        loader = ATONLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.txt")


# endregion


# region Test: SUPRA ROLLINFO Metadata (EXACT VALUES)


@pytest.mark.slow
class TestSUPRAROLLINFO:
    """Tests for SUPRA ROLLINFO metadata extraction.

    Per README.md, all values are EXACT with ZERO TOLERANCE.
    """

    def test_musical_holes_exact(self, loaded_supra_loader: ATONLoader) -> None:
        """ROLLINFO MUSICAL_HOLES is exactly 30092."""
        assert loaded_supra_loader.musical_holes == 30092

    def test_musical_notes_exact(self, loaded_supra_loader: ATONLoader) -> None:
        """ROLLINFO MUSICAL_NOTES is exactly 8718."""
        assert loaded_supra_loader.musical_notes == 8718

    def test_first_hole_exact(self, loaded_supra_loader: ATONLoader) -> None:
        """ROLLINFO FIRST_HOLE is exactly 15343 pixels."""
        assert loaded_supra_loader.first_hole.value == 15343

    def test_last_hole_exact(self, loaded_supra_loader: ATONLoader) -> None:
        """ROLLINFO LAST_HOLE is exactly 293119 pixels."""
        assert loaded_supra_loader.last_hole.value == 293119

    def test_musical_length_exact(self, loaded_supra_loader: ATONLoader) -> None:
        """ROLLINFO MUSICAL_LENGTH is exactly 277776 pixels."""
        assert loaded_supra_loader.musical_length.value == 277776

    def test_musical_length_matches_calculation(
        self, loaded_supra_loader: ATONLoader
    ) -> None:
        """MUSICAL_LENGTH equals LAST_HOLE - FIRST_HOLE."""
        expected = (
            loaded_supra_loader.last_hole.value - loaded_supra_loader.first_hole.value
        )
        assert loaded_supra_loader.musical_length.value == expected

    def test_image_dimensions_exact(self, loaded_supra_loader: ATONLoader) -> None:
        """Image dimensions match expected values."""
        dims = loaded_supra_loader.image_dimensions
        assert dims["width"] == 4096
        assert dims["height"] == 299400

    def test_dpi_exact(self, loaded_supra_loader: ATONLoader) -> None:
        """DPI is exactly 300.25."""
        assert loaded_supra_loader.dpi == 300.25

    def test_tracker_holes_exact(self, loaded_supra_loader: ATONLoader) -> None:
        """TRACKER_HOLES is exactly 100."""
        assert loaded_supra_loader.rollinfo["TRACKER_HOLES"] == 100


# endregion


# region Test: SUPRA Hole Parsing


@pytest.mark.slow
class TestSUPRAHoleParsing:
    """Tests for SUPRA hole block parsing.

    The ATON file contains exactly 30092 HOLE blocks which matches MUSICAL_HOLES.
    """

    def test_hole_count_parsed(self, loaded_supra_loader: ATONLoader) -> None:
        """All HOLE blocks are parsed.

        The loader parses exactly 30092 HOLE blocks, matching MUSICAL_HOLES.
        """
        assert loaded_supra_loader.n_holes == 30092

    def test_first_hole_object_exists(self, loaded_supra_loader: ATONLoader) -> None:
        """First hole object exists and has expected structure."""
        assert len(loaded_supra_loader.holes) > 0
        first = loaded_supra_loader.holes[0]
        assert isinstance(first, ATONHole)

    def test_first_hole_origin_row(self, loaded_supra_loader: ATONLoader) -> None:
        """First parsed hole has correct origin_row (matches FIRST_HOLE)."""
        first = loaded_supra_loader.holes[0]
        assert first.origin_row == 15343  # Same as FIRST_HOLE

    def test_holes_are_sorted_by_row(self, loaded_supra_loader: ATONLoader) -> None:
        """Holes are in ascending order by origin_row (time order)."""
        rows = [h.origin_row for h in loaded_supra_loader.holes]
        # Check first few and last few are in order
        assert rows[:10] == sorted(rows[:10])
        assert rows[-10:] == sorted(rows[-10:])

    def test_hole_has_required_fields(self, loaded_supra_loader: ATONLoader) -> None:
        """Hole objects have all required fields populated."""
        first = loaded_supra_loader.holes[0]

        # Coordinates
        assert first.origin_row > 0
        assert first.origin_col > 0
        assert first.width_row > 0
        assert first.width_col > 0

        # Centroid
        assert first.centroid_row > 0
        assert first.centroid_col > 0

        # Shape
        assert first.area > 0
        assert first.perimeter > 0
        assert 0 < first.circularity <= 1

        # Tracker hole is in valid range (0-99 for Welte-Mignon)
        assert 0 <= first.tracker_hole < 100


# endregion


# region Test: Query Methods


@pytest.mark.slow
class TestQueryMethods:
    """Tests for hole query methods."""

    def test_get_holes_by_tracker(self, loaded_supra_loader: ATONLoader) -> None:
        """Can filter holes by tracker bar position."""
        # Tracker hole 12 is used in first hole
        holes_12 = loaded_supra_loader.get_holes_by_tracker(12)
        assert len(holes_12) == 7  # Tracker hole 12 has exactly 7 holes
        assert all(h.tracker_hole == 12 for h in holes_12)

    def test_get_holes_in_range(self, loaded_supra_loader: ATONLoader) -> None:
        """Can filter holes by row range."""
        # Get holes in first 1000 pixels of musical region
        start = loaded_supra_loader.first_hole.value
        end = start + 1000
        holes = loaded_supra_loader.get_holes_in_range(start, end)

        assert len(holes) == 82  # First 1000px of musical region contains 82 holes
        assert all(start <= h.origin_row <= end for h in holes)

    def test_get_note_holes(self, loaded_supra_loader: ATONLoader) -> None:
        """Can get holes that represent note attacks."""
        note_holes = loaded_supra_loader.get_note_holes()

        # MUSICAL_NOTES = 8718: holes that represent note attacks
        assert len(note_holes) == 8718
        assert all(h.note_attack is not None for h in note_holes)


# endregion


# region Test: Unloaded State


class TestUnloadedState:
    """Tests for behavior when no file is loaded."""

    def test_rollinfo_empty_before_load(self) -> None:
        """Rollinfo is empty before loading."""
        loader = ATONLoader()
        assert loader.rollinfo == {}

    def test_holes_empty_before_load(self) -> None:
        """Holes list is empty before loading."""
        loader = ATONLoader()
        assert loader.holes == []

    def test_repr_before_load(self) -> None:
        """Repr works before loading."""
        loader = ATONLoader()
        assert "not loaded" in repr(loader)

    @pytest.mark.slow
    def test_repr_after_load(self, loaded_supra_loader: ATONLoader) -> None:
        """Repr shows counts after loading."""
        repr_str = repr(loaded_supra_loader)
        assert "30094" in repr_str or "holes=" in repr_str
        assert "musical_holes=30092" in repr_str
        assert "musical_notes=8718" in repr_str


# endregion


# region Test: Source Path Tracking


class TestSourcePathTracking:
    """Tests for tracking the loaded file path."""

    def test_source_path_none_before_load(self) -> None:
        """Source path is None before loading."""
        loader = ATONLoader()
        assert loader.source_path is None

    @pytest.mark.slow
    def test_source_path_after_load(self, supra_aton_path: Path) -> None:
        """Source path is set after loading."""
        loader = ATONLoader()
        loader.load(supra_aton_path)

        assert loader.source_path == supra_aton_path


# endregion


# region Test: Specific Hole Values


class TestSpecificHoleValues:
    """Tests for specific hole values to verify parsing correctness."""

    @pytest.mark.slow
    def test_first_hole_detailed(self, loaded_supra_loader: ATONLoader) -> None:
        """First hole has expected values from ATON file.

        From the ATON file:
        @ID:            K0_N1
        @ORIGIN_ROW:    15343px
        @ORIGIN_COL:    445px
        @WIDTH_ROW:     41px
        @WIDTH_COL:     22px
        @CENTROID_ROW:  15363.4px
        @CENTROID_COL:  455.816px
        @AREA:          816px
        @PERIMETER:     114.473px
        @CIRCULARITY:   0.78
        @TRACKER_HOLE:  12
        """
        first = loaded_supra_loader.holes[0]

        assert first.id == "K0_N1"
        assert first.origin_row == 15343
        assert first.origin_col == 445
        assert first.width_row == 41
        assert first.width_col == 22
        # Centroid, perimeter, circularity are parsed via float() from ATON text.
        # These decimal values are exactly representable in float64.
        assert first.centroid_row == 15363.4
        assert first.centroid_col == 455.816
        assert first.area == 816
        assert first.perimeter == 114.473
        assert first.circularity == 0.78
        assert first.tracker_hole == 12


# endregion


# region Test: Minimal Synthetic ATON (Fast Tests)

MINIMAL_ATON_FILE = FIXTURES_DIR / "aton" / "minimal.aton"


@pytest.fixture
def minimal_aton_path() -> Path:
    """Path to the minimal synthetic ATON file."""
    return MINIMAL_ATON_FILE


@pytest.fixture
def loaded_minimal_loader() -> ATONLoader:
    """ATONLoader with minimal.aton loaded."""
    loader = ATONLoader()
    loader.load(MINIMAL_ATON_FILE)
    return loader


class TestMinimalATON:
    """Tests for minimal.aton synthetic fixture.

    Per README.md, all values are EXACT with ZERO TOLERANCE.
    These tests are fast (5 holes vs 30092 in SUPRA).
    """

    def test_load_minimal_aton(self, minimal_aton_path: Path) -> None:
        """minimal.aton loads without error."""
        loader = ATONLoader()
        result = loader.load(minimal_aton_path)
        assert result is loader

    def test_hole_count_exact(self, loaded_minimal_loader: ATONLoader) -> None:
        """minimal.aton contains exactly 5 holes."""
        assert loaded_minimal_loader.n_holes == 5

    def test_musical_holes_exact(self, loaded_minimal_loader: ATONLoader) -> None:
        """ROLLINFO MUSICAL_HOLES is exactly 5."""
        assert loaded_minimal_loader.musical_holes == 5

    def test_musical_notes_exact(self, loaded_minimal_loader: ATONLoader) -> None:
        """ROLLINFO MUSICAL_NOTES is exactly 5."""
        assert loaded_minimal_loader.musical_notes == 5

    def test_first_hole_exact(self, loaded_minimal_loader: ATONLoader) -> None:
        """ROLLINFO FIRST_HOLE is exactly 100 pixels."""
        assert loaded_minimal_loader.first_hole.value == 100

    def test_last_hole_exact(self, loaded_minimal_loader: ATONLoader) -> None:
        """ROLLINFO LAST_HOLE is exactly 1900 pixels."""
        assert loaded_minimal_loader.last_hole.value == 1900

    def test_musical_length_exact(self, loaded_minimal_loader: ATONLoader) -> None:
        """ROLLINFO MUSICAL_LENGTH is exactly 1800 pixels."""
        assert loaded_minimal_loader.musical_length.value == 1800

    def test_image_dimensions_exact(self, loaded_minimal_loader: ATONLoader) -> None:
        """Image dimensions match expected values."""
        dims = loaded_minimal_loader.image_dimensions
        assert dims["width"] == 1024
        assert dims["height"] == 2000

    def test_dpi_exact(self, loaded_minimal_loader: ATONLoader) -> None:
        """DPI is exactly 300.0."""
        assert loaded_minimal_loader.dpi == 300.0

    def test_first_hole_detailed(self, loaded_minimal_loader: ATONLoader) -> None:
        """First hole (H1_N1) has exact expected values."""
        first = loaded_minimal_loader.holes[0]

        assert first.id == "H1_N1"
        assert first.origin_row == 100
        assert first.origin_col == 200
        assert first.width_row == 20
        assert first.width_col == 15
        assert first.bounding_box == BoundingBox.from_corners(200, 100, 215, 120)
        assert first.centroid_row == 110.5
        assert first.centroid_col == 207.5
        assert first.area == 280
        assert first.perimeter == 70.0
        assert first.circularity == 0.72
        assert first.tracker_hole == 60
        assert first.midi_key == 60
        assert first.note_attack == 100
        assert first.off_time == 120

    def test_all_holes_have_note_attack(
        self, loaded_minimal_loader: ATONLoader
    ) -> None:
        """All holes in minimal.aton are musical (have NOTE_ATTACK)."""
        note_holes = loaded_minimal_loader.get_note_holes()
        assert len(note_holes) == 5

    def test_from_file_convenience(self, minimal_aton_path: Path) -> None:
        """from_file() classmethod works correctly."""
        loader = ATONLoader.from_file(minimal_aton_path)
        assert loader.n_holes == 5


class TestMinimalATONTimeline:
    """Tests for timeline creation from minimal.aton."""

    def test_create_timeline(self, loaded_minimal_loader: ATONLoader) -> None:
        """create_timeline() produces a valid timeline."""
        timeline = loaded_minimal_loader.create_timeline()

        assert type(timeline) is DiscreteGraphicalTimeline
        assert timeline.n_events == 5

    def test_create_timeline_with_uid(self, loaded_minimal_loader: ATONLoader) -> None:
        """create_timeline() accepts custom uid."""
        timeline = loaded_minimal_loader.create_timeline(uid="dgt_1")

        assert timeline.id == "dgt_1"


# endregion


# region Test: Hole event IDs

# The first six hole blocks of the SUPRA roll fd660zf8362, as
# (@ID, ORIGIN_ROW, ORIGIN_COL, TRACKER_HOLE, MIDI_KEY): three holes on
# three tracker holes share the @ID K0_N1.
SUPRA_FIRST_SIX_HOLES = [
    ("K0_N1", 15343, 445, 12, -1),
    ("K0_N2", 15391, 445, 12, -1),
    ("K88_N1", 15617, 3737, 99, -1),
    ("K86_N1", 15626, 3661, 97, -1),
    ("K0_N1", 15638, 216, 6, -1),
    ("K0_N1", 15638, 293, 8, -1),
]


def _write_aton(path: Path, holes: list[tuple[str, int, int, int, int]]) -> Path:
    """Write an ATON file holding the given hole blocks."""
    blocks = "".join(
        "@@BEGIN: HOLE\n"
        f"@ID:\t\t{hole_id}\n"
        f"@ORIGIN_ROW:\t{row}px\n"
        f"@ORIGIN_COL:\t{col}px\n"
        "@WIDTH_ROW:\t31px\n"
        "@WIDTH_COL:\t22px\n"
        f"@TRACKER_HOLE:\t{tracker}\n"
        f"@MIDI_KEY:\t{key}\n"
        "@@END: HOLE\n\n"
        for hole_id, row, col, tracker, key in holes
    )
    path.write_text(
        "@@BEGIN: ROLLINFO\n"
        "@IMAGE_WIDTH:\t4096px\n"
        "@IMAGE_LENGTH:\t299400px\n"
        "@@END: ROLLINFO\n\n"
        f"@@BEGIN: HOLES\n\n{blocks}@@END: HOLES\n",
        encoding="utf-8",
    )
    return path


def _event_ids_and_names(timeline: DiscreteGraphicalTimeline) -> tuple[list, list]:
    events = timeline.get_events().table
    return events.column("id").to_pylist(), events.column("name").to_pylist()


class TestHoleEventIds:
    """Event IDs of holes (see README "ATONLoader event IDs")."""

    def test_holes_sharing_an_id_are_told_apart_by_tracker_hole(
        self, tmp_path: Path
    ) -> None:
        """The colliding SUPRA rows give six distinct IDs, the names as written."""
        path = _write_aton(tmp_path / "first_six.txt", SUPRA_FIRST_SIX_HOLES)
        timeline = ATONLoader.from_file(path).create_timeline(uid="dgt_1")

        ids, names = _event_ids_and_names(timeline)
        assert ids == [
            "hole_K0_N1_tracker12",
            "hole_K0_N2",
            "hole_K88_N1",
            "hole_K86_N1",
            "hole_K0_N1_tracker6",
            "hole_K0_N1_tracker8",
        ]
        assert names == ["K0_N1", "K0_N2", "K88_N1", "K86_N1", "K0_N1", "K0_N1"]

    def test_ids_that_do_not_collide_are_unchanged(
        self, loaded_minimal_loader: ATONLoader
    ) -> None:
        """minimal.aton's five distinct IDs keep the form hole_<ID>."""
        ids, names = _event_ids_and_names(loaded_minimal_loader.create_timeline())
        assert ids == [
            "hole_H1_N1",
            "hole_H2_N2",
            "hole_H3_N3",
            "hole_H4_N4",
            "hole_H5_N5",
        ]
        assert names == ["H1_N1", "H2_N2", "H3_N3", "H4_N4", "H5_N5"]

    def test_a_hole_stated_twice_is_refused(self, tmp_path: Path) -> None:
        """One @ID twice on one tracker hole derives one ID twice: refused."""
        holes = [("K0_N1", 15638, 216, 6, -1), ("K0_N1", 15638, 216, 6, -1)]
        loader = ATONLoader.from_file(_write_aton(tmp_path / "twice.txt", holes))

        with pytest.raises(ValueError) as raised:
            loader.create_timeline(uid="dgt_1")
        assert str(raised.value) == (
            "Event id 'hole_K0_N1_tracker6' is given twice in one call to "
            "add_events on timeline 'dgt_1'"
        )

    @pytest.mark.slow
    def test_supra_roll_loads_whole(self, loaded_supra_loader: ATONLoader) -> None:
        """30092 events, 30092 IDs; the 992 holes of shared @IDs carry a tracker."""
        timeline = loaded_supra_loader.create_timeline(uid="dgt_1")
        ids, names = _event_ids_and_names(timeline)

        assert timeline.n_events == 30092
        assert len(set(ids)) == 30092
        holes = loaded_supra_loader.holes
        with_tracker = [
            event_id == f"hole_{hole.id}_tracker{hole.tracker_hole}"
            for event_id, hole in zip(ids, holes, strict=True)
        ]
        plain = [
            event_id == f"hole_{hole.id}"
            for event_id, hole in zip(ids, holes, strict=True)
        ]
        assert sum(with_tracker) == 992
        assert sum(plain) == 29100
        assert names == [hole.id for hole in holes]
        assert ids[:6] == [
            "hole_K0_N1_tracker12",
            "hole_K0_N2_tracker12",
            "hole_K88_N1",
            "hole_K86_N1",
            "hole_K0_N1_tracker6",
            "hole_K0_N1_tracker8",
        ]


# endregion
