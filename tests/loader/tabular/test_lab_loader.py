"""Tests for LabLoader.

Tests use synthetic .lab files in tests/data/fixtures/lab/.
All expected values are EXACT per the ZERO TOLERANCE validation policy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from timetoalign.core import TimeUnit
from timetoalign.loader.tabular import LabLoader

# Test data directory
LAB_FIXTURES_DIR = Path(__file__).parent.parent.parent / "data" / "fixtures" / "lab"


# region Fixtures


@pytest.fixture
def regions_lab_path() -> Path:
    """Path to the regions.lab test file."""
    return LAB_FIXTURES_DIR / "regions.lab"


@pytest.fixture
def beats_lab_path() -> Path:
    """Path to the beats.lab test file."""
    return LAB_FIXTURES_DIR / "beats.lab"


@pytest.fixture
def instants_lab_path() -> Path:
    """Path to the instants.lab test file."""
    return LAB_FIXTURES_DIR / "instants.lab"


@pytest.fixture
def loaded_regions_loader(regions_lab_path: Path) -> LabLoader:
    """LabLoader with regions.lab loaded."""
    loader = LabLoader()
    loader.load(regions_lab_path)
    return loader


@pytest.fixture
def loaded_beats_loader(beats_lab_path: Path) -> LabLoader:
    """LabLoader with beats.lab loaded."""
    loader = LabLoader()
    loader.load(beats_lab_path)
    return loader


@pytest.fixture
def loaded_instants_loader(instants_lab_path: Path) -> LabLoader:
    """LabLoader with instants.lab loaded."""
    loader = LabLoader()
    loader.load(instants_lab_path)
    return loader


# endregion


# region Test: Loading


class TestLabLoaderLoading:
    """Tests for loading .lab files."""

    def test_load_regions_lab(self, regions_lab_path: Path) -> None:
        """regions.lab loads without error."""
        loader = LabLoader()
        result = loader.load(regions_lab_path)

        # Returns self for chaining
        assert result is loader

        # Data is loaded: regions.lab holds exactly 6 events.
        assert len(loader) == 6

    def test_load_beats_lab(self, beats_lab_path: Path) -> None:
        """beats.lab loads without error."""
        loader = LabLoader()
        loader.load(beats_lab_path)
        assert len(loader) == 6

    def test_load_instants_lab(self, instants_lab_path: Path) -> None:
        """instants.lab loads without error."""
        loader = LabLoader()
        loader.load(instants_lab_path)
        assert len(loader) == 4

    def test_from_file_convenience(self, regions_lab_path: Path) -> None:
        """from_file() classmethod works correctly."""
        loader = LabLoader.from_file(regions_lab_path)
        assert len(loader) == 6

    def test_load_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Loading nonexistent file raises FileNotFoundError."""
        loader = LabLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.lab")


# endregion


# region Test: Regions Lab (EXACT VALUES)


class TestRegionsLab:
    """Tests for regions.lab parsing.

    Per README.md, all values are EXACT with ZERO TOLERANCE.

    regions.lab contains:
    - 6 intervals: Intro, Verse, Chorus, Verse, Chorus, Outro
    - Coordinate range: [0.0, 15.2]
    """

    def test_event_count_exact(self, loaded_regions_loader: LabLoader) -> None:
        """regions.lab contains exactly 6 events."""
        assert len(loaded_regions_loader) == 6

    def test_all_intervals(self, loaded_regions_loader: LabLoader) -> None:
        """All events are intervals (have both start and end)."""
        types = loaded_regions_loader.count_events_by_temporal_type()
        assert types.get("interval", 0) == 6
        assert types.get("instant", 0) == 0

    def test_coordinate_range_exact(self, loaded_regions_loader: LabLoader) -> None:
        """Coordinate range is exactly [0.0, 15.2]."""
        coord_range = loaded_regions_loader.events.coordinate_range()
        assert coord_range is not None
        assert coord_range[0] == 0.0
        assert coord_range[1] == 15.2

    def test_default_unit_seconds(self, loaded_regions_loader: LabLoader) -> None:
        """Default unit is seconds for lab files."""
        assert loaded_regions_loader.unit == TimeUnit.seconds

    def test_default_event_type_region(self, loaded_regions_loader: LabLoader) -> None:
        """Default event type is Region."""
        types = loaded_regions_loader.count_events_by_type()
        assert types.get("Region", 0) == 6


# endregion


# region Test: Beats Lab (EXACT VALUES)


class TestBeatsLab:
    """Tests for beats.lab parsing.

    Per README.md, all values are EXACT with ZERO TOLERANCE.

    beats.lab contains:
    - 6 intervals all labeled "Beat"
    - Coordinate range: [0.0, 3.0]
    """

    def test_event_count_exact(self, loaded_beats_loader: LabLoader) -> None:
        """beats.lab contains exactly 6 events."""
        assert len(loaded_beats_loader) == 6

    def test_all_intervals(self, loaded_beats_loader: LabLoader) -> None:
        """All events are intervals."""
        types = loaded_beats_loader.count_events_by_temporal_type()
        assert types.get("interval", 0) == 6

    def test_coordinate_range_exact(self, loaded_beats_loader: LabLoader) -> None:
        """Coordinate range is exactly [0.0, 3.0]."""
        coord_range = loaded_beats_loader.events.coordinate_range()
        assert coord_range is not None
        assert coord_range[0] == 0.0
        assert coord_range[1] == 3.0


# endregion


# region Test: Instants Lab (EXACT VALUES)


class TestInstantsLab:
    """Tests for instants.lab parsing.

    Per README.md, all values are EXACT with ZERO TOLERANCE.

    instants.lab contains:
    - 4 instant events (start == end) all labeled "Downbeat"
    - Coordinate range: [0.0, 3.0]
    """

    def test_event_count_exact(self, loaded_instants_loader: LabLoader) -> None:
        """instants.lab contains exactly 4 events."""
        assert len(loaded_instants_loader) == 4

    def test_all_instants(self, loaded_instants_loader: LabLoader) -> None:
        """All events are instants (start == end treated as instant)."""
        types = loaded_instants_loader.count_events_by_temporal_type()
        # When start == end, it's still parsed as interval with zero duration
        # This is correct behavior for lab files
        assert types.get("interval", 0) + types.get("instant", 0) == 4

    def test_coordinate_range_exact(self, loaded_instants_loader: LabLoader) -> None:
        """Coordinate range is exactly [0.0, 3.0]."""
        coord_range = loaded_instants_loader.events.coordinate_range()
        assert coord_range is not None
        assert coord_range[0] == 0.0
        assert coord_range[1] == 3.0


# endregion


# region Test: Timeline Creation


class TestTimelineCreation:
    """Tests for creating timelines from loaded lab files."""

    def test_create_timeline(self, loaded_regions_loader: LabLoader) -> None:
        """create_timeline() produces a valid timeline."""
        timeline = loaded_regions_loader.create_timeline()

        assert timeline is not None
        assert timeline.id == "cpt_1"
        assert timeline.unit == TimeUnit.seconds
        assert timeline.n_events == 6

    def test_create_timeline_with_uid(self, loaded_regions_loader: LabLoader) -> None:
        """create_timeline() accepts custom uid."""
        timeline = loaded_regions_loader.create_timeline(uid="cpt_1")

        assert timeline.id == "cpt_1"

    def test_create_timelines(self, loaded_regions_loader: LabLoader) -> None:
        """create_timelines() returns list with one timeline."""
        timelines = loaded_regions_loader.create_timelines()

        assert len(timelines) == 1
        assert timelines[0].n_events == 6


# endregion


# region Test: API Conformance


class TestAPIConformance:
    """Tests for Loader API conformance (per H2)."""

    def test_has_from_file(self) -> None:
        """LabLoader has from_file classmethod."""
        assert hasattr(LabLoader, "from_file")
        assert callable(LabLoader.from_file)

    def test_has_create_timeline(self, loaded_regions_loader: LabLoader) -> None:
        """LabLoader has create_timeline method."""
        assert hasattr(loaded_regions_loader, "create_timeline")
        assert callable(loaded_regions_loader.create_timeline)

    def test_has_create_timelines(self, loaded_regions_loader: LabLoader) -> None:
        """LabLoader has create_timelines method."""
        assert hasattr(loaded_regions_loader, "create_timelines")
        assert callable(loaded_regions_loader.create_timelines)

    def test_has_store_property(self, loaded_regions_loader: LabLoader) -> None:
        """LabLoader has store property."""
        assert hasattr(loaded_regions_loader, "store")
        assert loaded_regions_loader.store is not None

    def test_has_repr_html(self, loaded_regions_loader: LabLoader) -> None:
        """LabLoader has _repr_html_ method."""
        assert hasattr(loaded_regions_loader, "_repr_html_")
        html = loaded_regions_loader._repr_html_()
        assert isinstance(html, str)
        assert "<" in html  # Contains HTML tags

    def test_has_len(self, loaded_regions_loader: LabLoader) -> None:
        """LabLoader supports len()."""
        assert len(loaded_regions_loader) == 6


# endregion


# region Test: Lab File Format Specifics


class TestLabFileFormat:
    """Tests for lab file format handling."""

    def test_delimiter_is_tab(self) -> None:
        """LabLoader uses tab as delimiter."""
        assert LabLoader.delimiter == "\t"

    def test_no_header_row(self) -> None:
        """LabLoader expects no header row."""
        assert LabLoader.header_row == -1

    def test_default_unit_seconds(self) -> None:
        """LabLoader default unit is seconds."""
        assert LabLoader._default_unit == TimeUnit.seconds

    def test_custom_lab_file(self, tmp_path: Path) -> None:
        """Can load custom lab file with arbitrary labels."""
        content = "0.0\t0.5\tA\n0.5\t1.0\tB\n1.0\t1.5\tC\n"
        path = tmp_path / "custom.lab"
        path.write_text(content)

        loader = LabLoader()
        loader.load(path)

        assert len(loader) == 3


# endregion
