"""Tests for RepoVizzLoader (XML manifest and legacy CSV modes).

This module tests the RepoVizzLoader against the EEP multimodal dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from timetoalign.core import TimeUnit
from timetoalign.loader.physical.repovizz import RepoVizzInfo, RepoVizzLoader
from timetoalign.loader.physical.repovizz_catalogue import CatalogueEntry
from timetoalign.loader.physical.repovizz_store import RepovizzDictStore
from timetoalign.testdata import ensure_data
from timetoalign.timelines import DiscretePhysicalTimeline, TimelineGroup

# region Test Data Paths


@pytest.fixture(scope="module")
def multimodal_dir() -> Path:
    """Path to the multimodal test data directory."""
    return ensure_data("score") / "beethoven_op18-4iv_multimodal"


@pytest.fixture(scope="module")
def normal_recording_dir(multimodal_dir: Path) -> Path:
    """Path to the Normal recording directory."""
    return multimodal_dir / "StringQuartetEEP_I_Normal"


@pytest.fixture(scope="module")
def xml_manifest_path(normal_recording_dir: Path) -> Path:
    """Path to the XML manifest file."""
    return normal_recording_dir / "StringQuartetEEP_I_Normal.xml"


@pytest.fixture
def csv_file_path(normal_recording_dir: Path) -> Path:
    """Path to a sample CSV file (bowing velocity)."""
    return normal_recording_dir / "vln1_bow_vel.csv"


@pytest.fixture(scope="module")
def audio_timeline_group(xml_manifest_path: Path) -> TimelineGroup | None:
    """Create the XML audio TimelineGroup once for read-only assertions."""
    loader = RepoVizzLoader.from_file(xml_manifest_path)
    if "audio" not in loader.groups:
        return None
    return loader.create_group(category="audio")


# endregion


# region CatalogueEntry Tests


class TestCatalogueEntry:
    """Tests for the CatalogueEntry dataclass."""

    def test_basic_creation(self) -> None:
        """CatalogueEntry can be created with required fields."""
        entry = CatalogueEntry(
            xml_id="test_id",
            name="Test Entry",
            category="Audio",
            group="audio",
            subgroup="ambient",
            filename="test.wav",
            file_type="BWF",
            sample_rate=44100.0,
            n_samples=441000,
        )
        assert entry.xml_id == "test_id"
        assert entry.name == "Test Entry"
        assert entry.is_signal is True
        assert entry.duration_seconds == 10.0

    def test_non_signal_entry(self) -> None:
        """CatalogueEntry with no samples is not a signal."""
        entry = CatalogueEntry(
            xml_id="annotation",
            name="Score Annotation",
            category="Annotation",
            group="score",
            subgroup="",
            filename="score.notes",
            file_type="NOTES",
            sample_rate=0.0,
            n_samples=0,
        )
        assert entry.is_signal is False
        assert entry.duration_seconds == 0.0

    def test_frozen_immutable(self) -> None:
        """CatalogueEntry is immutable (frozen dataclass)."""
        entry = CatalogueEntry(
            xml_id="test",
            name="Test",
            category="Test",
            group="test",
            subgroup="",
            filename=None,
            file_type="",
            sample_rate=0.0,
            n_samples=0,
        )
        with pytest.raises(AttributeError):
            entry.xml_id = "changed"  # type: ignore[misc]


# endregion


# region RepovizzDictStore Tests


class TestRepovizzDictStore:
    """Tests for the RepovizzDictStore."""

    def test_empty_store(self) -> None:
        """Empty store has no entries."""
        store = RepovizzDictStore()
        assert len(store.catalogue) == 0
        assert store.groups == []
        assert store.audio == []
        assert store.score == []
        assert store.descriptors == []
        assert store.mocap == []

    def test_catalogue_with_entries(self) -> None:
        """Store with catalogue returns correct group lists."""
        entries = {
            "audio1": CatalogueEntry(
                xml_id="audio1",
                name="Audio 1",
                category="Ambient",
                group="audio",
                subgroup="ambient",
                filename="audio.wav",
                file_type="BWF",
                sample_rate=44100.0,
                n_samples=44100,
            ),
            "score1": CatalogueEntry(
                xml_id="score1",
                name="Score 1",
                category="Annotation",
                group="score",
                subgroup="",
                filename="score.notes",
                file_type="NOTES",
                sample_rate=0.0,
                n_samples=0,
            ),
        }
        store = RepovizzDictStore(catalogue=entries)
        assert store.audio == ["audio1"]
        assert store.score == ["score1"]
        assert set(store.groups) == {"audio", "score"}


# endregion


# region RepoVizzLoader XML Mode Tests


class TestRepoVizzLoaderXmlMode:
    """Tests for RepoVizzLoader in XML manifest mode."""

    def test_load_xml_manifest(self, xml_manifest_path: Path) -> None:
        """XML manifest loads successfully."""
        loader = RepoVizzLoader.from_file(xml_manifest_path)
        assert loader.is_xml_mode is True
        assert len(loader.catalogue) > 0
        assert len(loader.timeline_ids) > 0

    def test_xml_groups(self, xml_manifest_path: Path) -> None:
        """XML manifest has expected groups."""
        loader = RepoVizzLoader.from_file(xml_manifest_path)
        groups = loader.groups
        # The EEP dataset has these categories
        assert "audio" in groups
        assert "score" in groups
        # May also have descriptors and mocap

    def test_catalogue_entry_lookup(self, xml_manifest_path: Path) -> None:
        """Catalogue entries can be looked up by ID."""
        loader = RepoVizzLoader.from_file(xml_manifest_path)
        first_id = loader.timeline_ids[0]
        entry = loader.get_entry(first_id)
        assert isinstance(entry, CatalogueEntry)
        assert entry.xml_id == first_id

    def test_create_timeline_xml(self, xml_manifest_path: Path) -> None:
        """Timeline can be created from XML catalogue entry."""
        loader = RepoVizzLoader.from_file(xml_manifest_path)

        # Find an audio entry (which has samples)
        audio_ids = loader.store.audio
        if audio_ids:
            tl = loader.create_timeline(entry=audio_ids[0])
            assert isinstance(tl, DiscretePhysicalTimeline)
            assert tl.unit == TimeUnit.samples
            # Audio entry should have samples
            entry = loader.get_entry(audio_ids[0])
            if entry.is_signal:
                assert tl.length.value == entry.n_samples

            overridden = loader.create_timeline(
                entry=audio_ids[0], uid="repovizz-entry-roundtrip"
            )
            assert overridden.id == "repovizz-entry-roundtrip"

    def test_create_timelines_filtered(self, xml_manifest_path: Path) -> None:
        """Multiple timelines can be created with ID pattern filter."""
        loader = RepoVizzLoader.from_file(xml_manifest_path)

        # Filter to audio entries only
        audio_ids = loader.store.audio
        if len(audio_ids) >= 2:
            timelines = loader.create_timelines(entries=audio_ids[:2])
            assert len(timelines) == 2

    @pytest.mark.slow
    def test_create_group(self, audio_timeline_group: TimelineGroup | None) -> None:
        """TimelineGroup can be created from XML loader."""
        # Get audio group if available
        if audio_timeline_group is not None:
            group = audio_timeline_group
            assert group.id == "repovizz:audio"
            assert len(group) > 0

    def test_create_group_entries(self, xml_manifest_path: Path) -> None:
        """TimelineGroup accepts explicit catalogue entries."""
        loader = RepoVizzLoader.from_file(xml_manifest_path)
        entries = loader.store.audio[:2]
        group = loader.create_group(entries=entries)
        assert [timeline.id for timeline in group] == entries

    def test_find_descriptor(self, xml_manifest_path: Path) -> None:
        """Descriptor can be found by name."""
        loader = RepoVizzLoader.from_file(xml_manifest_path)

        # Try to find a bowing descriptor
        desc_id = loader.find_descriptor("bow_vel")
        if desc_id:
            entry = loader.get_entry(desc_id)
            assert (
                "bow_vel" in entry.name.lower() or "bow_vel" in entry.filename.lower()
            )

    def test_repr_xml_mode(self, xml_manifest_path: Path) -> None:
        """Repr shows XML mode info."""
        loader = RepoVizzLoader.from_file(xml_manifest_path)
        repr_str = repr(loader)
        assert "xml" in repr_str.lower()
        assert "entries" in repr_str.lower()


# endregion


# region RepoVizzLoader Legacy CSV Mode Tests


class TestRepoVizzLoaderCsvMode:
    """Tests for RepoVizzLoader in legacy CSV mode."""

    def test_load_csv_file(self, csv_file_path: Path) -> None:
        """CSV file loads successfully."""
        loader = RepoVizzLoader.from_file(csv_file_path)
        assert loader.is_xml_mode is False
        assert loader.n_samples > 0
        assert loader.frame_rate > 0

    def test_csv_info_properties(self, csv_file_path: Path) -> None:
        """CSV loader provides info properties."""
        loader = RepoVizzLoader.from_file(csv_file_path)
        info = loader.info
        assert isinstance(info, RepoVizzInfo)
        assert info.n_samples == loader.n_samples
        assert info.frame_rate == loader.frame_rate
        assert info.duration_seconds == loader.n_samples / loader.frame_rate

    def test_create_timeline_csv(self, csv_file_path: Path) -> None:
        """Timeline can be created from CSV."""
        loader = RepoVizzLoader.from_file(csv_file_path)
        tl = loader.create_timeline()

        assert isinstance(tl, DiscretePhysicalTimeline)
        assert tl.unit == TimeUnit.samples
        assert tl.length.value == loader.n_samples

    def test_csv_timeline_has_cmap(self, csv_file_path: Path) -> None:
        """CSV timeline has SamplesToSeconds conversion map."""
        loader = RepoVizzLoader.from_file(csv_file_path)
        tl = loader.create_timeline()

        # Check that a conversion map was attached
        assert len(tl._conversion_maps) > 0

    def test_csv_timeline_custom_uid(self, csv_file_path: Path) -> None:
        """CSV timeline can have custom uid."""
        loader = RepoVizzLoader.from_file(csv_file_path)
        tl = loader.create_timeline(uid="custom_id")
        assert tl.id == "custom_id"

    def test_repr_csv_mode(self, csv_file_path: Path) -> None:
        """Repr shows CSV mode info."""
        loader = RepoVizzLoader.from_file(csv_file_path)
        repr_str = repr(loader)
        assert "samples" in repr_str.lower() or str(loader.n_samples) in repr_str

    def test_csv_invalid_file(self, normal_recording_dir: Path) -> None:
        """Invalid CSV file raises appropriate error."""
        # Try to load the XML as CSV (should fail because no framerate)
        xml_path = normal_recording_dir / "StringQuartetEEP_I_Normal.xml"
        with pytest.raises((ValueError, FileNotFoundError)):
            loader = RepoVizzLoader()
            # Force CSV mode by not using .xml extension
            loader._load_source(xml_path)


# endregion


# region Edge Cases and Error Handling


class TestRepoVizzLoaderErrors:
    """Tests for error handling in RepoVizzLoader."""

    def test_not_loaded_info_raises(self) -> None:
        """Accessing info without loading raises RuntimeError."""
        loader = RepoVizzLoader()
        with pytest.raises(RuntimeError, match="No CSV file loaded"):
            _ = loader.info

    def test_xml_mode_requires_uid_for_multiple(self, xml_manifest_path: Path) -> None:
        """XML mode with multiple entries requires uid for create_timeline."""
        loader = RepoVizzLoader.from_file(xml_manifest_path)
        if len(loader.timeline_ids) > 1:
            with pytest.raises(ValueError, match="Must specify entry_id"):
                loader.create_timeline()

    def test_invalid_entry_id_raises(self, xml_manifest_path: Path) -> None:
        """Invalid catalogue entry ID raises KeyError."""
        loader = RepoVizzLoader.from_file(xml_manifest_path)
        with pytest.raises(KeyError):
            loader.get_entry("nonexistent_id_12345")

    def test_clear_resets_state(self, xml_manifest_path: Path) -> None:
        """Clear resets all loader state."""
        loader = RepoVizzLoader.from_file(xml_manifest_path)
        assert len(loader.catalogue) > 0

        loader.clear()
        assert len(loader.catalogue) == 0
        assert len(loader.sources) == 0
        assert loader.is_xml_mode is False


# endregion


# region Integration Tests


class TestRepoVizzLoaderIntegration:
    """Integration tests for RepoVizzLoader."""

    def test_multiple_recordings(self, multimodal_dir: Path) -> None:
        """Two recordings load into identically-shaped catalogues that
        nevertheless carry distinct audio content."""
        normal_xml = (
            multimodal_dir
            / "StringQuartetEEP_I_Normal"
            / "StringQuartetEEP_I_Normal.xml"
        )
        mechanical_xml = (
            multimodal_dir
            / "StringQuartetEEP_I_Mechanical"
            / "StringQuartetEEP_I_Mechanical.xml"
        )

        loader1 = RepoVizzLoader.from_file(normal_xml)
        loader2 = RepoVizzLoader.from_file(mechanical_xml)

        # Each recording catalogues exactly 376 entries, 232 of them timelines.
        assert len(loader1.catalogue) == 376
        assert len(loader2.catalogue) == 376
        assert len(loader1.timeline_ids) == 232
        assert len(loader2.timeline_ids) == 232

        # Identical modality structure across the two recordings.
        assert loader1.groups == ["audio", "descriptors", "mocap", "score"]
        assert loader1.groups == loader2.groups
        assert len(loader1.store.audio) == 144
        assert len(loader2.store.audio) == 144
        assert len(loader1.store.score) == 4
        assert len(loader2.store.score) == 4

        # ...but the recordings themselves differ: the shared cardioid-ambient
        # audio entry is longer in the Mechanical take than in the Normal one.
        cardioid_id = "ROOT0_Audi1_Audi0_Ambi0"
        normal_cardioid = loader1.get_entry(cardioid_id)
        mechanical_cardioid = loader2.get_entry(cardioid_id)
        assert normal_cardioid.name == "Cardioid microphone"
        assert mechanical_cardioid.name == "Cardioid microphone"
        assert normal_cardioid.sample_rate == 44100.0
        assert mechanical_cardioid.sample_rate == 44100.0
        assert normal_cardioid.n_samples == 11753638
        assert mechanical_cardioid.n_samples == 12426696

    @pytest.mark.slow
    def test_timeline_group_iteration(
        self, audio_timeline_group: TimelineGroup | None
    ) -> None:
        """TimelineGroup from loader can be iterated."""
        if audio_timeline_group is not None:
            group = audio_timeline_group
            timelines = list(group)
            assert len(timelines) > 0
            for tl in timelines:
                assert isinstance(tl, DiscretePhysicalTimeline)


# endregion
