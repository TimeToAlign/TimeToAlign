"""Tests for IIIFManifestLoader.

Tests use the SUPRA piano roll IIIF manifest as the primary test case.
All expected values are EXACT per the ZERO TOLERANCE validation policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from timetoalign.loader.graphical.iiif import (
    IIIFCanvasInfo,
    IIIFManifestLoader,
)
from timetoalign.testdata import ensure_data
from timetoalign.timelines import DiscreteGraphicalTimeline

# Test data directory
SUPRA_DIR = ensure_data("supra")
IIIF_MANIFEST = SUPRA_DIR / "image" / "ifff_manifest.json"


# region Fixtures


@pytest.fixture
def supra_manifest_path() -> Path:
    """Path to the SUPRA IIIF manifest."""
    return IIIF_MANIFEST


@pytest.fixture
def loaded_supra_loader() -> IIIFManifestLoader:
    """IIIFManifestLoader with SUPRA manifest loaded."""
    loader = IIIFManifestLoader()
    loader.load(IIIF_MANIFEST)
    return loader


# endregion


# region Test: Loading


class TestIIIFManifestLoading:
    """Tests for loading IIIF manifests."""

    def test_load_supra_manifest(self, supra_manifest_path: Path) -> None:
        """SUPRA manifest loads without error."""
        loader = IIIFManifestLoader()
        result = loader.load(supra_manifest_path)

        # Returns self for chaining
        assert result is loader

        # Manifest is now loaded
        assert loader._manifest_info is not None

    def test_load_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Loading nonexistent file raises FileNotFoundError."""
        loader = IIIFManifestLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.json")

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        """Loading invalid JSON raises JSONDecodeError."""
        bad_file = tmp_path / "invalid.json"
        bad_file.write_text("not valid json {")

        loader = IIIFManifestLoader()
        with pytest.raises(json.JSONDecodeError):
            loader.load(bad_file)

    def test_load_manifest_no_canvases_raises(self, tmp_path: Path) -> None:
        """Loading manifest with no canvases raises ValueError."""
        empty_manifest = tmp_path / "empty.json"
        empty_manifest.write_text(
            json.dumps(
                {
                    "@context": "http://iiif.io/api/presentation/2/context.json",
                    "@id": "http://example.org/manifest",
                    "sequences": [{"canvases": []}],
                }
            )
        )

        loader = IIIFManifestLoader()
        with pytest.raises(ValueError, match="no canvases"):
            loader.load(empty_manifest)


# endregion


# region Test: SUPRA Manifest Dimensions (EXACT VALUES)


class TestSUPRAManifestDimensions:
    """Tests for SUPRA manifest dimension extraction.

    Per README.md, all values are EXACT with ZERO TOLERANCE.
    """

    def test_supra_dimensions_exact(
        self, loaded_supra_loader: IIIFManifestLoader
    ) -> None:
        """SUPRA manifest extracts exact dimensions from canvas."""
        dims = loaded_supra_loader.dimensions

        # EXACT values from SUPRA metadata
        assert dims["width"] == 4096
        assert dims["height"] == 299400

    def test_supra_width_exact(self, loaded_supra_loader: IIIFManifestLoader) -> None:
        """Width property returns exact value."""
        assert loaded_supra_loader.width == 4096

    def test_supra_height_exact(self, loaded_supra_loader: IIIFManifestLoader) -> None:
        """Height property returns exact value."""
        assert loaded_supra_loader.height == 299400

    def test_supra_single_canvas(self, loaded_supra_loader: IIIFManifestLoader) -> None:
        """SUPRA manifest has exactly one canvas."""
        assert loaded_supra_loader.n_canvases == 1

    def test_create_timeline_returns_discrete_graphical_timeline(
        self, loaded_supra_loader: IIIFManifestLoader
    ) -> None:
        """IIIF pixel material uses the discrete graphical concrete class."""
        timeline = loaded_supra_loader.create_timeline()

        assert type(timeline) is DiscreteGraphicalTimeline
        assert timeline.length.value == 299400


# endregion


# region Test: Manifest Metadata


class TestManifestMetadata:
    """Tests for manifest metadata extraction."""

    def test_supra_manifest_label(
        self, loaded_supra_loader: IIIFManifestLoader
    ) -> None:
        """SUPRA manifest has a label."""
        # The label is "Meistersinger von Nürnberg : Vorspiel"
        label = loaded_supra_loader.label
        assert label is not None
        assert "Meistersinger" in label

    def test_supra_manifest_id(self, loaded_supra_loader: IIIFManifestLoader) -> None:
        """SUPRA manifest has Stanford PURL as ID."""
        manifest_id = loaded_supra_loader.manifest_info.id
        assert "stanford.edu" in manifest_id
        assert "fd660zf8362" in manifest_id

    def test_supra_metadata_title(
        self, loaded_supra_loader: IIIFManifestLoader
    ) -> None:
        """SUPRA manifest metadata includes title."""
        # Check manifest-level metadata
        metadata = loaded_supra_loader.manifest_info.metadata
        assert "Title" in metadata

    def test_supra_metadata_contributor(
        self, loaded_supra_loader: IIIFManifestLoader
    ) -> None:
        """SUPRA manifest metadata includes Contributor key."""
        metadata = loaded_supra_loader.manifest_info.metadata
        # Note: IIIF 2.x manifests can have duplicate keys (e.g., multiple Contributors)
        # Our parser currently keeps only the last value for each key.
        # Just verify the Contributor key exists.
        assert "Contributor" in metadata

    def test_get_metadata_helper(self, loaded_supra_loader: IIIFManifestLoader) -> None:
        """get_metadata helper returns correct values."""
        title = loaded_supra_loader.get_metadata("Title")
        assert title is not None
        assert "Meistersinger" in title


# endregion


# region Test: Canvas Access


class TestCanvasAccess:
    """Tests for accessing individual canvas information."""

    def test_get_canvas_by_index(self, loaded_supra_loader: IIIFManifestLoader) -> None:
        """Can access canvas by index."""
        canvas = loaded_supra_loader.get_canvas(0)

        assert isinstance(canvas, IIIFCanvasInfo)
        assert canvas.width == 4096
        assert canvas.height == 299400

    def test_get_canvas_out_of_range_raises(
        self, loaded_supra_loader: IIIFManifestLoader
    ) -> None:
        """Accessing canvas out of range raises IndexError."""
        with pytest.raises(IndexError):
            loaded_supra_loader.get_canvas(999)

    def test_canvas_has_id(self, loaded_supra_loader: IIIFManifestLoader) -> None:
        """Canvas has a valid ID."""
        canvas = loaded_supra_loader.get_canvas(0)
        assert canvas.id is not None
        assert len(canvas.id) > 0


# endregion


# region Test: Unloaded State


class TestUnloadedState:
    """Tests for behavior when no manifest is loaded."""

    def test_manifest_info_raises_before_load(self) -> None:
        """Accessing manifest_info before loading raises RuntimeError."""
        loader = IIIFManifestLoader()
        with pytest.raises(RuntimeError, match="No manifest loaded"):
            _ = loader.manifest_info

    def test_dimensions_raises_before_load(self) -> None:
        """Accessing dimensions before loading raises RuntimeError."""
        loader = IIIFManifestLoader()
        with pytest.raises(RuntimeError, match="No manifest loaded"):
            _ = loader.dimensions

    def test_repr_before_load(self) -> None:
        """Repr works before loading."""
        loader = IIIFManifestLoader()
        assert "not loaded" in repr(loader)

    def test_repr_after_load(self, loaded_supra_loader: IIIFManifestLoader) -> None:
        """Repr shows dimensions after loading."""
        repr_str = repr(loaded_supra_loader)
        assert "4096" in repr_str
        assert "299400" in repr_str


# endregion


# region Test: IIIF 3.x Format


class TestIIIF3Format:
    """Tests for IIIF Presentation API 3.x format support."""

    def test_parse_v3_manifest(self, tmp_path: Path) -> None:
        """Can parse IIIF 3.x format manifests."""
        v3_manifest = tmp_path / "v3_manifest.json"
        v3_manifest.write_text(
            json.dumps(
                {
                    "@context": "http://iiif.io/api/presentation/3/context.json",
                    "id": "http://example.org/manifest",
                    "type": "Manifest",
                    "label": {"en": ["Test Manifest"]},
                    "items": [
                        {
                            "id": "http://example.org/canvas/1",
                            "type": "Canvas",
                            "width": 1000,
                            "height": 2000,
                            "label": {"en": ["Page 1"]},
                        }
                    ],
                }
            )
        )

        loader = IIIFManifestLoader()
        loader.load(v3_manifest)

        assert loader.width == 1000
        assert loader.height == 2000
        assert loader.label == "Test Manifest"
        assert loader.n_canvases == 1

    def test_parse_v3_multi_canvas(self, tmp_path: Path) -> None:
        """Can parse IIIF 3.x manifests with multiple canvases."""
        v3_manifest = tmp_path / "v3_multi.json"
        v3_manifest.write_text(
            json.dumps(
                {
                    "@context": "http://iiif.io/api/presentation/3/context.json",
                    "id": "http://example.org/manifest",
                    "type": "Manifest",
                    "items": [
                        {
                            "id": "http://example.org/canvas/1",
                            "type": "Canvas",
                            "width": 100,
                            "height": 200,
                        },
                        {
                            "id": "http://example.org/canvas/2",
                            "type": "Canvas",
                            "width": 300,
                            "height": 400,
                        },
                    ],
                }
            )
        )

        loader = IIIFManifestLoader()
        loader.load(v3_manifest)

        assert loader.n_canvases == 2
        # dimensions returns first canvas
        assert loader.width == 100
        assert loader.height == 200
        # Can access second canvas
        assert loader.get_canvas(1).width == 300


# endregion


# region Test: Source Path Tracking


class TestSourcePathTracking:
    """Tests for tracking the loaded manifest path."""

    def test_source_path_none_before_load(self) -> None:
        """Source path is None before loading."""
        loader = IIIFManifestLoader()
        assert loader.source_path is None

    def test_source_path_after_load(self, supra_manifest_path: Path) -> None:
        """Source path is set after loading."""
        loader = IIIFManifestLoader()
        loader.load(supra_manifest_path)

        assert loader.source_path == supra_manifest_path


# endregion
