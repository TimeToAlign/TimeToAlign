"""Tests for IIIFManifestLoader.

Tests use the SUPRA piano roll IIIF manifest as the primary test case.
All expected values are EXACT per the ZERO TOLERANCE validation policy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from timetoalign.loader.graphical.iiif import (
    IIIFCanvasInfo,
    IIIFManifestLoader,
)
from timetoalign.testdata import ensure_data
from timetoalign.timelines import DiscreteGraphicalTimeline, SegmentLine, Timeline

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

        # Manifest is now loaded: it holds one canvas.
        assert loader.n_canvases == 1

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
        assert label == "Meistersinger von Nürnberg : Vorspiel"

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
        assert canvas.id == (
            "https://purl.stanford.edu/fd660zf8362/iiif/canvas/"
            "cocina-fileSet-fd660zf8362-fd660zf8362_1"
        )


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


# region Real-world volumes (trimmed copies of the published manifests)

# Gallica, Presentation 2: six page images of one printed score.
GALLICA_MANIFEST_ID = (
    "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688/manifest.json"
)
GALLICA_LABEL = (
    "Exemplaire localisé à la Médiathèque Hector Berlioz (CNSMDP), D-9658 (1)"
)
GALLICA_CANVAS = "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688/canvas/f{}"
# (@id, label, width, height) as the file states them.
GALLICA_PAGES = [
    (GALLICA_CANVAS.format(1), "NP", 4564, 6174),
    (GALLICA_CANVAS.format(2), "2", 4512, 6174),
    (GALLICA_CANVAS.format(3), "3", 4564, 6174),
    (GALLICA_CANVAS.format(4), "4", 4512, 6174),
    (GALLICA_CANVAS.format(5), "5", 4564, 6174),
    (GALLICA_CANVAS.format(6), "NP", 4577, 6174),
]

# BnF, Presentation 3: a disc's four sides of sound, then four images.
BNF_MANIFEST_ID = (
    "https://openapi.bnf.fr/iiif/presentation/v3/ark:/12148/bpt6k88448791/manifest.json"
)
BNF_PREFIX = "https://openapi.bnf.fr/iiif/presentation/v3/ark:/12148/bpt6k88448791/"
# (id, duration) of the sound canvases.
BNF_SOUNDS = [
    (BNF_PREFIX + "canvas/4-4-6-2-4", 221.0),
    (BNF_PREFIX + "canvas/4-4-6-4-4", 219.0),
    (BNF_PREFIX + "canvas/4-6-6-2-4", 257.0),
    (BNF_PREFIX + "canvas/4-6-6-4-4", 155.0),
]
# (id, label, width, height) of the image canvases.
BNF_IMAGES = [
    (BNF_PREFIX + "f1/canvas", "5", 2677, 2677),
    (BNF_PREFIX + "f2/canvas", "6", 2677, 2677),
    (BNF_PREFIX + "f3/canvas", "recto", 2940, 2985),
    (BNF_PREFIX + "f4/canvas", "verso", 2940, 2985),
]


def _write(path: Path, manifest: dict) -> Path:
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.fixture
def gallica_path(tmp_path: Path) -> Path:
    """The Gallica manifest, trimmed to what the loader reads."""
    return _write(
        tmp_path / "bpt6k11620688_manifest.json",
        {
            "@context": "http://iiif.io/api/presentation/2/context.json",
            "@id": GALLICA_MANIFEST_ID,
            "@type": "sc:Manifest",
            "label": GALLICA_LABEL,
            "sequences": [
                {
                    "@type": "sc:Sequence",
                    "canvases": [
                        {
                            "@id": canvas_id,
                            "@type": "sc:Canvas",
                            "label": label,
                            "width": width,
                            "height": height,
                        }
                        for canvas_id, label, width, height in GALLICA_PAGES
                    ],
                }
            ],
        },
    )


@pytest.fixture
def bnf_path(tmp_path: Path) -> Path:
    """The BnF disc manifest, trimmed to what the loader reads."""
    sounds = [
        {"id": canvas_id, "type": "Canvas", "duration": duration}
        for canvas_id, duration in BNF_SOUNDS
    ]
    images = [
        {
            "id": canvas_id,
            "type": "Canvas",
            "label": {"fr": [label]},
            "width": width,
            "height": height,
        }
        for canvas_id, label, width, height in BNF_IMAGES
    ]
    return _write(
        tmp_path / "bpt6k88448791_manifest.json",
        {
            "@context": "http://iiif.io/api/presentation/3/context.json",
            "id": BNF_MANIFEST_ID,
            "type": "Manifest",
            "label": {"none": ["4 Mélodies classiques"]},
            "items": sounds + images,
        },
    )


def _layout(volume: Timeline) -> list[tuple[str, int, int, str, dict]]:
    """Each child's id, offset, length, name and statements, in order."""
    return [
        (
            child.id,
            offset.value,
            child.length.value,
            child.name,
            child.statements,
        )
        for _, offset, child in volume.iter_segments()
    ]


def _source(iri: str) -> dict[str, list[dict[str, str]]]:
    return {"dcterms:source": [{"id": iri}]}


class TestWholeVolume:
    """``canvases="all"``: every image laid end to end as a segment line."""

    def test_pages_along_the_height(self, gallica_path: Path) -> None:
        volume = IIIFManifestLoader().load(gallica_path).create_timeline(canvases="all")

        assert type(volume) is SegmentLine[DiscreteGraphicalTimeline]
        assert isinstance(volume, DiscreteGraphicalTimeline)
        assert volume.unit.value == "pixels"
        assert volume.id == "dgt_1"
        assert volume.name == GALLICA_LABEL
        assert volume.length.value == 37044
        assert volume.statements == _source(GALLICA_MANIFEST_ID)
        assert _layout(volume) == [
            ("dgt_2", 0, 6174, "NP", _source(GALLICA_CANVAS.format(1))),
            ("dgt_3", 6174, 6174, "2", _source(GALLICA_CANVAS.format(2))),
            ("dgt_4", 12348, 6174, "3", _source(GALLICA_CANVAS.format(3))),
            ("dgt_5", 18522, 6174, "4", _source(GALLICA_CANVAS.format(4))),
            ("dgt_6", 24696, 6174, "5", _source(GALLICA_CANVAS.format(5))),
            ("dgt_7", 30870, 6174, "NP", _source(GALLICA_CANVAS.format(6))),
        ]

    def test_pages_along_the_width(self, gallica_path: Path) -> None:
        volume = (
            IIIFManifestLoader()
            .load(gallica_path)
            .create_timeline(uid="score", name="volume", axis="width", canvases="all")
        )

        assert volume.id == "score"
        assert volume.name == "volume"
        assert volume.length.value == 27293
        assert [(offset, length) for _, offset, length, _, _ in _layout(volume)] == [
            (0, 4564),
            (4564, 4512),
            (9076, 4564),
            (13640, 4512),
            (18152, 4564),
            (22716, 4577),
        ]

    def test_other_dimension_stays_on_the_loader(self, gallica_path: Path) -> None:
        loader = IIIFManifestLoader().load(gallica_path)
        loader.create_timeline(canvases="all")

        assert [
            (canvas.id, canvas.label, canvas.width, canvas.height, canvas.duration)
            for canvas in loader.manifest_info.canvases
        ] == [(*page, None) for page in GALLICA_PAGES]

    def test_first_is_the_default(self, gallica_path: Path) -> None:
        timeline = IIIFManifestLoader().load(gallica_path).create_timeline()

        assert type(timeline) is DiscreteGraphicalTimeline
        assert timeline.length.value == 6174
        assert timeline.n_children == 0
        assert timeline.statements == {}

    def test_unknown_canvases_value_raises(self, gallica_path: Path) -> None:
        loader = IIIFManifestLoader().load(gallica_path)
        with pytest.raises(ValueError, match="canvases must be 'first' or 'all'"):
            loader.create_timeline(canvases="some")  # type: ignore[arg-type]

    def test_round_trip_through_a_manifest_file(
        self, gallica_path: Path, tmp_path: Path
    ) -> None:
        volume = IIIFManifestLoader().load(gallica_path).create_timeline(canvases="all")

        path = volume.to_file(tmp_path / "volume.yaml")
        read = Timeline.from_file(path)

        assert type(read) is SegmentLine[DiscreteGraphicalTimeline]
        assert read.id == "dgt_1"
        assert read.name == GALLICA_LABEL
        assert read.length.value == 37044
        assert read.statements == _source(GALLICA_MANIFEST_ID)
        assert _layout(read) == _layout(volume)


class TestTimeBasedCanvases:
    """Presentation 3 canvases with a duration and no width or height."""

    def test_parses_and_exposes_durations(self, bnf_path: Path) -> None:
        loader = IIIFManifestLoader().load(bnf_path)

        assert loader.n_canvases == 8
        assert loader.manifest_info.id == BNF_MANIFEST_ID
        assert [
            (canvas.id, canvas.width, canvas.height, canvas.duration)
            for canvas in loader.manifest_info.canvases
        ] == [
            (canvas_id, None, None, duration) for canvas_id, duration in BNF_SOUNDS
        ] + [
            (canvas_id, width, height, None)
            for canvas_id, _, width, height in BNF_IMAGES
        ]

    def test_first_canvas_time_based_refuses(self, bnf_path: Path) -> None:
        loader = IIIFManifestLoader().load(bnf_path)
        message = re.escape(
            f"Canvas {BNF_SOUNDS[0][0]} is time-based (duration 221.0) and states "
            "no width and height, so it has no extent in pixels"
        )
        with pytest.raises(ValueError, match=message):
            loader.create_timeline()
        with pytest.raises(ValueError, match=message):
            loader.create_timeline(axis="width", canvases="first")
        with pytest.raises(ValueError, match=message):
            _ = loader.dimensions
        with pytest.raises(ValueError, match=message):
            _ = loader.width
        with pytest.raises(ValueError, match=message):
            _ = loader.height

    def test_repr_on_time_based_first_canvas(self, bnf_path: Path) -> None:
        loader = IIIFManifestLoader().load(bnf_path)
        assert repr(loader) == "IIIFManifestLoader(canvases=8, duration=221.0)"

    def test_volume_holds_only_the_images(self, bnf_path: Path) -> None:
        volume = IIIFManifestLoader().load(bnf_path).create_timeline(canvases="all")

        assert volume.name == "4 Mélodies classiques"
        assert volume.length.value == 11324
        assert volume.statements == _source(BNF_MANIFEST_ID)
        assert _layout(volume) == [
            ("dgt_2", 0, 2677, "5", _source(BNF_IMAGES[0][0])),
            ("dgt_3", 2677, 2677, "6", _source(BNF_IMAGES[1][0])),
            ("dgt_4", 5354, 2985, "recto", _source(BNF_IMAGES[2][0])),
            ("dgt_5", 8339, 2985, "verso", _source(BNF_IMAGES[3][0])),
        ]

    def test_volume_without_images_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "sound.json",
            {
                "@context": "http://iiif.io/api/presentation/3/context.json",
                "id": BNF_MANIFEST_ID,
                "type": "Manifest",
                "items": [
                    {"id": canvas_id, "type": "Canvas", "duration": duration}
                    for canvas_id, duration in BNF_SOUNDS
                ],
            },
        )
        loader = IIIFManifestLoader().load(path)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Manifest {BNF_MANIFEST_ID} has no canvas with a width and height: "
                "all 4 canvas(es) are time-based"
            ),
        ):
            loader.create_timeline(canvases="all")

    def test_canvas_without_extent_or_duration_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "bare.json",
            {
                "@context": "http://iiif.io/api/presentation/3/context.json",
                "id": "http://example.org/manifest",
                "type": "Manifest",
                "items": [
                    {"id": "http://example.org/canvas/1", "type": "Canvas", "width": 10}
                ],
            },
        )
        with pytest.raises(
            ValueError,
            match="Canvas http://example.org/canvas/1 states neither width and "
            "height nor a duration",
        ):
            IIIFManifestLoader().load(path)


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
