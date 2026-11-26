"""Tests for IIIFManifestLoader.

Tests use the SUPRA piano roll IIIF manifest as the primary test case.
All expected values are EXACT per the ZERO TOLERANCE validation policy.
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from tests.loader.minting import shape
from timetoalign.loader.graphical.iiif import (
    IIIFCanvasInfo,
    IIIFManifestLoader,
)
from timetoalign.loader.graphical.paths import (
    HorizontalLinePath,
    PiecewisePath,
    TimeAxisPath,
    VerticalLinePath,
)
from timetoalign.testdata import ensure_data
from timetoalign.timelines import DiscreteGraphicalTimeline, SegmentLine, Timeline

# Test data directory
SUPRA_DIR = ensure_data("supra")
IIIF_MANIFEST = SUPRA_DIR / "image" / "ifff_manifest.json"
SUPRA_CANVAS = (
    "https://purl.stanford.edu/fd660zf8362/iiif/canvas/"
    "cocina-fileSet-fd660zf8362-fd660zf8362_1"
)


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

    def test_roll_along_a_vertical_line(
        self, loaded_supra_loader: IIIFManifestLoader
    ) -> None:
        """A piano roll is one plain timeline as long as its vertical path."""
        timelines = loaded_supra_loader.create_timelines(
            VerticalLinePath(x=2048, y0=0, y1=299400, tolerance=2048)
        )

        assert len(timelines) == 1
        (roll,) = timelines
        assert type(roll) is DiscreteGraphicalTimeline
        assert roll.unit.value == "pixels"
        assert shape(roll.id) == "dgt"
        assert roll.name == "display image"
        assert roll.length.value == 299400
        assert type(roll.length.value) is int
        assert roll.is_locked is True
        assert roll.n_children == 0
        assert roll.meta == {"image": SUPRA_CANVAS}
        assert roll.statements == {"dcterms:source": [{"id": SUPRA_CANVAS}]}

    def test_roll_length_is_the_path_not_the_image(
        self, loaded_supra_loader: IIIFManifestLoader
    ) -> None:
        (half,) = loaded_supra_loader.create_timelines(
            VerticalLinePath(x=2048, y0=0, y1=149700, tolerance=2048)
        )

        assert half.length.value == 149700


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


_NEUMA = "https://neuma.huma-num.fr/home/opus/all:collabscore:royaumont:concert:C130_0"


def _fr(text: str) -> dict[str, list[str]]:
    return {"fr": [text]}


#: Neuma's synchronisation manifest of *Tristesse* (C130_0), reduced: no
#: ``@context``, Presentation 3 language maps and ``items``.
NEUMA_COMBINED: dict[str, Any] = {
    "id": _NEUMA,
    "type": "Manifest",
    "label": _fr("[Tristesse] "),
    "metadata": [
        {"label": _fr("title"), "value": _fr("[Tristesse] ")},
        {"label": _fr("creator"), "value": _fr("Camille  Saint-Saëns (1835, 1921)")},
        {
            "label": _fr("MusicXML"),
            "value": _fr(
                "https://neuma.huma-num.fr//rest/collections/"
                "all:collabscore:royaumont:concert:C130_0/_sources/musicxml/_file/"
            ),
        },
        {
            "label": _fr("MEI"),
            "value": _fr(
                "https://neuma.huma-num.fr//rest/collections/"
                "all:collabscore:royaumont:concert:C130_0/_sources/mei/_file/"
            ),
        },
    ],
    "items": [
        {
            "id": f"{_NEUMA}/canvas",
            "type": "Canvas",
            "label": _fr("Combined image-audio canvas"),
            "height": 6248,
            "width": 4772,
            "duration": 167.0,
            "items": [
                {
                    "id": f"{_NEUMA}/list-media",
                    "type": "AnnotationPage",
                    "items": [
                        {
                            "id": f"{_NEUMA}/video",
                            "type": "Annotation",
                            "motivation": "painting",
                            "body": {
                                "id": "https://mediaserver.lecnam.net/downloads/file/"
                                "v126b13dde87aoc3161j/?url=media_1080_gbLK8JPpJe.mp4",
                                "type": "Video",
                                "duration": 167.0,
                                "format": "video/mpeg",
                            },
                            "target": f"{_NEUMA}/canvas",
                        },
                        {
                            "id": f"{_NEUMA}/img3",
                            "type": "Annotation",
                            "motivation": "painting",
                            "body": {
                                "id": "https://gallica.bnf.fr/iiif/ark:/12148/"
                                "bpt6k11619682/f3/full/full/0/native.jpg",
                                "type": "Image",
                                "height": 6248,
                                "width": 4772,
                                "format": "image/jpeg",
                            },
                            "target": f"{_NEUMA}/canvas#t=0.0,43.50492598474121",
                        },
                    ],
                }
            ],
            "annotations": [
                {
                    "id": f"{_NEUMA}/synchro",
                    "type": "AnnotationPage",
                    "items": [
                        {
                            "id": f"{_NEUMA}/m1",
                            "type": "Annotation",
                            "motivation": "highlighting",
                            "body": {"type": "TextualBody", "value": "resource"},
                            "target": {
                                "type": "SpecificResource",
                                "source": f"{_NEUMA}/list-media",
                                "selector": [
                                    {
                                        "type": "SvgSelector",
                                        "value": '<svg xmlns="http://www.w3.org/2000/svg">'
                                        '<polygon points="1060,2049 2172,2049 1060,2852 '
                                        '2176,2852"/></svg>',
                                    },
                                    {
                                        "type": "FragmentSelector",
                                        "conformsTo": "https://www.w3.org/TR/media-frags/",
                                        "value": "t=0.0,6.167964144958496",
                                    },
                                ],
                            },
                        }
                    ],
                }
            ],
        }
    ],
}


def _version_refusal(path: Path, lists: str) -> str:
    return re.escape(
        f"Cannot tell the IIIF Presentation version of {path}: its @context names "
        f"none, and it lists {lists}"
    )


class TestManifestWithoutContext:
    """A manifest stating no ``@context`` is read by its structure."""

    def test_neuma_combined_manifest(self, tmp_path: Path) -> None:
        path = tmp_path / "combined_manifest.json"
        path.write_text(json.dumps(NEUMA_COMBINED), encoding="utf-8")

        loader = IIIFManifestLoader.from_file(path)
        info = loader.manifest_info
        assert info.id == _NEUMA
        assert info.label == "[Tristesse] "
        assert info.metadata["title"] == "[Tristesse] "
        assert info.metadata["creator"] == "Camille  Saint-Saëns (1835, 1921)"
        assert [
            (c.id, c.width, c.height, c.duration, c.label, c.is_spatial)
            for c in info.canvases
        ] == [
            (
                f"{_NEUMA}/canvas",
                4772,
                6248,
                167.0,
                "Combined image-audio canvas",
                True,
            )
        ]
        assert loader.dimensions == {"width": 4772, "height": 6248}

    def test_presentation_2_by_its_sequences(self, tmp_path: Path) -> None:
        path = tmp_path / "v2.json"
        canvases = [
            {
                "@id": f"http://example.org/canvas/{n}",
                "label": str(n),
                "width": w,
                "height": h,
            }
            for n, w, h in ((1, 100, 200), (2, 300, 400))
        ]
        path.write_text(
            json.dumps(
                {"@id": "http://example.org/m", "sequences": [{"canvases": canvases}]}
            )
        )

        info = IIIFManifestLoader.from_file(path).manifest_info
        assert [(c.id, c.width, c.height, c.label) for c in info.canvases] == [
            ("http://example.org/canvas/1", 100, 200, "1"),
            ("http://example.org/canvas/2", 300, 400, "2"),
        ]

    def test_neither_structure_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "reverie_audio_manifest.json"
        path.write_text(
            json.dumps(
                {
                    "opus_ref": "C144_0",
                    "source_ref": "audio",
                    "nb_frames": 0,
                    "time_frames": [],
                }
            )
        )
        with pytest.raises(
            ValueError,
            match="^"
            + _version_refusal(
                path,
                "neither 'items' (Presentation 3) nor 'sequences' (Presentation 2)",
            )
            + "$",
        ):
            IIIFManifestLoader.from_file(path)

    def test_both_structures_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "both.json"
        path.write_text(json.dumps({"items": [], "sequences": []}))
        with pytest.raises(
            ValueError,
            match="^"
            + _version_refusal(
                path, "both 'items' (Presentation 3) and 'sequences' (Presentation 2)"
            )
            + "$",
        ):
            IIIFManifestLoader.from_file(path)


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


def _layout(line: Timeline) -> list[tuple[str, int, int, str]]:
    """Each segment's id, offset, length and name, in order."""
    return [
        (segment.id, offset.value, segment.length.value, segment.name)
        for _, offset, segment in line.iter_segments()
    ]


def _minted_layout(line: Timeline) -> list[tuple[str, int, int, str]]:
    """:func:`_layout`, each segment's id minted and read as its shape."""
    return [(shape(id_), *rest) for id_, *rest in _layout(line)]


def _source(iri: str) -> dict[str, list[dict[str, str]]]:
    return {"dcterms:source": [{"id": iri}]}


def _stated(line: Timeline) -> list[tuple[dict, dict]]:
    """Each segment's ``meta`` and statements, in order."""
    return [
        (segment.meta, segment.statements) for _, _, segment in line.iter_segments()
    ]


def _strip(x: int, y: int, width: int, height: int) -> dict[str, int]:
    return {"x": x, "y": y, "width": width, "height": height}


def _systems(*extents: tuple[int, int], first: HorizontalLinePath | None = None):
    """One line per system over its x extent; system k at y 1000 k, strip 600 high."""
    pieces = [
        HorizontalLinePath(x0=x0, x1=x1, y=1000 * k, tolerance=300)
        for k, (x0, x1) in enumerate(extents, start=1)
    ]
    if first is not None:
        pieces[0] = first
    return PiecewisePath(tuple(pieces))


# The systems of the scan's four music pages, which lie on canvases 1 to 4.
SCAN_PATHS: dict[int, TimeAxisPath] = {
    1: _systems(
        (1221, 4101),
        (971, 4112),
        (983, 4125),
        (994, 4142),
        first=HorizontalLinePath(
            x0=1221, x1=4101, y=Fraction(3855, 2), tolerance=Fraction(677, 2)
        ),
    ),
    2: _systems((519, 3661), (522, 3659), (522, 3659), (523, 3658), (522, 3661)),
    3: _systems((913, 4060), (918, 4054), (913, 4051), (907, 4048)),
    4: _systems((508, 3650), (504, 3643), (494, 3634), (489, 3628)),
}
F2, F3, F4, F5 = (GALLICA_CANVAS.format(n) for n in (2, 3, 4, 5))
PAGE_1_ACCOLADES = [
    ("accolade_1", 0, 2880, _strip(1221, 1589, 2880, 677)),
    ("accolade_2", 2880, 3141, _strip(971, 1700, 3141, 600)),
    ("accolade_3", 6021, 3142, _strip(983, 2700, 3142, 600)),
    ("accolade_4", 9163, 3148, _strip(994, 3700, 3148, 600)),
]


def _gallica(path: Path) -> IIIFManifestLoader:
    return IIIFManifestLoader().load(path)


def _line(x0: int = 0, x1: int = 100, y: int = 50) -> HorizontalLinePath:
    return HorizontalLinePath(x0=x0, x1=x1, y=y, tolerance=10)


class TestVolumeAlongPaths:
    """``create_timeline``: the selected canvases, each along its time path."""

    def test_lengths_are_the_paths(self, gallica_path: Path) -> None:
        volume = _gallica(gallica_path).create_timeline(
            SCAN_PATHS, canvases=[1, 2, 3, 4]
        )

        assert volume.unit.value == "pixels"
        assert volume.length.value == 53123
        assert _minted_layout(volume) == [
            ("dgt", 0, 12311, "2"),
            ("dgt", 12311, 15690, "3"),
            ("dgt", 28001, 12562, "4"),
            ("dgt", 40563, 12560, "5"),
        ]
        pages = [page for _, _, page in volume.iter_segments()]
        assert _minted_layout(pages[0]) == [
            ("dgt", offset, length, name)
            for name, offset, length, _ in PAGE_1_ACCOLADES
        ]
        assert _minted_layout(pages[1]) == [
            ("dgt", 0, 3142, "accolade_1"),
            ("dgt", 3142, 3137, "accolade_2"),
            ("dgt", 6279, 3137, "accolade_3"),
            ("dgt", 9416, 3135, "accolade_4"),
            ("dgt", 12551, 3139, "accolade_5"),
        ]
        assert [length for _, _, length, _ in _layout(pages[2])] == [
            3147,
            3136,
            3138,
            3141,
        ]
        assert [length for _, _, length, _ in _layout(pages[3])] == [
            3142,
            3139,
            3140,
            3139,
        ]

    def test_shape_and_locks(self, gallica_path: Path) -> None:
        volume = _gallica(gallica_path).create_timeline(
            SCAN_PATHS, canvases=[1, 2, 3, 4]
        )

        assert type(volume) is SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]
        assert volume.is_locked is True
        for _, _, page in volume.iter_segments():
            assert type(page) is SegmentLine[DiscreteGraphicalTimeline]
            assert page.is_locked is True
            for _, _, accolade in page.iter_segments():
                assert type(accolade) is DiscreteGraphicalTimeline
                assert accolade.is_locked is False

    def test_every_timeline_names_what_it_measures(self, gallica_path: Path) -> None:
        volume = _gallica(gallica_path).create_timeline(
            SCAN_PATHS, canvases=[1, 2, 3, 4]
        )

        assert shape(volume.id) == "dgt"
        assert volume.name == GALLICA_LABEL
        assert volume.meta == {}
        assert volume.statements == _source(GALLICA_MANIFEST_ID)
        assert _stated(volume) == [
            ({"image": canvas}, _source(canvas)) for canvas in (F2, F3, F4, F5)
        ]
        page_1 = volume.get_child(volume.list_children()[0])
        assert _stated(page_1) == [
            ({"box": box, "image": F2}, {}) for _, _, _, box in PAGE_1_ACCOLADES
        ]

    def test_given_identity(self, gallica_path: Path) -> None:
        volume = _gallica(gallica_path).create_timeline(
            {1: SCAN_PATHS[1], 2: SCAN_PATHS[2]},
            canvases=[1, 2],
            uid="scan",
            name="Aimons-nous",
            page_ids="p{page}",
            accolade_ids="p{page}_s{accolade}",
        )

        assert volume.id == "scan"
        assert volume.name == "Aimons-nous"
        assert [(page_id, name) for page_id, _, _, name in _layout(volume)] == [
            ("p1", "2"),
            ("p2", "3"),
        ]
        assert [
            accolade_id
            for _, _, page in volume.iter_segments()
            for accolade_id, _, _, _ in _layout(page)
        ] == ["p1_s1", "p1_s2", "p1_s3", "p1_s4"] + [
            "p2_s1",
            "p2_s2",
            "p2_s3",
            "p2_s4",
            "p2_s5",
        ]

    def test_round_trip_through_a_manifest_file(
        self, gallica_path: Path, tmp_path: Path
    ) -> None:
        volume = _gallica(gallica_path).create_timeline(
            SCAN_PATHS, canvases=[1, 2, 3, 4]
        )

        path = volume.to_file(tmp_path / "volume.yaml")
        read = Timeline.from_file(path)

        assert type(read) is SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]
        assert shape(read.id) == "dgt"
        assert read.name == GALLICA_LABEL
        assert read.length.value == 53123
        assert read.is_locked is True
        assert read.statements == _source(GALLICA_MANIFEST_ID)
        assert _layout(read) == _layout(volume)
        assert _stated(read) == _stated(volume)
        for (_, _, page), (_, _, page_read) in zip(
            volume.iter_segments(), read.iter_segments(), strict=True
        ):
            assert type(page_read) is SegmentLine[DiscreteGraphicalTimeline]
            assert page_read.is_locked is True
            assert _layout(page_read) == _layout(page)
            assert page_read.list_children() == page.list_children()
            assert _stated(page_read) == _stated(page)
        assert read.list_children() == volume.list_children()
        assert _stated(read.get_child(read.list_children()[0]))[0] == (
            {"box": _strip(1221, 1589, 2880, 677), "image": F2},
            {},
        )

    def test_canvas_sizes_stay_on_the_loader(self, gallica_path: Path) -> None:
        loader = _gallica(gallica_path)
        loader.create_timeline(SCAN_PATHS, canvases=[1, 2, 3, 4])

        assert [
            (canvas.id, canvas.label, canvas.width, canvas.height, canvas.duration)
            for canvas in loader.manifest_info.canvases
        ] == [(*page, None) for page in GALLICA_PAGES]


class TestPagesAsRoots:
    """``create_timelines``: one root per selected canvas."""

    def test_one_locked_root_per_canvas(self, gallica_path: Path) -> None:
        pages = _gallica(gallica_path).create_timelines(
            SCAN_PATHS, canvases=[1, 2, 3, 4]
        )

        assert [
            (type(page), shape(page.id), page.name, page.length.value, page.is_locked)
            for page in pages
        ] == [
            (SegmentLine[DiscreteGraphicalTimeline], "dgt", "2", 12311, True),
            (SegmentLine[DiscreteGraphicalTimeline], "dgt", "3", 15690, True),
            (SegmentLine[DiscreteGraphicalTimeline], "dgt", "4", 12562, True),
            (SegmentLine[DiscreteGraphicalTimeline], "dgt", "5", 12560, True),
        ]
        assert [(page.meta, page.statements) for page in pages] == [
            ({"image": canvas}, _source(canvas)) for canvas in (F2, F3, F4, F5)
        ]
        assert [
            (name, offset, length) for _, offset, length, name in _layout(pages[0])
        ] == [(name, offset, length) for name, offset, length, _ in PAGE_1_ACCOLADES]
        assert _stated(pages[0]) == [
            ({"box": box, "image": F2}, {}) for _, _, _, box in PAGE_1_ACCOLADES
        ]
        assert all(page._parent is None for page in pages)


class TestCanvasSelection:
    """Which canvases are read, and which path each one takes."""

    @pytest.mark.parametrize(
        "canvases",
        [[1, 2], [2, 1], [F2, F3], [F3, 1], (1, F3, 2)],
        ids=["positions", "reversed", "ids", "mixed", "named twice"],
    )
    def test_manifest_order_whatever_names_them(
        self, gallica_path: Path, canvases: Any
    ) -> None:
        pages = _gallica(gallica_path).create_timelines(
            {1: SCAN_PATHS[1], 2: SCAN_PATHS[2]}, canvases=canvases
        )

        assert [
            (page.name, page.length.value, page.meta["image"]) for page in pages
        ] == [("2", 12311, F2), ("3", 15690, F3)]

    def test_paths_keyed_by_id(self, gallica_path: Path) -> None:
        pages = _gallica(gallica_path).create_timelines(
            {F3: SCAN_PATHS[2], 1: SCAN_PATHS[1]}, canvases=[1, 2]
        )

        assert [
            (page.name, page.length.value, page.meta["image"]) for page in pages
        ] == [("2", 12311, F2), ("3", 15690, F3)]

    def test_none_selects_every_image_canvas(self, bnf_path: Path) -> None:
        volume = (
            IIIFManifestLoader()
            .load(bnf_path)
            .create_timeline(
                {
                    4: HorizontalLinePath(x0=100, x1=2600, y=1000, tolerance=10),
                    5: HorizontalLinePath(x0=0, x1=2677, y=1000, tolerance=10),
                    6: HorizontalLinePath(x0=0, x1=2940, y=1000, tolerance=10),
                    7: HorizontalLinePath(x0=40, x1=2940, y=1000, tolerance=10),
                }
            )
        )

        assert type(volume) is SegmentLine[DiscreteGraphicalTimeline]
        assert volume.name == "4 Mélodies classiques"
        assert volume.length.value == 11017
        assert volume.statements == _source(BNF_MANIFEST_ID)
        assert _minted_layout(volume) == [
            ("dgt", 0, 2500, "5"),
            ("dgt", 2500, 2677, "6"),
            ("dgt", 5177, 2940, "recto"),
            ("dgt", 8117, 2900, "verso"),
        ]
        assert _stated(volume) == [
            (
                {"box": _strip(x, 990, width, 20), "image": canvas},
                _source(canvas),
            )
            for (canvas, _, _, _), (x, width) in zip(
                BNF_IMAGES, [(100, 2500), (0, 2677), (0, 2940), (40, 2900)]
            )
        ]

    def test_one_path_for_the_one_selected_canvas(self, gallica_path: Path) -> None:
        (page,) = _gallica(gallica_path).create_timelines(SCAN_PATHS[1], canvases=[F2])

        assert (page.name, page.length.value, page.meta) == ("2", 12311, {"image": F2})


class TestRefusals:
    """Every refusal is whole, and nothing is built or minted before it."""

    @staticmethod
    def _nothing_was_built(loader: IIIFManifestLoader) -> None:
        """The loader mints what a fresh loader of its file mints first."""
        (page,) = loader.create_timelines(_line(), canvases=[1])
        fresh = IIIFManifestLoader().load(*loader.sources)
        (first,) = fresh.create_timelines(_line(), canvases=[1])
        assert page.id == first.id

    def test_selected_canvases_without_a_path(self, gallica_path: Path) -> None:
        loader = _gallica(gallica_path)
        with pytest.raises(ValueError) as error:
            loader.create_timeline(SCAN_PATHS)
        assert str(error.value) == (
            "canvases 0, 5 have no time path: give each one or leave them out "
            "with canvases="
        )
        with pytest.raises(ValueError) as error:
            loader.create_timelines(SCAN_PATHS, canvases=[0, 1, 2, 3, 4])
        assert str(error.value) == (
            "canvas 0 has no time path: give it one or leave it out with canvases="
        )
        self._nothing_was_built(loader)

    def test_path_for_an_unknown_or_unselected_canvas(self, gallica_path: Path) -> None:
        loader = _gallica(gallica_path)
        with pytest.raises(KeyError) as error:
            loader.create_timelines(
                {1: _line(), 0: _line(), 9: _line(), "nope": _line()}, canvases=[1]
            )
        assert error.value.args[0] == (
            "paths gives a time path to 9, 'nope' (not in the manifest); 0 (not "
            f"selected by canvases=). Selected canvases: 1: {F2!r}"
        )
        self._nothing_was_built(loader)

    def test_path_for_a_time_based_canvas(self, bnf_path: Path) -> None:
        loader = IIIFManifestLoader().load(bnf_path)
        images = [canvas for canvas, _, _, _ in BNF_IMAGES]
        refusal = "paths gives a time path to 0 (time-based: no width and height). "

        with pytest.raises(KeyError) as error:
            loader.create_timelines({0: _line(), 4: _line()})
        assert error.value.args[0] == (
            refusal + f"Selected canvases: 4: {images[0]!r}, 5: {images[1]!r}, "
            f"6: {images[2]!r}, 7: {images[3]!r}"
        )

        with pytest.raises(KeyError) as error:
            loader.create_timelines({0: _line(), 4: _line()}, canvases=[0, 4])
        assert error.value.args[0] == (
            refusal + f"Selected canvases: 0: {BNF_SOUNDS[0][0]!r}, 4: {images[0]!r}"
        )

    def test_one_path_for_a_time_based_canvas(self, bnf_path: Path) -> None:
        loader = IIIFManifestLoader().load(bnf_path)
        sound = BNF_SOUNDS[0][0]
        for key in (0, sound):
            with pytest.raises(KeyError) as error:
                loader.create_timelines(_line(), canvases=[key])
            assert error.value.args[0] == (
                "paths gives a time path to 0 (time-based: no width and height). "
                f"Selected canvases: 0: {sound!r}"
            )

    def test_one_path_for_several_canvases(self, gallica_path: Path) -> None:
        loader = _gallica(gallica_path)
        with pytest.raises(ValueError) as error:
            loader.create_timelines(_line(), canvases=[1, 2])
        assert str(error.value) == (
            f"One time path was given and 2 canvases are selected (1: {F2!r}, 2: "
            f"{F3!r}): give a mapping from each canvas to its path, or select one "
            "canvas with canvases=."
        )
        self._nothing_was_built(loader)

    def test_two_paths_for_one_canvas(self, gallica_path: Path) -> None:
        loader = _gallica(gallica_path)
        with pytest.raises(ValueError) as error:
            loader.create_timelines({1: _line(), F2: _line()}, canvases=[1])
        assert str(error.value) == (
            f"paths gives canvas 1 two time paths, as 1 and as {F2!r}."
        )

    def test_paths_of_another_type(self, gallica_path: Path) -> None:
        loader = _gallica(gallica_path)
        with pytest.raises(TypeError) as error:
            loader.create_timelines([_line()], canvases=[1])  # type: ignore[arg-type]
        assert str(error.value) == (
            "paths takes a time path or a mapping from canvases to time paths, "
            "not list."
        )

    def test_bare_string_canvases(self, gallica_path: Path) -> None:
        loader = _gallica(gallica_path)
        with pytest.raises(TypeError) as error:
            loader.create_timeline(SCAN_PATHS, canvases="all")  # type: ignore[arg-type]
        assert str(error.value) == (
            "canvases takes a sequence of positions or ids, not the bare string "
            "'all': write canvases=['all']."
        )

    def test_canvases_naming_no_canvas(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "two.json",
            {
                "@context": "http://iiif.io/api/presentation/3/context.json",
                "id": "http://example.org/manifest",
                "type": "Manifest",
                "items": [
                    {"id": "c/1", "type": "Canvas", "width": 100, "height": 200},
                    {"id": "c/2", "type": "Canvas", "duration": 3.5},
                ],
            },
        )
        loader = IIIFManifestLoader().load(path)
        with pytest.raises(KeyError) as error:
            loader.create_timelines(_line(), canvases=[0, 2, "c/3", -1])
        assert error.value.args[0] == (
            "canvases names 2, 'c/3', -1, which the manifest does not hold. "
            "Available canvases: 0: 'c/1', 1: 'c/2'"
        )

    def test_removed_arguments_have_no_alias(self, gallica_path: Path) -> None:
        loader = _gallica(gallica_path)
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            loader.create_timeline()  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="unexpected keyword argument 'axis'"):
            loader.create_timeline(SCAN_PATHS, axis="width")  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="takes 2 positional arguments"):
            loader.create_timeline(SCAN_PATHS, "scan")  # type: ignore[misc]
        assert not hasattr(loader.get_canvas(0), "extent")

    @pytest.mark.parametrize(
        "path",
        [
            HorizontalLinePath(x0=0, x1=4512, y=6174, tolerance=10),
            HorizontalLinePath(x0=0, x1=4512, y=0, tolerance=10),
            VerticalLinePath(x=4512, y0=0, y1=6174, tolerance=10),
            VerticalLinePath(x=0, y0=0, y1=6174, tolerance=10),
        ],
        ids=["bottom right", "top", "right edge", "left edge"],
    )
    def test_line_on_the_edge_of_its_canvas(
        self, gallica_path: Path, path: TimeAxisPath
    ) -> None:
        (page,) = _gallica(gallica_path).create_timelines(path, canvases=[1])

        assert page.length.value == path.length

    @pytest.mark.parametrize(
        ("path", "needed"),
        [
            (
                HorizontalLinePath(x0=0, x1=4513, y=50, tolerance=10),
                "0 <= x0, x1 <= 4512 and 0 <= y <= 6174",
            ),
            (
                HorizontalLinePath(x0=-1, x1=100, y=50, tolerance=10),
                "0 <= x0, x1 <= 4512 and 0 <= y <= 6174",
            ),
            (
                HorizontalLinePath(x0=0, x1=100, y=6175, tolerance=10),
                "0 <= x0, x1 <= 4512 and 0 <= y <= 6174",
            ),
            (
                HorizontalLinePath(x0=0, x1=100, y=-1, tolerance=10),
                "0 <= x0, x1 <= 4512 and 0 <= y <= 6174",
            ),
            (
                VerticalLinePath(x=50, y0=0, y1=6175, tolerance=10),
                "0 <= y0, y1 <= 6174 and 0 <= x <= 4512",
            ),
            (
                VerticalLinePath(x=50, y0=-1, y1=100, tolerance=10),
                "0 <= y0, y1 <= 6174 and 0 <= x <= 4512",
            ),
            (
                VerticalLinePath(x=4513, y0=0, y1=100, tolerance=10),
                "0 <= y0, y1 <= 6174 and 0 <= x <= 4512",
            ),
        ],
        ids=["x1", "x0", "y", "y above", "y1", "y0", "x"],
    )
    def test_line_leaving_its_canvas(
        self, gallica_path: Path, path: TimeAxisPath, needed: str
    ) -> None:
        loader = _gallica(gallica_path)
        with pytest.raises(ValueError) as error:
            loader.create_timelines(path, canvases=[1])
        assert str(error.value) == (
            f"Canvas 1 ({F2!r}, 4512 x 6174 pixels): its time path, {path!r}, "
            f"leaves the canvas; a line on it needs {needed}."
        )
        self._nothing_was_built(loader)

    def test_piece_leaving_its_canvas_is_named(self, gallica_path: Path) -> None:
        # Page 2's systems reach x 3661; canvas 0 would hold them, but the
        # second piece here is moved past canvas 2's width of 4564.
        wide = HorizontalLinePath(x0=522, x1=4565, y=2000, tolerance=300)
        path = PiecewisePath((SCAN_PATHS[2].pieces[0], wide))
        loader = _gallica(gallica_path)
        with pytest.raises(ValueError) as error:
            loader.create_timeline({1: SCAN_PATHS[1], 2: path}, canvases=[1, 2])
        assert str(error.value) == (
            f"Canvas 2 ({F3!r}, 4564 x 6174 pixels): piece 2 of its time path, "
            f"{wide!r}, leaves the canvas; a line on it needs 0 <= x0, x1 <= 4564 "
            "and 0 <= y <= 6174."
        )
        self._nothing_was_built(loader)


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

    def test_first_canvas_time_based_has_no_dimensions(self, bnf_path: Path) -> None:
        loader = IIIFManifestLoader().load(bnf_path)
        message = re.escape(
            f"Canvas {BNF_SOUNDS[0][0]} is time-based (duration 221.0) and states "
            "no width and height, so it has no extent in pixels"
        )
        with pytest.raises(ValueError, match=message):
            _ = loader.dimensions
        with pytest.raises(ValueError, match=message):
            _ = loader.width
        with pytest.raises(ValueError, match=message):
            _ = loader.height

    def test_repr_on_time_based_first_canvas(self, bnf_path: Path) -> None:
        loader = IIIFManifestLoader().load(bnf_path)
        assert repr(loader) == "IIIFManifestLoader(canvases=8, duration=221.0)"

    def test_manifest_of_sound_alone_selects_nothing(self, tmp_path: Path) -> None:
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

        assert loader.create_timelines({}) == []
        with pytest.raises(ValueError) as error:
            loader.create_timeline({})
        assert str(error.value) == "A volume needs at least one page."

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
