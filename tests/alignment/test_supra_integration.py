"""SUPRA Piano Roll Integration Tests.

Tests the full SUPRA piano roll alignment workflow using:
- IIIFManifestLoader for image dimensions
- ATONLoader for hole punch data
- AlignmentBundle for coordinate transfer with partial alignment

Per ZERO TOLERANCE policy, all assertions use exact expected values.

NOTE: Partial alignment is spelled as a child: the holes region is a child
of the image timeline at the first hole, and the bundle reaches the image
through that exact offset.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from timetoalign.alignment import AlignmentBundle
from timetoalign.loader.graphical.aton import ATONLoader
from timetoalign.loader.graphical.iiif import IIIFManifestLoader
from timetoalign.timelines import Timeline

# Test data directory
SUPRA_DIR = Path(__file__).parent.parent / "data" / "supra"


# region Fixtures


@pytest.fixture
def iiif_loader() -> IIIFManifestLoader:
    """Loaded IIIF manifest loader."""
    loader = IIIFManifestLoader()
    loader.load(SUPRA_DIR / "image" / "ifff_manifest.json")
    return loader


@pytest.fixture
def aton_loader() -> ATONLoader:
    """Loaded ATON hole punch loader."""
    loader = ATONLoader()
    loader.load(SUPRA_DIR / "image" / "fd660zf8362_analysis.txt")
    return loader


@pytest.fixture
def image_timeline(iiif_loader: IIIFManifestLoader) -> Timeline:
    """Graphical timeline representing the piano roll image."""
    return Timeline(
        length=iiif_loader.height,  # 299400 pixels
        uid="dgt1_image",
        name="Piano Roll Image (WM 990)",
    )


@pytest.fixture
def holes_timeline(aton_loader: ATONLoader) -> Timeline:
    """Graphical timeline for the musical region (hole punches)."""
    return Timeline(
        length=aton_loader.musical_length.value,  # 277776 pixels
        uid="dgt1_holes",
        name="Musical Holes Region",
    )


@pytest.fixture
def midi_raw_timeline(aton_loader: ATONLoader) -> Timeline:
    """Logical timeline representing raw MIDI (one note per hole)."""
    # For testing, we use a proportional length based on musical notes
    # In reality, this would be loaded from the MIDI file
    return Timeline(
        length=aton_loader.musical_notes * 100,  # 8718 * 100 = 871800 ticks (simulated)
        uid="dlt1_raw",
        name="MIDI Raw (fd660zf8362_raw.mid)",
    )


# endregion


# region Test: Data Loading


class TestSUPRADataLoading:
    """Tests for loading SUPRA data with exact expected values."""

    def test_iiif_dimensions_exact(self, iiif_loader: IIIFManifestLoader) -> None:
        """IIIF loader extracts exact image dimensions."""
        assert iiif_loader.width == 4096
        assert iiif_loader.height == 299400

    @pytest.mark.slow
    def test_aton_metadata_exact(self, aton_loader: ATONLoader) -> None:
        """ATON loader extracts exact metadata."""
        assert aton_loader.musical_holes == 30092
        assert aton_loader.musical_notes == 8718
        # first_hole, last_hole, musical_length now return Coordinate objects
        assert aton_loader.first_hole.value == 15343
        assert aton_loader.last_hole.value == 293119
        assert aton_loader.musical_length.value == 277776

    @pytest.mark.slow
    def test_aton_musical_length_calculation(self, aton_loader: ATONLoader) -> None:
        """Musical length equals last_hole - first_hole."""
        expected = aton_loader.last_hole.value - aton_loader.first_hole.value
        assert aton_loader.musical_length.value == expected == 277776


# endregion


# region Test: Timeline Creation


class TestSUPRATimelineCreation:
    """Tests for creating SUPRA timelines."""

    def test_image_timeline_length(self, image_timeline: Timeline) -> None:
        """Image timeline has correct length."""
        assert image_timeline.length.value == 299400

    @pytest.mark.slow
    def test_holes_timeline_length(self, holes_timeline: Timeline) -> None:
        """Holes timeline has correct length."""
        assert holes_timeline.length.value == 277776


# endregion


# region Test: Alignment Bundle


def _holes_bundle(
    image: Timeline, holes: Timeline, first_hole: int, **bundle_kwargs: str
) -> AlignmentBundle:
    """Bundle the holes region as the child of the image it is part of."""
    image.add_child(holes, offset=first_hole)
    bundle = AlignmentBundle(**bundle_kwargs)
    bundle.add_timeline(holes, as_group="holes")
    return bundle


class TestSUPRAAlignmentBundle:
    """Tests for the SUPRA workflow using AlignmentBundle with partial alignment.

    The holes region takes part as a child of the image, placed at the first
    hole, in a group of its own; the bundle registers the child and, with it,
    the image it belongs to.
    """

    def test_create_supra_bundle(
        self,
        image_timeline: Timeline,
        holes_timeline: Timeline,
        aton_loader: ATONLoader,
    ) -> None:
        """Can create a bundle with SUPRA timelines using partial alignment."""
        bundle = _holes_bundle(
            image_timeline,
            holes_timeline,
            aton_loader.first_hole.value,
            name="SUPRA WM 990",
        )

        # The grouped child brings its root along, ungrouped.
        assert bundle.n_timelines == 2
        assert bundle.n_groups == 1
        assert "dgt1_image" in bundle.timeline_ids
        assert "dgt1_holes" in bundle.timeline_ids
        assert holes_timeline.length.value == (
            aton_loader.last_hole.value - aton_loader.first_hole.value
        )

    def test_transfer_holes_to_image(
        self,
        image_timeline: Timeline,
        holes_timeline: Timeline,
        aton_loader: ATONLoader,
    ) -> None:
        """Transfer from holes region to full image coordinates."""
        bundle = _holes_bundle(
            image_timeline, holes_timeline, aton_loader.first_hole.value
        )

        # Coordinate 0 in holes -> first_hole in image - EXACT
        result_start = bundle.transfer(0.0, "dgt1_holes", "dgt1_image")
        assert result_start is not None
        assert result_start == 15343.0  # EXACT, no tolerance

        # Coordinate musical_length in holes -> last_hole in image - EXACT
        result_end = bundle.transfer(277776.0, "dgt1_holes", "dgt1_image")
        assert result_end is not None
        assert result_end == 293119.0  # EXACT, no tolerance

        # Midpoint: 138888 in holes -> 154231 in image - EXACT
        # (277776/2 = 138888 exactly, 15343 + 138888 = 154231 exactly)
        result_mid = bundle.transfer(138888.0, "dgt1_holes", "dgt1_image")
        assert result_mid is not None
        assert result_mid == 154231.0  # EXACT, no tolerance

    def test_transfer_image_to_holes(
        self,
        image_timeline: Timeline,
        holes_timeline: Timeline,
        aton_loader: ATONLoader,
    ) -> None:
        """Transfer from full image to holes region coordinates."""
        bundle = _holes_bundle(
            image_timeline, holes_timeline, aton_loader.first_hole.value
        )

        # first_hole in image -> 0 in holes - EXACT
        result_start = bundle.transfer(15343.0, "dgt1_image", "dgt1_holes")
        assert result_start is not None
        assert result_start == 0.0  # EXACT, no tolerance

        # last_hole in image -> musical_length in holes - EXACT
        result_end = bundle.transfer(293119.0, "dgt1_image", "dgt1_holes")
        assert result_end is not None
        assert result_end == 277776.0  # EXACT, no tolerance


# endregion


# region Test: Order Independence


class TestSUPRAOrderIndependence:
    """Tests verifying order-independence with SUPRA data.

    The bundle must produce identical transfer results regardless
    of the order in which timelines are added.
    """

    @pytest.mark.slow
    def test_order_1_image_first(
        self,
        aton_loader: ATONLoader,
    ) -> None:
        """Image registered first, then the holes child."""
        image = Timeline(length=299400, uid="dgt_1")
        holes = Timeline(length=277776, uid="dgt1_holes")
        image.add_child(holes, offset=15343)

        bundle = AlignmentBundle(id="order1")
        bundle.add_timeline(image)
        bundle.add_timeline(holes, as_group="holes")

        result = bundle.transfer(100000.0, "dgt1_holes", "dgt_1")
        assert result is not None
        # 100000 + 15343 = 115343
        assert result == 115343.0

    @pytest.mark.slow
    def test_order_both_produce_same_result(
        self,
        aton_loader: ATONLoader,
    ) -> None:
        """Registering the child before or after its root gives one answer."""
        image_tl_1 = Timeline(length=299400, uid="img1")
        holes_tl_1 = Timeline(length=277776, uid="holes1")
        image_tl_1.add_child(holes_tl_1, offset=15343)
        b1 = AlignmentBundle(id="b1")
        b1.add_timeline(image_tl_1)
        b1.add_timeline(holes_tl_1, as_group="holes")

        image_tl_2 = Timeline(length=299400, uid="img2")
        holes_tl_2 = Timeline(length=277776, uid="holes2")
        image_tl_2.add_child(holes_tl_2, offset=15343)
        b2 = AlignmentBundle(id="b2")
        b2.add_timeline(holes_tl_2, as_group="holes")  # brings its root along

        test_coord = 100000.0

        result1 = b1.transfer(test_coord, "holes1", "img1")
        result2 = b2.transfer(test_coord, "holes2", "img2")

        assert result1 == result2 == 115343.0

    def test_three_timeline_same_partial_alignment(self) -> None:
        """Two members sharing one image region transfer identically in any order.

        The holes region is a child of the image; the MIDI timeline is grouped
        with it, so both span the same region of the image.
        """
        # Order 1: holes first, then midi joins its group
        img1 = Timeline(length=299400, uid="dgt_1")
        holes1 = Timeline(length=277776, uid="dgt1_holes")
        midi1 = Timeline(length=871800, uid="dlt_1")  # Simulated MIDI ticks
        img1.add_child(holes1, offset=15343)
        b1 = AlignmentBundle(id="b1")
        b1.add_timeline(holes1, as_group="roll")
        b1.add_timeline(midi1, grouped_with="dgt1_holes")

        # Order 2: midi first, then holes joins its group
        img2 = Timeline(length=299400, uid="dgt_1")
        holes2 = Timeline(length=277776, uid="dgt1_holes")
        midi2 = Timeline(length=871800, uid="dlt_1")
        img2.add_child(holes2, offset=15343)
        b2 = AlignmentBundle(id="b2")
        b2.add_timeline(midi2, as_group="roll")
        b2.add_timeline(holes2, grouped_with="dlt_1")

        result1_holes_to_midi = b1.transfer(100000.0, "dgt1_holes", "dlt_1")
        result2_holes_to_midi = b2.transfer(100000.0, "dgt1_holes", "dlt_1")

        assert result1_holes_to_midi is not None
        assert result2_holes_to_midi is not None
        assert result1_holes_to_midi == result2_holes_to_midi

        # holes spans [0, 277776], midi spans [0, 871800], over the same region
        expected = float(Fraction(100000 * 871800, 277776))
        assert result1_holes_to_midi == expected

        # The image is reached through the child's offset in both orders.
        assert b1.transfer(100000.0, "dlt_1", "dgt_1") == b2.transfer(
            100000.0, "dlt_1", "dgt_1"
        )
        assert b1.transfer(0.0, "dlt_1", "dgt_1") == 15343.0


# endregion


# region Test: Summary


@pytest.mark.slow
class TestSUPRASummary:
    """Tests for bundle summary with SUPRA data."""

    def test_summary_structure(
        self,
        image_timeline: Timeline,
        holes_timeline: Timeline,
        aton_loader: ATONLoader,
    ) -> None:
        """Summary has correct structure."""
        bundle = _holes_bundle(
            image_timeline,
            holes_timeline,
            aton_loader.first_hole.value,
            id="supra_bundle",
            name="SUPRA WM 990",
        )

        summary = bundle.summary()

        assert summary["id"] == "supra_bundle"
        assert summary["name"] == "SUPRA WM 990"
        assert summary["n_timelines"] == 2
        assert summary["n_groups"] == 1
        assert "dgt1_image" in summary["timelines"]
        assert "dgt1_holes" in summary["timelines"]

    def test_summary_is_deterministic(
        self,
        aton_loader: ATONLoader,
    ) -> None:
        """Summary output is deterministic (same input -> same output)."""
        img1 = Timeline(length=299400, uid="img1")
        holes1 = Timeline(length=277776, uid="holes1")

        b1 = _holes_bundle(img1, holes1, 15343, id="test", name="Test")

        summary1 = b1.summary()
        summary2 = b1.summary()

        # Same bundle, same summary
        assert summary1 == summary2


# endregion
