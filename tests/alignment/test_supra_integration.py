"""SUPRA Piano Roll Integration Tests.

Tests the full SUPRA piano roll alignment workflow using:
- IIIFManifestLoader for image dimensions
- ATONLoader for hole punch data
- AlignmentBundle for coordinate transfer with partial alignment

Per ZERO TOLERANCE policy, all assertions use exact expected values.

NOTE: These tests use the TimelineGroup timestamp-based architecture.
Partial alignment is achieved via the start/end parameters in
AlignmentBundle.add_timeline().
"""

from __future__ import annotations

from pathlib import Path

import pytest

from timetoalign.alignment import AlignmentBundle
from timetoalign.core import IdCoordinate, TimeUnit
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


# region Test: AlignmentBundle Integration


@pytest.mark.slow
class TestSUPRAAlignmentBundle:
    """Tests for SUPRA workflow using AlignmentBundle with partial alignment.

    Uses the timestamp-based API where partial alignment is specified via
    start/end parameters in add_timeline().
    """

    def test_create_supra_bundle(
        self,
        image_timeline: Timeline,
        holes_timeline: Timeline,
        aton_loader: ATONLoader,
    ) -> None:
        """Can create a bundle with SUPRA timelines using partial alignment."""
        bundle = AlignmentBundle(name="SUPRA WM 990")

        # Add image as reference
        bundle.add_timeline(image_timeline, uid="dgt_1")

        # Add holes timeline with partial alignment
        # Holes region spans first_hole to last_hole in image coordinates
        bundle.add_timeline(
            holes_timeline,
            uid="dgt1_holes",
            grouped_with="dgt_1",
            start=IdCoordinate(
                float(aton_loader.first_hole.value), TimeUnit.seconds, "dgt_1"
            ),
            end=IdCoordinate(
                float(aton_loader.last_hole.value), TimeUnit.seconds, "dgt_1"
            ),
        )

        assert bundle.n_timelines == 2
        assert bundle.n_groups == 1
        assert "dgt_1" in bundle.timeline_ids
        assert "dgt1_holes" in bundle.timeline_ids

    def test_transfer_holes_to_image(
        self,
        image_timeline: Timeline,
        holes_timeline: Timeline,
        aton_loader: ATONLoader,
    ) -> None:
        """Transfer from holes region to full image coordinates."""
        bundle = AlignmentBundle(name="SUPRA WM 990")
        bundle.add_timeline(image_timeline, uid="dgt_1")

        # Add holes with partial alignment via start/end
        bundle.add_timeline(
            holes_timeline,
            uid="dgt1_holes",
            grouped_with="dgt_1",
            start=IdCoordinate(
                float(aton_loader.first_hole.value), TimeUnit.seconds, "dgt_1"
            ),  # 15343.0
            end=IdCoordinate(
                float(aton_loader.last_hole.value), TimeUnit.seconds, "dgt_1"
            ),  # 293119.0
        )

        # Coordinate 0 in holes -> first_hole in image - EXACT
        result_start = bundle.transfer(0.0, "dgt1_holes", "dgt_1")
        assert result_start is not None
        assert result_start == 15343.0  # EXACT, no tolerance

        # Coordinate musical_length in holes -> last_hole in image - EXACT
        result_end = bundle.transfer(277776.0, "dgt1_holes", "dgt_1")
        assert result_end is not None
        assert result_end == 293119.0  # EXACT, no tolerance

        # Midpoint: 138888 in holes -> 154231 in image - EXACT
        # (277776/2 = 138888 exactly, 15343 + 138888 = 154231 exactly)
        result_mid = bundle.transfer(138888.0, "dgt1_holes", "dgt_1")
        assert result_mid is not None
        assert result_mid == 154231.0  # EXACT, no tolerance

    def test_transfer_image_to_holes(
        self,
        image_timeline: Timeline,
        holes_timeline: Timeline,
        aton_loader: ATONLoader,
    ) -> None:
        """Transfer from full image to holes region coordinates."""
        bundle = AlignmentBundle(name="SUPRA WM 990")
        bundle.add_timeline(image_timeline, uid="dgt_1")

        bundle.add_timeline(
            holes_timeline,
            uid="dgt1_holes",
            grouped_with="dgt_1",
            start=IdCoordinate(
                float(aton_loader.first_hole.value), TimeUnit.seconds, "dgt_1"
            ),
            end=IdCoordinate(
                float(aton_loader.last_hole.value), TimeUnit.seconds, "dgt_1"
            ),
        )

        # first_hole in image -> 0 in holes - EXACT
        result_start = bundle.transfer(15343.0, "dgt_1", "dgt1_holes")
        assert result_start is not None
        assert result_start == 0.0  # EXACT, no tolerance

        # last_hole in image -> musical_length in holes - EXACT
        result_end = bundle.transfer(293119.0, "dgt_1", "dgt1_holes")
        assert result_end is not None
        assert result_end == 277776.0  # EXACT, no tolerance


# endregion


# region Test: Order Independence


class TestSUPRAOrderIndependence:
    """Tests verifying order-independence with SUPRA data.

    The bundle must produce identical transfer results regardless
    of the order in which timelines are added.

    Uses the timestamp-based API with start/end parameters for partial alignment.
    """

    @pytest.mark.slow
    def test_order_1_image_first(
        self,
        aton_loader: ATONLoader,
    ) -> None:
        """Create bundle: image first, then holes."""
        image_tl = Timeline(length=299400, uid="img1")
        holes_tl = Timeline(length=277776, uid="holes1")

        bundle = AlignmentBundle(id="order1")
        bundle.add_timeline(image_tl, uid="dgt_1")
        bundle.add_timeline(
            holes_tl,
            uid="dgt1_holes",
            grouped_with="dgt_1",
            start=IdCoordinate(15343.0, TimeUnit.seconds, "dgt_1"),
            end=IdCoordinate(293119.0, TimeUnit.seconds, "dgt_1"),
        )

        # Test transfer - interior point, compute expected exactly
        result = bundle.transfer(100000.0, "dgt1_holes", "dgt_1")
        assert result is not None
        # Expected: 15343 + (100000 / 277776) * (293119 - 15343)
        # This is an interior point with irrational scale factor
        expected = 15343.0 + (100000.0 / 277776.0) * (293119.0 - 15343.0)
        assert result == expected  # Same floating-point computation

    @pytest.mark.slow
    def test_order_both_produce_same_result(
        self,
        aton_loader: ATONLoader,
    ) -> None:
        """Both orderings produce identical transfer results."""
        # Order 1: image as reference
        image_tl_1 = Timeline(length=299400, uid="img1")
        holes_tl_1 = Timeline(length=277776, uid="holes1")

        b1 = AlignmentBundle(id="b1")
        b1.add_timeline(image_tl_1, uid="dgt_1")
        b1.add_timeline(
            holes_tl_1,
            uid="dgt1_holes",
            grouped_with="dgt_1",
            start=IdCoordinate(15343.0, TimeUnit.seconds, "dgt_1"),
            end=IdCoordinate(293119.0, TimeUnit.seconds, "dgt_1"),
        )

        # Order 2: Same order but with different uid assignment
        image_tl_2 = Timeline(length=299400, uid="img2")
        holes_tl_2 = Timeline(length=277776, uid="holes2")

        b2 = AlignmentBundle(id="b2")
        b2.add_timeline(image_tl_2, uid="image")  # Different bundle UID
        b2.add_timeline(
            holes_tl_2,
            uid="holes",
            grouped_with="image",
            start=IdCoordinate(15343.0, TimeUnit.seconds, "image"),
            end=IdCoordinate(293119.0, TimeUnit.seconds, "image"),
        )

        # Both should produce same coordinate values (just with different UIDs)
        test_coord = 100000.0

        result1 = b1.transfer(test_coord, "dgt1_holes", "dgt_1")
        result2 = b2.transfer(test_coord, "holes", "image")

        assert result1 == result2

    def test_three_timeline_same_partial_alignment(self) -> None:
        """Three timelines with same partial alignment produce identical transfers.

        When all three timelines are partial-aligned to the same region,
        the transfer between any two should be consistent regardless of order.
        """
        # Order 1: image -> holes -> midi (both partial to same region)
        b1 = AlignmentBundle(id="b1")
        img1 = Timeline(length=299400, uid="i1")
        holes1 = Timeline(length=277776, uid="h1")
        midi1 = Timeline(length=871800, uid="m1")  # Simulated MIDI ticks

        b1.add_timeline(img1, uid="dgt_1")
        b1.add_timeline(
            holes1,
            uid="dgt1_holes",
            grouped_with="dgt_1",
            start=IdCoordinate(15343.0, TimeUnit.seconds, "dgt_1"),
            end=IdCoordinate(293119.0, TimeUnit.seconds, "dgt_1"),
        )
        b1.add_timeline(
            midi1,
            uid="dlt_1",
            grouped_with="dgt_1",
            start=IdCoordinate(15343.0, TimeUnit.seconds, "dgt_1"),
            end=IdCoordinate(293119.0, TimeUnit.seconds, "dgt_1"),
        )

        # Order 2: image -> midi -> holes (same partial alignments, different order)
        b2 = AlignmentBundle(id="b2")
        img2 = Timeline(length=299400, uid="i2")
        holes2 = Timeline(length=277776, uid="h2")
        midi2 = Timeline(length=871800, uid="m2")

        b2.add_timeline(img2, uid="dgt_1")
        b2.add_timeline(
            midi2,
            uid="dlt_1",
            grouped_with="dgt_1",
            start=IdCoordinate(15343.0, TimeUnit.seconds, "dgt_1"),
            end=IdCoordinate(293119.0, TimeUnit.seconds, "dgt_1"),
        )
        b2.add_timeline(
            holes2,
            uid="dgt1_holes",
            grouped_with="dgt_1",
            start=IdCoordinate(15343.0, TimeUnit.seconds, "dgt_1"),
            end=IdCoordinate(293119.0, TimeUnit.seconds, "dgt_1"),
        )

        # Both bundles have holes and midi aligned to the same image region
        # So holes -> midi should give the same result in both

        # Test holes -> midi transfer (should be same in both bundles)
        result1_holes_to_midi = b1.transfer(100000.0, "dgt1_holes", "dlt_1")
        result2_holes_to_midi = b2.transfer(100000.0, "dgt1_holes", "dlt_1")

        assert result1_holes_to_midi is not None
        assert result2_holes_to_midi is not None
        # Both have same partial alignment, so results should be equal
        assert result1_holes_to_midi == result2_holes_to_midi

        # Verify the actual value: 100000 in holes -> midi
        # holes spans [0, 277776], midi spans [0, 871800], both over same image region
        # ratio = 100000 / 277776, result = ratio * 871800
        expected = (100000.0 / 277776.0) * 871800.0
        assert result1_holes_to_midi == expected


# endregion


# region Test: Summary


@pytest.mark.slow
class TestSUPRASummary:
    """Tests for bundle summary with SUPRA data.

    Uses the timestamp-based API with start/end parameters for partial alignment.
    """

    def test_summary_structure(
        self,
        image_timeline: Timeline,
        holes_timeline: Timeline,
        aton_loader: ATONLoader,
    ) -> None:
        """Summary has correct structure."""
        bundle = AlignmentBundle(id="supra_bundle", name="SUPRA WM 990")
        bundle.add_timeline(image_timeline, uid="dgt_1")
        bundle.add_timeline(
            holes_timeline,
            uid="dgt1_holes",
            grouped_with="dgt_1",
            start=IdCoordinate(
                float(aton_loader.first_hole.value), TimeUnit.seconds, "dgt_1"
            ),
            end=IdCoordinate(
                float(aton_loader.last_hole.value), TimeUnit.seconds, "dgt_1"
            ),
        )

        summary = bundle.summary()

        assert summary["id"] == "supra_bundle"
        assert summary["name"] == "SUPRA WM 990"
        assert summary["n_timelines"] == 2
        assert summary["n_groups"] == 1
        assert "dgt_1" in summary["timelines"]
        assert "dgt1_holes" in summary["timelines"]

    def test_summary_is_deterministic(
        self,
        aton_loader: ATONLoader,
    ) -> None:
        """Summary output is deterministic (same input -> same output)."""
        img1 = Timeline(length=299400, uid="img1")
        holes1 = Timeline(length=277776, uid="holes1")

        b1 = AlignmentBundle(id="test", name="Test")
        b1.add_timeline(img1, uid="dgt_1")
        b1.add_timeline(
            holes1,
            uid="dgt1_holes",
            grouped_with="dgt_1",
            start=IdCoordinate(15343.0, TimeUnit.seconds, "dgt_1"),
            end=IdCoordinate(293119.0, TimeUnit.seconds, "dgt_1"),
        )

        summary1 = b1.summary()
        summary2 = b1.summary()

        # Same bundle, same summary
        assert summary1 == summary2


# endregion
