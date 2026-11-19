"""Error handling tests for loaders.

These tests verify that loaders handle corrupt, malformed, and invalid input
gracefully by raising appropriate exceptions with helpful messages.

IMPORTANT: These tests ensure research pipelines don't silently produce
incorrect results from corrupt data.
"""

from __future__ import annotations

import pytest

from timetoalign.testdata import ensure_data

# Test data paths
CORRUPT_DIR = ensure_data("fixtures") / "corrupt"


class TestMidiLoaderErrorHandling:
    """Error handling tests for MIDI loaders."""

    @pytest.fixture
    def perf_loader(self):
        """Get PerformanceMidiLoader."""
        from timetoalign.loader.midi import PerformanceMidiLoader

        return PerformanceMidiLoader()

    @pytest.fixture
    def score_loader(self):
        """Get ScoreMidiLoader."""
        from timetoalign.loader.midi import ScoreMidiLoader

        return ScoreMidiLoader()

    def test_invalid_midi_header_raises(self, perf_loader):
        """Text file with .mid extension raises error.

        This catches the common case of a file that isn't actually MIDI.
        """
        invalid_file = CORRUPT_DIR / "invalid_header.mid"

        # PerformanceMidiLoader wraps all mido errors as ValueError
        # (see loader/midi/performance.py lines 106-110)
        with pytest.raises(ValueError):
            perf_loader.load(invalid_file)

    def test_truncated_midi_raises(self, perf_loader):
        """Truncated MIDI file raises error.

        This catches the case of a download that was interrupted.
        """
        truncated_file = CORRUPT_DIR / "truncated.mid"

        # PerformanceMidiLoader wraps all mido errors as ValueError
        with pytest.raises(ValueError):
            perf_loader.load(truncated_file)

    def test_empty_midi_raises(self, perf_loader):
        """Empty file raises error.

        This catches the case of an empty or zero-byte file.
        """
        empty_file = CORRUPT_DIR / "empty.mid"

        # PerformanceMidiLoader wraps all mido errors as ValueError
        with pytest.raises(ValueError):
            perf_loader.load(empty_file)

    def test_nonexistent_file_raises(self, perf_loader):
        """Loading nonexistent file raises error (wrapped in ValueError)."""
        nonexistent = CORRUPT_DIR / "this_file_does_not_exist.mid"

        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            perf_loader.load(nonexistent)

    def test_directory_instead_of_file_raises(self, perf_loader):
        """Loading a directory raises error."""
        with pytest.raises((IsADirectoryError, OSError, ValueError)):
            perf_loader.load(CORRUPT_DIR)


class TestScoreLoaderErrorHandling:
    """Error handling tests for score loaders (MusicXML, etc.)."""

    @pytest.fixture
    def partitura_loader(self):
        """Get PartituraLoader."""
        from timetoalign.loader.score import PartituraLoader

        return PartituraLoader()

    @pytest.fixture
    def music21_loader(self):
        """Get Music21Loader if available."""
        try:
            from timetoalign.loader.score import Music21Loader

            return Music21Loader()
        except ImportError:
            pytest.skip("Music21Loader requires music21")

    def test_malformed_xml_raises_partitura(self, partitura_loader):
        """Malformed XML raises parsing error.

        This catches the case of corrupted or incomplete XML.
        """
        malformed_file = CORRUPT_DIR / "malformed.musicxml"

        # Partitura delegates to lxml, which raises XMLSyntaxError
        from lxml.etree import XMLSyntaxError

        with pytest.raises(XMLSyntaxError):
            partitura_loader.load(malformed_file)

    def test_malformed_xml_raises_music21(self, music21_loader):
        """Malformed XML raises parsing error with Music21."""
        malformed_file = CORRUPT_DIR / "malformed.musicxml"

        # Music21 delegates to xml.etree, which raises ParseError
        from xml.etree.ElementTree import ParseError

        with pytest.raises(ParseError):
            music21_loader.load(malformed_file)

    def test_nonexistent_xml_raises(self, partitura_loader):
        """Loading nonexistent XML raises FileNotFoundError."""
        nonexistent = CORRUPT_DIR / "this_file_does_not_exist.musicxml"

        with pytest.raises((FileNotFoundError, OSError)):
            partitura_loader.load(nonexistent)


class TestEventStoreErrorHandling:
    """Error handling tests for EventStore operations."""

    def test_from_dicts_allows_null_id(self):
        """from_dicts allows null id (lenient parsing)."""
        from timetoalign.core import TimeUnit
        from timetoalign.storage import EventData

        # Missing 'id' field - implementation is lenient
        rows = [
            {"temporal_type": "instant", "event_type": "Note"},
        ]

        store = EventData.from_dicts(rows, unit=TimeUnit.ticks)
        assert store.count == 1  # Allowed, but id is null

    def test_from_dicts_with_invalid_temporal_type(self):
        """from_dicts accepts invalid temporal_type (lenient parsing).

        The schema accepts any string for temporal_type.
        Validation happens at higher levels if needed.
        """
        from timetoalign.core import TimeUnit
        from timetoalign.storage import EventData

        rows = [
            {"id": "n1", "temporal_type": "invalid_type", "event_type": "Note"},
        ]

        # Implementation is lenient - accepts any string
        store = EventData.from_dicts(rows, unit=TimeUnit.ticks)
        assert store.count == 1


class TestConversionMapErrorHandling:
    """Error handling tests for ConversionMap operations."""

    def test_linear_map_zero_scalar_raises(self):
        """LinearMap with zero scalar raises (not invertible)."""
        from timetoalign.maps.linear import LinearMap

        with pytest.raises(ValueError, match="not invertible"):
            LinearMap(scalar=0.0)

    def test_table_map_mismatched_lengths_raises(self):
        """TableMap with mismatched x/y lengths raises."""
        from timetoalign.maps.table import TableMap

        with pytest.raises(ValueError, match="same length"):
            TableMap(x_values=[0, 1, 2], y_values=[0, 1])

    def test_table_map_non_monotonic_x_raises(self):
        """TableMap with non-monotonic x values raises."""
        from timetoalign.maps.table import TableMap

        with pytest.raises(ValueError, match="monotonically increasing"):
            TableMap(x_values=[0, 2, 1], y_values=[0, 2, 1])

    def test_table_map_single_point_raises(self):
        """TableMap with single point raises (need at least 2)."""
        from timetoalign.maps.table import TableMap

        with pytest.raises(ValueError, match="at least 2"):
            TableMap(x_values=[0], y_values=[0])

    def test_table_map_extrapolation_error_policy(self):
        """TableMap with error extrapolation raises on out-of-bounds."""
        from timetoalign.core.enums import ExtrapolationPolicy
        from timetoalign.maps.table import TableMap

        m = TableMap(
            x_values=[0, 10], y_values=[0, 20], extrapolate=ExtrapolationPolicy.error
        )

        # In-bounds works
        assert m(5) == 10.0

        # Out-of-bounds raises
        with pytest.raises(ValueError, match="outside table bounds"):
            m(20)

        with pytest.raises(ValueError, match="outside table bounds"):
            m(-1)


class TestCoordinateErrorHandling:
    """Error handling tests for Coordinate operations."""

    def test_coordinate_incompatible_unit_arithmetic(self):
        """Arithmetic between incompatible units raises."""
        from timetoalign.core import Coordinate, Duration, TimeUnit

        # Coordinate + Coordinate is no longer legal at all; demonstrate the
        # "incompatible unit" path via Coordinate + Duration instead.
        c = Coordinate(10, TimeUnit.ticks)
        d = Duration(5, TimeUnit.seconds)

        with pytest.raises((TypeError, ValueError)):
            _ = c + d

    def test_coordinate_incompatible_unit_comparison(self):
        """Comparison between incompatible units raises."""
        from timetoalign.core import Coordinate, TimeUnit

        c1 = Coordinate(10, TimeUnit.ticks)
        c2 = Coordinate(5, TimeUnit.seconds)

        with pytest.raises((TypeError, ValueError)):
            _ = c1 < c2


class TestTimelineErrorHandling:
    """Error handling tests for Timeline operations."""

    def test_timeline_invalid_unit_for_domain(self):
        """Timeline rejects invalid units for its domain."""
        from timetoalign.core import TimeUnit
        from timetoalign.timelines import ContinuousLogicalTimeline

        # Logical timeline should reject physical units
        with pytest.raises(ValueError):
            ContinuousLogicalTimeline(unit=TimeUnit.seconds)

    def test_timeline_add_child_wrong_domain(self):
        """Adding child with incompatible domain raises."""
        from timetoalign.timelines import (
            ContinuousLogicalTimeline,
            ContinuousPhysicalTimeline,
        )

        parent = ContinuousLogicalTimeline.empty()
        child = ContinuousPhysicalTimeline.empty()

        with pytest.raises((ValueError, TypeError)):
            parent.add_child(child, offset=parent.make_coordinate(0))

    def test_timeline_expansion_when_locked(self):
        """Expanding locked timeline raises."""
        from fractions import Fraction

        from timetoalign.timelines import ContinuousLogicalTimeline

        tl = ContinuousLogicalTimeline(length=Fraction(10), locked=True)

        # Try to add event beyond length (using scalar for instant, not dict)
        with pytest.raises((ValueError, RuntimeError)):
            tl.add_events(
                [
                    {
                        "id": "n1",
                        "temporal_type": "instant",
                        "event_type": "Note",
                        "instant": 20.0,  # Scalar value, not dict
                    }
                ],
                allow_expansion=False,
            )
