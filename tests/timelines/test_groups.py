"""Tests for TimelineGroup."""

from __future__ import annotations

import pytest

from timetoalign.core import Coordinate, IdCoordinate, Interval
from timetoalign.core.enums import NumberType, TimeUnit
from timetoalign.core.fields import field_metadata
from timetoalign.core.timestamp import Stamp, TimeStamp
from timetoalign.maps import TableMap
from timetoalign.storage.events import EventData
from timetoalign.timelines import (
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    TimelineGroup,
)

# region Fixtures


@pytest.fixture
def dgt_timeline() -> DiscreteGraphicalTimeline:
    """Create a discrete graphical timeline (4875 pixels)."""
    return DiscreteGraphicalTimeline(length=4875, unit="pixels", uid="dgt_1")


@pytest.fixture
def audio_timeline() -> ContinuousPhysicalTimeline:
    """Create a continuous physical timeline (150 seconds)."""
    return ContinuousPhysicalTimeline(length=150.0, unit="seconds", uid="audio")


@pytest.fixture
def score_timeline() -> ContinuousPhysicalTimeline:
    """Create a score timeline (100 units, e.g., beats)."""
    return ContinuousPhysicalTimeline(length=100.0, unit="seconds", uid="score")


# endregion


# region Event ID Timestamp Lookup Tests


class TestEventIdTimestampLookup:
    """Event IDs resolve through group timestamp coordinates."""

    @staticmethod
    def _group() -> TimelineGroup:
        """Create two linearly aligned timelines with two source events."""
        audio = ContinuousPhysicalTimeline(length=10.0, unit="seconds", uid="audio")
        audio.add_events(
            [
                {"id": "event:one", "event_type": "Marker", "instant": 2.0},
                {"id": "event:two", "event_type": "Marker", "instant": 6.0},
            ]
        )
        score = ContinuousPhysicalTimeline(length=20.0, unit="seconds", uid="score")
        return TimelineGroup(id="paired", timelines=[audio, score])

    def test_get_timestamp_for_returns_source_coordinate_fields(self) -> None:
        """The event coordinate identifies its source and mapped member coordinates."""
        timestamp = self._group().get_timestamp_for("event:one")

        assert type(timestamp) is TimeStamp
        assert timestamp.axis.value == 2.0
        assert timestamp.source_id == "audio"
        assert timestamp.present_timelines == ["audio", "score"]
        assert timestamp.get_coordinate_for("audio", format="float") == 2.0
        assert timestamp.get_coordinate_for("score", format="float") == 4.0

    def test_get_timestamp_for_missing_event_raises_key_error(self) -> None:
        """Absent IDs retain the lookup API's KeyError contract."""
        with pytest.raises(KeyError):
            self._group().get_timestamp_for("missing")

    def test_get_timestamps_for_preserves_requested_order(self) -> None:
        """Bulk lookup yields one stamp per requested ID in the original order."""
        timestamps = self._group().get_timestamps_for(["event:two", "event:one"])

        assert len(timestamps) == 2
        assert [stamp.source_id for stamp in timestamps] == ["audio", "audio"]
        assert [
            stamp.get_coordinate_for("audio", format="float") for stamp in timestamps
        ] == [
            6.0,
            2.0,
        ]
        assert [
            stamp.get_coordinate_for("score", format="float") for stamp in timestamps
        ] == [
            12.0,
            4.0,
        ]

    def test_get_timestamps_for_missing_event_raises_atomically(self) -> None:
        """One unknown ID aborts the batch instead of emitting an empty row."""
        with pytest.raises(KeyError):
            self._group().get_timestamps_for(["event:one", "missing"])


# endregion


# region TimelineGroup Creation Tests


class TestTimelineGroupCreation:
    """Tests for TimelineGroup creation and initialization."""

    def test_create_empty_group(self) -> None:
        """Test creating an empty group."""
        group = TimelineGroup(id="test_group")

        assert group.id == "test_group"
        assert group.n_timelines == 0
        assert group.get_timestamp_table().num_rows == 0

    def test_create_group_with_initial_timeline(
        self, dgt_timeline: DiscreteGraphicalTimeline
    ) -> None:
        """Test creating a group with an initial timeline."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline])

        assert group.n_timelines == 1
        assert group.get_timestamp_table().num_rows == 2  # Start and end
        assert "dgt_1" in group

    def test_create_group_with_multiple_timelines(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test creating a group with multiple initial timelines."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        assert group.n_timelines == 2
        assert group.get_timestamp_table().num_rows == 2
        assert "dgt_1" in group
        assert "audio" in group

    def test_auto_generated_id(self) -> None:
        """An omitted id is minted as ``group_N``, N the smallest free number."""
        assert TimelineGroup().id == "group_1"
        held = ContinuousPhysicalTimeline(length=1.0, uid="group_1")
        assert TimelineGroup(timelines=[held]).id == "group_2"


class TestTimelineGroupEvents:
    """Event aggregation preserves Arrow data and source provenance."""

    def test_get_events_returns_promoted_event_data(self) -> None:
        """Member-specific fields are null-filled in one EventData table."""
        first = ContinuousPhysicalTimeline(length=10.0, uid="first")
        first.add_events([{"start": 1.0, "name": "a"}])
        second = ContinuousPhysicalTimeline(length=10.0, uid="second")
        second.add_events([{"start": 2.0, "velocity": 64}])

        events = TimelineGroup(id="group", timelines=[first, second]).get_events()

        assert isinstance(events, EventData)
        assert events.table.num_rows == 2
        assert events.table.column("timeline_id").to_pylist() == ["first", "second"]
        assert events.table.column("name").to_pylist() == ["a", None]
        assert events.table.column("velocity").to_pylist() == [None, 64]

    def test_get_events_empty_group_returns_empty_event_data(self) -> None:
        """An empty group has an empty Arrow-backed result."""
        events = TimelineGroup(id="empty").get_events()

        assert isinstance(events, EventData)
        assert events.table.num_rows == 0
        assert events.table.column("timeline_id").to_pylist() == []

    def test_get_events_rejects_incompatible_shared_column_types(self) -> None:
        """A shared column cannot silently change its Arrow type."""
        first = ContinuousPhysicalTimeline(length=10.0, uid="first")
        first.add_events([{"start": 1.0, "tag": 1}])
        second = ContinuousPhysicalTimeline(length=10.0, uid="second")
        second.add_events([{"start": 2.0, "tag": "x"}])

        group = TimelineGroup(id="group", timelines=[first, second])
        with pytest.raises(
            ValueError,
            match=r"^Conflicting Arrow types for column 'tag': int64 and string\.$",
        ):
            group.get_events()

    def test_get_events_overwrites_member_timeline_id(self) -> None:
        """Group membership is authoritative provenance."""
        timeline = ContinuousPhysicalTimeline(length=10.0, uid="authoritative")
        timeline.add_events([{"start": 1.0, "timeline_id": "stale"}])

        events = TimelineGroup(id="group", timelines=[timeline]).get_events()

        assert events.table.column_names.count("timeline_id") == 1
        assert events.table.column("timeline_id").to_pylist() == ["authoritative"]

    def test_get_events_propagates_timeline_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A member query failure is visible to the group caller."""
        timeline = ContinuousPhysicalTimeline(length=10.0, uid="broken")

        def fail(**kwargs: object) -> EventData:
            raise RuntimeError("member query failed")

        monkeypatch.setattr(timeline, "get_events", fail)
        group = TimelineGroup(id="group", timelines=[timeline])

        with pytest.raises(RuntimeError, match="member query failed"):
            group.get_events()


# endregion


# region Add Timeline Tests


class TestAddTimeline:
    """Tests for add_timeline() method."""

    def test_add_first_timeline(self, dgt_timeline: DiscreteGraphicalTimeline) -> None:
        """Test adding the first timeline to an empty group."""
        group = TimelineGroup(id="test_group")
        group.add_timeline(dgt_timeline)

        assert group.n_timelines == 1

        # The stored rows are the common start and end
        rows = group.get_timestamp_table(format="dataframe")
        assert rows["dgt_1 (pixels)"].tolist() == [0, 4875]

    def test_add_second_timeline_default_mapping(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test adding a second timeline with default (full extent) mapping."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline])
        group.add_timeline(audio_timeline)

        assert group.n_timelines == 2

        # Both span the full extent: one start row, one end row
        rows = group.get_timestamp_table(format="dataframe")
        assert rows["dgt_1 (pixels)"].tolist() == [0, 4875]
        assert rows["audio (seconds)"].tolist() == [0.0, 150.0]

    def test_partial_participation_is_a_child(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
        score_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """A score covering audio seconds 45-135 is grouped with that part.

        The part is a child of the audio at offset 45 with length 90; the
        score spans the whole of it. The score's 0 and 100 are the part's 0
        and 90, which the offset places at audio 45 and 135.
        """
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])
        section = ContinuousPhysicalTimeline(length=90.0, uid="audio_section")
        audio_timeline.add_child(section, offset=45.0)

        score_group = TimelineGroup(
            id="score_group", timelines=[section, score_timeline]
        )

        assert group.n_timelines == 2
        assert score_group.get_interval_for("score").start.value == 0.0
        assert score_group.get_interval_for("score").end.value == 100.0
        # audio 75 is the section's 30, a third of the way, so score 100/3
        at_75 = group.get_timestamp_at(75.0, "audio")
        assert at_75.get_coordinate_for("audio_section", format="float") == 30.0
        assert score_group.get_coordinate_at(30.0, "audio_section") == IdCoordinate(
            30.0, TimeUnit.seconds, "audio_section"
        )
        assert (
            score_group.get_timestamp_at(30.0, "audio_section").get_coordinate_for(
                "score", format="float"
            )
            == 33.333333333333336
        )
        # Before the section starts, the audio's stamp holds no section cell
        with pytest.raises(KeyError):
            group.get_timestamp_at(30.0, "audio").get_coordinate_for("audio_section")

    def test_add_duplicate_timeline_raises(
        self, dgt_timeline: DiscreteGraphicalTimeline
    ) -> None:
        """Test that adding a duplicate timeline raises an error."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline])

        with pytest.raises(ValueError, match="already in group"):
            group.add_timeline(dgt_timeline)


# endregion


# region Timestamp Access Tests


class TestTimestampAccess:
    """Tests for timestamp access methods."""

    def test_end_row_stamp_contract(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """The stamp at the common end exposes the same contract as other stamps."""
        dgt_timeline.add_conversion_map(
            TableMap(
                x_values=[0.0, 4875.0],
                y_values=[0.0, 150.0],
                source_unit=TimeUnit.pixels,
                target_unit=TimeUnit.seconds,
                uid="dgt_1-seconds",
            )
        )
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        gt = group.get_timestamp_at(4875, "dgt_1")

        assert isinstance(gt, Stamp)
        assert gt.source is group
        assert gt.source_id == "dgt_1"
        assert gt.is_interpolated is False
        assert gt.axis.value == 4875
        assert gt.get_coordinate_for("dgt_1", format="coordinate") == Coordinate(
            4875, TimeUnit.pixels
        )
        assert gt.get_unit(TimeUnit.seconds, format="float") == 150.0
        assert set(gt.to_dict()) == {"dgt_1", "audio"}

    def test_get_timestamp_table(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test get_timestamp_table() returns PyArrow table."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        table = group.get_timestamp_table()

        assert table.num_rows == 2
        assert "dgt_1" in table.column_names
        assert "audio" in table.column_names

    def test_get_timestamp_table_filtered(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test get_timestamp_table() with filter."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        table = group.get_timestamp_table(timeline_filter={"dgt_1"})

        assert table.num_rows == 2
        assert "dgt_1" in table.column_names
        assert "audio" not in table.column_names

    def test_dataframe_format(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """format="dataframe" returns a pandas frame with units in column names."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        df = group.get_timestamp_table(format="dataframe")

        assert len(df) == 2
        # Column names now include units like "dgt1 (pixels)"
        assert "dgt_1 (pixels)" in df.columns
        assert "audio (seconds)" in df.columns

        # Test units=False returns raw column names
        df_no_units = group.get_timestamp_table(format="dataframe", units=False)
        assert "dgt_1" in df_no_units.columns
        assert "audio" in df_no_units.columns


# endregion


# region Interpolation Tests


class TestGetTimestampAt:
    """Tests for get_timestamp_at() interpolation."""

    def test_get_timestamps_at_uses_each_embedded_timeline_id(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Batch lookup accepts independently qualified coordinate rows."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        stamps = group.get_timestamps_at(
            [
                IdCoordinate(75.0, TimeUnit.seconds, "audio"),
                IdCoordinate(2437.5, TimeUnit.pixels, "dgt_1"),
            ]
        )

        assert [
            stamp.get_coordinate_for("audio", format="float") for stamp in stamps
        ] == [
            75.0,
            75.01538461538462,
        ]
        assert [
            stamp.get_coordinate_for("dgt_1", format="int") for stamp in stamps
        ] == [
            2438,
            2438,
        ]

    def test_get_timestamps_at_requires_id_for_plain_values(
        self,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Batch lookup rejects raw rows when no timeline ID is supplied."""
        group = TimelineGroup(id="test_group", timelines=[audio_timeline])

        with pytest.raises(
            ValueError,
            match="timeline_id is required unless coordinate is an IdCoordinate",
        ):
            group.get_timestamps_at([25.0])

    def test_exact_boundary_match(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test get_timestamp_at() at exact boundary."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        ts = group.get_timestamp_at(0.0, "audio")
        assert ts.get_coordinate_for("audio", format="float") == 0.0
        assert ts.get_coordinate_for("dgt_1", format="float") == 0.0
        assert ts.is_interpolated is False

    def test_interpolation_midpoint(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test interpolation at midpoint."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        # 75 seconds is halfway through 150 seconds
        ts = group.get_timestamp_at(75.0, "audio")

        assert ts.get_coordinate_for("audio", format="float") == 75.0
        # Discrete timelines round to nearest integer: 4875/2 = 2437.5 → 2438
        assert ts.get_coordinate_for("dgt_1", format="int") == 2438
        assert ts.is_interpolated is True

    def test_interpolation_arbitrary_point(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test interpolation at an arbitrary point."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        # 30 seconds = 20% of 150 seconds
        ts = group.get_timestamp_at(30.0, "audio")

        assert ts.get_coordinate_for("audio", format="float") == 30.0
        assert ts.get_coordinate_for("dgt_1", format="int") == 975

    def test_interpolation_reverse_lookup(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test interpolation looking up by different timeline."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        # 2437.5 pixels is halfway through 4875 pixels
        ts = group.get_timestamp_at(2437.5, "dgt_1")

        # Discrete timelines round to nearest integer: 2437.5 → 2438
        assert ts.get_coordinate_for("dgt_1", format="int") == 2438
        # Every projection of the stamp is a projection of that one pixel, so
        # the audio reading is pixel 2438's, not the half-pixel's.
        assert ts.get_coordinate_for("audio", format="float") == 2438 / 4875 * 150
        assert ts.get_coordinate_for("audio", format="float") == 75.01538461538462

    def test_interpolation_out_of_range_raises(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test that querying outside range raises ValueError."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        with pytest.raises(ValueError, match="outside range"):
            group.get_timestamp_at(200.0, "audio")

    def test_discrete_out_of_range_coordinate_uses_integer_display(self) -> None:
        """A discrete coordinate is rendered as the integer it represents."""
        score = DiscreteLogicalTimeline(length=12000, uid="score")
        group = TimelineGroup(id="test_group", timelines=[score])

        with pytest.raises(ValueError) as error:
            group.get_timestamp_at(IdCoordinate(12673.0, TimeUnit.ticks, "score"))

        assert str(error.value) == "Coordinate 12673 outside range for 'score'"

    def test_continuous_out_of_range_coordinate_keeps_fractional_part(
        self,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """A continuous coordinate retains a meaningful fractional part."""
        group = TimelineGroup(id="test_group", timelines=[audio_timeline])

        with pytest.raises(ValueError) as error:
            group.get_timestamp_at(150.25, "audio")

        assert str(error.value) == "Coordinate 150.25 outside range for 'audio'"

    def test_out_of_range_coordinate_avoids_scientific_notation(
        self,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """A large coordinate is expanded to fixed-point display."""
        group = TimelineGroup(id="test_group", timelines=[audio_timeline])

        with pytest.raises(ValueError) as error:
            group.get_timestamp_at(1e20, "audio")

        assert str(error.value) == (
            "Coordinate 100000000000000000000 outside range for 'audio'"
        )

    def test_interpolation_nonexistent_timeline_raises(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
    ) -> None:
        """Test that querying non-existent timeline raises KeyError."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline])

        with pytest.raises(KeyError):
            group.get_timestamp_at(50.0, "nonexistent")


# endregion


# region Conversion Tests


class TestCoordinateRetrieval:
    """Tests for typed cross-member coordinate retrieval."""

    def test_convert_basic(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test basic coordinate conversion."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        # 75 seconds -> 2438 pixels (discrete: 2437.5 rounds to 2438)
        result = group.get_coordinate_at(
            IdCoordinate(75.0, TimeUnit.seconds, "audio"),
            timeline_id="dgt_1",
            format="coordinate",
        )
        assert result == Coordinate(2438, TimeUnit.pixels)

        # 2437.5 pixels names pixel 2438, so the answer is pixel 2438's second
        # position. A pixel axis holds integers; there is no half-pixel to
        # convert from.
        result = group.get_coordinate_at(
            IdCoordinate(2437.5, TimeUnit.pixels, "dgt_1"),
            timeline_id="audio",
            format="coordinate",
        )
        assert result == Coordinate(2438 / 4875 * 150, TimeUnit.seconds)
        assert result == Coordinate(75.01538461538462, TimeUnit.seconds)

    def test_convert_same_timeline(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
    ) -> None:
        """Test conversion to same timeline returns same value."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline])

        result = group.get_coordinate_at(
            IdCoordinate(1000, TimeUnit.pixels, "dgt_1"),
            timeline_id="dgt_1",
            format="coordinate",
        )
        assert result == Coordinate(1000, TimeUnit.pixels)

    def test_convert_accepts_coordinate_objects(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """convert() accepts raw value, Coordinate, and IdCoordinate forms.

        The endpoints are named by ``source``/``target``, so an IdCoordinate's
        own timeline_id is informational only; its value is what is converted.
        """
        from timetoalign.core import Coordinate, IdCoordinate
        from timetoalign.core.enums import TimeUnit

        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        raw = group.get_coordinate_at(2438, timeline_id="dgt_1", format="coordinate")
        assert raw == Coordinate(2438, TimeUnit.pixels)
        from_coord = group.get_coordinate_at(
            Coordinate(2438, TimeUnit.pixels),
            timeline_id="dgt_1",
            format="coordinate",
        )
        assert from_coord == Coordinate(2438, TimeUnit.pixels)
        from_id = group.get_coordinate_at(
            IdCoordinate(75.0, TimeUnit.seconds, "audio"),
            timeline_id="dgt_1",
            format="coordinate",
        )
        assert from_id == Coordinate(2438, TimeUnit.pixels)

    def test_convert_rejects_unsupported_type(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """The positional getter raises TypeError for a key string."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])
        with pytest.raises(TypeError):
            group.get_coordinate_at("x", timeline_id="dgt_1")  # type: ignore[arg-type]


# endregion


# region Range and Utilities Tests


class TestIntervalAndUtilities:
    """Tests for typed member extents and utility methods."""

    def test_get_interval_for(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Typed member extents return exact bounds."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        dgt_range = group.get_interval_for("dgt_1")
        audio_range = group.get_interval_for("audio")

        assert (dgt_range.start.value, dgt_range.end.value) == (0, 4875)
        assert (audio_range.start.value, audio_range.end.value) == (0.0, 150.0)

    def test_get_interval_for_nonexistent_raises(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
    ) -> None:
        """An unknown member raises KeyError."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline])

        with pytest.raises(KeyError):
            group.get_interval_for("nonexistent")

    def test_timeline_ids(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test timeline_ids property."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        ids = group.timeline_ids
        assert "dgt_1" in ids
        assert "audio" in ids
        assert len(ids) == 2


# endregion


# region Remove Timeline Tests


class TestRemoveTimeline:
    """Tests for remove_timeline() method."""

    def test_remove_timeline(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test removing a timeline from the group."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        removed = group.remove_timeline("audio")

        assert removed is audio_timeline
        assert group.n_timelines == 1
        assert "audio" not in group
        assert "dgt_1" in group

    def test_remove_nonexistent_raises(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
    ) -> None:
        """Test removing non-existent timeline raises KeyError."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline])

        with pytest.raises(KeyError):
            group.remove_timeline("nonexistent")

    def test_remove_all_timelines(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
    ) -> None:
        """Test removing all timelines results in empty group."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline])

        group.remove_timeline("dgt_1")

        assert group.n_timelines == 0
        assert group.get_timestamp_table().num_rows == 0


# endregion


# region Integration Tests


class TestGroupIntegration:
    """Integration tests for TimelineGroup with real timelines."""

    def test_thoresen_poc_setup(self) -> None:
        """Test setup similar to Thoresen PoC from spec.

        DGT1 (2009): 4875 pixels total
        DGT2 (2010): 4328 pixels total
        Both should map to 150 seconds of audio.
        """
        # Create timelines
        dgt1 = DiscreteGraphicalTimeline(length=4875, unit="pixels", uid="dgt_1")
        dgt2 = DiscreteGraphicalTimeline(length=4328, unit="pixels", uid="dgt_2")
        audio = ContinuousPhysicalTimeline(length=150.0, unit="seconds", uid="audio")

        # Create group 1: DGT1 + audio
        group1 = TimelineGroup(id="DGT1_Group", timelines=[dgt1, audio])

        # Create group 2: DGT2 + audio
        group2 = TimelineGroup(id="DGT2_Group", timelines=[dgt2, audio])

        # Verify conversions in group1
        assert group1.get_coordinate_at(
            IdCoordinate(0, TimeUnit.pixels, "dgt_1"),
            timeline_id="audio",
            format="coordinate",
        ) == Coordinate(0.0, TimeUnit.seconds)
        assert group1.get_coordinate_at(
            IdCoordinate(4875, TimeUnit.pixels, "dgt_1"),
            timeline_id="audio",
            format="coordinate",
        ) == Coordinate(150.0, TimeUnit.seconds)
        # 2437.5 names pixel 2438 on an integer axis.
        assert group1.get_coordinate_at(
            IdCoordinate(2437.5, TimeUnit.pixels, "dgt_1"),
            timeline_id="audio",
            format="coordinate",
        ) == Coordinate(2438 / 4875 * 150, TimeUnit.seconds)

        # Verify conversions in group2
        assert group2.get_coordinate_at(
            IdCoordinate(0, TimeUnit.pixels, "dgt_2"),
            timeline_id="audio",
            format="coordinate",
        ) == Coordinate(0.0, TimeUnit.seconds)
        assert group2.get_coordinate_at(
            IdCoordinate(4328, TimeUnit.pixels, "dgt_2"),
            timeline_id="audio",
            format="coordinate",
        ) == Coordinate(150.0, TimeUnit.seconds)
        assert group2.get_coordinate_at(
            IdCoordinate(2164, TimeUnit.pixels, "dgt_2"),
            timeline_id="audio",
            format="coordinate",
        ) == Coordinate(75.0, TimeUnit.seconds)

    def test_three_timeline_group(self) -> None:
        """Two parts of one audio timeline, each grouped with what spans it.

        The image spans audio seconds 0-150 and the score audio seconds
        45-135: each such part is a child of the audio, and each group pairs
        a part with the timeline spanning it.
        """
        audio = ContinuousPhysicalTimeline(length=180.0, unit="seconds", uid="audio")
        dgt = DiscreteGraphicalTimeline(length=4875, unit="pixels", uid="dgt")
        score = ContinuousPhysicalTimeline(length=100.0, unit="seconds", uid="score")
        first = ContinuousPhysicalTimeline(length=150.0, uid="audio_first")
        section = ContinuousPhysicalTimeline(length=90.0, uid="audio_section")
        audio.add_child(first, offset=0.0)
        audio.add_child(section, offset=45.0)

        image_group = TimelineGroup(id="image_group", timelines=[first, dgt])
        score_group = TimelineGroup(id="score_group", timelines=[section, score])

        # At audio 75.0: the section starts at 45, so the score is
        # (75-45)/90 = 1/3 of the way through its 100 seconds
        ts = audio.get_timestamp_at(75.0)
        assert ts.get_coordinate_for("audio", format="float") == 75.0
        # 100 * 30/90 = 33.333... is a non-terminating ratio, so this remains
        # a genuine float comparison.
        assert (
            score_group.get_timestamp_at(
                ts.get_coordinate_for("audio_section"), "audio_section"
            ).get_coordinate_for("score", format="float")
            == 33.333333333333336
        )
        # 75 of 150 seconds is half of 4875 pixels, 2437.5, which names 2438
        assert (
            image_group.get_timestamp_at(
                ts.get_coordinate_for("audio_first"), "audio_first"
            ).get_coordinate_for("dgt", format="int")
            == 2438
        )

        # At audio 30.0 (before the score's part starts)
        ts_early = audio.get_timestamp_at(30.0)
        with pytest.raises(KeyError):
            ts_early.get_coordinate_for("audio_section")
        assert ts_early.get_coordinate_for("audio_first").timeline_id == "audio_first"

        # At audio 160.0 (after the image's part ends)
        ts_late = audio.get_timestamp_at(160.0)
        with pytest.raises(KeyError):
            ts_late.get_coordinate_for("audio_first")
        with pytest.raises(KeyError):
            ts_late.get_coordinate_for("audio_section")

    def test_round_trip_conversion(self) -> None:
        """Test that conversions are reversible within quantization error.

        Discrete timelines round interpolated coordinates to integers,
        introducing a quantization error of at most ±0.5 in the discrete
        unit.  The round-trip tolerance in seconds is therefore
        0.5 * (150 / 4875) ≈ 0.0154 seconds.
        """
        audio = ContinuousPhysicalTimeline(length=150.0, unit="seconds", uid="audio")
        dgt = DiscreteGraphicalTimeline(length=4875, unit="pixels", uid="dgt")

        group = TimelineGroup(id="test_group", timelines=[audio, dgt])

        # Round trip: audio -> dgt -> audio
        original = 67.5
        dgt_coord = group.get_coordinate_at(
            IdCoordinate(original, TimeUnit.seconds, "audio"),
            timeline_id="dgt",
            format="coordinate",
        )
        assert isinstance(dgt_coord.value, int)  # discrete → integer
        back = group.get_coordinate_at(
            IdCoordinate.from_coordinate(dgt_coord, "dgt"),
            timeline_id="audio",
            format="coordinate",
        )

        assert back.value == 67.50769230769231


# endregion


# region Summary Test


class TestTimelineGroupSummary:
    """Tests for TimelineGroup.summary()."""

    def test_summary(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test summary() method."""
        group = TimelineGroup(
            id="test_group",
            name="Test Group",
            timelines=[dgt_timeline, audio_timeline],
        )

        summary = group.summary()

        assert summary["id"] == "test_group"
        assert summary["name"] == "Test Group"
        assert summary["n_timelines"] == 2
        assert "dgt_1" in summary["timeline_ids"]
        assert "audio" in summary["timeline_ids"]


# endregion


# region Unified Timestamp API Tests


class TestTimelineGroupTimestampAt:
    """Tests for the unified TimeStamp API via get_timestamp_at()."""

    def test_get_timestamp_at_basic(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test basic timestamp creation."""
        from timetoalign.core import TimeStamp

        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        ts = group.get_timestamp_at(75.0, "audio")
        assert isinstance(ts, TimeStamp)
        assert ts.axis.value == 75.0
        assert ts.source_id == "audio"

    def test_get_timestamp_at_without_conversion_maps(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Disabled conversion maps leave member timeline access available."""
        dgt_timeline.add_conversion_map(
            TableMap(
                x_values=[0.0, 4875.0],
                y_values=[0.0, 100.0],
                source_unit=TimeUnit.pixels,
                target_unit=TimeUnit.whole_notes,
                uid="pixels-to-beats",
            )
        )
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        ts = group.get_timestamp_at(4875.0, "dgt_1", conversion_maps=False)

        with pytest.raises(KeyError):
            ts.get_unit(TimeUnit.whole_notes)
        assert ts.get_coordinate_for("dgt_1", format="int") == 4875

    def test_get_timestamp_at_with_restricted_conversion_maps(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Restricted conversion maps expose only their target unit."""
        dgt_timeline.add_conversion_map(
            TableMap(
                x_values=[0.0, 4875.0],
                y_values=[0.0, 100.0],
                source_unit=TimeUnit.pixels,
                target_unit=TimeUnit.whole_notes,
                uid="pixels-to-beats",
            )
        )
        dgt_timeline.add_conversion_map(
            TableMap(
                x_values=[0.0, 4875.0],
                y_values=[0.0, 200.0],
                source_unit=TimeUnit.pixels,
                target_unit=TimeUnit.frames,
                uid="pixels-to-frames",
            )
        )
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        ts = group.get_timestamp_at(
            4875.0,
            "dgt_1",
            conversion_maps=[TimeUnit.whole_notes],
        )

        assert ts.get_unit(TimeUnit.whole_notes, format="float") == 100.0
        with pytest.raises(KeyError):
            ts.get_unit(TimeUnit.frames)

    def test_get_timestamp_at_coordinate_conversion(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test coordinate conversion via timestamp."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        # audio: 0 -> 150, dgt: 0 -> 4875
        # At audio=75 (half), dgt should be 4875/2 = 2437.5 -> rounded to 2438
        ts = group.get_timestamp_at(75.0, "audio")

        dgt_coord = ts.get_coordinate_for("dgt_1", format="int")
        # Discrete timelines are rounded to nearest integer
        assert dgt_coord == 2438

    def test_get_timestamp_at_bidirectional(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test bidirectional coordinate conversion."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        # From dgt to audio. 2437.5 names pixel 2438 on an integer axis.
        ts = group.get_timestamp_at(2437.5, "dgt_1")
        audio_coord = ts.get_coordinate_for("audio", format="float")
        assert audio_coord == 2438 / 4875 * 150

    def test_get_timestamp_at_unknown_timeline_raises(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
    ) -> None:
        """Test that unknown timeline raises KeyError."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline])

        with pytest.raises(KeyError, match="nonexistent"):
            group.get_timestamp_at(50.0, "nonexistent")

    def test_get_timestamp_at_empty_group_raises(self) -> None:
        """Test that empty group raises KeyError (no timelines)."""
        group = TimelineGroup(id="empty_group")

        # Empty group has no timelines, so KeyError is raised first
        with pytest.raises(KeyError, match="not in group"):
            group.get_timestamp_at(50.0, "any_timeline")

    def test_interval_stamp_from_get_timestamp_at(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Compose a TimeIntervalStamp from two get_timestamp_at() calls."""
        from timetoalign.core import TimeIntervalStamp

        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        start = group.get_timestamp_at(0.0, "audio")
        end = group.get_timestamp_at(100.0, "audio")
        interval = TimeIntervalStamp(
            intervals={
                timeline_id: Interval(
                    start=coordinate,
                    end=end.coordinates[timeline_id],
                )
                for timeline_id, coordinate in start.coordinates.items()
                if timeline_id in end.coordinates
            },
            source=group,
            source_id="audio",
        )

        assert isinstance(interval, TimeIntervalStamp)
        assert interval.duration.value == 100.0
        assert interval.source_id == "audio"

        # Check interval on dgt
        dgt_interval = interval.get_interval("dgt_1")
        # audio 0->100 maps to dgt 0->3250 (100/150 * 4875)
        assert dgt_interval.start.value == 0
        assert dgt_interval.end.value == 3250

    def test_timestamp_with_three_timelines(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
        score_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test coordinate conversion with three timelines."""
        group = TimelineGroup(
            id="test_group",
            timelines=[dgt_timeline, audio_timeline, score_timeline],
        )

        # All timelines aligned start-to-start and end-to-end
        # At audio midpoint (75.0):
        # - dgt should be 2437.5 -> rounded to 2438 (discrete timeline)
        # - score should be 50.0 (midpoint of 0-100)

        ts = group.get_timestamp_at(75.0, "audio")

        # Discrete timelines are rounded to nearest integer
        assert ts.get_coordinate_for("dgt_1", format="int") == 2438
        assert ts.get_coordinate_for("score", format="float") == 50.0
        assert ts.get_coordinate_for("audio", format="float") == 75.0

    def test_timestamp_same_timeline_returns_axis(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test that getting the same timeline returns axis value."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        ts = group.get_timestamp_at(75.0, "audio")

        # Getting the source timeline should return axis
        assert ts.get_coordinate_for("audio", format="float") == 75.0

    def test_implements_timestamp_source_protocol(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test that TimelineGroup implements TimeStampSource protocol."""
        from timetoalign.core.timestamp import TimeStampSource

        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        # Check protocol compliance
        assert isinstance(group, TimeStampSource)
        assert hasattr(group, "_get_unit_map_for_timeline")
        assert hasattr(group, "_get_unit_map")
        assert hasattr(group, "_get_related_timeline_ids")
        assert hasattr(group, "_get_available_units")

    def test_get_related_timeline_ids(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test _get_related_timeline_ids method."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        ids = group._get_related_timeline_ids()
        assert "dgt_1" in ids
        assert "audio" in ids
        assert len(ids) == 2

    def test_get_available_units_returns_empty(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
    ) -> None:
        """Test _get_available_units returns empty (groups don't have C-Maps)."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline])

        units = group._get_available_units()
        assert units == []


# endregion


# region TimelineGroup Unit Metadata Tests


class TestTimelineGroupUnitMetadata:
    """Tests for unit metadata in TimelineGroup timestamp tables."""

    def test_get_unit_for_timeline(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test _get_unit_for_timeline returns correct unit for each timeline."""
        from timetoalign.core.enums import TimeUnit

        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        dgt_unit = group._get_unit_for_timeline("dgt_1")
        assert dgt_unit == TimeUnit.pixels

        audio_unit = group._get_unit_for_timeline("audio")
        assert audio_unit == TimeUnit.seconds

    def test_get_unit_for_unknown_timeline_returns_none(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
    ) -> None:
        """Test _get_unit_for_timeline returns None for unknown timeline."""
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline])

        unit = group._get_unit_for_timeline("nonexistent")
        assert unit is None

    def test_timestamp_table_has_unit_metadata(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test that get_timestamp_table includes unit metadata on fields."""
        # Creating group with two timelines auto-creates boundaries
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        table = group.get_timestamp_table()

        # Check each timeline field has correct unit metadata
        dgt_field = table.schema.field("dgt_1")
        assert dgt_field.metadata is not None
        assert field_metadata(dgt_field)["unit"] == "pixels"
        assert field_metadata(dgt_field)["timeline_id"] == "dgt_1"

        audio_field = table.schema.field("audio")
        assert audio_field.metadata is not None
        assert field_metadata(audio_field)["unit"] == "seconds"
        assert field_metadata(audio_field)["timeline_id"] == "audio"

    def test_timestamp_table_preserves_metadata_after_add(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
        score_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test that unit metadata is preserved when adding timelines."""
        # Start with two timelines
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline, audio_timeline])

        # Add a third timeline
        group.add_timeline(score_timeline)

        table = group.get_timestamp_table()

        # Metadata should still be present after adding timeline
        dgt_field = table.schema.field("dgt_1")
        assert dgt_field.metadata is not None
        assert field_metadata(dgt_field)["unit"] == "pixels"

        audio_field = table.schema.field("audio")
        assert audio_field.metadata is not None
        assert field_metadata(audio_field)["unit"] == "seconds"

        # New timeline also has metadata
        score_field = table.schema.field("score")
        assert score_field.metadata is not None
        assert field_metadata(score_field)["unit"] == "seconds"
        assert field_metadata(score_field)["timeline_id"] == "score"

    def test_add_timeline_creates_metadata_for_new_column(
        self,
        dgt_timeline: DiscreteGraphicalTimeline,
        audio_timeline: ContinuousPhysicalTimeline,
    ) -> None:
        """Test that adding a timeline creates field with correct metadata."""
        # Start with one timeline
        group = TimelineGroup(id="test_group", timelines=[dgt_timeline])

        table_before = group.get_timestamp_table()
        assert "audio" not in table_before.schema.names

        # Add second timeline
        group.add_timeline(audio_timeline)

        table_after = group.get_timestamp_table()

        # New field should have metadata
        audio_field = table_after.schema.field("audio")
        assert audio_field.metadata is not None
        assert field_metadata(audio_field)["unit"] == "seconds"
        assert field_metadata(audio_field)["timeline_id"] == "audio"


# endregion


# region EventData field typing


def test_event_data_coerces_volta_to_integer() -> None:
    """Carried volta values retain the ordinal ending-number type."""
    null_only = EventData.from_dicts(
        [{"event_type": "Measure", "start": 0, "end": 1, "volta": None}],
        TimeUnit.quarters,
        NumberType.fraction,
    )
    with_volta = EventData.from_dicts(
        [{"event_type": "Measure", "start": 1, "end": 2, "volta": "2"}],
        TimeUnit.quarters,
        NumberType.fraction,
    )

    assert str(null_only._table.schema.field("volta").type) == "int64"
    assert with_volta._table.column("volta").to_pylist() == [2]
    null_only.extend(with_volta)
    assert null_only._table.column("volta").to_pylist() == [None, 2]


# endregion


# region TestTimelineGroupUnfold


@pytest.mark.slow
class TestTimelineGroupUnfold:
    """Tests for TimelineGroup.apply_flow() — group-level unfolding.

    Uses the Beethoven Op.18 No.4 iv multimodal score group containing:
    - CLT1: ContinuousLogicalTimeline (ABC v2.6, 878.5 quarters)
    - OpenScore: ContinuousLogicalTimeline (4th movement, 878.5 quarters)

    The FlowMap has 11 PlaythroughSections producing 1116 unfolded QB.
    Gold standard values from the test_unfolding.py module.
    """

    # Gold standard constants (same as test_unfolding.py)
    FOLDED_QB = 878.5
    UNFOLDED_QB = 1116
    N_SECTIONS = 11

    @pytest.fixture(scope="class")
    def score_data(self):
        """Build score group + flow data for unfolding tests."""
        from pathlib import Path

        from timetoalign.core.enums import FlowMode
        from timetoalign.loader.score import Ms3Loader
        from timetoalign.timelines.flow import ScoreFlowController

        data_dir = (
            Path(__file__).parent.parent
            / "data"
            / "score"
            / "beethoven_op18-4iv_multimodal"
        )
        abc_dir = data_dir / "ABC"
        os_dir = data_dir / "OpenScoreSQ"

        if not abc_dir.exists():
            pytest.skip(f"Beethoven test data not found: {abc_dir}")

        # CLT1
        abc_loader = Ms3Loader.from_file(
            abc_dir / "n04op18-4_04.notes.tsv",
            abc_dir / "n04op18-4_04.measures.tsv",
            abc_dir / "n04op18-4_04.harmonies.tsv",
        )
        clt1 = abc_loader.create_timeline(uid="clt_1")

        # OpenScore (4th movement only)
        os_loader = Ms3Loader.from_file(
            os_dir / "sq8913219.notes.tsv",
            os_dir / "sq8913219.measures.tsv",
        )
        os_full = os_loader.create_timeline(uid="openscore_full")
        os_controller = ScoreFlowController(os_loader.store.measures)
        boundaries = os_controller.get_section_boundary_coordinates()
        os_full.create_regions_from_boundaries(
            [0, *[float(b) for b in boundaries], float(os_full.length.value)],
            prefix="movement",
        )
        openscore = os_full.create_child_from_region("region_4", uid="openscore")

        # Group
        group = TimelineGroup(
            id="score",
            name="Score (ABC + OpenScore)",
            timelines=[clt1, openscore],
        )

        # Flow
        controller = abc_loader.create_flow_controller()
        flow = controller.compute_flow(FlowMode.default)

        return {
            "group": group,
            "clt_1": clt1,
            "openscore": openscore,
            "controller": controller,
            "flow": flow,
            "abc_loader": abc_loader,
        }

    @pytest.fixture(scope="class")
    def unfolded_group(self, score_data):
        """Unfold the score group once for flattened-result assertions."""
        return score_data["group"].apply_flow(
            score_data["flow"], score_data["controller"], "clt_1"
        )

    def test_unfold_returns_timeline_group(self, unfolded_group):
        """apply_flow() returns a TimelineGroup."""
        assert isinstance(unfolded_group, TimelineGroup)

    def test_unfold_preserves_timeline_ids(self, score_data, unfolded_group):
        """Unfolded group has the same timeline IDs as the original."""
        group = score_data["group"]
        assert set(unfolded_group.timeline_ids) == set(group.timeline_ids)

    def test_unfold_preserves_timeline_count(self, score_data, unfolded_group):
        """Unfolded group has the same number of timelines."""
        group = score_data["group"]
        assert unfolded_group.n_timelines == group.n_timelines

    def test_unfold_group_name(self, unfolded_group):
        """Unfolded group has a descriptive name."""
        assert "unfolded" in unfolded_group.name.lower()

    def test_unfold_custom_name(self, score_data):
        """Custom name is used when specified."""
        group = score_data["group"]
        result = group.apply_flow(
            score_data["flow"],
            score_data["controller"],
            "clt_1",
            name="My Unfolded Group",
        )
        assert result.name == "My Unfolded Group"

    def test_unfold_clt1_length(self, unfolded_group):
        """Unfolded CLT1 has exact unfolded length (1116 QB)."""
        clt1 = unfolded_group.get_timeline("clt_1")
        assert clt1.length.value == self.UNFOLDED_QB

    def test_unfold_openscore_length(self, unfolded_group):
        """Unfolded OpenScore has exact unfolded length (1116 QB)."""
        openscore = unfolded_group.get_timeline("openscore")
        assert openscore.length.value == self.UNFOLDED_QB

    def test_unfold_child_count(self, unfolded_group):
        """Each unfolded member has exactly N_SECTIONS appended children."""
        for tl_id in unfolded_group.timeline_ids:
            tl = unfolded_group.get_timeline(tl_id)
            assert tl.n_children == self.N_SECTIONS, (
                f"{tl_id}: n_children={tl.n_children}, " f"expected {self.N_SECTIONS}"
            )

    def test_unfold_children_are_named_by_section_with_minted_ids(self, unfolded_group):
        """Section labels name the children; every timeline id is minted, once."""
        import re

        from timetoalign.timelines.groups import _held_ids, _tree

        labels = ["A", "A+B", "B+C", "C+D+D1", "D", "D2+E", "E+F+F1", "F"]
        labels += ["F2+G+G1", "G", "G2"]
        timelines = []
        for member in unfolded_group:
            names = [member.get_child(i).name for i in member.list_children()]
            assert names == labels
            for _, descendant in member._iter_descendants():
                assert re.fullmatch(r"clt_\d+", descendant.id)
            timelines += _tree(member)
        assert len({timeline.id for timeline in timelines}) == len(timelines)
        ids = [
            entry[0] for timeline in timelines for entry in timeline._identity_entries()
        ]
        assert len(_held_ids(unfolded_group)) == len(ids)

    def test_unfold_members_keep_concrete_type(self, score_data, unfolded_group):
        """Each unfolded member is the same concrete type as its source."""
        group = score_data["group"]
        for tl_id in unfolded_group.timeline_ids:
            src = group.get_timeline(tl_id)
            out = unfolded_group.get_timeline(tl_id)
            assert type(out) is type(src)

    def test_unfold_flattened_has_events(self, unfolded_group):
        """Unfolded CLT1 has note events (now living in its appended children)."""
        clt1 = unfolded_group.get_timeline("clt_1")
        events = clt1.get_events(event_type="Note", include_children=True)
        assert len(events) > 0

    def test_unfold_reference_has_flow_maps(self, unfolded_group):
        """Reference timeline in unfolded group has FlowMaps attached."""
        clt1 = unfolded_group.get_timeline("clt_1")
        assert clt1.n_flow_maps >= 2
        assert clt1.has_flow_map("source")

    def test_unfold_invalid_reference_raises(self, score_data):
        """KeyError when reference_timeline_id is not in the group."""
        group = score_data["group"]
        with pytest.raises(KeyError, match="nonexistent"):
            group.apply_flow(
                score_data["flow"],
                score_data["controller"],
                "nonexistent",
            )

    def test_unfold_consistency_with_single_timeline(self, score_data, unfolded_group):
        """Group unfold produces same length as create_unfolded_timeline.

        Verifies that the group-based approach is consistent with the
        single-timeline function for the reference timeline: both yield a
        same-type timeline with one appended child per section.
        """
        from timetoalign.timelines.flow import create_unfolded_timeline

        controller = score_data["controller"]
        flow = score_data["flow"]
        clt1 = score_data["clt_1"]

        group_clt1 = unfolded_group.get_timeline("clt_1")
        single_clt1 = create_unfolded_timeline(clt1, flow, controller)

        assert group_clt1.length.value == single_clt1.length.value
        assert group_clt1.n_children == single_clt1.n_children


# endregion


# region TestScoreLoaderCreateFlowController


class TestScoreLoaderCreateFlowController:
    """Tests for ScoreLoader.create_flow_controller()."""

    @pytest.fixture(scope="class")
    def abc_loader(self):
        """Load the Beethoven ABC score."""
        from pathlib import Path

        from timetoalign.loader.score import Ms3Loader

        data_dir = (
            Path(__file__).parent.parent
            / "data"
            / "score"
            / "beethoven_op18-4iv_multimodal"
            / "ABC"
        )
        if not data_dir.exists():
            pytest.skip(f"Beethoven test data not found: {data_dir}")

        return Ms3Loader.from_file(
            data_dir / "n04op18-4_04.notes.tsv",
            data_dir / "n04op18-4_04.measures.tsv",
        )

    def test_returns_score_flow_controller(self, abc_loader):
        """create_flow_controller() returns a ScoreFlowController."""
        from timetoalign.timelines.flow import ScoreFlowController

        controller = abc_loader.create_flow_controller()
        assert isinstance(controller, ScoreFlowController)

    def test_controller_can_compute_flow(self, abc_loader):
        """The returned controller can compute a flow."""
        from timetoalign.core.enums import FlowMode

        controller = abc_loader.create_flow_controller()
        flow = controller.compute_flow(FlowMode.default)
        assert len(flow.sections) == 11

    def test_raises_without_measures(self):
        """ValueError when no measure data has been loaded."""
        from timetoalign.loader.score import Ms3Loader

        loader = Ms3Loader()
        with pytest.raises(ValueError, match="no measure data"):
            loader.create_flow_controller()

    def test_equivalent_to_manual_construction(self, abc_loader):
        """Produces the same result as manual ScoreFlowController()."""
        from timetoalign.core.enums import FlowMode
        from timetoalign.timelines.flow import ScoreFlowController

        # API way
        ctrl_api = abc_loader.create_flow_controller()
        flow_api = ctrl_api.compute_flow(FlowMode.default)

        # Manual way
        ctrl_manual = ScoreFlowController(abc_loader.store.measures)
        flow_manual = ctrl_manual.compute_flow(FlowMode.default)

        assert len(flow_api.sections) == len(flow_manual.sections)


# endregion
