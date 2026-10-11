"""One stamp surface and one table surface across every stamp producer.

Validation logic is documented in ``tests/core/README.md`` under "One stamp
surface, one table surface". Timelines, groups and bundles answer the same
four precise questions under the same four names, choose among them by the
same dispatch rule, and publish the same two table formats over the same
coordinate-struct cells.
"""

from __future__ import annotations

from fractions import Fraction

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.helpers import table_column
from timetoalign.alignment import AlignmentBundle, MatchClaim
from timetoalign.alignment.claims import AlignmentAnchor, MatchClaimField
from timetoalign.alignment.graph import MatchStamp
from timetoalign.core import Coordinate, IdCoordinate, TimeStamp, TimeUnit
from timetoalign.maps import ScalarMap
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    Timeline,
    TimelineGroup,
)

# region Fixtures


def _timeline() -> Timeline:
    """A seconds timeline carrying two named instants."""
    timeline = Timeline(length=10.0, unit=TimeUnit.seconds, uid="cpt_1")
    timeline.add_events(
        [
            {"id": "e1", "event_type": "Beat", "instant": 2.0},
            {"id": "e2", "event_type": "Beat", "instant": 6.0},
        ]
    )
    return timeline


def _group() -> TimelineGroup:
    """A two-member group: 4000 pixels spanning 10 seconds."""
    pixels = DiscreteGraphicalTimeline(length=4000, unit="pixels", uid="dgt_1")
    audio = ContinuousPhysicalTimeline(length=10.0, unit="seconds", uid="cpt_1")
    audio.add_events([{"id": "e1", "event_type": "Beat", "instant": 2.5}])
    return TimelineGroup(id="pair", timelines=[pixels, audio])


def _ratio_bundle(*, columnar: bool) -> AlignmentBundle:
    """A bundle whose anchors sit at exact thirds of a quarters axis.

    Built twice over the two claim stores — the per-claim Python list and the
    columnar ``MatchClaimField`` — because they reach the table by different
    reads and only the columnar one goes through a bulk four-column read.
    """
    score = ContinuousLogicalTimeline(length=Fraction(10), uid="clt_1")
    performance = ContinuousPhysicalTimeline(length=20.0, unit="seconds", uid="cpt_1")
    bundle = AlignmentBundle(id="ratios")
    bundle.add_timeline(score, uid="clt_1", as_group="score_group")
    bundle.add_timeline(performance, uid="cpt_1", as_group="perf_group")

    claims = [
        MatchClaim(
            timeline_a_id="clt_1",
            timeline_b_id="cpt_1",
            start_anchor=AlignmentAnchor(
                timeline_a_id="clt_1",
                coordinate_a=Coordinate(Fraction(third, 3), TimeUnit.quarters),
                timeline_b_id="cpt_1",
                coordinate_b=Coordinate(float(third), TimeUnit.seconds),
            ),
            is_synchronous=True,
        )
        for third in (0, 5, 15)
    ]
    if columnar:
        bundle.add_match_claim_field(MatchClaimField.from_claims(claims))
    else:
        bundle.add_match_claims(claims)
    return bundle


def _bundle() -> AlignmentBundle:
    """Two number-native timelines aligned two-to-one across groups."""
    score = Timeline(length=100, uid="score", unit=TimeUnit.number)
    score.add_events([{"id": "e1", "event_type": "Beat", "instant": 50}])
    performance = Timeline(length=200, uid="perf", unit=TimeUnit.number)

    bundle = AlignmentBundle(id="pair")
    bundle.add_timeline(score, uid="score", as_group="score_group")
    bundle.add_timeline(performance, uid="perf", as_group="perf_group")
    bundle.add_match_claims(
        [
            MatchClaim(
                timeline_a_id="score",
                timeline_b_id="perf",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="score",
                    coordinate_a=Coordinate(position, TimeUnit.number),
                    timeline_b_id="perf",
                    coordinate_b=Coordinate(position * 2, TimeUnit.number),
                ),
                is_synchronous=True,
            )
            for position in (0, 50, 100)
        ]
    )
    return bundle


# endregion


# region Dispatcher branch matrix


class TestTimelineDispatcher:
    """``Timeline.get_timestamp`` selects among the four precise getters."""

    def test_scalar_coordinate_selects_the_positional_getter(self) -> None:
        """A position routes to get_timestamp_at and answers identically."""
        timeline = _timeline()

        dispatched = timeline.get_timestamp(2.0)
        precise = timeline.get_timestamp_at(2.0)

        assert isinstance(dispatched, TimeStamp)
        assert dispatched.coordinates == precise.coordinates
        assert dispatched.source_id == precise.source_id

    def test_coordinate_collection_selects_the_plural_getter(self) -> None:
        """A position collection routes to get_timestamps_at and gives a list."""
        timeline = _timeline()

        dispatched = timeline.get_timestamp([2.0, 6.0])

        assert isinstance(dispatched, list)
        assert [stamp.axis.value for stamp in dispatched] == [2.0, 6.0]

    def test_string_selects_the_key_getter(self) -> None:
        """An event ID routes to get_timestamp_for and answers identically."""
        timeline = _timeline()

        dispatched = timeline.get_timestamp("e2")
        precise = timeline.get_timestamp_for("e2")

        assert dispatched.coordinates == precise.coordinates
        assert dispatched.axis.value == 6.0

    def test_key_collection_selects_the_plural_key_getter(self) -> None:
        """A key collection routes to get_timestamps_for and gives a list."""
        timeline = _timeline()

        dispatched = timeline.get_timestamp(["e2", "e1"])

        assert isinstance(dispatched, list)
        assert [stamp.axis.value for stamp in dispatched] == [6.0, 2.0]

    def test_mixed_collection_is_rejected(self) -> None:
        """No element-by-element semantic dispatch."""
        with pytest.raises(TypeError):
            _timeline().get_timestamp(["e1", 2.0])

    def test_boolean_is_not_a_coordinate(self) -> None:
        """bool subclasses int and is still never a position."""
        with pytest.raises(TypeError):
            _timeline().get_timestamp(True)

    def test_empty_collection_gives_an_empty_list(self) -> None:
        """An empty collection is a coordinate collection with no members."""
        assert _timeline().get_timestamp([]) == []


class TestTimelineGroupDispatcher:
    """``TimelineGroup.get_timestamp`` selects among the four precise getters."""

    def test_scalar_coordinate_selects_the_positional_getter(self) -> None:
        """A position routes to get_timestamp_at and answers identically."""
        group = _group()

        dispatched = group.get_timestamp(2.5, "cpt_1")
        precise = group.get_timestamp_at(2.5, "cpt_1")

        assert dispatched.coordinates == precise.coordinates
        assert dispatched.get_coordinate_for("dgt_1", format="int") == 1000

    def test_coordinate_collection_selects_the_plural_getter(self) -> None:
        """A position collection routes to get_timestamps_at and gives a list."""
        dispatched = _group().get_timestamp([0.0, 10.0], "cpt_1")

        assert isinstance(dispatched, list)
        assert [
            stamp.get_coordinate_for("dgt_1", format="int") for stamp in dispatched
        ] == [0, 4000]

    def test_string_selects_the_key_getter(self) -> None:
        """An event ID routes to get_timestamp_for and answers identically."""
        group = _group()

        dispatched = group.get_timestamp("e1")
        precise = group.get_timestamp_for("e1")

        assert dispatched.coordinates == precise.coordinates
        assert dispatched.get_coordinate_for("dgt_1", format="int") == 1000

    def test_key_collection_selects_the_plural_key_getter(self) -> None:
        """A key collection routes to get_timestamps_for and gives a list."""
        dispatched = _group().get_timestamp(["e1"])

        assert isinstance(dispatched, list)
        assert len(dispatched) == 1
        assert dispatched[0].get_coordinate_for("cpt_1", format="float") == 2.5

    def test_a_named_member_re_anchors_an_event_stamp(self) -> None:
        """An event's coordinate is on ITS axis; another member gets a transfer.

        Reinterpreting 2.5 seconds as 2.5 pixels would be the alternative,
        and it is the reason this case is pinned.
        """
        group = _group()

        owned = group.get_timestamp_for("e1")
        re_anchored = group.get_timestamp_for("e1", "dgt_1")

        assert owned.source_id == "cpt_1"
        assert re_anchored.source_id == "dgt_1"
        assert re_anchored.get_coordinate_for("dgt_1", format="int") == 1000
        assert re_anchored.get_coordinate_for("cpt_1", format="float") == 2.5

    def test_mixed_collection_is_rejected(self) -> None:
        """No element-by-element semantic dispatch."""
        with pytest.raises(TypeError):
            _group().get_timestamp(["e1", 2.5], "cpt_1")

    def test_boolean_is_not_a_coordinate(self) -> None:
        """bool subclasses int and is still never a position."""
        with pytest.raises(TypeError):
            _group().get_timestamp(True, "cpt_1")

    def test_empty_collection_gives_an_empty_list(self) -> None:
        """An empty collection is a coordinate collection with no members."""
        assert _group().get_timestamp([], "cpt_1") == []


class TestAlignmentBundleDispatcher:
    """``AlignmentBundle.get_matchstamp`` selects among the four getters."""

    def test_scalar_coordinate_selects_the_positional_getter(self) -> None:
        """A position routes to get_matchstamp_at and answers identically."""
        bundle = _bundle()

        dispatched = bundle.get_matchstamp(50, "score")
        precise = bundle.get_matchstamp_at(50, "score")

        assert isinstance(dispatched, MatchStamp)
        assert dispatched.coordinates == precise.coordinates
        assert dispatched.get_coordinate_for("perf", format="float") == 100.0

    def test_coordinate_collection_selects_the_plural_getter(self) -> None:
        """A position collection routes to get_matchstamps_at and gives a list."""
        dispatched = _bundle().get_matchstamp([0, 50, 100], "score")

        assert isinstance(dispatched, list)
        assert [
            stamp.get_coordinate_for("perf", format="float") for stamp in dispatched
        ] == [0.0, 100.0, 200.0]

    def test_string_selects_the_key_getter(self) -> None:
        """An event ID routes to get_matchstamp_for and answers identically."""
        bundle = _bundle()

        dispatched = bundle.get_matchstamp("e1")
        precise = bundle.get_matchstamp_for("e1")

        assert dispatched.coordinates == precise.coordinates
        assert dispatched.get_coordinate_for("perf", format="float") == 100.0

    def test_key_collection_selects_the_plural_key_getter(self) -> None:
        """A key collection routes to get_matchstamps_for and gives a list."""
        dispatched = _bundle().get_matchstamp(["e1"])

        assert isinstance(dispatched, list)
        assert dispatched[0].get_coordinate_for("perf", format="float") == 100.0

    def test_mixed_collection_is_rejected(self) -> None:
        """No element-by-element semantic dispatch."""
        with pytest.raises(TypeError):
            _bundle().get_matchstamp(["e1", 50], "score")

    def test_boolean_is_not_a_coordinate(self) -> None:
        """bool subclasses int and is still never a position."""
        with pytest.raises(TypeError):
            _bundle().get_matchstamp(True, "score")

    def test_empty_collection_gives_an_empty_list(self) -> None:
        """An empty collection is a coordinate collection with no members."""
        assert _bundle().get_matchstamp([], "score") == []


# endregion


# region Deleted names and shadowing


class TestDeletedNames:
    """Superseded spellings are absent, not aliased."""

    @pytest.mark.parametrize("name", ["get_timestamp_of", "get_timestamps_of"])
    def test_the_of_suffix_is_gone(self, name: str) -> None:
        """One key semantics, one suffix: ``_for``."""
        with pytest.raises(AttributeError):
            getattr(_timeline(), name)
        with pytest.raises(AttributeError):
            getattr(_group(), name)

    def test_to_dataframe_is_gone(self) -> None:
        """The frame is a format of the table, not a second method."""
        with pytest.raises(AttributeError):
            _timeline().to_dataframe
        with pytest.raises(AttributeError):
            _group().to_dataframe

    def test_as_fractions_is_not_accepted(self) -> None:
        """A caller may not override an axis's declared representation."""
        with pytest.raises(TypeError):
            _timeline().get_timestamp_table(format="dataframe", as_fractions=True)

    def test_get_matchstamps_no_longer_takes_positions(self) -> None:
        """The bare plural means the bundle's claims, not a position batch."""
        with pytest.raises(TypeError):
            _bundle().get_matchstamps(coordinates=[0, 50])

    def test_get_matchstamps_lists_the_bundle_claims(self) -> None:
        """Three synchronous claims, three stamps."""
        assert len(_bundle().get_matchstamps()) == 3


class TestDispatcherDoesNotShadow:
    """``get_timestamp`` and ``get_timestamp_at`` are separate methods."""

    def test_they_are_distinct_functions(self) -> None:
        """One name, one job: a dispatcher may not be its own target."""
        assert Timeline.get_timestamp is not Timeline.get_timestamp_at

    def test_the_implementation_lives_on_the_precise_name(self) -> None:
        """``unit=`` is keyword-only, so slot two is the result-axis validator.

        The precise getter matches ``get_coordinate_at`` position for
        position, which is what makes the two lanes readable side by side.
        """
        timeline = Timeline(length=960, unit=TimeUnit.ticks, uid="dlt_1")
        timeline.add_conversion_map(
            ScalarMap(
                scalar=Fraction(1, 480),
                source_unit=TimeUnit.ticks,
                target_unit=TimeUnit.quarters,
            )
        )

        assert timeline.get_timestamp_at(1, unit=TimeUnit.quarters).axis.value == 480
        assert timeline.get_timestamp_at(480, "dlt_1").axis.value == 480
        with pytest.raises(KeyError, match="Unknown result timeline ID"):
            timeline.get_timestamp_at(480, "other")


# endregion


# region Table format vocabulary


class TestTableFormatVocabulary:
    """``format=`` is closed, and shaping options belong to one of its values."""

    def test_table_format_returns_arrow(self) -> None:
        """The default is the Arrow table."""
        assert isinstance(_timeline().get_timestamp_table(), pa.Table)
        assert isinstance(_group().get_timestamp_table(), pa.Table)
        assert isinstance(_bundle().get_matchstamp_table(), pa.Table)

    def test_dataframe_format_returns_pandas(self) -> None:
        """The frame is reachable by name on every table method."""
        assert isinstance(
            _timeline().get_timestamp_table(format="dataframe"), pd.DataFrame
        )
        assert isinstance(
            _group().get_timestamp_table(format="dataframe"), pd.DataFrame
        )
        assert isinstance(
            _bundle().get_matchstamp_table(format="dataframe"), pd.DataFrame
        )

    @pytest.mark.parametrize("bad", ["pandas", "arrow", "TABLE", ""])
    def test_unknown_format_names_both_accepted_values(self, bad: str) -> None:
        """The error tells the caller what the vocabulary actually is."""
        with pytest.raises(ValueError) as exc_info:
            _timeline().get_timestamp_table(format=bad)

        assert "table" in str(exc_info.value)
        assert "dataframe" in str(exc_info.value)

    def test_dataframe_options_are_rejected_for_an_arrow_result(self) -> None:
        """The message names exactly the options that do not apply."""
        with pytest.raises(ValueError) as exc_info:
            _timeline().get_timestamp_table(units=False, include_ids=False)

        message = str(exc_info.value)
        assert "units" in message
        assert "include_ids" in message
        assert "fields" not in message
        assert 'format="dataframe"' in message

    def test_group_and_bundle_reject_the_same_options(self) -> None:
        """One rule, applied by every table method."""
        with pytest.raises(ValueError, match="units"):
            _group().get_timestamp_table(units=True)
        with pytest.raises(ValueError, match="fields"):
            _bundle().get_matchstamp_table(fields=[])


# endregion


# region Coordinate-struct cells


class TestStructCells:
    """Every timestamp table stores coordinates as structs."""

    STRUCT = "struct<value: double, numerator: int64, denominator: int64>"

    def test_timeline_columns_are_coordinate_structs(self) -> None:
        """Axis and timeline columns carry unit, number type and identity."""
        table = _timeline().get_timestamp_table()

        assert table.column_names == ["cpt_1"]
        for name in table.column_names:
            field = table.schema.field(name)
            assert str(field.type) == self.STRUCT
            metadata = _metadata(field)
            assert metadata["unit"] == "seconds"
            assert metadata["number_type"] == "float"
            assert metadata["timeline_id"] == "cpt_1"

    def test_group_columns_are_coordinate_structs(self) -> None:
        """A group's published columns carry each member's declared type."""
        table = _group().get_timestamp_table()

        pixels = _metadata(table.schema.field("dgt_1"))
        assert str(table.schema.field("dgt_1").type) == self.STRUCT
        assert pixels["unit"] == "pixels"
        assert pixels["number_type"] == "int"
        assert pixels["timeline_id"] == "dgt_1"

    def test_matchstamp_columns_are_coordinate_structs(self) -> None:
        """The bundle table uses the same cell shape as the timeline table."""
        table = _bundle().get_matchstamp_table()

        for name in ("score", "perf"):
            field = table.schema.field(name)
            assert str(field.type) == self.STRUCT
            assert _metadata(field)["timeline_id"] == name

    def test_frame_cells_are_scalars_not_dicts(self) -> None:
        """A struct is the storage shape; a frame reader gets one number."""
        frames = [
            _timeline().get_timestamp_table(format="dataframe"),
            _group().get_timestamp_table(format="dataframe"),
            _bundle().get_matchstamp_table(format="dataframe"),
        ]
        for frame in frames:
            assert len(frame) > 0
            for name in frame.columns:
                for cell in frame[name]:
                    assert not isinstance(cell, dict)

    # The dyadic of float(5/3). Named once so every assertion below can say
    # what it is refusing rather than repeating a sixteen-digit ratio.
    DYADIC_OF_FIVE_THIRDS = Fraction(7505999378950827, 4503599627370496)

    def test_authored_ratio_survives_the_frame(self) -> None:
        """The defect under repair: 5/3 must not become its dyadic.

        ``Fraction(5, 3)`` and ``Fraction(float(Fraction(5, 3)))`` agree to
        sixteen digits and only one of them is what the caller wrote. A double
        column cannot tell them apart, which is why the column is a struct.

        All three ways an axis is sourced are covered, because they reach the
        column by different routes and only one of them was ever asserted:
        positions the caller passes, positions collected from events, and
        positions collected from timeline boundaries. The event route is the
        one every caller of a bare ``get_timestamp_table()`` gets.
        """
        assert self.DYADIC_OF_FIVE_THIRDS == Fraction(float(Fraction(5, 3)))
        assert self.DYADIC_OF_FIVE_THIRDS != Fraction(5, 3)

        authored = ContinuousLogicalTimeline(length=Fraction(10), uid="clt_1")
        table = authored.get_timestamp_table([Fraction(5, 3)])
        frame = authored.get_timestamp_table([Fraction(5, 3)], format="dataframe")

        cell = table.column("clt_1").combine_chunks()
        assert cell.field("numerator").to_pylist() == [5]
        assert cell.field("denominator").to_pylist() == [3]
        assert frame["clt_1 (quarters)"].tolist() == [Fraction(5, 3)]
        assert frame["clt_1 (quarters)"].tolist() != [self.DYADIC_OF_FIVE_THIRDS]

    def test_authored_ratio_survives_the_event_derived_axis(self) -> None:
        """The default table path collects from events and must stay exact.

        The event store keeps the ratio; reducing the column to its float
        member on the way to the table threw it away again, so a timeline
        whose stamp getter answered 5/3 had a table cell reading the dyadic.
        """
        timeline = ContinuousLogicalTimeline(length=Fraction(10), uid="clt_1")
        timeline.add_events(
            [{"id": "e1", "event_type": "Note", "instant": Fraction(5, 3)}]
        )

        table = timeline.get_timestamp_table()
        frame = timeline.get_timestamp_table(format="dataframe")

        cell = table.column("clt_1").combine_chunks()
        assert cell.field("numerator").to_pylist() == [5]
        assert cell.field("denominator").to_pylist() == [3]
        assert frame["clt_1 (quarters)"].tolist() == [Fraction(5, 3)]
        assert frame["clt_1 (quarters)"].tolist() != [self.DYADIC_OF_FIVE_THIRDS]
        # The two lanes of one position agree, which is the point.
        assert timeline.get_timestamp_for("e1").axis.value == Fraction(5, 3)

    def test_authored_ratio_survives_the_boundary_derived_axis(self) -> None:
        """Boundaries are computed from exact lengths and offsets.

        A child placed at 5/3 puts that ratio on the parent axis without any
        event carrying it, so the boundary collector has to stay exact on its
        own account.
        """
        parent = ContinuousLogicalTimeline(length=Fraction(10), uid="clt_1")
        child = ContinuousLogicalTimeline(length=Fraction(1, 3), uid="clt_2")
        parent.add_child(child, offset=Fraction(5, 3))

        frame = parent.get_timestamp_table(
            include_events=False, include_boundaries=True, format="dataframe"
        )

        # child spans [5/3, 2) on the parent axis: 5/3 + 1/3 == 2.
        assert frame["clt_1 (quarters)"].tolist() == [
            Fraction(0),
            Fraction(5, 3),
            Fraction(2),
            Fraction(10),
        ]
        assert self.DYADIC_OF_FIVE_THIRDS not in frame["clt_1 (quarters)"].tolist()
        assert frame["clt_2 (quarters)"].tolist()[1] == Fraction(0)
        assert frame["clt_2 (quarters)"].tolist()[2] == Fraction(1, 3)

    def test_authored_ratio_survives_the_claim_table(self) -> None:
        """Both claim stores tabulate the ratio their anchors were given.

        The columnar store is the one that reads four Arrow columns in bulk;
        it held the exact numerator and denominator all along and handed over
        only the float member.
        """
        for from_graph in (False, True):
            per_claim = _ratio_bundle(columnar=False).get_matchstamp_table(
                from_graph=from_graph, format="dataframe"
            )
            columnar = _ratio_bundle(columnar=True).get_matchstamp_table(
                from_graph=from_graph, format="dataframe"
            )
            assert per_claim["clt_1 (quarters)"].tolist() == [
                Fraction(0),
                Fraction(5, 3),
                Fraction(5),
            ]
            assert (
                columnar["clt_1 (quarters)"].tolist()
                == per_claim["clt_1 (quarters)"].tolist()
            )

    def test_group_stored_boundary_rows_are_a_known_float_lane(self) -> None:
        """Documented limitation: a group's STORED rows come from a float store.

        A group's timestamp store is ``float64`` by design — interpolation
        between members runs on doubles — so a boundary that was authored as
        an exact ratio is recorded as the double nearest to it and reads back
        as that double's dyadic. Everything the group computes from a query is
        exact: the stamp lane, the coordinate lane, and ``at=``-queried table
        rows all answer ``5/3``. Only the rows the store already holds are
        affected.

        Pinned rather than hidden, so that the day the store becomes typed
        this test fails and says exactly what changed.
        """
        score = ContinuousLogicalTimeline(length=Fraction(5, 3), uid="clt_1")
        audio = ContinuousPhysicalTimeline(length=20.0, unit="seconds", uid="cpt_1")
        group = TimelineGroup(id="ratio_pair", timelines=[score, audio])

        stored = group.get_timestamp_table(format="dataframe")
        queried = group.get_timestamp_table(
            [Fraction(5, 3)], "clt_1", format="dataframe"
        )

        # The store itself is doubles, which is where the loss happens.
        assert group._timestamp_table.column("clt_1").to_pylist() == [
            0.0,
            float(Fraction(5, 3)),
        ]
        assert stored["clt_1 (quarters)"].tolist()[1] == self.DYADIC_OF_FIVE_THIRDS

        # Every lane that resolves a query rather than replaying a stored row
        # is exact.
        assert queried["clt_1 (quarters)"].tolist() == [Fraction(5, 3)]
        assert group.get_timestamp_at(Fraction(5, 3), "clt_1").axis.value == Fraction(
            5, 3
        )
        assert group.get_coordinate_at(Fraction(5, 3), "clt_1").value == Fraction(5, 3)

    def test_authored_ratio_survives_a_parquet_round_trip(self, tmp_path) -> None:
        """Storage is where an unwritten ratio would be lost for good."""
        timeline = ContinuousLogicalTimeline(length=Fraction(10), uid="clt_1")
        table = timeline.get_timestamp_table([Fraction(5, 3), Fraction(7, 3)])

        path = tmp_path / "timestamps.parquet"
        pq.write_table(table, path)
        restored = pq.read_table(path)

        assert restored.schema == table.schema
        assert table_column(restored, "clt_1") == [Fraction(5, 3), Fraction(7, 3)]


def _metadata(field: pa.Field) -> dict[str, object]:
    """Read one field's TimeToAlign! metadata blob."""
    from timetoalign.core.fields import field_metadata

    return field_metadata(field)


# endregion


# region Frame parity and atomicity


class TestFrameParity:
    """The frame lane carries what the deleted frame methods carried."""

    def test_group_query_frame_matches_the_stamp_lane(self) -> None:
        """Positions asked of the group answer the same in both lanes."""
        group = _group()
        positions = [0.0, 2.5, 10.0]

        stamps = group.get_timestamps_at(positions, "cpt_1")
        frame = group.get_timestamp_table(positions, "cpt_1", format="dataframe")

        assert list(frame.columns) == ["dgt_1 (pixels)", "cpt_1 (seconds)"]
        assert frame["cpt_1 (seconds)"].tolist() == positions
        assert frame["dgt_1 (pixels)"].tolist() == [0, 1000, 4000]
        assert [
            stamp.get_coordinate_for("dgt_1", format="int") for stamp in stamps
        ] == frame["dgt_1 (pixels)"].tolist()

    def test_timeline_frame_matches_the_decoded_table(self) -> None:
        """The frame is the table, named and decoded — not a second source."""
        timeline = _timeline()

        frame = timeline.get_timestamp_table(
            format="dataframe", units=False, include_ids=False
        )
        table = timeline.get_timestamp_table()

        assert frame.columns.tolist() == table.column_names
        for name in table.column_names:
            assert frame[name].tolist() == table_column(table, name)

    def test_timeline_frame_indexes_by_event_id(self) -> None:
        """The absorbed include_ids behaviour, unchanged."""
        frame = _timeline().get_timestamp_table(format="dataframe", units=False)

        assert frame.index.name == "id"
        assert frame.index.tolist() == ["e1", "e2"]

    def test_timeline_table_accepts_event_keys(self) -> None:
        """Event IDs give one row each, in the order they were asked for."""
        timeline = _timeline()

        frame = timeline.get_timestamp_table(["e2", "e1"], format="dataframe")

        assert frame.index.tolist() == ["e2", "e1"]
        assert frame["cpt_1 (seconds)"].tolist() == [6.0, 2.0]

    def test_a_child_id_coordinate_still_applies_the_child_offset(self) -> None:
        """Naming a child in the query keeps the automatic offset."""
        parent = Timeline(length=20.0, unit=TimeUnit.seconds, uid="parent")
        child = Timeline(length=5.0, unit=TimeUnit.seconds, uid="child")
        parent.add_child(child, offset=10.0)

        frame = parent.get_timestamp_table(
            [IdCoordinate(2.0, TimeUnit.seconds, "child")],
            conversion_maps=False,
            format="dataframe",
        )

        assert frame["parent (seconds)"].tolist() == [12.0]
        assert frame["child (seconds)"].tolist() == [2.0]


class TestAtomicity:
    """A plural stamp getter answers completely or raises."""

    def test_an_unresolvable_position_aborts_the_batch(self) -> None:
        """No partial list is returned for a position outside the range."""
        group = _group()

        with pytest.raises(ValueError):
            group.get_timestamps_at([0.0, 99.0], "cpt_1")

    def test_a_missing_key_raises_rather_than_emitting_a_null_row(self) -> None:
        """A NaN row for an unknown event would report an answer there is not."""
        with pytest.raises(KeyError):
            _group().get_timestamps_for(["e1", "missing"])
        with pytest.raises(KeyError):
            _timeline().get_timestamps_for(["e1", "missing"])

    def test_the_table_lane_raises_on_the_same_input(self) -> None:
        """One input, one behaviour — whichever exit the caller took.

        Swallowing an invalid query into a null-filled row would make a null
        ambiguous: a reader could not tell "this member does not reach this
        position" from "your query was invalid". Those must never look alike.
        """
        group = _group()

        with pytest.raises(ValueError):
            group.get_timestamp_table([0.0, 99.0], "cpt_1")
        with pytest.raises(ValueError):
            group.get_timestamp_table([0.0, 99.0], "cpt_1", format="dataframe")

    def test_a_null_cell_means_only_that_a_member_is_unreached(self) -> None:
        """The surviving meaning of a null, pinned so it stays the only one."""
        pixels = DiscreteGraphicalTimeline(length=1000, unit="pixels", uid="dgt_1")
        partial = DiscreteGraphicalTimeline(length=400, unit="pixels", uid="dgt_2")
        group = TimelineGroup(id="gapped", timelines=[pixels])
        group.add_timeline(partial, start=0, end=500)

        frame = group.get_timestamp_table(format="dataframe")

        assert frame["dgt_2 (pixels)"].isna().any()
        assert frame["dgt_2 (pixels)"].dropna().tolist() == [0, 400]


# endregion
