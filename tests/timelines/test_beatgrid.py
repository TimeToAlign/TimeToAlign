"""Exact validation for beat grids.

The validation logic these tests follow is stated in ``README.md``
(section ``test_beatgrid.py - Beat Grids``): the boundary rule and its
knife edge, the two measure numberings, the four conversion directions,
exact tempo integration, and the atomic raises.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from timetoalign.alignment import TimeSkeleton
from timetoalign.core import (
    BeatPolicy,
    Coordinate,
    NumberType,
    RegularMeasure,
    TimeUnit,
)
from timetoalign.timelines import BeatGrid, BeatGridSegment, GridBeat, SectionHierarchy

FOUR_FOUR = BeatPolicy.from_time_signature("4/4").as_divisions()


def _segment(
    start: str | int | Fraction,
    bpm: int | Fraction,
    battito: int = 1,
    metro: str = "4/4",
) -> BeatGridSegment:
    """Build a segment from the facts a grid export states."""
    return BeatGridSegment(
        start=Fraction(str(start)),
        bpm=Fraction(bpm),
        policy=BeatPolicy.from_time_signature(metro).as_divisions(),
        battito=battito,
    )


def _pickup_grid() -> BeatGrid:
    """The grid of the pickup / grid-change / trailing-measure layout.

    ``(0, 60, 4/4, 4)``, ``(3.5, 120, 4/4, 2)``, ``(7, 120, 4/4, 1)`` over
    eight seconds. Grid 1's beat at 3.0 s sits exactly half a beat before
    the 3.5 s boundary -- the knife edge of the boundary rule.
    """
    return BeatGrid(
        [
            _segment(0, 60, battito=4),
            _segment("3.5", 120, battito=2),
            _segment(7, 120, battito=1),
        ],
        extent=8,
    )


def _phantom_grid() -> BeatGrid:
    """Three 120 BPM grids whose anchors sit 0.01 s and 0.30 s past a beat."""
    return BeatGrid(
        [
            _segment("0.5", 120),
            _segment("10.51", 120),
            _segment("20.31", 120),
        ],
        extent=30,
    )


def _reanchor_grid() -> BeatGrid:
    """A 60 BPM grid re-anchored mid-measure by a 120 BPM grid on beat 3."""
    return BeatGrid([_segment(0, 60), _segment("6.01", 120, battito=3)], extent=12)


def _restated_index_grid() -> BeatGrid:
    """A grid whose measure 15 states beats 3 and 4 twice.

    Segment 1 runs 120 BPM from zero, so measure 15 opens at 28.0 and its
    beats fall at 28.0, 28.5, 29.0, 29.5. Segment 2 anchors on beat 3 at
    30.0 with a 60/128 = 0.46875 s beat, restating beats 3 and 4 before
    its first downbeat at 30.9375 opens measure 16.
    """
    return BeatGrid(
        [_segment(0, 120), _segment(30, 128, battito=3)],
        extent=32,
    )


class TestSegmentAssembly:
    """Segments state source facts; the grid orders and bounds them."""

    def test_segments_are_sorted_and_bounded_by_their_successor(self) -> None:
        """Each segment ends where the next one opens; the last at the extent."""
        grid = BeatGrid(
            [_segment("10.51", 120), _segment("0.5", 120), _segment("20.31", 120)],
            extent=30,
        )

        assert [segment.start for segment in grid.segments] == [
            Fraction("0.5"),
            Fraction("10.51"),
            Fraction("20.31"),
        ]
        assert [segment.end for segment in grid.segments] == [
            Fraction("10.51"),
            Fraction("20.31"),
            Fraction(30),
        ]
        assert grid.extent == Fraction(30)
        assert grid.n_segments == 3

    def test_lengths_are_derived_from_tempo_and_policy(self) -> None:
        """Beat, bar and first downbeat come out of the stated facts."""
        segment = _segment("0.145", 150, battito=2)

        assert segment.beat_seconds == Fraction(60, 150)
        assert segment.bar_seconds == Fraction(60, 150) * 4
        # Battito 2 needs three 0.4 s beats to reach the next downbeat.
        assert segment.first_downbeat == Fraction("1.345")
        assert segment.quarters_per_second == Fraction(150, 60)

    def test_anchor_on_the_downbeat_is_its_own_first_downbeat(self) -> None:
        """Battito 1 means the anchor opens a measure."""
        assert _segment("0.5", 120).first_downbeat == Fraction("0.5")

    def test_segment_rejects_facts_it_cannot_hold(self) -> None:
        """A non-positive tempo and an out-of-bar anchor are refused."""
        with pytest.raises(ValueError, match="tempo must be positive"):
            BeatGridSegment(start=0, bpm=0, policy=FOUR_FOUR, battito=1)
        with pytest.raises(ValueError, match="outside 1..4"):
            BeatGridSegment(start=0, bpm=120, policy=FOUR_FOUR, battito=5)
        with pytest.raises(ValueError, match="cannot end at"):
            BeatGridSegment(start=2, bpm=120, policy=FOUR_FOUR, battito=1, end=1)

    def test_grid_rejects_segments_it_cannot_order(self) -> None:
        """No segments, or two at one instant, leave the order undefined."""
        with pytest.raises(ValueError, match="at least one segment"):
            BeatGrid([])
        with pytest.raises(ValueError, match="distinct instant"):
            BeatGrid([_segment(1, 120), _segment(1, 90)], extent=10)

    def test_a_stated_end_that_matches_the_bound_survives(self) -> None:
        """Segments abut, so stating the bound the grid computes is fine."""
        first = replace(_segment(0, 120), end=Fraction(4))
        grid = BeatGrid([first, _segment(4, 120)], extent=8)

        assert [segment.end for segment in grid.segments] == [
            Fraction(4),
            Fraction(8),
        ]

    def test_a_stated_end_that_disagrees_is_refused(self) -> None:
        """A shorter or longer end would leave a gap or an overlap."""
        with pytest.raises(
            ValueError, match=r"states end=3, but the grid bounds it at 4"
        ):
            BeatGrid(
                [replace(_segment(0, 120), end=Fraction(3)), _segment(4, 120)], extent=8
            )
        with pytest.raises(
            ValueError, match=r"states end=6, but the grid bounds it at 8"
        ):
            BeatGrid([replace(_segment(0, 120), end=Fraction(6))], extent=8)
        with pytest.raises(
            ValueError,
            match=r"states end=6, but the grid bounds it at the grid's extent",
        ):
            BeatGrid([replace(_segment(0, 120), end=Fraction(6))])

    def test_equality_is_structural_over_segments_and_extent(self) -> None:
        """Two grids of the same facts are equal; a different tempo is not."""
        assert _phantom_grid() == _phantom_grid()
        assert BeatGrid([_segment(0, 120)], extent=8) != BeatGrid(
            [_segment(0, 90)], extent=8
        )
        assert BeatGrid([_segment(0, 120)], extent=8) != BeatGrid(
            [_segment(0, 120)], extent=9
        )
        assert BeatGrid([_segment(0, 120)], extent=8) != "not a grid"

    def test_repr_names_segments_extent_and_measures(self) -> None:
        """The repr states what the grid holds, unbounded grids included."""
        assert repr(_phantom_grid()) == "BeatGrid(3 segments, extent=30, 15 measures)"
        assert repr(BeatGrid([_segment(0, 120)])) == "BeatGrid(1 segment, unbounded)"

    def test_metro_is_read_as_one_beat_per_counted_value(self) -> None:
        """6/8 counts six eighth beats, not two dotted ones."""
        policy = BeatPolicy.from_time_signature("6/8").as_divisions()

        assert policy.n_beats == 6
        assert policy.division == Fraction(1, 2)
        assert policy.span == Fraction(3)
        assert policy.name == "6/8"
        assert BeatPolicy.from_time_signature("6/8").n_beats == 2

    def test_unreadable_metro_raises(self) -> None:
        """A meter the grid cannot read is refused, never defaulted."""
        with pytest.raises(ValueError, match="Cannot read time signature"):
            BeatPolicy.from_time_signature("common").as_divisions()
        with pytest.raises(ValueError, match="Cannot read time signature"):
            BeatPolicy.from_time_signature("0/4").as_divisions()

    def test_a_value_that_is_not_a_number_raises(self) -> None:
        """Coordinate input accepts numbers and coordinates, nothing else."""
        with pytest.raises(TypeError, match="A grid tempo must be a number"):
            BeatGridSegment(start=0, bpm=None, policy=FOUR_FOUR, battito=1)
        with pytest.raises(TypeError, match="A grid start must be a number"):
            BeatGridSegment(start=True, bpm=120, policy=FOUR_FOUR, battito=1)

    def test_segment_repr_names_its_facts(self) -> None:
        """A segment shows its anchor, bound, tempo, policy and index."""
        grid = BeatGrid([_segment(0, 120, battito=2)], extent=8)

        assert repr(grid.segments[0]) == (
            "BeatGridSegment(start=0, end=8, bpm=120, policy=4/4, battito=2)"
        )
        assert repr(_segment(0, 120)) == (
            "BeatGridSegment(start=0, unbounded, bpm=120, policy=4/4, battito=1)"
        )


class TestRestatedBeatIndex:
    """A measure a segment re-anchors inside can state one index twice."""

    def test_the_measure_states_six_beats(self) -> None:
        """Beats 1,2,3,4 from one segment and 3,4 again from the next."""
        beats = [
            beat for beat in _restated_index_grid().iter_beats() if beat.measure == 15
        ]

        assert [(beat.seconds, beat.segment, beat.beat) for beat in beats] == [
            (Fraction(28), 0, 1),
            (Fraction("28.5"), 0, 2),
            (Fraction(29), 0, 3),
            (Fraction("29.5"), 0, 4),
            (Fraction(30), 1, 3),
            (Fraction(30) + Fraction(60, 128), 1, 4),
        ]
        # The next segment's first downbeat opens measure 16.
        assert _restated_index_grid().seconds_at(16).value == 30.9375

    def test_a_restated_index_raises_and_names_its_candidates(self) -> None:
        """Answering one of two instants would resolve a question not asked."""
        grid = _restated_index_grid()

        with pytest.raises(
            ValueError,
            match=r"Measure 15 states beat 3 at more than one instant: 29, 30",
        ):
            grid.seconds_at(15, 3)
        with pytest.raises(ValueError, match="more than one instant"):
            grid.segment_seconds_at(15, 4)

    def test_an_unambiguous_index_in_the_same_measure_still_answers(self) -> None:
        """Only the restated indices are refused."""
        grid = _restated_index_grid()

        assert grid.seconds_at(15, 2).value == 28.5
        assert grid.seconds_at(15).value == 28.0
        assert grid.segment_seconds_at(15, 2).value == 28.5

    def test_interpolation_uses_the_unique_neighbours(self) -> None:
        """A fractional beat needs both ends to name one instant each."""
        grid = _restated_index_grid()

        # Beats 1 and 2 are unique: 28.0 + 0.5 * 0.5 = 28.25.
        assert grid.seconds_at(15, 1.5).value == 28.25
        # Beat 3 is restated, so 2.5 cannot be interpolated.
        with pytest.raises(ValueError, match="beat 3 at more than one instant"):
            grid.seconds_at(15, 2.5)

    def test_instants_stay_unambiguous(self) -> None:
        """Reading a position and tabulating are unaffected by the restatement."""
        grid = _restated_index_grid()

        assert grid.position_at(30).beat == 3
        assert grid.position_at(Fraction("29.2")).beat == 3
        table = grid.get_beat_table()
        measure_15 = table[table["measure"] == 15]
        assert len(measure_15) == 6
        assert measure_15["beat"].tolist() == [1, 2, 3, 4, 3, 4]
        assert measure_15["seconds"].tolist() == [
            28.0,
            28.5,
            29.0,
            29.5,
            30.0,
            30.46875,
        ]


class TestBeatCoordinates:
    """A beat stores the exact ratio and publishes the seconds coordinate."""

    def test_seconds_is_the_exact_ratio(self) -> None:
        """The stored field is what the grid's arithmetic produced."""
        beat = _reanchor_grid().position_at(Fraction("6.5"))

        assert isinstance(beat.seconds, Fraction)
        assert beat.seconds == Fraction("6.01")
        # 601/100 has no exact float, so a float lane would show here.
        assert beat.seconds != Fraction(6.01)

    def test_the_instant_is_a_float_seconds_coordinate(self) -> None:
        """The published coordinate follows the seconds axis's own type."""
        beat = _reanchor_grid().position_at(Fraction("6.5"))

        assert isinstance(beat.instant, Coordinate)
        assert beat.instant.unit is TimeUnit.seconds
        assert beat.instant.number_type is NumberType.float
        assert beat.instant.value == 6.01
        assert beat.instant == Coordinate(6.01, TimeUnit.seconds)

    def test_the_two_directions_of_the_grid_compare_equal(self) -> None:
        """Reading a position and naming a measure answer the same value.

        Measure 16 of the restated-index grid opens at 30.9375, a value
        every lane represents exactly, so the assertion pins the equality
        rather than a shared rounding.
        """
        grid = _restated_index_grid()

        assert grid.seconds_at(16).value == 30.9375
        assert grid.position_at(30.9375).instant == grid.seconds_at(16)
        # And it is a real comparison: a different beat is not equal.
        assert grid.position_at(30).instant != grid.seconds_at(16)

    def test_the_equality_holds_for_ratios_without_an_exact_float(self) -> None:
        """Both directions convert the same ratio, so both land together."""
        grid = _reanchor_grid()

        assert grid.position_at(Fraction("7.5")).instant == grid.seconds_at(3)
        assert grid.seconds_at(3).value == 7.01

    def test_a_beat_is_built_from_its_exact_seconds(self) -> None:
        """Construction takes the ratio; equality compares the stored fields."""
        built = GridBeat(
            seconds=Fraction("6.01"), segment=1, measure=2, segment_measure=0, beat=3
        )

        assert built == _reanchor_grid().position_at(Fraction("6.5"))
        assert built.instant == Coordinate(6.01, TimeUnit.seconds)


class TestBoundaryRule:
    """A beat within half a beat of the next segment is the next anchor."""

    def test_a_beat_far_before_the_boundary_is_kept(self) -> None:
        """0.30 s before the next anchor, at a 0.25 s half-beat: kept."""
        grid = _phantom_grid()
        second = [beat for beat in grid.iter_beats() if beat.segment == 1]

        assert len(second) == 20
        assert second[-1].seconds == Fraction("20.01")
        assert Fraction("20.31") - second[-1].seconds == Fraction("0.3")

    def test_a_beat_just_before_the_boundary_is_dropped(self) -> None:
        """0.01 s before the next anchor: the displaced anchor, not a beat."""
        grid = _phantom_grid()
        first = [beat for beat in grid.iter_beats() if beat.segment == 0]

        assert len(first) == 20
        assert first[-1].seconds == Fraction(10)
        assert Fraction("10.5") not in {beat.seconds for beat in grid.iter_beats()}

    def test_a_beat_exactly_half_a_beat_before_the_boundary_is_kept(self) -> None:
        """The knife edge: 60 BPM, boundary at 3.5, beat at 3.0 -- kept."""
        grid = _pickup_grid()
        first = [beat for beat in grid.iter_beats() if beat.segment == 0]

        assert [beat.seconds for beat in first] == [
            Fraction(0),
            Fraction(1),
            Fraction(2),
            Fraction(3),
        ]
        # The gap to the boundary is exactly half of the 60 BPM beat.
        assert Fraction("3.5") - first[-1].seconds == grid.segments[0].beat_seconds / 2
        assert grid.position_at(Fraction("3.2")).seconds == Fraction(3)

    def test_the_rule_does_not_apply_at_the_end_of_the_grid(self) -> None:
        """The last segment keeps every beat strictly below the extent."""
        grid = _phantom_grid()
        last = [beat for beat in grid.iter_beats() if beat.segment == 2]

        assert last[-1].seconds == Fraction("29.81")
        assert Fraction(30) - last[-1].seconds < Fraction(1, 4)

    def test_a_beat_exactly_at_the_extent_is_not_generated(self) -> None:
        """The extent bounds the grid half-openly."""
        grid = BeatGrid([_segment(0, 120)], extent=2)

        assert [beat.seconds for beat in grid.iter_beats()] == [
            Fraction(0),
            Fraction(1, 2),
            Fraction(1),
            Fraction(3, 2),
        ]


class TestNumbering:
    """One walk produces both measure numberings."""

    def test_beats_before_the_first_downbeat_are_measure_zero(self) -> None:
        """A Battito 4 anchor opens the represented tail of bar 0."""
        beats = list(_pickup_grid().iter_beats())

        assert (beats[0].seconds, beats[0].measure, beats[0].beat) == (
            Fraction(0),
            0,
            4,
        )
        assert (beats[1].seconds, beats[1].measure, beats[1].beat) == (
            Fraction(1),
            1,
            1,
        )

    def test_the_two_numberings_differ_at_a_mid_measure_re_anchor(self) -> None:
        """The anchor continues set measure 2 and opens segment measure 0."""
        beats = {beat.seconds: beat for beat in _reanchor_grid().iter_beats()}

        anchor = beats[Fraction("6.01")]
        assert (
            anchor.segment,
            anchor.measure,
            anchor.segment_measure,
            anchor.beat,
        ) == (
            1,
            2,
            0,
            3,
        )
        downbeat = beats[Fraction("7.01")]
        assert (
            downbeat.segment,
            downbeat.measure,
            downbeat.segment_measure,
            downbeat.beat,
        ) == (1, 3, 1, 1)
        assert downbeat.is_downbeat and not anchor.is_downbeat

    def test_measure_numbering_continues_across_segments(self) -> None:
        """A new segment never resets the grid-wide count."""
        beats = list(_phantom_grid().iter_beats())
        downbeats = [beat for beat in beats if beat.is_downbeat]

        assert len(downbeats) == 15
        assert [beat.measure for beat in downbeats] == list(range(1, 16))
        assert [beat.segment_measure for beat in downbeats] == [1, 2, 3, 4, 5] * 3

    def test_n_measures_counts_a_leading_partial_measure(self) -> None:
        """Fifteen downbeats and no pickup are fifteen measures."""
        assert _phantom_grid().n_measures == 15
        # Three downbeats plus the Battito 4 pickup that precedes the first.
        assert _pickup_grid().n_measures == 4


class TestConversions:
    """All four directions, on hand-derived values."""

    def test_seconds_at_answers_the_labelled_beat(self) -> None:
        """Measure 6 of the phantom grid opens at the second anchor."""
        grid = _phantom_grid()

        answer = grid.seconds_at(6)
        assert isinstance(answer, Coordinate)
        assert answer.unit is TimeUnit.seconds
        assert answer.value == 10.51
        assert grid.seconds_at(6, 3).value == 11.51

    def test_a_fractional_beat_interpolates_between_its_neighbours(self) -> None:
        """Beat 2.5 at 120 BPM is half a beat past beat 2."""
        grid = BeatGrid([_segment(0, 120)], extent=8)

        assert grid.seconds_at(1, 2.5).value == 0.75
        assert grid.seconds_at(1, Fraction(3, 2)).value == 0.25

    def test_segment_seconds_at_subtracts_the_segment_start(self) -> None:
        """The two readings of one instant differ by the segment's start."""
        grid = _phantom_grid()

        assert grid.seconds_at(7).value == 12.51
        assert grid.segment_seconds_at(7).value == 2.0

    def test_position_at_floors_to_the_beat_that_is_sounding(self) -> None:
        """A position between beats names the earlier one."""
        grid = _phantom_grid()

        found = grid.position_at(10.505)
        assert isinstance(found, GridBeat)
        assert (found.seconds, found.segment, found.measure, found.beat) == (
            Fraction(10),
            0,
            5,
            4,
        )
        assert grid.position_at(Fraction("10.51")).seconds == Fraction("10.51")

    def test_position_at_carries_both_numberings(self) -> None:
        """The re-anchored grid answers set and segment measures together."""
        grid = _reanchor_grid()

        assert grid.position_at(Fraction("6.5")) == GridBeat(
            seconds=Fraction("6.01"),
            segment=1,
            measure=2,
            segment_measure=0,
            beat=3,
        )
        assert grid.position_at(Fraction("6.6")).beat == 4

    def test_position_at_accepts_a_seconds_coordinate(self) -> None:
        """Coordinate input is the same query as a raw number."""
        grid = _phantom_grid()

        assert grid.position_at(
            Coordinate(10.505, TimeUnit.seconds)
        ) == grid.position_at(Fraction("10.505"))

    def test_a_caller_policy_reads_the_index_as_a_quarter_offset(self) -> None:
        """Eighth-note counting puts beat 3 one quarter past the downbeat."""
        grid = BeatGrid([_segment(0, 120)], extent=8)
        eighths = BeatPolicy.uniform(Fraction(1, 2), 8)

        # 2 x 1/2 = 1 quarter; at 120 BPM in 4/4 that is 0.5 s.
        assert grid.seconds_at(1, 3, policy=eighths).value == 0.5
        assert grid.seconds_at(1, Fraction(7, 2), policy=eighths).value == 0.625
        # Second 1.0 is quarter offset 2, the fifth eighth and the third quarter.
        assert grid.position_at(1.0, policy=eighths).beat == 5
        assert grid.position_at(1.0).beat == 3

    def test_a_caller_policy_integrates_across_a_tempo_change(self) -> None:
        """Beat 3 of measure 2 is two quarters into a 60 BPM stretch."""
        grid = _reanchor_grid()

        assert grid.seconds_at(2, 3, policy=FOUR_FOUR).value == 6.0
        assert grid.seconds_at(2, 4, policy=FOUR_FOUR).value == 6.505

    def test_segment_at_names_the_half_open_span(self) -> None:
        """An anchor belongs to the segment it opens."""
        grid = _reanchor_grid()

        assert grid.segment_at(0) == 0
        assert grid.segment_at(6.0) == 0
        assert grid.segment_at(Fraction("6.01")) == 1
        assert grid.segment_at(Fraction("11.999")) == 1


class TestQuartersBetween:
    """Exact integration, not an average tempo."""

    def test_a_span_across_a_tempo_change_is_neither_tempo(self) -> None:
        """2.01 quarters at 60 BPM plus 2 quarters at 120 BPM."""
        grid = _reanchor_grid()

        assert grid.quarters_between(4, Fraction("7.01")) == Fraction(401, 100)
        assert grid.quarters_between(4, Fraction("6.01")) == Fraction(201, 100)
        assert grid.quarters_between(Fraction("6.01"), Fraction("7.01")) == Fraction(2)

    def test_positions_before_the_first_segment_use_its_tempo(self) -> None:
        """A grid opening late still reads the seconds before it."""
        grid = _phantom_grid()

        assert grid.quarters_between(0, Fraction("0.5")) == Fraction(1)

    def test_an_empty_span_is_zero_and_a_backwards_span_raises(self) -> None:
        """A span is oriented; running it backwards is a caller error."""
        grid = _phantom_grid()

        assert grid.quarters_between(4, 4) == Fraction(0)
        with pytest.raises(ValueError, match="runs backwards"):
            grid.quarters_between(5, 4)

    def test_the_domain_is_closed_at_the_extent(self) -> None:
        """No beat sounds at the extent, but the grid states a tempo up to it."""
        grid = _phantom_grid()

        # The last measure runs from its downbeat to the extent: 1.69 s.
        assert grid.quarters_between(Fraction("28.31"), 30) == Fraction(169, 50)
        with pytest.raises(
            ValueError, match=r"end at 30001/1000 lies beyond the grid's extent 30"
        ):
            grid.quarters_between(0, Fraction("30.001"))
        with pytest.raises(ValueError, match="lies before zero"):
            grid.quarters_between(-1, 4)

    def test_an_unbounded_grid_has_no_upper_bound(self) -> None:
        """Without an extent the tempo runs on, and so does the integration."""
        grid = BeatGrid.from_tempo(120)

        assert grid.quarters_between(0, 1000) == Fraction(2000)
        with pytest.raises(ValueError, match="lies before zero"):
            grid.quarters_between(-1, 0)

    def test_the_counted_value_governs_the_quarter_reading(self) -> None:
        """A 6/8 bar is three quarters however many beats count it."""
        grid = BeatGrid([_segment(0, 120, metro="6/8")], extent=10)

        # Six eighth beats of 0.5 s span 3 s and three quarters.
        assert grid.quarters_between(0, 3) == Fraction(3)


class TestBeatTable:
    """The table renders the same walk the scalar getters read."""

    def test_columns_and_row_count_match_the_generated_beats(self) -> None:
        """One row per beat, five columns, in time order."""
        grid = _phantom_grid()

        table = grid.get_beat_table()

        assert list(table.columns) == [
            "seconds",
            "segment",
            "segment_seconds",
            "measure",
            "beat",
        ]
        assert len(table) == 60
        assert table["seconds"].tolist()[:3] == [0.5, 1.0, 1.5]
        assert table["seconds"].iloc[-1] == 29.81
        assert table["measure"].iloc[-1] == 15
        assert table["beat"].iloc[-1] == 4

    def test_rows_agree_with_the_scalar_getters(self) -> None:
        """The table is a rendering, so it cannot answer differently."""
        grid = _phantom_grid()
        table = grid.get_beat_table()

        row = table[(table["measure"] == 6) & (table["beat"] == 1)]
        assert row["seconds"].iloc[0] == grid.seconds_at(6).value
        assert row["segment_seconds"].iloc[0] == grid.segment_seconds_at(6).value
        assert row["segment"].iloc[0] == 1

    def test_the_segment_filter_keeps_one_segment(self) -> None:
        """Restricting to a segment keeps its beats and its numbering."""
        grid = _phantom_grid()

        table = grid.get_beat_table(segment=0)

        assert len(table) == 20
        assert set(table["segment"]) == {0}
        assert table["measure"].tolist()[-1] == 5
        assert table["segment_seconds"].tolist()[:2] == [0.0, 0.5]

    def test_segment_numbering_restarts_per_segment(self) -> None:
        """The segment reading opens a re-anchored segment at measure 0."""
        grid = _reanchor_grid()

        table = grid.get_beat_table(numbering="segment", segment=1)

        assert table["measure"].tolist()[:3] == [0, 0, 1]
        assert grid.get_beat_table(segment=1)["measure"].tolist()[:3] == [2, 2, 3]

    def test_an_unknown_segment_or_numbering_raises(self) -> None:
        """A table cannot be built for something the grid does not hold."""
        grid = _phantom_grid()

        with pytest.raises(ValueError, match="Segment 3 is outside 0..2"):
            grid.get_beat_table(segment=3)
        with pytest.raises(ValueError, match="Unknown numbering"):
            grid.get_beat_table(numbering="bars")  # type: ignore[arg-type]


class TestFromTempo:
    """The one-segment convenience states the same facts."""

    def test_from_tempo_builds_one_bounded_segment(self) -> None:
        """Defaults are 4/4 from zero, bounded by the extent."""
        grid = BeatGrid.from_tempo(120, extent=8)

        assert grid.n_segments == 1
        assert grid.segments[0].start == Fraction(0)
        assert grid.segments[0].policy.name == "4/4"
        assert grid.segments[0].end == Fraction(8)
        assert len(list(grid.iter_beats())) == 16

    def test_from_tempo_takes_a_start_anchor_and_a_meter(self) -> None:
        """A Battito 2 anchor at 150 BPM reaches its downbeat at 1.345 s."""
        grid = BeatGrid.from_tempo(
            150, metro="4/4", start="0.145", battito=2, extent=220
        )

        assert grid.segments[0].first_downbeat == Fraction("1.345")
        assert grid.seconds_at(1).value == 1.345

    def test_from_tempo_accepts_an_explicit_policy(self) -> None:
        """An explicit policy overrides the meter string."""
        grid = BeatGrid.from_tempo(120, policy=BeatPolicy.uniform(Fraction(1), 3))

        assert grid.segments[0].policy.n_beats == 3
        assert grid.segments[0].policy.name is None

    def test_from_tempo_without_an_extent_is_unbounded(self) -> None:
        """An unbounded grid generates beats for as long as it is asked."""
        grid = BeatGrid.from_tempo(120)

        assert grid.extent is None
        assert grid.segments[0].end is None
        assert [beat.seconds for beat in grid.iter_beats(stop=2)] == [
            Fraction(0),
            Fraction(1, 2),
            Fraction(1),
            Fraction(3, 2),
        ]
        assert grid.seconds_at(100).value == 198.0

    def test_a_float_contributes_its_exact_binary_value(self) -> None:
        """Floats are read exactly, never rounded to a tidier ratio."""
        grid = BeatGrid.from_tempo(120, start=0.1, extent=8)

        assert grid.segments[0].start == Fraction(0.1)
        assert grid.segments[0].start != Fraction(1, 10)


class TestExport:
    """Annotation exports delegate to the table."""

    def test_sonic_visualiser_labels_every_beat(self, tmp_path: Path) -> None:
        """Two fields, a header, and M<measure>B<beat> labels."""
        grid = BeatGrid.from_tempo(120, extent=2)
        path = tmp_path / "beats.csv"

        assert grid.export_to_csv(path, format="sonic_visualiser") == 4
        assert path.read_text(encoding="utf-8").splitlines() == [
            "TIME,LABEL",
            "0.0,M1B1",
            "0.5,M1B2",
            "1.0,M1B3",
            "1.5,M1B4",
        ]

    def test_sonic_visualiser_measures_and_both(self, tmp_path: Path) -> None:
        """Measure labels mark downbeats; both interleaves in time order."""
        grid = BeatGrid.from_tempo(120, extent=2)

        measures = tmp_path / "measures.csv"
        assert (
            grid.export_to_csv(measures, format="sonic_visualiser", labels="measures")
            == 1
        )
        assert measures.read_text(encoding="utf-8").splitlines() == [
            "TIME,LABEL",
            "0.0,M1",
        ]

        both = tmp_path / "both.csv"
        assert grid.export_to_csv(both, format="sonic_visualiser", labels="both") == 5
        assert both.read_text(encoding="utf-8").splitlines()[:3] == [
            "TIME,LABEL",
            "0.0,M1B1",
            "0.0,M1",
        ]

    def test_tilia_writes_four_fields(self, tmp_path: Path) -> None:
        """Time, measure, beat and the downbeat flag."""
        grid = BeatGrid.from_tempo(120, extent=2)
        path = tmp_path / "tilia.csv"

        assert grid.export_to_csv(path, format="tilia") == 4
        assert path.read_text(encoding="utf-8").splitlines() == [
            "time,measure,beat,is_first_in_measure",
            "0.0,1,1,True",
            "0.5,1,2,False",
            "1.0,1,3,False",
            "1.5,1,4,False",
        ]

    def test_unknown_format_or_labels_raise(self, tmp_path: Path) -> None:
        """An unrecognised spelling is refused before anything is written."""
        grid = BeatGrid.from_tempo(120, extent=2)

        with pytest.raises(ValueError, match="Unknown format"):
            grid.export_to_csv(tmp_path / "x.csv", format="default")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown labels"):
            grid.export_to_csv(
                tmp_path / "x.csv",
                format="sonic_visualiser",
                labels="downbeats",  # type: ignore[arg-type]
            )


class TestAtomicRaises:
    """An unresolvable query raises rather than answering approximately."""

    def test_a_position_outside_the_grid_raises(self) -> None:
        """Before the first segment and at or after the extent."""
        grid = _phantom_grid()

        with pytest.raises(ValueError, match="lies before the grid"):
            grid.position_at(Fraction("0.4"))
        with pytest.raises(ValueError, match="at or after the grid's extent"):
            grid.position_at(30)
        with pytest.raises(ValueError, match="lies before the grid"):
            grid.segment_at(0)

    def test_a_measure_or_beat_the_grid_does_not_state_raises(self) -> None:
        """No silent nearest match, and the candidates are named."""
        grid = _phantom_grid()

        with pytest.raises(ValueError, match="Measure 16 is not one this grid states"):
            grid.seconds_at(16)
        with pytest.raises(ValueError, match="has no beat 5"):
            grid.seconds_at(1, 5)
        with pytest.raises(ValueError, match="below the downbeat"):
            grid.seconds_at(1, 0)

    def test_a_fractional_beat_without_an_upper_neighbour_raises(self) -> None:
        """Interpolation needs both ends; the measure's last beat has one end."""
        grid = _phantom_grid()

        with pytest.raises(ValueError, match="has no beat 5"):
            grid.seconds_at(1, 4.5)

    def test_a_coordinate_in_another_unit_raises(self) -> None:
        """A grid is measured in seconds and says so."""
        grid = _phantom_grid()

        with pytest.raises(ValueError, match="measured in 'seconds', not 'quarters'"):
            grid.position_at(Coordinate(Fraction(4), TimeUnit.quarters))

    def test_a_policy_reading_past_the_measure_raises(self) -> None:
        """A caller policy cannot address past its own bar or its measure."""
        grid = BeatGrid([_segment(0, 120)], extent=8)

        with pytest.raises(ValueError, match="outside 1..4"):
            grid.seconds_at(1, 5, policy=FOUR_FOUR)
        # Half-note beats: beat 3 is 4 quarters in, which is measure 2.
        with pytest.raises(ValueError, match="at or beyond 2"):
            grid.seconds_at(1, 3, policy=BeatPolicy.uniform(Fraction(2), 4))
        # And beyond the grid entirely, the integration itself runs out:
        # beat 4 of measure 4 is 6 quarters past its 6.0 s downbeat.
        with pytest.raises(ValueError, match="lies beyond the grid's extent 8"):
            grid.seconds_at(4, 4, policy=BeatPolicy.uniform(Fraction(2), 4))

    def test_a_policy_needs_a_measure_the_grid_opens(self) -> None:
        """The partial measure before the first downbeat has no downbeat."""
        grid = _pickup_grid()

        assert grid.position_at(0).measure == 0
        with pytest.raises(ValueError, match="Measure 0 has no downbeat"):
            grid.position_at(0, policy=FOUR_FOUR)
        with pytest.raises(ValueError, match="Measure 0 has no downbeat"):
            grid.seconds_at(0, 1, policy=FOUR_FOUR)

    def test_a_policy_offset_of_zero_is_the_downbeat(self) -> None:
        """Beat 1 under any policy is the downbeat itself."""
        grid = _reanchor_grid()

        assert grid.seconds_at(3, 1, policy=FOUR_FOUR).value == 7.01

    def test_the_last_measure_is_bounded_by_the_extent(self) -> None:
        """With no following downbeat, the extent bounds the reading."""
        grid = BeatGrid([_segment(0, 120)], extent=8)

        # Measure 4 opens at 6.0; two half-note beats later is 7.0 < 8.
        assert (
            grid.seconds_at(4, 2, policy=BeatPolicy.uniform(Fraction(2), 4)).value
            == 7.0
        )

    def test_a_policy_reaching_past_an_unbounded_grid(self) -> None:
        """An unbounded grid resolves a policy offset at its last tempo."""
        grid = BeatGrid.from_tempo(120)

        assert grid.seconds_at(3, 3, policy=FOUR_FOUR).value == 5.0

    def test_a_position_after_a_fully_displaced_first_segment_raises(self) -> None:
        """A segment whose every beat is the next anchor generates none."""
        # The 0.1 s segment is shorter than half of its own 0.5 s beat.
        grid = BeatGrid([_segment(0, 120), _segment("0.1", 120)], extent=4)

        assert [beat.segment for beat in grid.iter_beats()][:2] == [1, 1]
        with pytest.raises(ValueError, match="lies before the grid's first beat"):
            grid.position_at(Fraction("0.05"))

    def test_an_unbounded_grid_cannot_be_tabulated_or_counted(self) -> None:
        """Both would have to generate beats without end."""
        grid = BeatGrid.from_tempo(120)

        with pytest.raises(ValueError, match="Cannot tabulate an unbounded beat grid"):
            grid.get_beat_table()
        with pytest.raises(ValueError, match="generates beats without end"):
            _ = grid.n_measures

    def test_a_grid_has_no_serialized_format(self) -> None:
        """Segments are the source facts; a second encoding of them would not be."""
        grid = _phantom_grid()

        assert not hasattr(grid, "to_dict")
        assert not hasattr(BeatGrid, "from_dict")


class TestStructuralView:
    """A structure read from a tempo source carries the grid that made it."""

    @staticmethod
    def _skeleton(grid: BeatGrid | None) -> TimeSkeleton:
        measures = [
            RegularMeasure(
                number=number,
                time_signature="4/4",
                nominal_length=Fraction(4),
                actual_length=Fraction(4),
            )
            for number in (1, 2)
        ]
        return TimeSkeleton(SectionHierarchy.from_measures(measures), beat_grid=grid)

    def test_create_beatgrid_returns_the_stored_grid(self) -> None:
        """The generation entry hands back the very object, not a rebuild."""
        grid = _phantom_grid()

        assert self._skeleton(grid).create_beatgrid() is grid

    def test_create_beatgrid_without_a_tempo_raises(self) -> None:
        """Measure lengths alone state no tempo, and none is invented."""
        with pytest.raises(ValueError, match="carries no tempo segments"):
            self._skeleton(None).create_beatgrid()

    def test_structural_equality_includes_the_grid(self) -> None:
        """Same measures, different tempi: not the same structure."""
        assert self._skeleton(None) == self._skeleton(None)
        assert self._skeleton(_phantom_grid()) == self._skeleton(_phantom_grid())
        assert self._skeleton(_phantom_grid()) != self._skeleton(None)
        assert self._skeleton(_phantom_grid()) != self._skeleton(_reanchor_grid())
