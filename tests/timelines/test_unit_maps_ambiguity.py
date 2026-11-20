"""Two maps, one unit.

Reasoning and rules: ``tests/timelines/README.md``,
section "test_unit_maps_ambiguity.py".
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign.core import TimeUnit
from timetoalign.maps import LinearMap, ScalarMap
from timetoalign.timelines import ContinuousLogicalTimeline, DiscreteGraphicalTimeline


def _two_seconds_maps() -> tuple[ContinuousLogicalTimeline, LinearMap, LinearMap]:
    """A quarters timeline reading seconds two ways: at 60 and at 120 bpm."""
    timeline = ContinuousLogicalTimeline(length=Fraction(16), uid="clt_1")
    slow = LinearMap(
        scalar=1.0,
        offset=0,
        source_unit=TimeUnit.quarters,
        target_unit=TimeUnit.seconds,
        uid="tempo_60",
        name="at_60_bpm",
    )
    fast = LinearMap(
        scalar=0.5,
        offset=0,
        source_unit=TimeUnit.quarters,
        target_unit=TimeUnit.seconds,
        uid="tempo_120",
        name="at_120_bpm",
    )
    timeline.add_conversion_map(slow)
    timeline.add_conversion_map(fast)
    return timeline, slow, fast


class TestNothingIsDisplaced:
    """A second map for a unit never replaces the first."""

    def test_both_maps_survive_in_both_registries(self) -> None:
        timeline, slow, fast = _two_seconds_maps()

        assert timeline.n_conversion_maps == 2
        assert timeline._get_unit_maps(TimeUnit.seconds) == [slow, fast]
        assert timeline._unit_maps[TimeUnit.seconds] == [slow, fast]

    def test_a_single_map_still_answers_without_a_name(self) -> None:
        timeline = ContinuousLogicalTimeline(length=Fraction(16), uid="clt_1")
        only = LinearMap(
            scalar=1.0,
            offset=0,
            source_unit=TimeUnit.quarters,
            target_unit=TimeUnit.seconds,
        )
        timeline.add_conversion_map(only)

        assert timeline._get_unit_map(TimeUnit.seconds) is only
        assert timeline.get_conversion_map(TimeUnit.seconds) is only

    def test_an_unmapped_unit_is_none(self) -> None:
        timeline, _slow, _fast = _two_seconds_maps()

        assert timeline._get_unit_maps(TimeUnit.ticks) == []
        assert timeline._get_unit_map(TimeUnit.ticks) is None
        assert timeline.get_conversion_map(TimeUnit.ticks) is None


class TestTheScalarLaneRefusesToGuess:
    """Two readings of one axis are not a thing to pick between."""

    def test_the_private_lane_raises_and_lists_both(self) -> None:
        timeline, _slow, _fast = _two_seconds_maps()

        with pytest.raises(ValueError) as error:
            timeline._get_unit_map(TimeUnit.seconds)

        message = str(error.value)
        assert "tempo_60 (at_60_bpm)" in message
        assert "tempo_120 (at_120_bpm)" in message

    def test_the_public_lookup_raises_the_same_way(self) -> None:
        timeline, _slow, _fast = _two_seconds_maps()

        with pytest.raises(ValueError, match="name the one you mean"):
            timeline.get_conversion_map(TimeUnit.seconds)

    def test_the_alias_spelling_raises_too(self) -> None:
        timeline, _slow, _fast = _two_seconds_maps()

        with pytest.raises(ValueError, match="name the one you mean"):
            timeline.get_conversion_map("s")


class TestNamingSelects:
    """An id or a name picks one of the readings."""

    def test_by_id(self) -> None:
        timeline, slow, fast = _two_seconds_maps()

        assert timeline._get_unit_map(TimeUnit.seconds, name="tempo_60") is slow
        assert timeline._get_unit_map(TimeUnit.seconds, name="tempo_120") is fast

    def test_by_name(self) -> None:
        timeline, slow, fast = _two_seconds_maps()

        assert timeline._get_unit_map(TimeUnit.seconds, name="at_60_bpm") is slow
        assert timeline._get_unit_map(TimeUnit.seconds, name="at_120_bpm") is fast

    def test_an_unknown_name_raises(self) -> None:
        timeline, _slow, _fast = _two_seconds_maps()

        with pytest.raises(KeyError, match="at_90_bpm"):
            timeline._get_unit_map(TimeUnit.seconds, name="at_90_bpm")

    def test_the_id_lane_still_finds_a_named_map(self) -> None:
        timeline, slow, _fast = _two_seconds_maps()

        assert timeline.get_conversion_map("tempo_60") is slow
        assert timeline.get_conversion_map("at_60_bpm") is slow


class TestTheStampShowsAllReadings:
    """A stamp reports both answers rather than choosing one."""

    def test_conversion_rows_carry_both(self) -> None:
        timeline, _slow, _fast = _two_seconds_maps()

        stamp = timeline.get_timestamp(Fraction(4))
        rows = stamp._conversion_rows()

        assert [value for _label, value, _suffix in rows] == [4.0, 2.0]

    def test_each_reading_is_reachable_by_name(self) -> None:
        timeline, _slow, _fast = _two_seconds_maps()

        stamp = timeline.get_timestamp(Fraction(4))

        assert stamp.get_conversion_for("at_60_bpm") == 4.0
        assert stamp.get_conversion_for("at_120_bpm") == 2.0

    def test_the_table_carries_one_column_per_reading(self) -> None:
        timeline, _slow, _fast = _two_seconds_maps()

        names = timeline.get_timestamp_table().column_names

        assert "at_60_bpm" in names
        assert "at_120_bpm" in names


class TestDuplicateIdsAreRefused:
    """One id, one map, on one timeline."""

    def test_attaching_the_same_map_twice_raises(self) -> None:
        timeline = ContinuousLogicalTimeline(length=Fraction(16), uid="clt_1")
        cmap = ScalarMap(
            scalar=2.0,
            source_unit=TimeUnit.quarters,
            target_unit=TimeUnit.seconds,
            uid="tempo",
        )
        timeline.add_conversion_map(cmap)

        with pytest.raises(ValueError, match="already attached"):
            timeline.add_conversion_map(cmap)

    def test_a_different_map_reusing_an_id_raises(self) -> None:
        timeline = ContinuousLogicalTimeline(length=Fraction(16), uid="clt_1")
        timeline.add_conversion_map(
            ScalarMap(
                scalar=2.0,
                source_unit=TimeUnit.quarters,
                target_unit=TimeUnit.seconds,
                uid="tempo",
            )
        )

        with pytest.raises(ValueError, match="already held by conversion map 'tempo'"):
            timeline.add_conversion_map(
                ScalarMap(
                    scalar=3.0,
                    source_unit=TimeUnit.quarters,
                    target_unit=TimeUnit.ticks,
                    uid="tempo",
                )
            )


class TestDetachRemovesByIdentity:
    """The temporary inverse goes; a pre-existing same-unit map stays."""

    def test_a_pre_existing_map_survives_child_conversion(self) -> None:
        parent = DiscreteGraphicalTimeline(length=1000, uid="dgt_1")
        parent.add_conversion_map(
            LinearMap(
                scalar=Fraction(1, 4),
                offset=0,
                source_unit=TimeUnit.pixels,
                target_unit=TimeUnit.quarters,
                uid="pixels_to_quarters",
            )
        )
        child = ContinuousLogicalTimeline(length=Fraction(100), uid="clt_1")
        survivor = LinearMap(
            scalar=8,
            offset=0,
            source_unit=TimeUnit.quarters,
            target_unit=TimeUnit.pixels,
            uid="child_own_map",
        )
        child.add_conversion_map(survivor)

        parent.add_child(child, offset=0, use_conversion_map=True)

        assert survivor.id in child._conversion_maps
        assert child._get_unit_maps(TimeUnit.pixels) == [survivor]

    def test_the_temporary_map_leaves_no_trace(self) -> None:
        parent = DiscreteGraphicalTimeline(length=1000, uid="dgt_1")
        parent.add_conversion_map(
            LinearMap(
                scalar=Fraction(1, 4),
                offset=0,
                source_unit=TimeUnit.pixels,
                target_unit=TimeUnit.quarters,
                uid="pixels_to_quarters",
            )
        )
        child = ContinuousLogicalTimeline(length=Fraction(100), uid="clt_1")

        parent.add_child(child, offset=0, use_conversion_map=True)

        assert child._get_unit_maps(TimeUnit.pixels) == []
        assert TimeUnit.pixels not in child._unit_maps
