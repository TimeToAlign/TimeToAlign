"""How a bar is counted in beats.

Reasoning, specimens and gold values: ``tests/core/README.md``,
section "test_beat_policy.py".
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from timetoalign.core import BeatPolicy


class TestFromTimeSignature:
    """A signature states which value is counted and how it groups."""

    def test_compound_six_eight_counts_two_dotted_beats(self) -> None:
        policy = BeatPolicy.from_time_signature("6/8")

        assert policy.grouping == (3, 3)
        assert policy.division == Fraction(1, 2)
        assert policy.rods == (Fraction(3, 2), Fraction(3, 2))
        assert policy.n_beats == 2
        assert policy.span == Fraction(3)

    def test_simple_four_four_counts_four_quarters(self) -> None:
        policy = BeatPolicy.from_time_signature("4/4")

        assert policy.grouping == (1, 1, 1, 1)
        assert policy.rods == (Fraction(1), Fraction(1), Fraction(1), Fraction(1))
        assert policy.span == Fraction(4)

    def test_three_eight_stays_simple(self) -> None:
        """The compound rule needs more than three divisions to fire."""
        policy = BeatPolicy.from_time_signature("3/8")

        assert policy.grouping == (1, 1, 1)
        assert policy.rods == (Fraction(1, 2),) * 3

    def test_three_four_stays_simple(self) -> None:
        policy = BeatPolicy.from_time_signature("3/4")

        assert policy.grouping == (1, 1, 1)
        assert policy.span == Fraction(3)

    def test_nine_eight_counts_three_dotted_beats(self) -> None:
        policy = BeatPolicy.from_time_signature("9/8")

        assert policy.grouping == (3, 3, 3)
        assert policy.rods == (Fraction(3, 2),) * 3
        assert policy.span == Fraction(9, 2)

    def test_composite_signature_spells_its_own_grouping(self) -> None:
        policy = BeatPolicy.from_time_signature("3/8+2/8+3/8")

        assert policy.grouping == (3, 2, 3)
        assert policy.rods == (Fraction(3, 2), Fraction(1), Fraction(3, 2))
        assert policy.offset_for(3) == Fraction(5, 2)
        assert policy.span == Fraction(4)

    def test_common_time(self) -> None:
        assert BeatPolicy.from_time_signature("C").rods == (Fraction(1),) * 4

    def test_cut_time(self) -> None:
        for spelling in ("C|", "cut"):
            assert BeatPolicy.from_time_signature(spelling).rods == (
                Fraction(2),
                Fraction(2),
            )

    @pytest.mark.parametrize("signature", ["", "waltz", "4/", "/4", "4-4", "0/4"])
    def test_unreadable_signature_raises(self, signature: str) -> None:
        """No silent fall back to 4/4: a default is a loader's decision."""
        with pytest.raises(ValueError):
            BeatPolicy.from_time_signature(signature)

    def test_mixed_denominators_raise(self) -> None:
        with pytest.raises(ValueError, match="denominator"):
            BeatPolicy.from_time_signature("3/8+1/4")


class TestUniform:
    """Count the same bar in a different value."""

    def test_eighths_in_three_four(self) -> None:
        policy = BeatPolicy.uniform(Fraction(1, 2), 6)

        assert policy.grouping == (1,) * 6
        assert policy.rods == (Fraction(1, 2),) * 6
        assert policy.span == Fraction(3)
        assert policy.span == BeatPolicy.from_time_signature("3/4").span


class TestDerivedQuantities:
    """Rods, offsets and indices all follow from grouping x division."""

    def test_rod_for_is_one_based(self) -> None:
        policy = BeatPolicy.from_time_signature("3/8+2/8+3/8")

        assert policy.rod_for(1) == Fraction(3, 2)
        assert policy.rod_for(2) == Fraction(1)
        assert policy.rod_for(3) == Fraction(3, 2)

    def test_offsets_run_from_the_downbeat(self) -> None:
        policy = BeatPolicy.from_time_signature("3/8+2/8+3/8")

        assert policy.offset_for(1) == Fraction(0)
        assert policy.offset_for(2) == Fraction(3, 2)
        assert policy.offset_for(3) == Fraction(5, 2)

    def test_index_at_finds_the_containing_beat(self) -> None:
        policy = BeatPolicy.from_time_signature("3/8+2/8+3/8")

        assert policy.index_at(Fraction(0)) == 1
        assert policy.index_at(Fraction(1)) == 1
        assert policy.index_at(Fraction(3, 2)) == 2
        assert policy.index_at(Fraction(5, 2)) == 3
        assert policy.index_at(Fraction(7, 2)) == 3

    @pytest.mark.parametrize("offset", [Fraction(-1), Fraction(4), Fraction(9, 2)])
    def test_index_at_refuses_a_position_outside_the_bar(
        self, offset: Fraction
    ) -> None:
        policy = BeatPolicy.from_time_signature("3/8+2/8+3/8")

        with pytest.raises(ValueError):
            policy.index_at(offset)

    @pytest.mark.parametrize("index", [0, 4])
    def test_out_of_range_index_raises(self, index: int) -> None:
        policy = BeatPolicy.from_time_signature("3/8+2/8+3/8")

        with pytest.raises(ValueError):
            policy.rod_for(index)


class TestValidation:
    """A policy must state something countable."""

    def test_empty_grouping_raises(self) -> None:
        with pytest.raises(ValidationError):
            BeatPolicy(grouping=(), division=Fraction(1))

    def test_zero_grouping_entry_raises(self) -> None:
        with pytest.raises(ValidationError):
            BeatPolicy(grouping=(3, 0), division=Fraction(1))

    def test_non_positive_division_raises(self) -> None:
        with pytest.raises(ValidationError):
            BeatPolicy(grouping=(1,), division=Fraction(0))

    def test_frozen(self) -> None:
        policy = BeatPolicy.from_time_signature("4/4")

        with pytest.raises(ValidationError):
            policy.division = Fraction(1, 2)
