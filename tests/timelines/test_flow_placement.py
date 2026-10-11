"""Tests for FlowMaps that place their spans instead of concatenating them.

An ordinary interval-built ``FlowMap`` stacks its played spans end to end. That
is right for assembling a performance, but wrong for undoing one: a recording
that omits two measures has to put the music *back* where it came from, leaving
the omitted stretch empty.

Three constructions describe that placement, and they are required to agree:

- **Gap entries** — ``Gap(6)`` mixed into the flow spec, pushing everything
  after it 6 quarters later. ``Gap()`` sizes itself from the neighbouring
  spans' source coordinates.
- **``at`` placements** — the target coordinate of each played span, stated
  outright.
- **A mapping** — ``{0: "A8_1", 129: "A8_2"}``, pairing each coordinate with
  the span that begins there; see ``FlowMap.from_dict``.

The motivating case throughout is A8, a recording of Satie's first Gymnopédie
(3/4, so 3 QB per measure) that skips measures 42 and 43. On the 234 QB score
that is a 6 QB hole: the performance plays ``[0, 123)`` and ``[129, 234)``,
concatenating to 228 QB. Inverting that flow must restore a 234 QB timeline
whose second span sits at QB 129 — the downbeat of measure 44.

Following the project's ZERO TOLERANCE policy, every assertion is an exact
``Fraction`` (or exact ``float`` on the timeline convenience surface) — no
ranges, no ``pytest.approx``.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign.core import struct_to_rational
from timetoalign.timelines.flow import FlowMap, Gap
from timetoalign.timelines.flow.sections import _coerce_flow_entries
from timetoalign.timelines.types import ContinuousLogicalTimeline, SegmentLine

# region Fixtures

SCORE_LENGTH = Fraction(234)  # 78 measures of 3/4
CUT_START = Fraction(123)  # downbeat of measure 42
CUT_END = Fraction(129)  # downbeat of measure 44
CUT_LENGTH = Fraction(6)  # measures 42 & 43
PERF_LENGTH = Fraction(228)  # 234 - 6


@pytest.fixture
def score() -> ContinuousLogicalTimeline:
    """The folded score, with the two played spans marked as Regions."""
    tl = ContinuousLogicalTimeline(length=SCORE_LENGTH, name="gymno_1")
    tl.create_region(0, CUT_START, id="A8_1", name="A8_1")
    tl.create_region(CUT_END, SCORE_LENGTH, id="A8_2", name="A8_2")
    tl.add_events(
        [
            {"id": "m1", "event_type": "Note", "instant": 0},
            {"id": "m42", "event_type": "Note", "instant": CUT_START},
            {"id": "m44", "event_type": "Note", "instant": CUT_END},
        ]
    )
    tl.create_flow_map(["A8_1", "A8_2"], id="A8")
    return tl


@pytest.fixture
def performance() -> ContinuousLogicalTimeline:
    """The 228 QB performance, with its two spans marked as Regions."""
    tl = ContinuousLogicalTimeline(length=PERF_LENGTH, name="A8")
    tl.create_region(0, CUT_START, id="A8_1", name="A8_1")
    tl.create_region(CUT_START, PERF_LENGTH, id="A8_2", name="A8_2")
    return tl


def sections(
    fm: FlowMap,
) -> list[tuple[str | None, Fraction, Fraction, Fraction, Fraction]]:
    """Every section as ``(label, source_start, source_end, target_start, target_end)``."""
    return [
        (s.label, s.source_start, s.source_end, s.target_start, s.target_end)
        for s in fm._sections
    ]


# endregion

# region Entry coercion


class TestCoerceFlowEntries:
    """`_coerce_flow_entries` mixes Gaps into the interval coercion."""

    def test_gaps_pass_through_between_spans(self) -> None:
        entries = _coerce_flow_entries([(0, 123), Gap(6), (123, 228)])
        assert entries == [
            (Fraction(0), Fraction(123), None),
            Gap(Fraction(6)),
            (Fraction(123), Fraction(228), None),
        ]

    def test_no_gaps_matches_plain_interval_coercion(self) -> None:
        assert _coerce_flow_entries([(0, 123), (129, 234)]) == [
            (Fraction(0), Fraction(123), None),
            (Fraction(129), Fraction(234), None),
        ]

    def test_singleton_interval_still_needs_no_list(self) -> None:
        assert _coerce_flow_entries((0, 123)) == [(Fraction(0), Fraction(123), None)]

    def test_name_string_is_never_char_iterated(self) -> None:
        regions = {"A8_1": (0, 123)}
        assert _coerce_flow_entries("A8_1", resolve=regions.get) == [
            (Fraction(0), Fraction(123), "A8_1")
        ]

    def test_gaps_alone_describe_nothing_played(self) -> None:
        with pytest.raises(ValueError, match="at least one played span"):
            _coerce_flow_entries([Gap(6), Gap(3)])

    def test_a_lone_gap_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one played span"):
            _coerce_flow_entries(Gap(6))

    def test_end_before_start_still_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be before start"):
            _coerce_flow_entries([(0, 123), Gap(6), (228, 123)])


class TestGap:
    """The `Gap` value object."""

    def test_duration_is_exact(self) -> None:
        assert Gap(6).duration == Fraction(6)
        assert Gap(1.5).duration == Fraction(3, 2)

    def test_auto_gap_has_no_duration(self) -> None:
        assert Gap().duration is None
        assert Gap().is_auto
        assert not Gap(6).is_auto

    def test_negative_duration_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            Gap(-1)

    def test_repr(self) -> None:
        assert repr(Gap(6)) == "Gap(6)"
        assert repr(Gap()) == "Gap(auto)"
        assert repr(Gap(6, "cut")) == "Gap(6, label='cut')"


# endregion

# region The two placement approaches


class TestPlacementApproachesAgree:
    """Gap entries and `at` placements build the same section table.

    Both describe the A8 restoration: the performance's two spans, ``[0, 123)``
    and ``[123, 228)``, replaced on a 234 QB axis with the second at QB 129.
    """

    @pytest.fixture
    def via_gaps(self, performance: ContinuousLogicalTimeline) -> FlowMap:
        return performance.create_flow_map(
            ["A8_1", Gap(CUT_LENGTH), "A8_2"], id="restored_gaps"
        )

    @pytest.fixture
    def via_at(self, performance: ContinuousLogicalTimeline) -> FlowMap:
        return performance.create_flow_map(
            ["A8_1", "A8_2"], at=[0, CUT_END], id="restored_at"
        )

    def test_identical_sections(self, via_gaps: FlowMap, via_at: FlowMap) -> None:
        assert sections(via_gaps) == sections(via_at)

    def test_exact_section_table(self, via_gaps: FlowMap) -> None:
        assert sections(via_gaps) == [
            ("A8_1", Fraction(0), CUT_START, Fraction(0), CUT_START),
            # The hole: no source extent, 6 QB of target extent.
            (None, CUT_START, CUT_START, CUT_START, CUT_END),
            ("A8_2", CUT_START, PERF_LENGTH, CUT_END, SCORE_LENGTH),
        ]

    def test_played_sections_counted_without_the_gap(self, via_gaps: FlowMap) -> None:
        assert via_gaps.n_sections == 2
        assert via_gaps.n_gaps == 1
        assert repr(via_gaps) == "FlowMap(restored_gaps: 2 sections, 1 gap)"

    def test_total_target_length_spans_the_hole(self, via_gaps: FlowMap) -> None:
        assert via_gaps.total_target_length == SCORE_LENGTH

    def test_second_span_lands_at_measure_44(self, via_gaps: FlowMap) -> None:
        # Performance QB 123 (the seam) is score QB 129, the downbeat of m. 44.
        assert via_gaps.unfold_coordinate(CUT_START) == [CUT_END]

    def test_first_span_is_unshifted(self, via_gaps: FlowMap) -> None:
        assert via_gaps.unfold_coordinate(50) == [Fraction(50)]

    def test_gap_reported(self, via_gaps: FlowMap) -> None:
        assert via_gaps.iter_gaps() == [(CUT_START, CUT_END, None)]

    def test_labelled_gap_keeps_its_name(
        self, performance: ContinuousLogicalTimeline
    ) -> None:
        fm = performance.create_flow_map(
            ["A8_1", Gap(CUT_LENGTH, "skipped_42_43"), "A8_2"], id="labelled"
        )
        assert fm.iter_gaps() == [(CUT_START, CUT_END, "skipped_42_43")]


class TestAutoSizedGap:
    """`Gap()` measures the hole its neighbours leave on the source axis."""

    def test_auto_gap_reads_the_hole_off_the_folded_source(
        self, score: ContinuousLogicalTimeline
    ) -> None:
        # Written against the folded score, whose spans already sit 6 QB apart,
        # an auto gap reproduces the source layout exactly.
        fm = score.create_flow_map(["A8_1", Gap(), "A8_2"], id="layout")
        assert sections(fm) == [
            ("A8_1", Fraction(0), CUT_START, Fraction(0), CUT_START),
            (None, CUT_START, CUT_START, CUT_START, CUT_END),
            ("A8_2", CUT_END, SCORE_LENGTH, CUT_END, SCORE_LENGTH),
        ]

    def test_auto_gap_is_the_identity_on_the_folded_axis(
        self, score: ContinuousLogicalTimeline
    ) -> None:
        fm = score.create_flow_map(["A8_1", Gap(), "A8_2"], id="layout")
        for coord in (Fraction(0), Fraction(50), CUT_END, Fraction(200)):
            assert fm.unfold_coordinate(coord) == [coord]

    def test_touching_spans_yield_a_zero_width_gap(self) -> None:
        fm = FlowMap([(0, 123), Gap(), (123, 228)], id="touching")
        assert fm.iter_gaps() == []
        assert fm.total_target_length == PERF_LENGTH

    def test_leading_auto_gap_has_nothing_to_measure(self) -> None:
        with pytest.raises(ValueError, match="at the start of the flow"):
            FlowMap([Gap(), (0, 123)])

    def test_trailing_auto_gap_has_nothing_to_measure(self) -> None:
        with pytest.raises(ValueError, match="at the end of the flow"):
            FlowMap([(0, 123), Gap()])

    def test_backwards_neighbours_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative hole"):
            FlowMap([(129, 234), Gap(), (0, 123)])


class TestFromDict:
    """`FlowMap.from_dict` pairs each target coordinate with the span there."""

    @pytest.fixture
    def via_dict(self, performance: ContinuousLogicalTimeline) -> FlowMap:
        return FlowMap.from_dict(
            {0: "A8_1", CUT_END: "A8_2"},
            id="restored_dict",
            resolve=performance.get_region,
        )

    def test_matches_the_other_constructions(
        self, via_dict: FlowMap, performance: ContinuousLogicalTimeline
    ) -> None:
        via_gaps = performance.create_flow_map(
            ["A8_1", Gap(CUT_LENGTH), "A8_2"], id="restored_gaps"
        )
        assert sections(via_dict) == sections(via_gaps)

    def test_second_span_lands_at_measure_44(self, via_dict: FlowMap) -> None:
        assert via_dict.unfold_coordinate(CUT_START) == [CUT_END]

    def test_gap_reported(self, via_dict: FlowMap) -> None:
        assert via_dict.iter_gaps() == [(CUT_START, CUT_END, None)]

    def test_keys_need_not_be_written_in_order(self) -> None:
        fm = FlowMap.from_dict({CUT_END: (CUT_START, PERF_LENGTH), 0: (0, CUT_START)})
        assert [s.target_start for s in fm._sections if not s.is_gap] == [
            Fraction(0),
            CUT_END,
        ]

    def test_a_value_may_hold_several_spans(self) -> None:
        # Two spans laid end to end from QB 129 on.
        fm = FlowMap.from_dict({0: (0, 60), CUT_END: [(60, 90), (90, 123)]})
        placed = [(s.target_start, s.target_end) for s in fm._sections if not s.is_gap]
        assert placed == [
            (Fraction(0), Fraction(60)),
            (CUT_END, Fraction(159)),
            (Fraction(159), Fraction(192)),
        ]

    def test_overlapping_placements_rejected(self) -> None:
        with pytest.raises(ValueError, match="overlaps the preceding span"):
            FlowMap.from_dict({0: (0, 123), 100: (123, 228)})

    def test_the_constructor_accepts_a_mapping_too(
        self, performance: ContinuousLogicalTimeline, via_dict: FlowMap
    ) -> None:
        # `create_flow_map` takes the same mapping, so it needs no separate path.
        fm = performance.create_flow_map({0: "A8_1", CUT_END: "A8_2"}, id="viadict")
        assert sections(fm) == sections(via_dict)

    def test_mapping_and_at_cannot_be_combined(self) -> None:
        with pytest.raises(ValueError, match="Cannot combine `at` with a mapping"):
            FlowMap({0: (0, 123)}, at=[0])


class TestAtPlacements:
    """`at` states each played span's target coordinate outright."""

    def test_none_entry_follows_its_predecessor(self) -> None:
        fm = FlowMap([(0, 123), (123, 228)], at=[0, None], id="follows")
        assert fm.iter_gaps() == []
        assert fm.total_target_length == PERF_LENGTH

    def test_leading_gap_is_expressible(self) -> None:
        # An anacrusis-style offset: nothing plays before QB 12.
        fm = FlowMap([(0, 123)], at=[12], id="offset")
        assert fm.iter_gaps() == [(Fraction(0), Fraction(12), None)]
        assert fm.total_target_length == Fraction(135)

    def test_cannot_combine_with_gap_entries(self) -> None:
        with pytest.raises(ValueError, match="Cannot combine"):
            FlowMap([(0, 123), Gap(6), (123, 228)], at=[0, 129])

    def test_length_must_match_played_span_count(self) -> None:
        with pytest.raises(ValueError, match="one target coordinate per played span"):
            FlowMap([(0, 123), (123, 228)], at=[0])

    def test_overlapping_placement_rejected(self) -> None:
        with pytest.raises(ValueError, match="overlaps the preceding span"):
            FlowMap([(0, 123), (123, 228)], at=[0, 100])


# endregion

# region Trailing gaps


class TestTargetLength:
    """A flow ending in a gap needs its target extent recorded."""

    def test_trailing_gap_needs_target_length(self) -> None:
        # A1 plays only the first 39 measures of the 78-measure score.
        fm = FlowMap([(0, 117)], id="A1", target_length=SCORE_LENGTH)
        assert fm.total_target_length == SCORE_LENGTH
        assert fm.iter_gaps() == [(Fraction(117), SCORE_LENGTH, None)]

    def test_without_it_the_flow_ends_at_its_last_section(self) -> None:
        fm = FlowMap([(0, 117)], id="A1")
        assert fm.total_target_length == Fraction(117)
        assert fm.iter_gaps() == []

    def test_timeline_records_its_own_length_as_source_length(
        self, score: ContinuousLogicalTimeline
    ) -> None:
        fm = score.get_flow_map("A8")
        assert fm.source_length == SCORE_LENGTH

    def test_inverse_swaps_the_recorded_extents(
        self, score: ContinuousLogicalTimeline
    ) -> None:
        fm = score.get_flow_map("A8")
        inverse = fm.inverse()
        assert inverse.target_length == SCORE_LENGTH
        assert inverse.source_length is None


# endregion

# region Folding across a hole


class TestFoldAcrossGaps:
    """A target coordinate inside a hole folds back to nothing."""

    @pytest.fixture
    def restored(self, performance: ContinuousLogicalTimeline) -> FlowMap:
        return performance.create_flow_map(
            ["A8_1", Gap(CUT_LENGTH), "A8_2"], id="restored"
        )

    def test_before_the_hole(self, restored: FlowMap) -> None:
        assert restored.fold(50) == Fraction(50)

    def test_after_the_hole(self, restored: FlowMap) -> None:
        # Score QB 150 is performance QB 144.
        assert restored.fold(150) == Fraction(144)

    def test_at_the_far_edge_of_the_hole(self, restored: FlowMap) -> None:
        assert restored.fold(CUT_END) == CUT_START

    def test_inside_the_hole_raises(self, restored: FlowMap) -> None:
        with pytest.raises(ValueError, match=r"falls in a gap of the flow"):
            restored.fold(125)

    def test_the_error_names_the_hole(self, restored: FlowMap) -> None:
        with pytest.raises(ValueError, match=r"\[123, 129\)"):
            restored.fold(125)

    def test_beyond_the_end_still_says_so(self, restored: FlowMap) -> None:
        with pytest.raises(ValueError, match="beyond the end of the flow"):
            restored.fold(SCORE_LENGTH)

    def test_implied_hole_folds_to_nothing_too(
        self, score: ContinuousLogicalTimeline
    ) -> None:
        # The inverse records no gap section; its hole is the space between two
        # placed spans, and must fold the same way.
        inverse = score.get_flow_map("A8").inverse()
        with pytest.raises(ValueError, match=r"falls in a gap"):
            inverse.fold(125)

    def test_nothing_unfolds_into_the_hole(self, restored: FlowMap) -> None:
        # No performance coordinate maps into 123-129; both neighbours skip it.
        assert restored.unfold_coordinate(CUT_START - 1) == [CUT_START - 1]
        assert restored.unfold_coordinate(CUT_START) == [CUT_END]


# endregion

# region apply_flow places its children


class TestApplyFlowPlacesChildren:
    """`apply_flow` puts each slice at its target coordinate."""

    @pytest.fixture
    def restored(
        self, performance: ContinuousLogicalTimeline
    ) -> ContinuousLogicalTimeline:
        performance.create_flow_map(["A8_1", Gap(CUT_LENGTH), "A8_2"], id="restored")
        return performance.apply_flow("restored")

    def test_length_spans_the_hole(self, restored: ContinuousLogicalTimeline) -> None:
        assert restored.length.value == SCORE_LENGTH

    def test_children_are_the_played_spans_only(
        self, restored: ContinuousLogicalTimeline
    ) -> None:
        assert restored.list_children() == ["A8_1", "A8_2"]

    def test_second_child_sits_at_measure_44(
        self, restored: ContinuousLogicalTimeline
    ) -> None:
        assert restored.get_child_offset("A8_1").value == Fraction(0)
        assert restored.get_child_offset("A8_2").value == CUT_END

    def test_regions_match_the_children(
        self, restored: ContinuousLogicalTimeline
    ) -> None:
        assert [r.name for r in restored.iter_regions()] == ["A8_1", "A8_2"]
        r2 = restored.get_region("region_2")
        assert (r2.start.value, r2.end.value) == (CUT_END, SCORE_LENGTH)

    def test_fill_gaps_closes_the_hole_with_an_empty_segment(
        self, performance: ContinuousLogicalTimeline
    ) -> None:
        performance.create_flow_map(["A8_1", Gap(CUT_LENGTH), "A8_2"], id="restored")
        result = performance.apply_flow("restored", fill_gaps=True)
        # The hole becomes a child in its own right, so the result tiles its
        # axis contiguously, in target order.
        assert result.list_children() == ["A8_1", "gap_1", "A8_2"]
        filler = result.get_child("gap_1")
        assert filler.length.value == CUT_LENGTH
        assert filler.n_events == 0
        assert result.get_child_offset("gap_1").value == CUT_START
        gap = result.get_region("region_2")
        assert gap.name == "gap_1"
        assert (gap.start.value, gap.end.value) == (CUT_START, CUT_END)

    def test_fill_gaps_uses_the_gap_label(
        self, performance: ContinuousLogicalTimeline
    ) -> None:
        performance.create_flow_map(
            ["A8_1", Gap(CUT_LENGTH, "skipped_42_43"), "A8_2"], id="restored"
        )
        result = performance.apply_flow("restored", fill_gaps=True)
        assert [r.name for r in result.iter_regions()] == [
            "A8_1",
            "skipped_42_43",
            "A8_2",
        ]

    def test_gaps_unmarked_by_default(
        self, restored: ContinuousLogicalTimeline
    ) -> None:
        assert [r.name for r in restored.iter_regions()] == ["A8_1", "A8_2"]

    def test_segment_line_needs_its_gaps_filled(self) -> None:
        # A SegmentLine admits no space between its segments, so a placing flow
        # can only be unfolded into one when the holes are closed.
        sl = SegmentLine[ContinuousLogicalTimeline](unit="q", name="sl")
        for i in range(4):
            sl.append_segment(ContinuousLogicalTimeline(length=10, unit="q"), f"s{i}")
        sl.create_region(0, 10, id="first", name="first")
        sl.create_region(30, 40, id="last", name="last")
        sl.create_flow_map({0: "first", 20: "last"}, id="placed")

        with pytest.raises(ValueError, match="fill_gaps=True"):
            sl.apply_flow("placed")

        result = sl.apply_flow("placed", fill_gaps=True)
        assert isinstance(result, SegmentLine)
        assert result.list_children() == ["first", "gap_1", "last"]
        # Contiguous, in order, and the empty segment holds the middle.
        assert [
            result.get_child_offset(name).value for name in result.list_children()
        ] == [Fraction(0), Fraction(10), Fraction(20)]
        assert result.length.value == Fraction(30)

    def test_trailing_gap_is_filled_too(self, score: ContinuousLogicalTimeline) -> None:
        # A1 plays only the opening; inverting it leaves the rest of the score
        # empty, and filling closes that trailing stretch with a segment.
        score.create_flow_map(["A8_1"], id="A1")
        restored = score.apply_flow("A1").apply_flow("source", fill_gaps=True)
        assert restored.length.value == SCORE_LENGTH
        assert restored.list_children() == ["A8_1", "gap_1"]
        assert restored.get_child("gap_1").length.value == SCORE_LENGTH - CUT_START

    def test_concatenating_flow_is_unchanged(
        self, score: ContinuousLogicalTimeline
    ) -> None:
        # No placement given: the spans still stack end to end, exactly as
        # before this mechanism existed.
        result = score.apply_flow("A8")
        assert result.length.value == PERF_LENGTH
        assert result.get_child_offset("A8_2").value == CUT_START


# endregion

# region The round trip: unfold, then invert


class TestInverseRestoresTheSource:
    """Applying the reverse FlowMap rebuilds a timeline laid out like the source.

    This is the payoff: unfolding A8 drops measures 42 and 43 and yields a
    228 QB performance. Applying that result's ``"source"`` map — which
    ``apply_flow`` attaches automatically — puts the two spans back at QB 0 and
    QB 129, restoring the 234 QB layout with a 6 QB hole in the middle. No gap
    or placement has to be stated: the inverse carries it.
    """

    @pytest.fixture
    def restored(self, score: ContinuousLogicalTimeline) -> ContinuousLogicalTimeline:
        return score.apply_flow("A8").apply_flow("source")

    def test_original_length_recovered(
        self, restored: ContinuousLogicalTimeline
    ) -> None:
        assert restored.length.value == SCORE_LENGTH

    def test_spans_return_to_their_score_coordinates(
        self, restored: ContinuousLogicalTimeline
    ) -> None:
        assert restored.get_child_offset("A8_1").value == Fraction(0)
        assert restored.get_child_offset("A8_2").value == CUT_END

    def test_regions_agree_with_the_children(
        self, restored: ContinuousLogicalTimeline
    ) -> None:
        # Before placement was honoured, the Regions said 129 while the
        # children sat at 123 — the result contradicted itself.
        for region in restored.iter_regions():
            assert region.start.value == restored.get_child_offset(region.name).value

    def test_events_land_back_on_their_score_coordinates(
        self, restored: ContinuousLogicalTimeline
    ) -> None:
        events = {
            e["id"]: struct_to_rational(e["start"])
            for e in restored.get_events(include_children=True)
        }
        # m42 was cut and stays gone; m1 and m44 return to 0 and 129.
        assert events == {"m1": Fraction(0), "m44": CUT_END}

    def test_the_hole_is_two_measures_wide(
        self, score: ContinuousLogicalTimeline
    ) -> None:
        inverse = score.apply_flow("A8").get_flow_map("source")
        assert inverse.iter_gaps() == [(CUT_START, CUT_END, None)]

    def test_inverse_of_inverse_returns_the_forward_map(
        self, score: ContinuousLogicalTimeline
    ) -> None:
        forward = score.get_flow_map("A8")
        assert sections(forward.inverse().inverse()) == sections(forward)

    def test_a_gap_inverts_to_dropped_material(
        self, performance: ContinuousLogicalTimeline
    ) -> None:
        # Inverting a restoration turns its hole back into a cut: the gap
        # section, which had target extent and no source extent, becomes an
        # elision with source extent and no target extent.
        restoring = performance.create_flow_map(
            ["A8_1", Gap(CUT_LENGTH), "A8_2"], id="restored"
        )
        cutting = restoring.inverse()
        elisions = [s for s in cutting._sections if s.is_elision]
        assert len(elisions) == 1
        assert (elisions[0].source_start, elisions[0].source_end) == (
            CUT_START,
            CUT_END,
        )
        # The elided stretch is played nowhere, so it unfolds to nothing.
        assert cutting.unfold_coordinate(125) == []
        assert cutting.iter_gaps() == []


# endregion
