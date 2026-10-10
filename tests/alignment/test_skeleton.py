"""End-to-end validation for the ``TimeSkeleton`` shared temporal structure.

These tests implement the ``README.md`` *TimeSkeleton Validation* contract for
skeleton-from-load (portable and real-specimen lanes), authored flows and gaps,
reference-timeline materialization, and participant membership. Every anchor is
an exact value derived from an actual measure map, never a range.

Claim-coordinate recording (``create_match_claim`` coordinate units) is owned by
a parallel regression suite and is intentionally out of scope here.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from timetoalign.alignment import TimeSkeleton
from timetoalign.core import Coordinate, Gap, Measure, NumberType, Tempo, TimeUnit
from timetoalign.loader.score.ms3 import Ms3Loader
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    MetricHierarchy,
    SectionHierarchy,
    TempoEntry,
    TempoMap,
)

# The published Satie reference score (203 bars of 3/4, section closers on the
# ``breaks`` column of mc 78 and mc 143).
SPECIMEN = Path(
    "/home/laser/git/tta/dashboard/specimens/IEEE1599/"
    "SatiePetriNets/notebooks/trois-gymnopedies.measures.tsv"
)

# The portable miniature: 7 bars of 3/4, section closers on mc 3 and mc 5,
# cutting the score into 3 / 2 / 2 measures (9 / 6 / 6 quarters, 21 total).
_MINI_COLUMNS = (
    "mc",
    "mn",
    "quarterbeats",
    "duration_qb",
    "keysig",
    "timesig",
    "act_dur",
    "mc_offset",
    "numbering_offset",
    "dont_count",
    "barline",
    "breaks",
    "repeats",
    "next",
)


def _write_mini_measures_tsv(directory: Path) -> Path:
    """Write a 7-bar 3/4 measures TSV with section closers on mc 3 and mc 5."""
    path = directory / "mini.measures.tsv"
    lines = ["\t".join(_MINI_COLUMNS)]
    for mc in range(1, 8):
        breaks = "section" if mc in (3, 5) else ""
        repeats = "firstMeasure" if mc == 1 else ("lastMeasure" if mc == 7 else "")
        successor = mc + 1 if mc < 7 else -1
        row = {
            "mc": mc,
            "mn": mc,
            "quarterbeats": (mc - 1) * 3,
            "duration_qb": "3.0",
            "keysig": 0,
            "timesig": "3/4",
            "act_dur": "3/4",
            "mc_offset": 0,
            "numbering_offset": 0,
            "dont_count": "False",
            "barline": "",
            "breaks": breaks,
            "repeats": repeats,
            "next": successor,
        }
        lines.append("\t".join(str(row[column]) for column in _MINI_COLUMNS))
    path.write_text("\n".join(lines) + "\n")
    return path


def _concrete_hierarchy(*counts: int) -> SectionHierarchy:
    """A concrete section hierarchy of 3/4 bars, ``counts`` measures per section."""

    def bars(count: int) -> list[Measure]:
        return [
            Measure(actual_length=Fraction(3), time_signature="3/4")
            for _ in range(count)
        ]

    return SectionHierarchy.from_measures([bars(count) for count in counts])


@pytest.fixture
def mini_timeline(tmp_path: Path) -> ContinuousLogicalTimeline:
    """A score timeline auto-attached to a skeleton parsed from the miniature."""
    path = _write_mini_measures_tsv(tmp_path)
    return Ms3Loader.from_file(path).create_timeline()


@pytest.fixture
def mini_skeleton(mini_timeline: ContinuousLogicalTimeline) -> TimeSkeleton:
    """The single skeleton auto-attached to the miniature score timeline."""
    return mini_timeline.skeleton


@pytest.fixture
def specimen_timeline() -> ContinuousLogicalTimeline:
    """A score timeline auto-attached to the real Satie skeleton (skip-guarded)."""
    if not SPECIMEN.exists():
        pytest.skip(f"Specimen not found: {SPECIMEN}")
    return Ms3Loader.from_file(SPECIMEN).create_timeline()


# region (d) Skeleton-from-load: portable lane


class TestSkeletonFromMiniatureLoad:
    """Loading the miniature auto-attaches a 3 / 2 / 2 skeleton."""

    def test_default_load_attaches_one_participant(
        self, mini_skeleton: TimeSkeleton
    ) -> None:
        assert mini_skeleton.section_hierarchy.n_sections == 3
        assert mini_skeleton.section_hierarchy.n_measures == 7
        assert mini_skeleton.n_participants == 1

    def test_source_flow_is_always_present_exactly_once(
        self, mini_skeleton: TimeSkeleton
    ) -> None:
        assert list(mini_skeleton.flows).count("source") == 1

    def test_section_measure_counts(self, mini_skeleton: TimeSkeleton) -> None:
        assert [
            section.n_measures for section in mini_skeleton.section_hierarchy.sections
        ] == [3, 2, 2]

    def test_section_quarter_spans(self, mini_skeleton: TimeSkeleton) -> None:
        spans = [
            section.measure_map.total_actual_length
            for section in mini_skeleton.section_hierarchy.sections
        ]
        assert spans == [Fraction(9), Fraction(6), Fraction(6)]

    def test_total_quarters(
        self,
        mini_skeleton: TimeSkeleton,
        mini_timeline: ContinuousLogicalTimeline,
    ) -> None:
        assert mini_timeline.length.value == Fraction(21)
        assert mini_skeleton.materialize(flow="source").length.value == Fraction(21)

    def test_flatten_opt_out_yields_no_attachment(self, tmp_path: Path) -> None:
        path = _write_mini_measures_tsv(tmp_path)
        timeline = Ms3Loader.from_file(path).create_timeline(flatten=True)
        assert timeline.skeletons == ()
        with pytest.raises(ValueError):
            _ = timeline.skeleton


# endregion


# region (e) Skeleton-from-load: real-specimen lane


class TestSkeletonFromSpecimenLoad:
    """Loading the real Satie score reproduces the published 78 / 65 / 60 shape."""

    def test_section_and_measure_counts(
        self, specimen_timeline: ContinuousLogicalTimeline
    ) -> None:
        skeleton = specimen_timeline.skeleton
        assert skeleton.section_hierarchy.n_sections == 3
        assert skeleton.section_hierarchy.n_measures == 203
        assert [
            section.n_measures for section in skeleton.section_hierarchy.sections
        ] == [78, 65, 60]

    def test_section_quarter_spans(
        self, specimen_timeline: ContinuousLogicalTimeline
    ) -> None:
        skeleton = specimen_timeline.skeleton
        spans = [
            section.measure_map.total_actual_length
            for section in skeleton.section_hierarchy.sections
        ]
        assert spans == [Fraction(234), Fraction(195), Fraction(180)]

    def test_total_quarters_and_participant(
        self, specimen_timeline: ContinuousLogicalTimeline
    ) -> None:
        skeleton = specimen_timeline.skeleton
        assert skeleton.materialize(flow="source").length.value == Fraction(609)
        assert skeleton.n_participants == 1
        assert list(skeleton.flows).count("source") == 1


# endregion


# region (f) Authored flows and gaps


class TestAuthoredFlows:
    """Measure-range flows sum their traversed measures and quarters exactly."""

    def test_elision_flow_on_specimen(
        self, specimen_timeline: ContinuousLogicalTimeline
    ) -> None:
        skeleton = specimen_timeline.skeleton
        skeleton.add_flow(["1-41", "44-78"], id="elision")
        # 41 + 35 == 76 measures; 76 * 3 == 228 quarters.
        assert skeleton.materialize(flow="elision").length.value == Fraction(228)
        sections = skeleton.flows["elision"].sections
        assert [(s.mc_start, s.mc_end) for s in sections] == [(1, 42), (44, 79)]

    def test_elision_flow_on_miniature(self, mini_skeleton: TimeSkeleton) -> None:
        # The miniature analogue skips m4 and m5: 3 + 2 == 5 bars == 15 quarters.
        mini_skeleton.add_flow(["1-3", "6-7"], id="elision")
        assert mini_skeleton.materialize(flow="elision").length.value == Fraction(15)

    def test_elision_does_not_merge_ranges(self, mini_skeleton: TimeSkeleton) -> None:
        mini_skeleton.add_flow(["1-3", "6-7"], id="elision")
        sections = mini_skeleton.flows["elision"].sections
        assert len(sections) == 2
        assert [(s.mc_start, s.mc_end) for s in sections] == [(1, 4), (6, 8)]
        assert [s.atomic_section_ids for s in sections] == [
            ("1-3",),
            ("6-7",),
        ]

    def test_step_order_and_gap_placement(self, mini_skeleton: TimeSkeleton) -> None:
        gap = Gap(6)
        mini_skeleton.add_flow(["1-3", gap, "6-7"], id="withgap")
        # The ordered step store keeps played ranges and the Gap marker in
        # authored order; a Gap owns no measure-id span, so it mints no section.
        assert mini_skeleton._flow_steps["withgap"] == ("1-3", gap, "6-7")
        sections = mini_skeleton.flows["withgap"].sections
        assert [(s.mc_start, s.mc_end) for s in sections] == [(1, 4), (6, 8)]

    def test_gap_spelling_stores_label_not_duration(self) -> None:
        gap = Gap("intro_extension")
        assert gap.label == "intro_extension"
        assert gap.duration is None
        assert gap.is_auto is True

    def test_gap_with_duration_contributes_its_length(
        self, mini_skeleton: TimeSkeleton
    ) -> None:
        mini_skeleton.add_flow(["1-3", Gap(6), "6-7"], id="withgap")
        # 15 played quarters plus the explicit 6-quarter gap.
        assert mini_skeleton.materialize(flow="withgap").length.value == Fraction(21)

    def test_duration_less_gap_contributes_nothing(
        self, mini_skeleton: TimeSkeleton
    ) -> None:
        mini_skeleton.add_flow(["1-3", Gap(), "6-7"], id="autogap")
        # A duration-less gap adds no quarters: only the 15 played remain.
        assert mini_skeleton.materialize(flow="autogap").length.value == Fraction(15)

    def test_duplicate_flow_id_raises(self, mini_skeleton: TimeSkeleton) -> None:
        mini_skeleton.add_flow(["1-3"], id="dup")
        with pytest.raises(ValueError, match="already exists"):
            mini_skeleton.add_flow(["4-5"], id="dup")

    def test_unknown_section_id_raises(self, mini_skeleton: TimeSkeleton) -> None:
        with pytest.raises(KeyError):
            mini_skeleton.add_flow(["sec9"], id="bad")

    def test_unknown_measure_id_raises(self, mini_skeleton: TimeSkeleton) -> None:
        with pytest.raises(KeyError):
            mini_skeleton.add_flow(["99"], id="bad")

    def test_backwards_range_raises(self, mini_skeleton: TimeSkeleton) -> None:
        with pytest.raises(ValueError, match="runs backwards"):
            mini_skeleton.add_flow(["5-2"], id="bad")


# endregion


# region (g) Reference-timeline materialization


class TestMaterialize:
    """Materialize yields a quarters/Fraction axis whose length is the traversal."""

    def test_flow_axis_is_logical_quarters_fraction(
        self, mini_skeleton: TimeSkeleton
    ) -> None:
        reference = mini_skeleton.materialize(flow="source")
        assert isinstance(reference, ContinuousLogicalTimeline)
        assert reference.unit == TimeUnit.quarters
        assert reference.number_type == NumberType.fraction

    def test_suite_reorder_length_on_miniature(
        self, mini_skeleton: TimeSkeleton
    ) -> None:
        # Reordered third, first, second: 6 + 9 + 6 == 21 quarters.
        mini_skeleton.add_flow(["sec3", "sec1", "sec2"], id="suite")
        assert mini_skeleton.materialize(flow="suite").length.value == Fraction(21)

    def test_suite_reorder_length_on_specimen(
        self, specimen_timeline: ContinuousLogicalTimeline
    ) -> None:
        skeleton = specimen_timeline.skeleton
        # Reordered third, first, second: 180 + 234 + 195 == 609 quarters.
        skeleton.add_flow(["sec3", "sec1", "sec2"], id="suite")
        assert skeleton.materialize(flow="suite").length.value == Fraction(609)

    def test_repeated_materialize_returns_cached_identity(
        self, mini_skeleton: TimeSkeleton
    ) -> None:
        first = mini_skeleton.materialize(flow="source")
        second = mini_skeleton.materialize(flow="source")
        assert first is second

    def test_no_flow_on_concrete_structure_raises(
        self, mini_skeleton: TimeSkeleton
    ) -> None:
        with pytest.raises(ValueError, match="Concrete structures"):
            mini_skeleton.materialize()

    def test_no_flow_on_abstract_structure_is_floating(self) -> None:
        skeleton = TimeSkeleton(SectionHierarchy.from_measure_counts([3, 2, 2]))
        reference = skeleton.materialize()
        # An all-abstract structure floats: length is measure count plus one.
        assert reference.unit == TimeUnit.floating_measures
        assert reference.number_type == NumberType.float
        assert reference.length.value == 8.0


# endregion


# region (h) Membership: attach, detach, and the single-attachment accessor


class TestMembership:
    """Participants attach to a flow; detach and the ``.skeleton`` accessor."""

    def test_attach_range_mints_one_step_flow(
        self, mini_skeleton: TimeSkeleton
    ) -> None:
        recording = ContinuousPhysicalTimeline(length=30.0, unit="seconds", uid="rec1")
        mini_skeleton.attach(recording, flow=("1", "5"))
        assert mini_skeleton.n_participants == 2
        assert len(mini_skeleton.flows) == 2
        minted = mini_skeleton.flows["1-5"]
        assert len(minted.sections) == 1
        assert minted.sections[0].mc_range == (1, 6)
        assert minted.sections[0].atomic_section_ids == ("1-5",)

    def test_same_range_attach_reuses_the_content_derived_flow(
        self, mini_skeleton: TimeSkeleton
    ) -> None:
        first = ContinuousPhysicalTimeline(length=30.0, unit="seconds", uid="rec1")
        second = ContinuousPhysicalTimeline(length=25.0, unit="seconds", uid="rec2")
        mini_skeleton.attach(first, flow=("1", "5"))
        mini_skeleton.attach(second, flow=("1", "5"))
        assert mini_skeleton.n_participants == 3
        assert list(mini_skeleton.flows) == ["source", "1-5"]

    def test_range_attach_refuses_a_differing_flow_under_its_id(
        self, mini_skeleton: TimeSkeleton
    ) -> None:
        mini_skeleton.add_flow(["1-3", "6-7"], id="1-5")
        recording = ContinuousPhysicalTimeline(length=30.0, unit="seconds", uid="rec1")
        with pytest.raises(ValueError, match="'1-5' already exists with steps"):
            mini_skeleton.attach(recording, flow=("1", "5"))

    def test_detach_restores_participant_count_and_structure(
        self, mini_skeleton: TimeSkeleton
    ) -> None:
        recording = ContinuousPhysicalTimeline(length=30.0, unit="seconds", uid="rec1")
        mini_skeleton.attach(recording, flow=("1", "5"))
        mini_skeleton.detach(recording)
        assert mini_skeleton.n_participants == 1
        assert recording not in mini_skeleton.participants
        assert mini_skeleton.section_hierarchy.n_sections == 3
        assert mini_skeleton.section_hierarchy.n_measures == 7
        spans = [
            section.measure_map.total_actual_length
            for section in mini_skeleton.section_hierarchy.sections
        ]
        assert spans == [Fraction(9), Fraction(6), Fraction(6)]

    @pytest.mark.xfail(
        strict=True,
        reason="implementation gap: detach leaves the emptied minted flow "
        "in the flow set instead of removing it",
    )
    def test_detach_removes_the_minted_flow(self, mini_skeleton: TimeSkeleton) -> None:
        recording = ContinuousPhysicalTimeline(length=30.0, unit="seconds", uid="rec1")
        mini_skeleton.attach(recording, flow=("1", "5"))
        mini_skeleton.detach(recording)
        assert list(mini_skeleton.flows) == ["source"]

    def test_timeline_side_attach_is_refused(self, mini_skeleton: TimeSkeleton) -> None:
        recording = ContinuousPhysicalTimeline(length=30.0, unit="seconds", uid="rec1")
        with pytest.raises(NotImplementedError):
            recording.attach(mini_skeleton)

    def test_skeleton_accessor_returns_the_single_attachment(
        self,
        mini_skeleton: TimeSkeleton,
        mini_timeline: ContinuousLogicalTimeline,
    ) -> None:
        assert mini_timeline.skeleton is mini_skeleton

    def test_skeleton_accessor_raises_on_zero(self) -> None:
        loose = ContinuousPhysicalTimeline(length=10.0, unit="seconds", uid="rec2")
        assert loose.skeletons == ()
        with pytest.raises(ValueError, match="skeleton attachments"):
            _ = loose.skeleton

    def test_skeleton_accessor_raises_on_two(self) -> None:
        recording = ContinuousPhysicalTimeline(length=10.0, unit="seconds", uid="rec3")
        first = TimeSkeleton(_concrete_hierarchy(3))
        second = TimeSkeleton(_concrete_hierarchy(3))
        first.attach(recording)
        second.attach(recording)
        assert len(recording.skeletons) == 2
        with pytest.raises(ValueError, match="skeleton attachments"):
            _ = recording.skeleton


# endregion


# region (i) Structural equality and identity-based attachment


class TestStructuralEquality:
    """Equality is the authored structure; identity and usage are excluded."""

    def test_equal_hierarchies_compare_equal_despite_distinct_uids(self) -> None:
        first = TimeSkeleton(_concrete_hierarchy(3, 2, 2))
        second = TimeSkeleton(_concrete_hierarchy(3, 2, 2))
        assert first.id != second.id
        assert first == second

    def test_differing_section_hierarchy_breaks_equality(self) -> None:
        assert TimeSkeleton(_concrete_hierarchy(3, 2, 2)) != TimeSkeleton(
            _concrete_hierarchy(4, 2, 2)
        )

    def test_non_skeleton_operand_is_not_equal(self) -> None:
        skeleton = TimeSkeleton(_concrete_hierarchy(3))
        assert skeleton.__eq__("skeleton") is NotImplemented
        assert (skeleton == "skeleton") is False

    def test_instances_are_unhashable(self) -> None:
        with pytest.raises(TypeError, match="unhashable"):
            hash(TimeSkeleton(_concrete_hierarchy(3)))

    def test_differing_tempo_maps_do_not_break_structural_equality(self) -> None:
        slow = Tempo(bpm=Fraction(60))
        fast = Tempo(bpm=Fraction(120))
        section = _concrete_hierarchy(3, 2, 2)

        def hierarchy(tempo: Tempo) -> MetricHierarchy:
            return MetricHierarchy(
                measure_map=section.measure_map,
                tempo_maps=(
                    TempoMap(
                        kind="tempo_indication",
                        entries=(
                            TempoEntry(
                                at=Coordinate(Fraction(0), "quarters"),
                                tempo=tempo,
                            ),
                        ),
                        provenance={},
                    ),
                ),
            )

        first = TimeSkeleton(section, hierarchy(slow))
        second = TimeSkeleton(section, hierarchy(fast))
        third = TimeSkeleton(section, hierarchy(slow))
        assert first == second
        assert first == third

    def test_authored_flows_are_compared_by_content(self) -> None:
        first = TimeSkeleton(_concrete_hierarchy(3, 2, 2))
        second = TimeSkeleton(_concrete_hierarchy(3, 2, 2))
        first.add_flow(["1-3", "6-7"], id="elision")
        assert first != second
        second.add_flow(["1-3", "6-7"], id="elision")
        assert first == second

    def test_participants_do_not_affect_equality(self) -> None:
        first = TimeSkeleton(_concrete_hierarchy(3, 2, 2))
        second = TimeSkeleton(_concrete_hierarchy(3, 2, 2))
        recording = ContinuousPhysicalTimeline(
            length=30.0, unit="seconds", uid="eq-rec"
        )
        first.attach(recording)
        assert first.n_participants == 1
        assert second.n_participants == 0
        assert first == second

    def test_same_range_attach_on_two_skeletons_preserves_equality(self) -> None:
        first = TimeSkeleton(_concrete_hierarchy(3, 2, 2))
        second = TimeSkeleton(_concrete_hierarchy(3, 2, 2))
        rec_a = ContinuousPhysicalTimeline(
            length=30.0, unit="seconds", uid="range-rec-a"
        )
        rec_b = ContinuousPhysicalTimeline(
            length=25.0, unit="seconds", uid="range-rec-b"
        )
        # The minted flow id is the range step itself, never the participant
        # id, so identical ranges keep the flow stores identical.
        first.attach(rec_a, flow=("1", "5"))
        second.attach(rec_b, flow=("1", "5"))
        assert first == second


class TestAttachmentIdentity:
    """Attachment bookkeeping distinguishes equal skeletons by identity."""

    def test_two_equal_skeletons_attach_without_aliasing(self) -> None:
        recording = ContinuousPhysicalTimeline(
            length=10.0, unit="seconds", uid="id-rec1"
        )
        first = TimeSkeleton(_concrete_hierarchy(3))
        second = TimeSkeleton(_concrete_hierarchy(3))
        assert first == second
        first.attach(recording)
        second.attach(recording)
        assert len(recording.skeletons) == 2
        assert recording.skeletons[0] is first
        assert recording.skeletons[1] is second

    def test_detach_removes_exactly_the_detached_skeleton(self) -> None:
        recording = ContinuousPhysicalTimeline(
            length=10.0, unit="seconds", uid="id-rec2"
        )
        first = TimeSkeleton(_concrete_hierarchy(3))
        second = TimeSkeleton(_concrete_hierarchy(3))
        first.attach(recording)
        second.attach(recording)
        first.detach(recording)
        assert len(recording.skeletons) == 1
        assert recording.skeletons[0] is second


# endregion


# region (j) Materialization round trip


class TestMaterializeRoundTrip:
    """A reference timeline harvests back its originating skeleton."""

    def test_abstract_reference_round_trips_to_the_same_skeleton(self) -> None:
        skeleton = TimeSkeleton(SectionHierarchy.from_measure_counts([3, 2, 2]))
        reference = skeleton.materialize()
        assert reference.create_skeleton() is skeleton
        assert reference.create_skeleton() == skeleton
        assert skeleton.materialize() is reference

    def test_flow_reference_round_trips_to_the_same_skeleton(
        self, mini_skeleton: TimeSkeleton
    ) -> None:
        reference = mini_skeleton.materialize(flow="source")
        assert reference.create_skeleton() is mini_skeleton
        assert reference.create_skeleton() == mini_skeleton
        assert mini_skeleton.materialize(flow="source") is reference

    def test_flow_reference_is_not_a_participant(
        self, mini_skeleton: TimeSkeleton
    ) -> None:
        reference = mini_skeleton.materialize(flow="source")
        assert mini_skeleton.n_participants == 1
        assert all(
            participant is not reference for participant in mini_skeleton.participants
        )

    def test_abstract_reference_is_not_a_participant(self) -> None:
        skeleton = TimeSkeleton(SectionHierarchy.from_measure_counts([3, 2, 2]))
        skeleton.materialize()
        assert skeleton.n_participants == 0


# endregion


def test_reordered_suite_claims_record_exact_reference_onsets(
    mini_skeleton: TimeSkeleton,
) -> None:
    from timetoalign.core import Coordinate, MeasureId

    mini_skeleton.add_flow(["sec3", "sec1", "sec2"], id="claim-suite")
    recording = ContinuousPhysicalTimeline(
        length=30.0,
        unit=TimeUnit.seconds,
        uid="claim-suite-recording",
    )
    mini_skeleton.attach(recording, flow="claim-suite")

    claims = [
        mini_skeleton.create_match_claim(recording.id, at=at, coordinate=0.0)
        for at in (
            MeasureId("6"),
            MeasureId("1"),
            MeasureId("4"),
            MeasureId("5"),
        )
    ]

    assert [claim.start_anchor.coordinate_b for claim in claims] == [
        Coordinate(Fraction(0), TimeUnit.quarters),
        Coordinate(Fraction(6), TimeUnit.quarters),
        Coordinate(Fraction(15), TimeUnit.quarters),
        Coordinate(Fraction(18), TimeUnit.quarters),
    ]
