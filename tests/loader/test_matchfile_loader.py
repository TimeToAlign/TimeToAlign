"""Tests for MatchfileLoader — Vienna Match (.match) alignment files.

This module tests the MatchfileLoader against the Vienna 1x22 dataset
(Chopin Op. 10 No. 3, 22 performances). It verifies:

- Header parsing and raw file format handling
- Score timeline construction (quarter-beat coordinates, C-Maps)
- Performance timeline construction (tick coordinates, C-Maps)
- MatchClaim generation (matched notes and deletion NOMATCH sentinels)
- AlignmentBundle assembly (single-file and multi-file)

All counts are exact per the Zero Tolerance Validation Policy.
See tests/data/vienna_1x22/README.md for gold standard values.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from timetoalign.alignment.bundle import AlignmentBundle, _reset_bundle_ids
from timetoalign.alignment.claims import _reset_anchor_ids, _reset_claim_ids
from timetoalign.core import TimeUnit
from timetoalign.loader.alignment.matchfile import MatchfileLoader
from timetoalign.maps.linear import ScalarMap, ShiftMap
from timetoalign.testdata import ensure_data
from timetoalign.timelines.groups import _reset_group_ids
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    DiscreteLogicalTimeline,
)

ensure_data("vienna_1x22", "score")

# region Test Data

VIENNA_DATA_DIR = Path(__file__).parent.parent / "data" / "vienna_1x22"
P01_MATCH = VIENNA_DATA_DIR / "Chopin_op10_no3_p01.match"
ALL_MATCH_FILES = sorted(VIENNA_DATA_DIR.glob("*.match"))

# Gold standard counts (from README.md)
SNOTE_COUNT = 454  # snote records per .match file (all 22 identical)
P01_DELETION_COUNT = 3  # deletions in p01 (n356, n359, n454)
P01_MATCH_COUNT = 451  # matched notes in p01 (454 - 3)
P01_PERF_NOTE_COUNT = 451  # performance notes in p01
MIDI_CLOCK_UNITS = 480  # ticks per quarter
MIDI_CLOCK_RATE = 500000  # microseconds per quarter (120 BPM)
TOTAL_MATCH_FILES = 22  # number of performances

# Per-performer deletion counts (from README.md gold standard table).
# Keyed by performer stem (e.g. "p01") → exact deletion count.
DELETION_COUNTS: dict[str, int] = {
    "p01": 3,
    "p02": 6,
    "p03": 2,
    "p04": 4,
    "p05": 4,
    "p06": 3,
    "p07": 3,
    "p08": 20,
    "p09": 18,
    "p10": 7,
    "p11": 4,
    "p12": 1,
    "p13": 2,
    "p14": 4,
    "p15": 5,
    "p16": 6,
    "p17": 3,
    "p18": 4,
    "p19": 2,
    "p20": 7,
    "p21": 3,
    "p22": 2,
}

# Derived totals
TOTAL_DELETIONS = sum(DELETION_COUNTS.values())  # 113
TOTAL_MATCHED = TOTAL_MATCH_FILES * SNOTE_COUNT - TOTAL_DELETIONS  # 9875

# endregion


# region Fixtures


@pytest.fixture(autouse=True)
def reset_ids():
    """Reset all ID generators before each test for isolation."""
    _reset_group_ids()
    _reset_anchor_ids()
    _reset_claim_ids()
    _reset_bundle_ids()


@pytest.fixture
def p01_loader() -> MatchfileLoader:
    """MatchfileLoader with p01 loaded."""
    loader = MatchfileLoader()
    loader.load(P01_MATCH)
    return loader


@pytest.fixture(scope="module")
def all_loader() -> MatchfileLoader:
    """MatchfileLoader with all 22 files loaded."""
    loader = MatchfileLoader()
    loader.load(*ALL_MATCH_FILES)
    return loader


# endregion


# region TestMatchfileFormat


class TestMatchfileFormat:
    """Unit tests for raw file parsing, independent of TTA domain objects."""

    def test_header_parsing(self):
        """Header fields are parsed correctly from p01."""
        header = MatchfileLoader._parse_header(P01_MATCH)
        assert header["midiClockUnits"] == MIDI_CLOCK_UNITS
        assert header["midiClockRate"] == MIDI_CLOCK_RATE
        assert header["piece"] == "Chopin_op10_no3"
        assert "scoreFileName" in header
        assert "midiFileName" in header

    def test_header_version(self):
        """Match file version is extracted."""
        header = MatchfileLoader._parse_header(P01_MATCH)
        assert header.get("matchFileVersion") == "1.0.0"

    def test_file_not_found_raises(self):
        """FileNotFoundError raised for nonexistent file."""
        loader = MatchfileLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(VIENNA_DATA_DIR / "nonexistent.match")

    def test_wrong_extension_raises(self):
        """ValueError raised for non-.match file."""
        loader = MatchfileLoader()
        musicxml = VIENNA_DATA_DIR / "Chopin_op10_no3.musicxml"
        with pytest.raises(ValueError, match="Not a .match file"):
            loader.load(musicxml)

    def test_load_returns_self(self):
        """load() returns Self for method chaining."""
        loader = MatchfileLoader()
        result = loader.load(P01_MATCH)
        assert result is loader


# endregion


# region TestMatchfileLoaderSingle


class TestMatchfileLoaderSingle:
    """Tests for single-file loading (p01)."""

    def test_score_timeline_type(self, p01_loader: MatchfileLoader):
        """Score timeline is ContinuousLogicalTimeline."""
        score_tl = p01_loader.create_timeline("score")
        assert isinstance(score_tl, ContinuousLogicalTimeline)

    def test_score_timeline_unit(self, p01_loader: MatchfileLoader):
        """Score timeline uses TimeUnit.quarters by default."""
        score_tl = p01_loader.create_timeline("score")
        assert score_tl.unit == TimeUnit.quarters

    def test_score_timeline_uid(self, p01_loader: MatchfileLoader):
        """Score timeline uses the role-prefixed logical timeline ID."""
        score_tl = p01_loader.create_timeline("score")
        assert score_tl.id == "score:clt_1"
        assert score_tl.name == "Chopin_op10_no3"

    def test_score_timeline_note_count(self, p01_loader: MatchfileLoader):
        """Score timeline has exactly 454 events (snote subset)."""
        score_tl = p01_loader.create_timeline("score")
        assert len(score_tl) == SNOTE_COUNT

    def test_score_timeline_nonnegative_coordinates(self, p01_loader: MatchfileLoader):
        """All score event coordinates are >= 0 after normalisation."""
        score_tl = p01_loader.create_timeline("score")
        events = score_tl.events
        table = events.table
        # start is a struct<value, numerator, denominator>; flatten to get
        # the 'value' sub-column at index 0.
        starts = table.column("start").combine_chunks().flatten()[0].to_pylist()
        assert all(s >= 0.0 for s in starts), (
            f"Found negative start coordinates after normalisation: "
            f"{[s for s in starts if s < 0.0]}"
        )

    def test_score_coordinates_preserve_exact_fractions(
        self, p01_loader: MatchfileLoader
    ):
        """Known score positions retain exact normalized beat fractions."""
        score_tl = p01_loader.create_timeline("score")
        expected = {
            "n1": (Fraction(0), Fraction(1, 2), Fraction(1, 2)),
            "n2": (Fraction(1, 2), Fraction(1, 2), Fraction(1)),
            "n12": (Fraction(3, 2), Fraction(5, 4), Fraction(11, 4)),
        }

        for note_id, (start, duration, end) in expected.items():
            event = score_tl.get_event(note_id)
            assert event is not None
            assert (
                Fraction(event["start"]["numerator"], event["start"]["denominator"])
                == start
            )
            assert (
                Fraction(
                    event["duration"]["numerator"], event["duration"]["denominator"]
                )
                == duration
            )
            assert (
                Fraction(event["end"]["numerator"], event["end"]["denominator"]) == end
            )

    def test_score_cache_matches_exact_timeline_coordinates(
        self, p01_loader: MatchfileLoader
    ):
        """The exact score cache and stored timeline use the same pairs."""
        score_tl = p01_loader.create_timeline("score")
        for note_id, (cached_start, cached_end) in p01_loader._score_events.items():
            event = score_tl.get_event(note_id)
            assert event is not None
            stored_start = Fraction(
                event["start"]["numerator"], event["start"]["denominator"]
            )
            stored_end = Fraction(
                event["end"]["numerator"], event["end"]["denominator"]
            )
            assert stored_start == cached_start
            assert stored_end == cached_end

    def test_anacrusis_offset(self, p01_loader: MatchfileLoader):
        """Anacrusis offset is 0.5 (the negation of min raw onset -0.5)."""
        assert p01_loader.anacrusis_offset == 0.5

    def test_score_cmap_shift(self, p01_loader: MatchfileLoader):
        """raw_quarters ShiftMap converts normalised coords to raw partitura."""
        score_tl = p01_loader.create_timeline("score")
        shift_map = score_tl.get_conversion_map("raw_quarters")
        assert shift_map is not None
        assert isinstance(shift_map, ShiftMap)
        assert shift_map.offset == -0.5

    def test_score_cmap_divs(self, p01_loader: MatchfileLoader):
        """quarters_to_divs ScalarMap is attached, maps 1.0 -> 480."""
        score_tl = p01_loader.create_timeline("score")
        divs_map = score_tl.get_conversion_map("quarters_to_divs")
        assert divs_map is not None
        assert isinstance(divs_map, ScalarMap)
        assert divs_map(1.0) == MIDI_CLOCK_UNITS

    def test_perf_timeline_type(self, p01_loader: MatchfileLoader):
        """Performance timeline is DiscreteLogicalTimeline."""
        perf_tl = p01_loader.create_timeline("perf:1")
        assert isinstance(perf_tl, DiscreteLogicalTimeline)

    def test_perf_timeline_unit(self, p01_loader: MatchfileLoader):
        """Performance timeline uses TimeUnit.ticks."""
        perf_tl = p01_loader.create_timeline("perf:1")
        assert perf_tl.unit == TimeUnit.ticks

    def test_perf_timeline_uid(self, p01_loader: MatchfileLoader):
        """Performance timeline uses a role/source/type ID."""
        perf_tl = p01_loader.create_timeline("perf:1")
        assert perf_tl.id == "perf:Chopin_op10_no3_p01:dlt_1"
        assert perf_tl.name == "Chopin_op10_no3_p01"

    def test_perf_timeline_note_count(self, p01_loader: MatchfileLoader):
        """Performance timeline has exactly 451 events (454 - 3 deletions)."""
        perf_tl = p01_loader.create_timeline("perf:1")
        assert len(perf_tl) == P01_PERF_NOTE_COUNT

    def test_perf_cmap_seconds(self, p01_loader: MatchfileLoader):
        """ticks_to_seconds ScalarMap attached; 480 ticks = 0.5s at 120 BPM."""
        perf_tl = p01_loader.create_timeline("perf:1")
        secs_map = perf_tl.get_conversion_map(TimeUnit.seconds)
        assert secs_map is not None
        assert isinstance(secs_map, ScalarMap)
        # At 120 BPM: 480 ticks = 1 quarter = 0.5 seconds
        expected = MIDI_CLOCK_RATE / (MIDI_CLOCK_UNITS * 1_000_000) * 480
        assert abs(secs_map(480) - expected) < 1e-10

    def test_total_claims_count(self, p01_loader: MatchfileLoader):
        """Total MatchClaims = 454 (451 match + 3 NOMATCH)."""
        assert len(p01_loader._claims) == SNOTE_COUNT

    def test_synchronous_claims_count(self, p01_loader: MatchfileLoader):
        """451 synchronous (matched) claims."""
        sync_claims = [c for c in p01_loader._claims if c.is_synchronous]
        assert len(sync_claims) == P01_MATCH_COUNT

    def test_nomatch_claims_count(self, p01_loader: MatchfileLoader):
        """3 non-synchronous (deletion/NOMATCH) claims."""
        nomatch_claims = [c for c in p01_loader._claims if not c.is_synchronous]
        assert len(nomatch_claims) == P01_DELETION_COUNT

    def test_match_metadata_agent(self, p01_loader: MatchfileLoader):
        """All claims carry the vienna_match software agent."""
        for claim in p01_loader._claims:
            assert claim.metadata is not None
            assert claim.metadata.agent.name == "vienna_match"

    def test_match_metadata_agent_version(self, p01_loader: MatchfileLoader):
        """All claims carry the vienna_match version as the agent identifier."""
        for claim in p01_loader._claims:
            assert claim.metadata.agent.identifier == "v1.0.0"

    def test_match_metadata_certainty(self, p01_loader: MatchfileLoader):
        """All claims carry certainty=1.0."""
        for claim in p01_loader._claims:
            assert claim.metadata.certainty == 1.0

    def test_synchronous_claims_have_anchors(self, p01_loader: MatchfileLoader):
        """Every synchronous claim has start_anchor and end_anchor."""
        sync_claims = [c for c in p01_loader._claims if c.is_synchronous]
        for claim in sync_claims:
            assert claim.start_anchor is not None
            assert claim.end_anchor is not None

    def test_nomatch_claims_have_no_anchors(self, p01_loader: MatchfileLoader):
        """Every NOMATCH claim has no anchors."""
        nomatch_claims = [c for c in p01_loader._claims if not c.is_synchronous]
        for claim in nomatch_claims:
            assert claim.start_anchor is None
            assert claim.end_anchor is None

    def test_claims_reference_correct_timelines(self, p01_loader: MatchfileLoader):
        """All claims connect score and p01 performance timelines."""
        score_id = "score:clt_1"
        perf_id = "perf:Chopin_op10_no3_p01:dlt_1"
        for claim in p01_loader._claims:
            assert claim.timeline_a_id == score_id
            assert claim.timeline_b_id == perf_id

    def test_no_rejected_files(self, p01_loader: MatchfileLoader):
        """Single file load should not reject any files."""
        assert len(p01_loader.rejected_files) == 0

    def test_sources_tracked(self, p01_loader: MatchfileLoader):
        """Sources list contains the loaded file."""
        assert len(p01_loader.sources) == 1
        assert p01_loader.sources[0] == P01_MATCH

    def test_loader_len(self, p01_loader: MatchfileLoader):
        """len(loader) returns number of performance timelines."""
        assert len(p01_loader) == 1

    def test_loader_repr(self, p01_loader: MatchfileLoader):
        """repr includes performance count and claim count."""
        r = repr(p01_loader)
        assert "performances=1" in r
        assert f"claims={SNOTE_COUNT}" in r


# endregion


# region TestMatchfileLoaderNormalization


class TestMatchfileLoaderNormalization:
    """Tests for anacrusis normalisation behaviour."""

    def test_no_normalisation_flag(self):
        """When normalize_anacrusis=False, no ShiftMap is attached."""
        loader = MatchfileLoader(normalize_anacrusis=False)
        loader.load(P01_MATCH)
        score_tl = loader.create_timeline("score")
        shift_map = score_tl.get_conversion_map("raw_quarters")
        assert shift_map is None

    def test_normalisation_still_shifts_coordinates(self):
        """Even with normalize_anacrusis=False, coordinates are still
        non-negative (normalisation is always applied to stored coords;
        the flag controls only whether the ShiftMap is attached)."""
        loader = MatchfileLoader(normalize_anacrusis=False)
        loader.load(P01_MATCH)
        score_tl = loader.create_timeline("score")
        events = score_tl.events
        starts = events.table.column("start").combine_chunks().flatten()[0].to_pylist()
        assert all(s >= 0.0 for s in starts)

    def test_anacrusis_offset_computed_from_file(self):
        """Offset is computed dynamically, not hardcoded."""
        loader = MatchfileLoader()
        loader.load(P01_MATCH)
        # Offset should be 0.5 for this piece (min onset = -0.5)
        assert loader.anacrusis_offset == 0.5


# endregion


# region TestMatchfileLoaderCreateTimeline


class TestMatchfileLoaderCreateTimeline:
    """Tests for create_timeline() role-based lookup."""

    def test_score_role(self, p01_loader: MatchfileLoader):
        """'score' role returns score timeline."""
        tl = p01_loader.create_timeline("score")
        assert tl.id == "score:clt_1"

    def test_perf_numeric_role(self, p01_loader: MatchfileLoader):
        """'perf:1' returns first performance timeline."""
        tl = p01_loader.create_timeline("perf:1")
        assert tl.id == "perf:Chopin_op10_no3_p01:dlt_1"

    def test_perf_uid_lookup(self, p01_loader: MatchfileLoader):
        """Full uid lookup works."""
        tl = p01_loader.create_timeline("perf:Chopin_op10_no3_p01:dlt_1")
        assert tl.id == "perf:Chopin_op10_no3_p01:dlt_1"

    def test_score_uid_lookup(self, p01_loader: MatchfileLoader):
        """Score timeline accessible by full uid."""
        tl = p01_loader.create_timeline("score:clt_1")
        assert tl.id == "score:clt_1"

    def test_invalid_role_raises(self, p01_loader: MatchfileLoader):
        """KeyError raised for unknown role/uid."""
        with pytest.raises(KeyError):
            p01_loader.create_timeline("nonexistent")

    def test_no_load_raises(self):
        """RuntimeError raised if load() not called."""
        loader = MatchfileLoader()
        with pytest.raises(RuntimeError, match="No files loaded"):
            loader.create_timeline("score")


# endregion


# region TestMatchfileLoaderCreateBundle


class TestMatchfileLoaderCreateBundle:
    """Tests for create_bundle() — single-file."""

    def test_returns_alignment_bundle(self, p01_loader: MatchfileLoader):
        """Returns an AlignmentBundle."""
        bundle = p01_loader.create_bundle()
        assert isinstance(bundle, AlignmentBundle)

    def test_bundle_timeline_count(self, p01_loader: MatchfileLoader):
        """Bundle has 2 timelines (1 score + 1 performance)."""
        bundle = p01_loader.create_bundle()
        assert len(bundle.timelines) == 2

    def test_bundle_score_timeline(self, p01_loader: MatchfileLoader):
        """Score timeline is in the bundle under its type-based uid."""
        bundle = p01_loader.create_bundle()
        assert "score:clt_1" in bundle.timelines

    def test_bundle_cross_group_claims(self, p01_loader: MatchfileLoader):
        """Bundle has 454 cross-group claims."""
        bundle = p01_loader.create_bundle()
        assert len(bundle.cross_group_claims) == SNOTE_COUNT

    def test_no_load_raises(self):
        """RuntimeError raised if load() not called."""
        loader = MatchfileLoader()
        with pytest.raises(RuntimeError, match="No files loaded"):
            loader.create_bundle()

    def test_bundle_score_in_group(self, p01_loader: MatchfileLoader):
        """Score timeline is in the 'score' group."""
        bundle = p01_loader.create_bundle()
        assert "score:clt_1" in bundle.timeline_to_group
        assert bundle.timeline_to_group["score:clt_1"] == "score"


# endregion


# region TestMatchfileLoaderCreateTimelines


class TestMatchfileLoaderCreateTimelines:
    """Tests for create_timelines() — all-at-once retrieval."""

    def test_returns_list(self, p01_loader: MatchfileLoader):
        """create_timelines() returns a list."""
        tls = p01_loader.create_timelines()
        assert isinstance(tls, list)

    def test_single_file_gives_two(self, p01_loader: MatchfileLoader):
        """Single file produces [score, perf] = 2 timelines."""
        tls = p01_loader.create_timelines()
        assert len(tls) == 2

    def test_id_pattern_filters_performance(self, p01_loader: MatchfileLoader):
        """The single-file fixture has one performance timeline."""
        tls = p01_loader.create_timelines(id_pattern=r"^perf:")
        assert len(tls) == 1
        assert tls[0].id.startswith("perf:")

    def test_id_pattern_no_match(self, p01_loader: MatchfileLoader) -> None:
        """An unmatched ID pattern returns no timelines."""
        assert p01_loader.create_timelines(id_pattern=r"^missing:") == []

    def test_score_is_first(self, p01_loader: MatchfileLoader):
        """Score timeline is always first."""
        tls = p01_loader.create_timelines()
        assert isinstance(tls[0], ContinuousLogicalTimeline)
        assert tls[0].id == "score:clt_1"

    def test_empty_before_load(self):
        """Empty list before load()."""
        loader = MatchfileLoader()
        assert loader.create_timelines() == []


# endregion


# region TestMatchfileLoaderMulti


@pytest.mark.slow
class TestMatchfileLoaderMulti:
    """Tests for multi-file loading (all 22 performances)."""

    def test_22_performances_loaded(self, all_loader: MatchfileLoader):
        """All 22 performances produce distinct timelines."""
        assert len(all_loader) == TOTAL_MATCH_FILES

    def test_22_sources_tracked(self, all_loader: MatchfileLoader):
        """All 22 source files tracked."""
        assert len(all_loader.sources) == TOTAL_MATCH_FILES

    def test_no_rejected_files(self, all_loader: MatchfileLoader):
        """All 22 files are compatible (no rejections)."""
        assert len(all_loader.rejected_files) == 0

    def test_shared_score_timeline(self, all_loader: MatchfileLoader):
        """All 22 performances share the same score timeline object."""
        score_tl = all_loader.create_timeline("score")
        tls = all_loader.create_timelines()
        # First element is the score timeline
        assert tls[0] is score_tl

    def test_score_note_count_unchanged(self, all_loader: MatchfileLoader):
        """Score timeline still has 454 events after 22 loads."""
        score_tl = all_loader.create_timeline("score")
        assert len(score_tl) == SNOTE_COUNT

    def test_distinct_performance_uids(self, all_loader: MatchfileLoader):
        """22 performance timelines have distinct UIDs."""
        tls = all_loader.create_timelines()
        perf_tls = tls[1:]  # skip score
        uids = [tl.id for tl in perf_tls]
        assert len(set(uids)) == TOTAL_MATCH_FILES

    def test_total_claims_count(self, all_loader: MatchfileLoader):
        """Total claims = sum of per-file claims across 22 files."""
        # Each file produces snote_count claims (matched + deletions)
        total = len(all_loader._claims)
        # Must be exactly 22 * snote_count = 9,988
        assert total == TOTAL_MATCH_FILES * SNOTE_COUNT

    def test_all_claims_reference_same_score(self, all_loader: MatchfileLoader):
        """All claims reference the shared score timeline ID."""
        score_id = "score:clt_1"
        for claim in all_loader._claims:
            assert claim.timeline_a_id == score_id

    def test_create_timelines_gives_23(self, all_loader: MatchfileLoader):
        """create_timelines() returns 23 (1 score + 22 performances)."""
        tls = all_loader.create_timelines()
        assert len(tls) == 1 + TOTAL_MATCH_FILES

    def test_id_pattern_filters_all_performances(self, all_loader: MatchfileLoader):
        """The full fixture has exactly 22 performance timelines."""
        tls = all_loader.create_timelines(id_pattern=r"^perf:")
        assert len(tls) == TOTAL_MATCH_FILES

    def test_bundle_timeline_count(self, all_loader: MatchfileLoader):
        """Bundle has 23 timelines."""
        bundle = all_loader.create_bundle()
        assert len(bundle.timelines) == 1 + TOTAL_MATCH_FILES

    def test_bundle_cross_group_claims_count(self, all_loader: MatchfileLoader):
        """Bundle has 9,988 cross-group claims."""
        bundle = all_loader.create_bundle()
        assert len(bundle.cross_group_claims) == TOTAL_MATCH_FILES * SNOTE_COUNT

    def test_perf_numeric_lookup_all(self, all_loader: MatchfileLoader):
        """perf:1 through perf:22 all resolve correctly."""
        for i in range(1, TOTAL_MATCH_FILES + 1):
            tl = all_loader.create_timeline(f"perf:{i}")
            assert tl is not None
            assert isinstance(tl, DiscreteLogicalTimeline)

    def test_incremental_load(self):
        """Loading files incrementally produces same result as batch."""
        batch_loader = MatchfileLoader()
        batch_loader.load(*ALL_MATCH_FILES)

        incremental_loader = MatchfileLoader()
        for f in ALL_MATCH_FILES:
            incremental_loader.load(f)

        assert len(batch_loader) == len(incremental_loader)
        assert len(batch_loader._claims) == len(incremental_loader._claims)


# endregion


# region TestMatchfileLoaderExternalScore


class TestMatchfileLoaderExternalScore:
    """Tests for external score timeline binding via
    create_bundle(score_timeline=...)."""

    def test_external_score_used_in_bundle(self):
        """Bundle uses the externally supplied score timeline."""
        loader = MatchfileLoader()
        loader.load(P01_MATCH)

        # Create an external score TL with the same uid
        external_score = ContinuousLogicalTimeline(
            length=100.0,
            unit=TimeUnit.quarters,
            uid="score:clt_1",
        )

        bundle = loader.create_bundle(score_timeline=external_score)
        assert bundle.timelines["score:clt_1"] is external_score

    def test_claims_still_reference_internal_uid(self):
        """MatchClaims are never rebound — they reference the internal
        score timeline's uid, which must match the external one."""
        loader = MatchfileLoader()
        loader.load(P01_MATCH)

        external_score = ContinuousLogicalTimeline(
            length=100.0,
            unit=TimeUnit.quarters,
            uid="score:clt_1",
        )

        bundle = loader.create_bundle(score_timeline=external_score)
        for claim in bundle.cross_group_claims:
            assert claim.timeline_a_id == "score:clt_1"

    def test_compatible_external_score_accepted(self, p01_loader: MatchfileLoader):
        """External score with matching events passes verification."""
        # Build an external timeline with matching coordinates
        internal_score = p01_loader.create_timeline("score")
        external_score = ContinuousLogicalTimeline(
            length=200.0,
            unit=TimeUnit.quarters,
            uid="score:clt_1",
        )
        # Copy a few events from internal to external
        events = internal_score.events
        sample_events = []
        for i, event in enumerate(events):
            if i >= 5:
                break
            sample_events.append(
                {
                    "id": event["id"],
                    "start": float(event["start"]["value"]),
                    "end": float(event["end"]["value"]),
                }
            )
        external_score.add_events(sample_events)

        # Should not raise
        bundle = p01_loader.create_bundle(score_timeline=external_score)
        assert bundle.timelines["score:clt_1"] is external_score

    def test_incompatible_external_score_raises(self, p01_loader: MatchfileLoader):
        """External score with mismatched coordinates raises ValueError."""
        external_score = ContinuousLogicalTimeline(
            length=200.0,
            unit=TimeUnit.quarters,
            uid="score:clt_1",
        )
        # Add an event with a known snote ID but wrong coordinate
        # Get a real snote ID from the loader's internal cache
        real_id = next(iter(p01_loader._score_events))
        real_start, real_end = p01_loader._score_events[real_id]
        external_score.add_events(
            [
                {
                    "id": real_id,
                    "start": real_start + 99.0,  # deliberately wrong
                    "end": real_end + 99.0,
                }
            ]
        )

        with pytest.raises(ValueError, match="incompatible event"):
            p01_loader.create_bundle(score_timeline=external_score)

    def test_verify_false_skips_check(self, p01_loader: MatchfileLoader):
        """verify=False allows incompatible external score without error."""
        external_score = ContinuousLogicalTimeline(
            length=200.0,
            unit=TimeUnit.quarters,
            uid="score:clt_1",
        )
        real_id = next(iter(p01_loader._score_events))
        real_start, real_end = p01_loader._score_events[real_id]
        external_score.add_events(
            [
                {
                    "id": real_id,
                    "start": real_start + 99.0,
                    "end": real_end + 99.0,
                }
            ]
        )

        # Should NOT raise with verify=False
        bundle = p01_loader.create_bundle(score_timeline=external_score, verify=False)
        assert bundle.timelines["score:clt_1"] is external_score

    def test_empty_external_score_accepted(self, p01_loader: MatchfileLoader):
        """External score with no events passes verification (absence tolerated)."""
        external_score = ContinuousLogicalTimeline(
            length=100.0,
            unit=TimeUnit.quarters,
            uid="score:clt_1",
        )
        # No events added — all lookups return None — tolerated
        bundle = p01_loader.create_bundle(score_timeline=external_score)
        assert bundle.timelines["score:clt_1"] is external_score


# endregion


# region TestTimelineGetEvent


class TestTimelineGetEvent:
    """Tests for Timeline.get_event() — single event lookup by ID."""

    def test_get_existing_event(self, p01_loader: MatchfileLoader):
        """get_event returns a dict for a known event ID."""
        score_tl = p01_loader.create_timeline("score")
        event = score_tl.get_event("n1")
        assert event is not None
        assert event["id"] == "n1"

    def test_get_nonexistent_event(self, p01_loader: MatchfileLoader):
        """get_event returns None for an unknown event ID."""
        score_tl = p01_loader.create_timeline("score")
        event = score_tl.get_event("nonexistent_note_xyz")
        assert event is None

    def test_event_has_coordinates(self, p01_loader: MatchfileLoader):
        """Returned event dict has start and end coordinates."""
        score_tl = p01_loader.create_timeline("score")
        event = score_tl.get_event("n1")
        assert event is not None
        assert "start" in event
        assert "end" in event

    def test_event_start_matches_cache(self, p01_loader: MatchfileLoader):
        """get_event coordinate matches the internal _score_events cache."""
        score_tl = p01_loader.create_timeline("score")
        for snote_id, (cached_start, _cached_end) in list(
            p01_loader._score_events.items()
        )[:10]:
            event = score_tl.get_event(snote_id)
            assert event is not None
            start_val = (
                float(event["start"]["value"])
                if isinstance(event["start"], dict)
                else float(event["start"])
            )
            assert abs(start_val - cached_start) < 1e-10

    def test_performance_timeline_get_event(self, p01_loader: MatchfileLoader):
        """get_event works on performance timelines too."""
        perf_tl = p01_loader.create_timeline("perf:1")
        # Performance notes have IDs like "n0", "n1", etc.
        # Get first event from the timeline
        events = perf_tl.events
        first_event = next(iter(events))
        first_id = first_event["id"]
        result = perf_tl.get_event(first_id)
        assert result is not None
        assert result["id"] == first_id


# endregion


# region TestTimelineGetConversionMapByName


class TestTimelineGetConversionMapByName:
    """Tests for Timeline.get_conversion_map() with name-based lookup."""

    def test_lookup_by_unit_still_works(self, p01_loader: MatchfileLoader):
        """Unit-based lookup (original API) still works."""
        perf_tl = p01_loader.create_timeline("perf:1")
        secs_map = perf_tl.get_conversion_map(TimeUnit.seconds)
        assert secs_map is not None
        assert isinstance(secs_map, ScalarMap)

    def test_lookup_by_unit_string(self, p01_loader: MatchfileLoader):
        """String unit lookup (e.g. 'seconds') still works."""
        perf_tl = p01_loader.create_timeline("perf:1")
        secs_map = perf_tl.get_conversion_map("seconds")
        assert secs_map is not None

    def test_lookup_by_name_shift_map(self, p01_loader: MatchfileLoader):
        """Name-based lookup finds the raw_quarters ShiftMap."""
        score_tl = p01_loader.create_timeline("score")
        shift_map = score_tl.get_conversion_map("raw_quarters")
        assert shift_map is not None
        assert isinstance(shift_map, ShiftMap)
        assert shift_map.offset == -0.5

    def test_lookup_by_name_scalar_map(self, p01_loader: MatchfileLoader):
        """Name-based lookup finds the quarters_to_divs ScalarMap."""
        score_tl = p01_loader.create_timeline("score")
        divs_map = score_tl.get_conversion_map("quarters_to_divs")
        assert divs_map is not None
        assert isinstance(divs_map, ScalarMap)

    def test_lookup_by_unknown_name_returns_none(self, p01_loader: MatchfileLoader):
        """Name-based lookup for a non-existent map returns None."""
        score_tl = p01_loader.create_timeline("score")
        result = score_tl.get_conversion_map("nonexistent_map")
        assert result is None

    def test_lookup_by_unknown_unit_returns_none(self, p01_loader: MatchfileLoader):
        """Unit-based lookup for a unit with no map returns None."""
        score_tl = p01_loader.create_timeline("score")
        result = score_tl.get_conversion_map(TimeUnit.milliseconds)
        assert result is None


# endregion


# region TestMatchfileLoaderCheckOrAddScoreEvent


class TestMatchfileLoaderCheckOrAddScoreEvent:
    """Tests for the _check_or_add_score_event() compatibility check."""

    def test_compatible_event_returns_true(self, p01_loader: MatchfileLoader):
        """Known event with matching coordinates is compatible."""
        real_id = next(iter(p01_loader._score_events))
        onset, end = p01_loader._score_events[real_id]
        assert p01_loader._check_or_add_score_event(real_id, onset, end, "test.match")

    def test_incompatible_event_returns_false(self, p01_loader: MatchfileLoader):
        """Known event with different onset is incompatible."""
        real_id = next(iter(p01_loader._score_events))
        onset, end = p01_loader._score_events[real_id]
        assert not p01_loader._check_or_add_score_event(
            real_id, onset + 99.0, end + 99.0, "test.match"
        )

    def test_new_event_added(self, p01_loader: MatchfileLoader):
        """Unknown event ID is added to the cache and timeline."""
        new_id = "n_brand_new_test_event"
        assert new_id not in p01_loader._score_events
        result = p01_loader._check_or_add_score_event(new_id, 42.0, 43.0, "test.match")
        assert result is True
        assert new_id in p01_loader._score_events
        assert p01_loader._score_events[new_id] == (42.0, 43.0)

    def test_new_event_appears_on_timeline(self, p01_loader: MatchfileLoader):
        """Newly added event is retrievable via Timeline.get_event()."""
        new_id = "n_timeline_lookup_test"
        p01_loader._check_or_add_score_event(new_id, 10.0, 11.0, "test.match")
        score_tl = p01_loader.create_timeline("score")
        event = score_tl.get_event(new_id)
        assert event is not None
        assert event["id"] == new_id

    def test_to_tta_coord_static(self):
        """_to_tta_coord is a pure offset addition."""
        assert MatchfileLoader._to_tta_coord(3.5, 0.5) == 4.0
        assert MatchfileLoader._to_tta_coord(-0.5, 0.5) == 0.0
        assert MatchfileLoader._to_tta_coord(0.0, 0.0) == 0.0


# endregion


# region TestPerPerformerDeletionCounts


class TestPerPerformerDeletionCounts:
    """Validate exact per-performer deletion and match counts.

    Gold standard values from README.md. ZERO TOLERANCE: exact counts only.
    """

    def test_all_files_have_known_deletion_counts(self):
        """Every match file has a documented deletion count."""
        for f in ALL_MATCH_FILES:
            performer = f.stem.split("_")[-1]
            assert (
                performer in DELETION_COUNTS
            ), f"No gold standard deletion count for {performer}"

    @pytest.mark.parametrize(
        "performer,expected_deletions",
        [
            (
                pytest.param(performer, expected, marks=pytest.mark.slow)
                if performer in {"p04", "p10", "p15", "p21"}
                else (performer, expected)
            )
            for performer, expected in DELETION_COUNTS.items()
        ],
    )
    def test_per_performer_deletion_count(
        self, performer: str, expected_deletions: int
    ):
        """Each performer's deletion count matches the gold standard."""
        match_file = VIENNA_DATA_DIR / f"Chopin_op10_no3_{performer}.match"
        loader = MatchfileLoader()
        loader.load(match_file)

        nomatch_claims = [c for c in loader._claims if not c.is_synchronous]
        assert len(nomatch_claims) == expected_deletions, (
            f"Performer {performer}: expected {expected_deletions} deletions, "
            f"got {len(nomatch_claims)}"
        )

    @pytest.mark.parametrize(
        "performer,expected_deletions",
        [
            (
                pytest.param(performer, expected, marks=pytest.mark.slow)
                if performer == "p05"
                else (performer, expected)
            )
            for performer, expected in DELETION_COUNTS.items()
        ],
    )
    def test_per_performer_matched_count(self, performer: str, expected_deletions: int):
        """Each performer's matched count = SNOTE_COUNT - deletions."""
        match_file = VIENNA_DATA_DIR / f"Chopin_op10_no3_{performer}.match"
        loader = MatchfileLoader()
        loader.load(match_file)

        sync_claims = [c for c in loader._claims if c.is_synchronous]
        expected_matched = SNOTE_COUNT - expected_deletions
        assert len(sync_claims) == expected_matched, (
            f"Performer {performer}: expected {expected_matched} matched, "
            f"got {len(sync_claims)}"
        )

    @pytest.mark.slow
    def test_total_deletions_across_all_performers(self, all_loader: MatchfileLoader):
        """Total NOMATCH claims across all 22 files = 113."""
        nomatch_claims = [c for c in all_loader._claims if not c.is_synchronous]
        assert len(nomatch_claims) == TOTAL_DELETIONS

    @pytest.mark.slow
    def test_total_matched_across_all_performers(self, all_loader: MatchfileLoader):
        """Total synchronous claims across all 22 files = 9875."""
        sync_claims = [c for c in all_loader._claims if c.is_synchronous]
        assert len(sync_claims) == TOTAL_MATCHED

    @pytest.mark.slow
    def test_p08_highest_deletion_count(self, all_loader: MatchfileLoader):
        """p08 has the most deletions (20) — outlier validation."""
        # Find claims referencing p08's performance timeline
        p08_claims = [
            c
            for c in all_loader._claims
            if c.timeline_b_id == "perf:Chopin_op10_no3_p08:dlt_1"
        ]
        nomatch_p08 = [c for c in p08_claims if not c.is_synchronous]
        assert len(nomatch_p08) == 20

    @pytest.mark.slow
    def test_p12_lowest_deletion_count(self, all_loader: MatchfileLoader):
        """p12 has the fewest deletions (1) — outlier validation."""
        p12_claims = [
            c
            for c in all_loader._claims
            if c.timeline_b_id == "perf:Chopin_op10_no3_p12:dlt_1"
        ]
        nomatch_p12 = [c for c in p12_claims if not c.is_synchronous]
        assert len(nomatch_p12) == 1


# endregion


# region TestMatchfileLoaderPerfPNNShorthand


class TestMatchfileLoaderPerfPNNShorthand:
    """Tests for the perf:pNN shorthand in create_timeline()."""

    def test_perf_p01_shorthand(self, p01_loader: MatchfileLoader):
        """'perf:p01' resolves to the first performance timeline."""
        tl = p01_loader.create_timeline("perf:p01")
        assert tl.id == "perf:Chopin_op10_no3_p01:dlt_1"

    @pytest.mark.slow
    def test_perf_p_shorthands_all_22(self, all_loader: MatchfileLoader):
        """'perf:p01' through 'perf:p22' all resolve for the 22-file set."""
        for i in range(1, TOTAL_MATCH_FILES + 1):
            tl = all_loader.create_timeline(f"perf:p{i:02d}")
            assert tl is not None
            assert isinstance(tl, DiscreteLogicalTimeline)

    @pytest.mark.slow
    def test_perf_p_shorthand_matches_numeric(self, all_loader: MatchfileLoader):
        """'perf:pNN' and 'perf:N' resolve to the same timeline."""
        for i in range(1, TOTAL_MATCH_FILES + 1):
            tl_numeric = all_loader.create_timeline(f"perf:{i}")
            tl_pnn = all_loader.create_timeline(f"perf:p{i:02d}")
            assert tl_numeric is tl_pnn

    def test_perf_p_invalid_raises(self, p01_loader: MatchfileLoader):
        """'perf:pXX' with non-numeric suffix raises KeyError."""
        with pytest.raises(KeyError):
            p01_loader.create_timeline("perf:pabc")

    def test_perf_p_out_of_range_raises(self, p01_loader: MatchfileLoader):
        """'perf:p99' with out-of-range index raises KeyError."""
        with pytest.raises(KeyError):
            p01_loader.create_timeline("perf:p99")


# endregion


# region TestMatchfileLoaderRejection


class TestMatchfileLoaderRejection:
    """Tests for file rejection during multi-file loading.

    Exercises the code path in _load_source() where a subsequent file
    has score events with mismatched coordinates.
    """

    def test_rejection_preserves_prior_state(self, p01_loader: MatchfileLoader):
        """After injecting a bad score event and loading a 2nd file,
        the loader rejects the 2nd file and preserves the p01 state."""
        # Tamper with the internal cache: give a known snote_id a wrong onset
        real_id = next(iter(p01_loader._score_events))
        original = p01_loader._score_events[real_id]
        p01_loader._score_events[real_id] = (original[0] + 99.0, original[1] + 99.0)

        # Load a second file (p02) — it will see the tampered coordinate
        # and reject the file
        p02_match = VIENNA_DATA_DIR / "Chopin_op10_no3_p02.match"
        p01_loader.load(p02_match)

        # p02 should be rejected
        assert len(p01_loader.rejected_files) == 1
        assert p01_loader.rejected_files[0] == p02_match

        # Performance count unchanged (only p01)
        assert len(p01_loader) == 1

        # Claims unchanged (only p01's claims)
        assert len(p01_loader._claims) == SNOTE_COUNT

    def test_rejection_does_not_add_performance_timeline(
        self, p01_loader: MatchfileLoader
    ):
        """Rejected file does not contribute a performance timeline."""
        real_id = next(iter(p01_loader._score_events))
        original = p01_loader._score_events[real_id]
        p01_loader._score_events[real_id] = (original[0] + 99.0, original[1] + 99.0)

        p02_match = VIENNA_DATA_DIR / "Chopin_op10_no3_p02.match"
        p01_loader.load(p02_match)

        timelines = p01_loader.create_timelines()
        assert len(timelines) == 2  # score + p01 only

    def test_rejected_file_tracked_in_sources(self, p01_loader: MatchfileLoader):
        """Rejected files appear in sources list (all files attempted)."""
        real_id = next(iter(p01_loader._score_events))
        original = p01_loader._score_events[real_id]
        p01_loader._score_events[real_id] = (original[0] + 99.0, original[1] + 99.0)

        p02_match = VIENNA_DATA_DIR / "Chopin_op10_no3_p02.match"
        p01_loader.load(p02_match)

        assert len(p01_loader.sources) == 2  # p01 + p02 both tracked
        assert p02_match in p01_loader.sources

    def test_rejection_allows_subsequent_compatible_files(
        self, p01_loader: MatchfileLoader
    ):
        """After a rejected file, a compatible file still loads normally."""
        real_id = next(iter(p01_loader._score_events))
        original = p01_loader._score_events[real_id]
        p01_loader._score_events[real_id] = (original[0] + 99.0, original[1] + 99.0)

        p02_match = VIENNA_DATA_DIR / "Chopin_op10_no3_p02.match"
        p01_loader.load(p02_match)

        assert len(p01_loader.rejected_files) == 1

        # Restore the original cache entry so p03 will be compatible
        p01_loader._score_events[real_id] = original

        p03_match = VIENNA_DATA_DIR / "Chopin_op10_no3_p03.match"
        p01_loader.load(p03_match)

        # p03 should be accepted
        assert len(p01_loader.rejected_files) == 1  # still only p02
        assert len(p01_loader) == 2  # p01 + p03


# endregion


# region TestPerPerformerPerfNoteCount


class TestPerPerformerPerfNoteCount:
    """Validate that each performer's performance timeline has the correct
    number of notes (= matched count = SNOTE_COUNT - deletions)."""

    @pytest.mark.parametrize(
        "performer,expected_deletions",
        list(DELETION_COUNTS.items()),
    )
    def test_perf_note_count(self, performer: str, expected_deletions: int):
        """Performance timeline note count = SNOTE_COUNT - deletions."""
        match_file = VIENNA_DATA_DIR / f"Chopin_op10_no3_{performer}.match"
        loader = MatchfileLoader()
        loader.load(match_file)

        perf_tl = loader.create_timeline("perf:1")
        expected = SNOTE_COUNT - expected_deletions
        assert len(perf_tl) == expected, (
            f"Performer {performer}: expected {expected} perf notes, "
            f"got {len(perf_tl)}"
        )


# endregion
