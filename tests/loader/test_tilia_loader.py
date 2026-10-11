"""Tests for TiliaJsonLoader: TiLiA JSON annotation export loader.

Test specimen:
    Bruckner5_Scherzo.json — TiLiA export with 7 timelines:
        [0] HIERARCHY_TIMELINE "Form (Harnoncourt)"  — 33 components
        [1] MARKER_TIMELINE    "Themen"              — 14 components
        [2] MARKER_TIMELINE    "Topik"               —  5 components
        [3] BEAT_TIMELINE      "Takte"               — 1146 components
        [4] MARKER_TIMELINE    "Tempo"               — 12 components
        [5] MARKER_TIMELINE    "Dramaturgie"         — 11 components
        [6] PDF_TIMELINE       "Partitur (PDF)"      — 19 components

See Also:
    timetoalign.loader.alignment.tilia.TiliaJsonLoader
    timetoalign.loader.alignment.tilia.TiliaDictStore
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from timetoalign.loader.alignment.tilia import TiliaDictStore, TiliaJsonLoader
from timetoalign.testdata import ensure_data

ensure_data("score")

# region Test data paths

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

BRUCKNER_PATH = (
    DATA_ROOT / "score" / "bruckner5_scherzo" / "harnoncourt" / "Bruckner5_Scherzo.json"
)

# endregion

# region Expected values — gold standard for zero-tolerance assertions

EXPECTED_N_TIMELINES = 7
EXPECTED_MEDIA_LENGTH = 788.0

EXPECTED_TIMELINE_KINDS = [
    "HIERARCHY_TIMELINE",
    "MARKER_TIMELINE",
    "MARKER_TIMELINE",
    "BEAT_TIMELINE",
    "MARKER_TIMELINE",
    "MARKER_TIMELINE",
    "PDF_TIMELINE",
]

EXPECTED_TIMELINE_NAMES = [
    "Form (Harnoncourt)",
    "Themen",
    "Topik",
    "Takte",
    "Tempo",
    "Dramaturgie",
    "Partitur (PDF)",
]

EXPECTED_COMPONENT_COUNTS = [33, 14, 5, 1146, 12, 11, 19]

EXPECTED_TIMELINE_IDS = [
    "cpt_1",
    "cpt_2",
    "cpt_3",
    "cpt_4",
    "cpt_5",
    "cpt_6",
    "cpt_7",
]

EXPECTED_SOURCE_TIMELINE_IDS = [
    "HIERARCHY_TIMELINE_0",
    "MARKER_TIMELINE_1",
    "MARKER_TIMELINE_2",
    "BEAT_TIMELINE_3",
    "MARKER_TIMELINE_4",
    "MARKER_TIMELINE_5",
    "PDF_TIMELINE_6",
]

# endregion


# region Fixtures


@pytest.fixture
def bruckner_loader() -> TiliaJsonLoader:
    """TiliaJsonLoader loaded with Bruckner5_Scherzo.json."""
    loader = TiliaJsonLoader()
    loader.load(BRUCKNER_PATH)
    return loader


# endregion


# region File and loading tests


class TestTiliaLoading:
    """Basic loading and structural verification."""

    def test_file_exists(self) -> None:
        assert BRUCKNER_PATH.exists(), f"Test data not found: {BRUCKNER_PATH}"

    def test_timeline_count(self, bruckner_loader: TiliaJsonLoader) -> None:
        assert len(bruckner_loader.timeline_ids) == EXPECTED_N_TIMELINES

    def test_media_length(self, bruckner_loader: TiliaJsonLoader) -> None:
        assert bruckner_loader.media_length == EXPECTED_MEDIA_LENGTH

    def test_timeline_ids(self, bruckner_loader: TiliaJsonLoader) -> None:
        assert bruckner_loader.timeline_ids == EXPECTED_TIMELINE_IDS

    def test_timeline_kinds(self, bruckner_loader: TiliaJsonLoader) -> None:
        specs = bruckner_loader.timeline_specs
        kinds = [s["kind"] for s in specs]
        assert kinds == EXPECTED_TIMELINE_KINDS

    def test_timeline_names(self, bruckner_loader: TiliaJsonLoader) -> None:
        specs = bruckner_loader.timeline_specs
        names = [s["name"] for s in specs]
        assert names == EXPECTED_TIMELINE_NAMES

    def test_component_counts(self, bruckner_loader: TiliaJsonLoader) -> None:
        specs = bruckner_loader.timeline_specs
        counts = [s["n_components"] for s in specs]
        assert counts == EXPECTED_COMPONENT_COUNTS

    def test_tables_created(self, bruckner_loader: TiliaJsonLoader) -> None:
        """Each timeline should have a corresponding pa.Table."""
        for tl_id in EXPECTED_SOURCE_TIMELINE_IDS:
            table = bruckner_loader.get_table(tl_id)
            assert isinstance(table, pa.Table)

    def test_table_row_counts_match_components(
        self, bruckner_loader: TiliaJsonLoader
    ) -> None:
        """pa.Table row count should exactly match component count."""
        for tl_id, expected_count in zip(
            EXPECTED_SOURCE_TIMELINE_IDS, EXPECTED_COMPONENT_COUNTS
        ):
            table = bruckner_loader.get_table(tl_id)
            assert (
                table.num_rows == expected_count
            ), f"Table '{tl_id}': expected {expected_count} rows, got {table.num_rows}"


# endregion


# region Store integration tests


class TestTiliaStore:
    """TiliaJsonLoader.store returns a TiliaDictStore."""

    def test_store_is_tilia_dict_store(self, bruckner_loader: TiliaJsonLoader) -> None:
        assert isinstance(bruckner_loader.store, TiliaDictStore)

    def test_store_keys_preserve_source_timeline_ids(
        self, bruckner_loader: TiliaJsonLoader
    ) -> None:
        assert set(bruckner_loader.store.keys()) == set(EXPECTED_SOURCE_TIMELINE_IDS)

    def test_store_len(self, bruckner_loader: TiliaJsonLoader) -> None:
        assert len(bruckner_loader.store) == EXPECTED_N_TIMELINES

    def test_store_getitem(self, bruckner_loader: TiliaJsonLoader) -> None:
        """Store[key] returns EventData wrapping the pa.Table."""
        from timetoalign.storage.events import EventData

        for tl_id in EXPECTED_SOURCE_TIMELINE_IDS:
            data = bruckner_loader.store[tl_id]
            assert isinstance(data, EventData)

    def test_store_hierarchy_property(self, bruckner_loader: TiliaJsonLoader) -> None:
        """store.hierarchy returns concatenation of all hierarchy tables."""
        hierarchy = bruckner_loader.store.hierarchy
        # One hierarchy timeline with 33 components
        assert hierarchy._table.num_rows == 33

    def test_store_beat_property(self, bruckner_loader: TiliaJsonLoader) -> None:
        """store.beat returns concatenation of all beat tables."""
        beat = bruckner_loader.store.beat
        # One beat timeline with 1146 components
        assert beat._table.num_rows == 1146

    def test_store_marker_property(self, bruckner_loader: TiliaJsonLoader) -> None:
        """store.marker returns concatenation of all marker tables.

        Bruckner specimen has 4 marker timelines:
        - Themen: 14 components
        - Topik: 5 components
        - Tempo: 12 components
        - Dramaturgie: 11 components
        Total: 42 components
        """
        marker = bruckner_loader.store.marker
        assert marker._table.num_rows == 42

    def test_store_pdf_property(self, bruckner_loader: TiliaJsonLoader) -> None:
        """store.pdf returns concatenation of all PDF tables."""
        pdf = bruckner_loader.store.pdf
        # One PDF timeline with 19 components
        assert pdf._table.num_rows == 19

    def test_store_harmony_property_empty(
        self, bruckner_loader: TiliaJsonLoader
    ) -> None:
        """store.harmony returns empty EventData when no harmony timelines exist."""
        harmony = bruckner_loader.store.harmony
        assert harmony._table.num_rows == 0

    def test_store_kind_map(self, bruckner_loader: TiliaJsonLoader) -> None:
        """kind_map should track all timeline types."""
        km = bruckner_loader.store.kind_map
        assert km["HIERARCHY_TIMELINE_0"] == "hierarchy"
        assert km["MARKER_TIMELINE_1"] == "marker"
        assert km["MARKER_TIMELINE_2"] == "marker"
        assert km["BEAT_TIMELINE_3"] == "beat"
        assert km["MARKER_TIMELINE_4"] == "marker"
        assert km["MARKER_TIMELINE_5"] == "marker"
        assert km["PDF_TIMELINE_6"] == "pdf"


# endregion


# region Timeline creation tests


class TestTiliaTimelineCreation:
    """Domain object creation: create_timeline, create_timelines."""

    def test_create_timeline_by_id(self, bruckner_loader: TiliaJsonLoader) -> None:
        tl = bruckner_loader.create_timeline("BEAT_TIMELINE_3")
        assert tl.id == "cpt_4"
        assert tl.name == "Takte"

    def test_create_timeline_by_name(self, bruckner_loader: TiliaJsonLoader) -> None:
        tl = bruckner_loader.create_timeline("Takte")
        assert tl.id == "cpt_4"

    def test_create_timeline_by_index(self, bruckner_loader: TiliaJsonLoader) -> None:
        tl = bruckner_loader.create_timeline(0)
        assert tl.id == "cpt_1"

    def test_create_timeline_event_counts(
        self, bruckner_loader: TiliaJsonLoader
    ) -> None:
        """Each timeline should have exactly the expected number of events."""
        for tl_id, expected_count in zip(
            EXPECTED_SOURCE_TIMELINE_IDS, EXPECTED_COMPONENT_COUNTS
        ):
            tl = bruckner_loader.create_timeline(tl_id)
            assert tl.n_events == expected_count, (
                f"Timeline '{tl_id}': expected {expected_count} events, "
                f"got {tl.n_events}"
            )

    def test_create_timelines_all(self, bruckner_loader: TiliaJsonLoader) -> None:
        timelines = bruckner_loader.create_timelines()
        assert len(timelines) == EXPECTED_N_TIMELINES

    def test_create_timelines_subset(self, bruckner_loader: TiliaJsonLoader) -> None:
        subset_ids = ["BEAT_TIMELINE_3", "HIERARCHY_TIMELINE_0"]
        timelines = bruckner_loader.create_timelines(uids=subset_ids)
        assert len(timelines) == 2
        assert timelines[0].id == "cpt_4"
        assert timelines[1].id == "cpt_1"

    def test_timeline_length_matches_media(
        self, bruckner_loader: TiliaJsonLoader
    ) -> None:
        """All timelines should use media_length as their extent."""
        for tl_id in EXPECTED_SOURCE_TIMELINE_IDS:
            tl = bruckner_loader.create_timeline(tl_id)
            assert float(tl.length.value) == EXPECTED_MEDIA_LENGTH, (
                f"Timeline '{tl_id}': expected length {EXPECTED_MEDIA_LENGTH}, "
                f"got {tl.length.value}"
            )

    def test_timeline_unit_is_seconds(self, bruckner_loader: TiliaJsonLoader) -> None:
        """All timelines should use seconds as their unit."""
        from timetoalign.core import TimeUnit

        for tl_id in EXPECTED_SOURCE_TIMELINE_IDS:
            tl = bruckner_loader.create_timeline(tl_id)
            assert tl.unit == TimeUnit.seconds

    def test_timeline_caching(self, bruckner_loader: TiliaJsonLoader) -> None:
        """Same id should return the same object (cached)."""
        tl1 = bruckner_loader.create_timeline("BEAT_TIMELINE_3")
        tl2 = bruckner_loader.create_timeline("BEAT_TIMELINE_3")
        assert tl1 is tl2

    def test_create_timeline_missing_raises(
        self, bruckner_loader: TiliaJsonLoader
    ) -> None:
        with pytest.raises(KeyError, match="No timeline with uid"):
            bruckner_loader.create_timeline("NONEXISTENT_99")


# endregion


# region Group creation tests


class TestTiliaGroupCreation:
    """create_group() produces a TimelineGroup with all timelines."""

    def test_create_group(self, bruckner_loader: TiliaJsonLoader) -> None:
        group = bruckner_loader.create_group()
        assert group.n_timelines == EXPECTED_N_TIMELINES

    def test_group_timeline_ids(self, bruckner_loader: TiliaJsonLoader) -> None:
        group = bruckner_loader.create_group()
        assert sorted(group.timeline_ids) == sorted(EXPECTED_TIMELINE_IDS)

    def test_group_has_name(self, bruckner_loader: TiliaJsonLoader) -> None:
        group = bruckner_loader.create_group()
        assert group.name == "Bruckner5_Scherzo"

    def test_group_subset(self, bruckner_loader: TiliaJsonLoader) -> None:
        subset = ["BEAT_TIMELINE_3", "HIERARCHY_TIMELINE_0"]
        group = bruckner_loader.create_group(uids=subset)
        assert group.n_timelines == 2


# endregion


# region AlignmentBundle creation tests


class TestTiliaAlignmentBundle:
    """create_bundle() wraps group in a bundle."""

    def test_create_bundle(self, bruckner_loader: TiliaJsonLoader) -> None:
        bundle = bruckner_loader.create_bundle()
        assert bundle.n_timelines == EXPECTED_N_TIMELINES

    def test_bundle_has_one_group(self, bruckner_loader: TiliaJsonLoader) -> None:
        bundle = bruckner_loader.create_bundle()
        assert bundle.n_groups == 1

    def test_bundle_has_no_claims(self, bruckner_loader: TiliaJsonLoader) -> None:
        bundle = bruckner_loader.create_bundle()
        assert len(bundle.cross_group_claims) == 0


# endregion


# region Error handling


class TestTiliaErrorHandling:
    """Error handling for TiliaJsonLoader."""

    def test_create_before_load_raises(self) -> None:
        loader = TiliaJsonLoader()
        with pytest.raises(RuntimeError, match="No data loaded"):
            loader.create_timeline("any")

    def test_create_group_before_load_raises(self) -> None:
        loader = TiliaJsonLoader()
        with pytest.raises(RuntimeError, match="No data loaded"):
            loader.create_group()

    def test_create_bundle_before_load_raises(self) -> None:
        loader = TiliaJsonLoader()
        with pytest.raises(RuntimeError, match="No data loaded"):
            loader.create_bundle()

    def test_repr_before_load(self) -> None:
        loader = TiliaJsonLoader()
        assert "not loaded" in repr(loader)

    def test_repr_after_load(self, bruckner_loader: TiliaJsonLoader) -> None:
        r = repr(bruckner_loader)
        assert "TiliaJsonLoader(" in r
        assert "cpt_4" in r


# endregion
