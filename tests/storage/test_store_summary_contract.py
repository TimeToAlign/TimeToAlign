"""Contract tests for summaries and notebook representations of stores."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from timetoalign.loader.format.json import JsonLoader
from timetoalign.loader.graphical import GraphicalLoader
from timetoalign.loader.midi import PerformanceMidiLoader
from timetoalign.loader.score.partitura import PartituraLoader
from timetoalign.loader.tabular import LabLoader
from timetoalign.storage.store import AlignmentStore, EventStore
from timetoalign.testdata import ensure_data

STORE_NAMES = ("single", "dict", "score", "midi", "graphical", "alignment")


@pytest.fixture(scope="module")
def loaded_stores() -> Mapping[str, tuple[EventStore, dict[str, Any]]]:
    """Load representative stores and their exact expected summaries."""
    fixtures_dir = ensure_data("fixtures")
    lab_loader = LabLoader()
    lab_loader.load(fixtures_dir / "lab" / "regions.lab")
    single_store = lab_loader.store

    audio_dir = ensure_data("audio")
    json_loader = JsonLoader(principal_keys=["audio"])
    json_loader.load(audio_dir / "hard_techno" / "dj_studio_data.json")
    dict_store = json_loader.store

    vienna_dir = ensure_data("vienna_1x22")
    score_path = vienna_dir / "Chopin_op10_no3.musicxml"
    score_loader = PartituraLoader()
    score_loader.load(score_path)
    score_store = score_loader.store

    midi_loader = PerformanceMidiLoader()
    midi_loader.load(vienna_dir / "Chopin_op10_no3_p01.mid")
    midi_store = midi_loader.store

    thoresen_dir = ensure_data("thoresen")
    graphical_loader = GraphicalLoader(metadata={"source": "Thoresen 2009"})
    source_index = graphical_loader.add_image(
        thoresen_dir / "thoresen_2009_sound-objects_p312_page1_1.jpeg"
    )
    graphical_loader.add_horizontal_segment(
        source_index,
        x0=2,
        x1=969,
        y=18,
        name="system_1",
    )
    graphical_store = graphical_loader.store

    alignment_store = AlignmentStore.empty()

    return {
        "single": (
            single_store,
            {
                "tables": {
                    "events": {
                        "unit": "seconds",
                        "count": 6,
                        "range": (0.0, 14.0),
                    }
                },
                "facts": {},
            },
        ),
        "dict": (
            dict_store,
            {
                "tables": {
                    "audio": {
                        "unit": "seconds",
                        "count": 3,
                        "range": (0.0, 0.0),
                    }
                },
                "facts": {},
            },
        ),
        "score": (
            score_store,
            {
                "tables": {
                    "notes": {
                        "unit": "quarters",
                        "count": 498,
                        "range": (0.0, 40.5),
                    },
                    "measures": {
                        "unit": "quarters",
                        "count": 22,
                        "range": (0.0, 40.5),
                    },
                    "controls": {
                        "unit": "quarters",
                        "count": 22,
                        "range": (0.0, 40.0),
                    },
                    "annotations": {
                        "unit": "quarters",
                        "count": 5,
                        "range": (0.0, 32.5),
                    },
                },
                "facts": {
                    "has_rests": False,
                    "format": "score",
                    "parser": "partitura",
                    "source": str(score_path),
                    "anacrusis_offset": 0.5,
                },
            },
        ),
        "midi": (
            midi_store,
            {
                "tables": {
                    "notes": {
                        "unit": "ticks",
                        "count": 451,
                        "range": (0.0, 78610.0),
                    },
                    "controls": {
                        "unit": "ticks",
                        "count": 3423,
                        "range": (0.0, 83034.0),
                    },
                },
                "facts": midi_store.metadata,
            },
        ),
        "graphical": (
            graphical_store,
            {
                "tables": {},
                "facts": {
                    "n_sources": 1,
                    "n_segments": 1,
                    "total_length": 967.0,
                    "source": "Thoresen 2009",
                },
            },
        ),
        "alignment": (
            alignment_store,
            {
                "tables": {
                    "events": {
                        "unit": "seconds",
                        "count": 0,
                        "range": (0.0, 0.0),
                    }
                },
                "facts": {
                    "event_count": 0,
                    "cmap_count": 0,
                    "domains": ["physical"],
                    "store_names": ["events"],
                },
            },
        ),
    }


@pytest.mark.parametrize("store_name", STORE_NAMES)
def test_loaded_store_html_is_non_empty(
    loaded_stores: Mapping[str, tuple[EventStore, dict[str, Any]]],
    store_name: str,
) -> None:
    """Every store exposed by a loader has a valid notebook representation."""
    store, _ = loaded_stores[store_name]

    html = store._repr_html_()

    assert isinstance(html, str)
    assert html
    assert "<table" in html


@pytest.mark.parametrize("store_name", STORE_NAMES)
def test_loaded_store_summary_has_exact_contract(
    loaded_stores: Mapping[str, tuple[EventStore, dict[str, Any]]],
    store_name: str,
) -> None:
    """Every store summary has the exact shared structure and values."""
    store, expected = loaded_stores[store_name]

    summary = store.summary()

    assert summary == expected
    assert tuple(summary) == ("tables", "facts")
    assert {
        name: tuple(table_summary) for name, table_summary in summary["tables"].items()
    } == dict.fromkeys(expected["tables"], ("unit", "count", "range"))


def test_partitura_score_store_html_regression() -> None:
    """A Partitura-loaded Vienna score renders as HTML without raising."""
    data_dir = ensure_data("vienna_1x22")
    loader = PartituraLoader()
    loader.load(data_dir / "Chopin_op10_no3.musicxml")

    html = loader.store._repr_html_()

    assert isinstance(html, str)
    assert html
    assert "<table" in html


def test_store_level_facts_remain_reachable(
    loaded_stores: Mapping[str, tuple[EventStore, dict[str, Any]]],
) -> None:
    """Specialized summaries retain every previously reported fact."""
    score_store, score_expected = loaded_stores["score"]
    midi_store, midi_expected = loaded_stores["midi"]
    graphical_store, graphical_expected = loaded_stores["graphical"]
    alignment_store, alignment_expected = loaded_stores["alignment"]

    assert score_store.summary()["facts"] == score_expected["facts"]
    assert midi_store.summary()["facts"] == midi_expected["facts"]
    assert graphical_store.summary()["facts"] == graphical_expected["facts"]
    assert alignment_store.summary()["facts"] == alignment_expected["facts"]
