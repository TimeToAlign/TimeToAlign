"""Tests for PerformancePrecisionLoader — CAAMP audio-to-score alignments.

This module tests ``PerformancePrecisionLoader`` against the Chopin
Nocturne Op. 9 No. 2 specimen in the ``performance_precision`` corpus: a
``.solo`` score, a Verovio timemap ``.json``, and an ``Alignments/``
directory holding three CSVs (note / bar / beat) for each of 7
recordings.

It verifies:

- the composed ``SoloLoader`` (2494 events),
- the ``MetricMap`` built from the timemap (38 measures, length 212.5),
- the measure+offset → absolute-quarter resolver (validated anchors),
- the score timeline (2494 events, quarters, pickup note at 0),
- per-performer MatchClaim counts (559 note = 480 sync + 79 NOMATCH;
  32 bar; 376 beat) and the bundle totals (8 timelines, 6769 claims).

All counts and coordinates are exact per the Zero Tolerance Validation
Policy.  Validation logic is documented in ``tests/loader/README.md``.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign.alignment.claims import MatchClaim
from timetoalign.core import TimeUnit
from timetoalign.loader.alignment import PerformancePrecisionLoader
from timetoalign.loader.tabular.solo import SoloLoader
from timetoalign.maps.meter import MetricalPositionMap, MetricMap
from timetoalign.testdata import ensure_data

SPECIMEN_DIR = ensure_data("performance_precision")
SOLO_PATH = SPECIMEN_DIR / "Chopin Nocturne Op. 9 No. 2.solo"

PERFORMERS = [
    "Chopin_Ashkenazy",
    "Chopin_Barenboim",
    "Chopin_Freire",
    "Chopin_Horowitz",
    "Chopin_Pollini",
    "Chopin_Rachmaninoff",
    "Chopin_Rubinstein",
]

SCORE_TL_ID = "score:clt_1"


@pytest.fixture
def loader() -> PerformancePrecisionLoader:
    return PerformancePrecisionLoader.from_file(SPECIMEN_DIR)


@pytest.fixture
def bundle(loader: PerformancePrecisionLoader):
    return loader.create_bundle()


# region Composed SoloLoader


def test_composed_solo_loader_event_count() -> None:
    """The internally-composed SoloLoader yields 2494 events."""
    solo = SoloLoader.from_file(SOLO_PATH)
    assert len(solo.events) == 2494


# endregion


# region MetricMap


def test_metric_map(loader: PerformancePrecisionLoader) -> None:
    mm = loader.metric_map
    assert isinstance(mm, MetricMap)
    assert mm.n_measures == 38
    assert mm.total_length == Fraction(425, 2)
    assert mm._starts_frac[0] == Fraction(0)
    assert mm._starts_frac[1] == Fraction(1, 2)
    assert mm._mns[0] == "1"
    assert mm._mns[-1] == "38"
    assert list(mm._mcs) == list(range(1, 39))


def test_metrical_position_map_exposed(loader: PerformancePrecisionLoader) -> None:
    assert isinstance(loader.metrical_position_map, MetricalPositionMap)


# endregion


# region Label resolver


@pytest.mark.parametrize(
    "label,expected",
    [
        ("0+11/8", Fraction(0)),
        ("1+0/1", Fraction(1, 2)),
        ("2+0/1", Fraction(13, 2)),
        ("3+0/1", Fraction(25, 2)),
        ("32+3/4", Fraction(379, 2)),
        ("37+3/2", Fraction(425, 2)),
    ],
)
def test_resolver_anchors(
    loader: PerformancePrecisionLoader, label: str, expected: Fraction
) -> None:
    """Validated measure+offset → absolute-quarter anchors."""
    timemap = SPECIMEN_DIR / "Chopin Nocturne Op. 9 No. 2.json"
    first_meter = loader._first_meter_quarters(timemap)
    starts, m0 = loader._build_measure_lookup(loader.metric_map, first_meter)

    measure_str, offset_str = label.split("+")
    numerator, denominator = offset_str.split("/")
    offset_wn = Fraction(int(numerator), int(denominator))
    got = loader._resolve_label(int(measure_str), offset_wn, starts, m0)
    assert got == expected


def test_anacrusis_virtual_downbeat(loader: PerformancePrecisionLoader) -> None:
    """LABEL measure 0's virtual downbeat sits at -5.5 quarters."""
    timemap = SPECIMEN_DIR / "Chopin Nocturne Op. 9 No. 2.json"
    first_meter = loader._first_meter_quarters(timemap)
    _starts, m0 = loader._build_measure_lookup(loader.metric_map, first_meter)
    assert m0 == Fraction(-11, 2)


# endregion


# region Score timeline


def test_score_timeline_basics(loader: PerformancePrecisionLoader) -> None:
    score = loader.create_timeline("score")
    assert len(score.events) == 2494
    assert score.unit == TimeUnit.quarters
    assert score.length.value == 212.5


def test_score_pickup_note_at_zero(loader: PerformancePrecisionLoader) -> None:
    """The pickup note (note_id 'n1b8xktz') resolves to quarter 0."""
    score = loader.create_timeline("score")
    table = score.events._table
    note_ids = table.column("note_id").to_pylist()
    starts = table.column("start").to_pylist()
    idx = note_ids.index("n1b8xktz")
    assert starts[idx]["value"] == 0.0
    assert starts[idx]["numerator"] == 0


def test_score_first_full_bar_note(loader: PerformancePrecisionLoader) -> None:
    """A 1+0/1 note resolves to quarter 1/2 (0.5)."""
    score = loader.create_timeline("score")
    starts = score.events._table.column("start").to_pylist()
    values = {s["value"] for s in starts}
    assert 0.5 in values


def test_score_carries_pitch(loader: PerformancePrecisionLoader) -> None:
    """The EnharmonicPitch view is carried through from the SoloLoader.

    A ``.solo`` file is number-only, so its most-expressive faithful
    pitch type is EnharmonicPitch, carried as a ``{midi_number}`` struct.
    The loader keeps that view on the score timeline (rather than
    degrading it to a raw int), and the keystone preserves it as a real
    struct column.  The pickup note's MIDI pitch is 70 in the ``.solo``
    file.
    """
    score = loader.create_timeline("score")
    table = score.events._table
    note_ids = table.column("note_id").to_pylist()
    pitches = table.column("pitch").to_pylist()
    idx = note_ids.index("n1b8xktz")
    assert pitches[idx] == {"midi_number": 70}


# endregion


# region Performer timelines & claim counts


def test_bundle_timeline_count(bundle) -> None:
    """1 score group + 7 standalone performers = 8 timelines."""
    assert len(bundle.timelines) == 8


def test_total_claim_counts(bundle) -> None:
    claims = bundle.cross_group_claims
    assert len(claims) == 6769
    sync = sum(1 for c in claims if c.is_synchronous)
    nomatch = sum(1 for c in claims if not c.is_synchronous)
    assert sync == 6216
    assert nomatch == 553


def _performer_claims(bundle, performer_key: str) -> list[MatchClaim]:
    perf_tl_id = f"perf:{performer_key}:cpt_1"
    return [c for c in bundle.cross_group_claims if c.timeline_b_id == perf_tl_id]


def _by_granularity(claims: list[MatchClaim], granularity: str) -> list[MatchClaim]:
    return [c for c in claims if c.metadata.agent.identifier == granularity]


@pytest.mark.parametrize("performer_key", PERFORMERS)
def test_per_performer_counts(bundle, performer_key: str) -> None:
    """Every performer: note 559 (480 sync + 79 N), bar 32, beat 376."""
    claims = _performer_claims(bundle, performer_key)

    note = _by_granularity(claims, "note")
    bar = _by_granularity(claims, "bar")
    beats = _by_granularity(claims, "beats")

    assert len(note) == 559
    assert sum(1 for c in note if c.is_synchronous) == 480
    assert sum(1 for c in note if not c.is_synchronous) == 79

    assert len(bar) == 32
    assert all(c.is_synchronous for c in bar)

    assert len(beats) == 376
    assert all(c.is_synchronous for c in beats)


def test_performer_physical_timeline(loader: PerformancePrecisionLoader) -> None:
    """Each performer timeline holds one event per aligned note row (480)."""
    perf = loader.create_timeline("Chopin_Ashkenazy")
    assert perf.unit == TimeUnit.seconds
    assert len(perf.events) == 480


def test_claim_metadata(bundle) -> None:
    """Claims carry the documented agent (name + per-granularity identifier)."""
    claims = _performer_claims(bundle, "Chopin_Ashkenazy")
    sample = claims[0]
    assert sample.metadata.agent.name == "performance_precision"
    assert sample.metadata.certainty == 1.0
    assert sample.metadata.agent.identifier in {"note", "bar", "beats"}


# endregion


# region Coordinate spot-check


def test_ashkenazy_bar_spot_check(bundle) -> None:
    """Ashkenazy bar claim for LABEL 1+0/1: score 0.5 quarters, 10.272 s."""
    bar_claims = _by_granularity(_performer_claims(bundle, "Chopin_Ashkenazy"), "bar")
    synchronous = [c for c in bar_claims if c.is_synchronous]
    # The first bar row of the file is LABEL 1+0/1 (the first downbeat).
    first = synchronous[0]
    assert first.start_anchor.coordinate_a.value == Fraction(1, 2)
    assert first.start_anchor.coordinate_a.unit is TimeUnit.quarters
    assert first.start_anchor.coordinate_b.value == 10.272
    assert first.start_anchor.coordinate_b.unit is TimeUnit.seconds


# endregion


# region Loader API


def test_create_timelines_returns_eight(loader: PerformancePrecisionLoader) -> None:
    timelines = loader.create_timelines()
    assert len(timelines) == 8
    assert timelines[0].id == SCORE_TL_ID


def test_create_timelines_id_pattern_filters(
    loader: PerformancePrecisionLoader,
) -> None:
    """The pinned 8-timeline bundle has 7 performer timelines."""
    assert len(loader.create_timelines(id_pattern=r"^score:")) == 1
    assert len(loader.create_timelines(id_pattern=r"^perf:")) == 7


def test_create_timelines_id_pattern_no_match(
    loader: PerformancePrecisionLoader,
) -> None:
    """An unmatched ID pattern returns no timelines."""
    assert loader.create_timelines(id_pattern=r"^missing:") == []


def test_create_timeline_by_score_role(loader: PerformancePrecisionLoader) -> None:
    assert loader.create_timeline("score").id == SCORE_TL_ID


def test_create_timeline_by_performer_uid(
    loader: PerformancePrecisionLoader,
) -> None:
    tl = loader.create_timeline("perf:Chopin_Ashkenazy:cpt_1")
    assert tl.id == "perf:Chopin_Ashkenazy:cpt_1"
    assert tl.name == "Chopin_Ashkenazy"


def test_create_timeline_unknown_raises(
    loader: PerformancePrecisionLoader,
) -> None:
    with pytest.raises(KeyError):
        loader.create_timeline("does_not_exist")


def test_load_source_not_used() -> None:
    loader = PerformancePrecisionLoader()
    with pytest.raises(NotImplementedError):
        loader._load_source(SOLO_PATH)


# endregion
