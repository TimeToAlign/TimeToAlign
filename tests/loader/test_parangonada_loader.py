"""Tests for ParangonadaLoader — parangonada CSV export → AlignmentBundle.

This module tests ``ParangonadaLoader`` against the parangonada export of
the ``Beethoven_Eroica_op35-cpjku`` dataset in the ``parangonar`` corpus:
five performances of Beethoven's Eroica Variations op. 35 (Var. XIV),
each contributing a ``part.csv`` / ``ppart.csv`` / ``align.csv`` triple
under ``match/match_transkun/``.

It verifies:

- discovery of exactly 5 performers from non-uniformly-suffixed
  directories (only two carry ``_parangonada``);
- the shared score timelines (``score:clt_1`` / ``score:dlt_1``, 251 notes
  each) and the divs→quarters ``LinearMap`` reproducing every
  ``onset_quarter`` exactly;
- per-performer performance-timeline event counts and MatchClaim counts;
- the bundle totals (12 timelines, 6 groups, 1275 claims);
- the measured ``Beat`` / ``Dynamics`` feature events on each
  performance's seconds timeline (63 of each per performer), including
  the ``.dyn`` ↔ ``.beats`` 1:1 onset join and feature spot-checks;
- a coordinate spot-check (Szegedi ``align`` row 0);
- the faithfully-preserved duplicate matchtype-0 rows in Brendel and
  Hewitt; and
- the ``SamplesToSeconds`` C-Map round-trip.

All counts and coordinates are exact per the Zero Tolerance Validation
Policy.  Validation logic is documented in ``tests/loader/README.md``.
"""

from __future__ import annotations

import csv
from fractions import Fraction
from pathlib import Path

import pytest

from timetoalign.alignment.claims import MatchClaim
from timetoalign.core import TimeUnit
from timetoalign.loader.alignment import ParangonadaLoader
from timetoalign.maps.convenience import SamplesToSeconds
from timetoalign.testdata import ensure_data

DATASET_DIR = ensure_data("parangonar") / "Beethoven_Eroica_op35-cpjku"
MATCH_DIR = DATASET_DIR / "match" / "match_transkun"
FEATURES_DIR = DATASET_DIR / "features"

#: Each performer's .beats / .dyn files hold exactly this many data rows.
FEATURE_ROW_COUNT = 63

SCORE_CLT_ID = "score:clt_1"
SCORE_DLT_ID = "score:dlt_1"

# Performers in chronological (sorted-key) order, with their on-disk
# subdirectory names — note only two carry the ``_parangonada`` suffix.
PERFORMERS = [
    ("1966_Szegedi", "1966_Szegedi_parangonada"),
    ("1970_Gould", "1970_Gould_parangonada"),
    ("1971_Curzon", "1971_Curzon"),
    ("1985_Brendel", "1985_Brendel"),
    ("2023_Hewitt", "2023_Hewitt"),
]
PERFORMER_KEYS = [key for key, _ in PERFORMERS]

# Zero-tolerance per-performer counts: ppart events, align total, mt0
# (synchronous), mt1+mt2 (NOMATCH).
PERFORMER_COUNTS = {
    "1966_Szegedi": {"ppart": 232, "total": 256, "sync": 227, "nomatch": 29},
    "1970_Gould": {"ppart": 253, "total": 257, "sync": 247, "nomatch": 10},
    "1971_Curzon": {"ppart": 246, "total": 255, "sync": 242, "nomatch": 13},
    "1985_Brendel": {"ppart": 249, "total": 253, "sync": 249, "nomatch": 4},
    "2023_Hewitt": {"ppart": 244, "total": 254, "sync": 243, "nomatch": 11},
}


@pytest.fixture
def loader() -> ParangonadaLoader:
    return ParangonadaLoader.from_file(DATASET_DIR)


@pytest.fixture
def bundle(loader: ParangonadaLoader):
    return loader.create_bundle()


def _subdir(performer_key: str) -> Path:
    name = dict(PERFORMERS)[performer_key]
    return MATCH_DIR / name


def _feature_file(performer_key: str, suffix: str) -> Path:
    """The single ``<key>_measure*.<suffix>`` feature file (globbed)."""
    matches = sorted(FEATURES_DIR.glob(f"{performer_key}_measure*.{suffix}"))
    assert len(matches) == 1, f"expected one {suffix} file, found {matches}"
    return matches[0]


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _performer_claims(bundle, performer_key: str) -> list[MatchClaim]:
    """Claims touching this performer's cpt_1 timeline (either orientation)."""
    cpt_id = f"perf:{performer_key}:cpt_1"
    return [c for c in bundle.cross_group_claims if c.connects(cpt_id)]


# region Bundle shape


def test_bundle_timeline_count(bundle) -> None:
    """2 score timelines + 5 performers x 2 = 12 timelines."""
    assert bundle.n_timelines == 12


def test_bundle_group_count(bundle) -> None:
    """1 score group + 5 performer groups = 6 groups."""
    assert bundle.n_groups == 6


def test_bundle_group_ids(bundle) -> None:
    expected = {"score"} | {f"perf:{key}" for key in PERFORMER_KEYS}
    assert set(bundle.group_ids) == expected


def test_discovers_five_performers(loader: ParangonadaLoader) -> None:
    """Exactly 5 performers, keyed and sorted chronologically."""
    assert loader.performer_keys == PERFORMER_KEYS


def test_repr_html_shows_claims_not_zero_events(loader: ParangonadaLoader) -> None:
    """The HTML card reports the real claim count, never the base Events: 0."""
    html = loader._repr_html_()
    n_claims = len(loader)
    assert f"<tr><td><b>Claims</b></td><td>{n_claims}</td></tr>" in html
    assert "Events" not in html
    assert "<tr><td><b>Performers</b></td><td>5</td></tr>" in html
    assert "<b>Dataset</b>" in html
    assert "in 6 group(s)" in html
    assert "<b>Performer keys</b>" in html


# endregion


# region Score timelines


def test_score_clt_event_count(loader: ParangonadaLoader) -> None:
    score = loader.create_timeline(SCORE_CLT_ID)
    assert len(score.events) == 251
    assert score.unit == TimeUnit.quarters


def test_score_dlt_event_count(loader: ParangonadaLoader) -> None:
    score = loader.create_timeline(SCORE_DLT_ID)
    assert len(score.events) == 251
    assert score.unit == TimeUnit.ticks


def test_score_clt_carries_pitch_and_voice(loader: ParangonadaLoader) -> None:
    """The first part.csv note (id 'nwqgcz5') carries MIDI pitch 63, voice 3.

    A number-only source represents pitch as its most-expressive faithful
    type, EnharmonicPitch, carried as a ``{midi_number}`` struct; the
    keystone preserves it as a real struct column (never a JSON string).
    """
    score = loader.create_timeline(SCORE_CLT_ID)
    table = score.events._table
    ids = table.column("id").to_pylist()
    pitches = table.column("pitch").to_pylist()
    voices = table.column("voice").to_pylist()
    idx = ids.index("nwqgcz5")
    assert pitches[idx] == {"midi_number": 63}
    assert int(voices[idx]) == 3


def test_divs_to_quarters_cmap_reproduces_all_onsets(
    loader: ParangonadaLoader,
) -> None:
    """The divs→quarters LinearMap reproduces every onset_quarter exactly.

    Asserts exact Fraction equality for all 251 notes: quarters =
    div / 32 - 1/2.
    """
    dlt = loader.create_timeline(SCORE_DLT_ID)
    cmap = dlt.get_conversion_map(TimeUnit.quarters)
    with open(_subdir("1971_Curzon") / "part.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 251
    for row in rows:
        onset_div = int(row["onset_div"])
        expected = Fraction(str(row["onset_quarter"]))
        assert cmap(onset_div) == expected


# endregion


# region Performance timelines & claim counts


@pytest.mark.parametrize("performer_key", PERFORMER_KEYS)
def test_performance_timeline_event_counts(
    loader: ParangonadaLoader, performer_key: str
) -> None:
    """cpt_1 and dpt_1 each hold one *Note* event per ppart.csv row.

    ``cpt_1`` additionally holds Beat/Dynamics feature events, so the Note
    count is filtered explicitly; ``dpt_1`` carries notes only.
    """
    expected = PERFORMER_COUNTS[performer_key]["ppart"]
    cpt = loader.create_timeline(f"perf:{performer_key}:cpt_1")
    dpt = loader.create_timeline(f"perf:{performer_key}:dpt_1")
    assert cpt.unit == TimeUnit.seconds
    assert dpt.unit == TimeUnit.samples
    assert len(cpt.events.filter(event_type="Note")) == expected
    assert len(dpt.events) == expected


@pytest.mark.parametrize("performer_key", PERFORMER_KEYS)
def test_per_performer_claim_counts(bundle, performer_key: str) -> None:
    """Per performer: total / synchronous (mt0) / NOMATCH (mt1+mt2)."""
    counts = PERFORMER_COUNTS[performer_key]
    claims = _performer_claims(bundle, performer_key)

    assert len(claims) == counts["total"]
    assert sum(1 for c in claims if c.is_synchronous) == counts["sync"]
    assert sum(1 for c in claims if not c.is_synchronous) == counts["nomatch"]


def test_total_claim_counts(bundle) -> None:
    """Totals across all 5 performers: 1275 = 1208 sync + 67 NOMATCH."""
    claims = bundle.cross_group_claims
    assert len(claims) == 1275
    assert sum(1 for c in claims if c.is_synchronous) == 1208
    assert sum(1 for c in claims if not c.is_synchronous) == 67


def test_claim_metadata(bundle) -> None:
    """Claims carry the documented agent (name + per-performer identifier)."""
    claims = _performer_claims(bundle, "1966_Szegedi")
    sample = claims[0]
    assert sample.metadata.agent.name == "parangonada"
    assert sample.metadata.certainty == 1.0
    assert sample.metadata.agent.identifier == "1966_Szegedi"


# endregion


# region Feature events (.beats / .dyn)


@pytest.mark.parametrize("performer_key", PERFORMER_KEYS)
def test_feature_event_counts(loader: ParangonadaLoader, performer_key: str) -> None:
    """Each cpt_1 holds exactly 63 Beat and 63 Dynamics feature events."""
    cpt = loader.create_timeline(f"perf:{performer_key}:cpt_1")
    assert len(cpt.events.filter(event_type="Beat")) == FEATURE_ROW_COUNT
    assert len(cpt.events.filter(event_type="Dynamics")) == FEATURE_ROW_COUNT


@pytest.mark.parametrize("performer_key", PERFORMER_KEYS)
def test_feature_events_only_on_cpt(
    loader: ParangonadaLoader, performer_key: str
) -> None:
    """Beat/Dynamics events live on cpt_1 (seconds) only, never on dpt_1."""
    dpt = loader.create_timeline(f"perf:{performer_key}:dpt_1")
    assert len(dpt.events.filter(event_type="Beat")) == 0
    assert len(dpt.events.filter(event_type="Dynamics")) == 0


@pytest.mark.parametrize("performer_key", PERFORMER_KEYS)
def test_beats_dyn_join_is_one_to_one(performer_key: str) -> None:
    """The .beats ↔ .dyn join on (measure_number, beat) is 1:1 with 63 matches.

    Asserts the keys are unique within each file and the two key sets are
    identical, so every .dyn row recovers exactly one .beats onset.
    """
    beats = _read_tsv(_feature_file(performer_key, "beats"))
    dyn = _read_tsv(_feature_file(performer_key, "dyn"))
    assert len(beats) == FEATURE_ROW_COUNT
    assert len(dyn) == FEATURE_ROW_COUNT

    beat_keys = [(r["measure_number"], r["beat"]) for r in beats]
    dyn_keys = [(r["measure_number"], r["beat"]) for r in dyn]
    assert len(set(beat_keys)) == FEATURE_ROW_COUNT
    assert len(set(dyn_keys)) == FEATURE_ROW_COUNT
    assert set(beat_keys) == set(dyn_keys)

    matched = sum(1 for k in dyn_keys if k in set(beat_keys))
    assert matched == FEATURE_ROW_COUNT


def test_szegedi_beat_spot_check(loader: ParangonadaLoader) -> None:
    """Szegedi .beats row 0 → a Beat event at 0.796354 s (measure 1, beat 1).

    Row 0 is ``1  1  0.000000  0.796354  83.660126  0  0``.  Carried
    scalar columns keep their native PyArrow types (``measure_number`` /
    ``beat`` as ints, ``bpm`` as a float); ``start`` is the coordinate
    struct.
    """
    cpt = loader.create_timeline("perf:1966_Szegedi:cpt_1")
    beats = cpt.events.filter(event_type="Beat").table.to_pylist()
    matching = [
        r
        for r in beats
        if r["start"]["value"] == 0.796354
        and r["measure_number"] == 1
        and r["beat"] == 1
        and r["bpm"] == 83.660126
    ]
    assert len(matching) == 1


def test_szegedi_dynamics_spot_check(loader: ParangonadaLoader) -> None:
    """Szegedi .dyn row 0 → a Dynamics event at 0.796354 s (onset joined).

    Row 0 is ``1  1  0.000000  48.500000  58.000000  0``; the onset is
    recovered from the .beats row with the same ``(1, 1)`` key.  Carried
    scalar columns keep their native PyArrow types.
    """
    cpt = loader.create_timeline("perf:1966_Szegedi:cpt_1")
    dyns = cpt.events.filter(event_type="Dynamics").table.to_pylist()
    matching = [
        r
        for r in dyns
        if r["start"]["value"] == 0.796354
        and r["measure_number"] == 1
        and r["beat"] == 1
        and r["velocity_mean"] == 48.5
        and r["velocity_max"] == 58.0
    ]
    assert len(matching) == 1


def test_feature_events_preserve_bundle_totals(bundle) -> None:
    """Feature events do not change the bundle's claim/timeline/group totals."""
    assert bundle.n_timelines == 12
    assert bundle.n_groups == 6
    claims = bundle.cross_group_claims
    assert len(claims) == 1275
    assert sum(1 for c in claims if c.is_synchronous) == 1208
    assert sum(1 for c in claims if not c.is_synchronous) == 67


# endregion


# region Coordinate spot-check


def test_szegedi_align_row_zero_spot_check(bundle) -> None:
    """Szegedi align idx 0 ('0,0,ngx1f26,n149') → synchronous claim.

    Source coordinate equals score_q_by_id['ngx1f26'] (40.0 quarters);
    target coordinate equals perf_sec_by_id['n149'] (38.183334 s).
    """
    cpt_id = "perf:1966_Szegedi:cpt_1"
    matching = [
        c
        for c in bundle.cross_group_claims
        if c.is_synchronous
        and c.start_anchor is not None
        and c.start_anchor.timeline_a_id == SCORE_CLT_ID
        and c.start_anchor.timeline_b_id == cpt_id
        and c.start_anchor.coordinate_a.value == 40.0
        and c.start_anchor.coordinate_a.unit is TimeUnit.quarters
        and c.start_anchor.coordinate_b.value == 38.183334
        and c.start_anchor.coordinate_b.unit is TimeUnit.seconds
    ]
    assert len(matching) == 1


# endregion


# region Known data quirk: duplicated matchtype-0 rows


@pytest.mark.parametrize(
    "performer_key,partid,ppartid,idx_a,idx_b",
    [
        ("1985_Brendel", "n1qocvw7", "n65", "74", "75"),
        ("2023_Hewitt", "n1qocvw7", "n64", "71", "72"),
    ],
)
def test_duplicate_matchtype0_row_is_present_in_source(
    performer_key: str,
    partid: str,
    ppartid: str,
    idx_a: str,
    idx_b: str,
) -> None:
    """Brendel and Hewitt each carry one exactly-duplicated mt0 align row.

    This is data, not a bug; the loader preserves it (no deduplication).
    The test pins that the duplicate is present in the raw ``align.csv``.
    """
    with open(_subdir(performer_key) / "align.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    dup = [
        r
        for r in rows
        if r["matchtype"] == "0" and r["partid"] == partid and r["ppartid"] == ppartid
    ]
    assert len(dup) == 2
    assert {r["idx"] for r in dup} == {idx_a, idx_b}


def test_duplicate_row_yields_duplicate_claim(bundle) -> None:
    """The duplicated Brendel row produces two identical synchronous claims."""
    cpt_id = "perf:1985_Brendel:cpt_1"
    # part note n1qocvw7 onset_quarter and ppart note n65 onset_sec.
    with open(_subdir("1985_Brendel") / "part.csv", encoding="utf-8", newline="") as f:
        part = {r["id"]: r for r in csv.DictReader(f)}
    with open(_subdir("1985_Brendel") / "ppart.csv", encoding="utf-8", newline="") as f:
        ppart = {r["id"]: r for r in csv.DictReader(f)}
    src = float(Fraction(str(part["n1qocvw7"]["onset_quarter"])))
    tgt = float(ppart["n65"]["onset_sec"])

    matching = [
        c
        for c in bundle.cross_group_claims
        if c.is_synchronous
        and c.start_anchor is not None
        and c.start_anchor.timeline_a_id == SCORE_CLT_ID
        and c.start_anchor.timeline_b_id == cpt_id
        and c.start_anchor.coordinate_a.value == src
        and c.start_anchor.coordinate_a.unit is TimeUnit.quarters
        and c.start_anchor.coordinate_b.value == tgt
        and c.start_anchor.coordinate_b.unit is TimeUnit.seconds
    ]
    assert len(matching) == 2


# endregion


# region SamplesToSeconds C-Map


def test_samples_to_seconds_cmap_on_dpt(loader: ParangonadaLoader) -> None:
    """The dpt_1 timeline's SamplesToSeconds map converts 44100 → 1.0 s."""
    dpt = loader.create_timeline("perf:1966_Szegedi:dpt_1")
    cmap = dpt.get_conversion_map(TimeUnit.seconds)
    assert isinstance(cmap, SamplesToSeconds)
    assert cmap.sample_rate == 44100
    assert cmap(44100) == 1.0
    assert cmap(88200) == 2.0


# endregion


# region Loader API


def test_create_timelines_returns_twelve(loader: ParangonadaLoader) -> None:
    timelines = loader.create_timelines()
    assert len(timelines) == 12
    assert timelines[0].id == SCORE_CLT_ID
    assert timelines[1].id == SCORE_DLT_ID


def test_create_timelines_id_pattern_filters(loader: ParangonadaLoader) -> None:
    """The pinned 12-timeline bundle has 10 performer timelines."""
    assert len(loader.create_timelines(id_pattern=r"^score:")) == 2
    assert len(loader.create_timelines(id_pattern=r"^perf:")) == 10


def test_create_timelines_id_pattern_no_match(loader: ParangonadaLoader) -> None:
    """An unmatched ID pattern returns no timelines."""
    assert loader.create_timelines(id_pattern=r"^missing:") == []


def test_create_timeline_unknown_raises(loader: ParangonadaLoader) -> None:
    with pytest.raises(KeyError):
        loader.create_timeline("does_not_exist")


def test_create_bundle_before_load_raises() -> None:
    loader = ParangonadaLoader()
    with pytest.raises(RuntimeError):
        loader.create_bundle()


def test_load_missing_directory_raises(tmp_path: Path) -> None:
    loader = ParangonadaLoader()
    with pytest.raises(FileNotFoundError):
        loader.load(tmp_path / "nonexistent")


def test_load_directory_without_match_raises(tmp_path: Path) -> None:
    (tmp_path / "empty_dataset").mkdir()
    loader = ParangonadaLoader()
    with pytest.raises(FileNotFoundError):
        loader.load(tmp_path / "empty_dataset")


def test_load_source_not_used() -> None:
    loader = ParangonadaLoader()
    with pytest.raises(NotImplementedError):
        loader._load_source(DATASET_DIR)


# endregion
