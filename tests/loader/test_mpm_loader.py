"""Tests for MpmLoader — an MPM-Toolbox MSM+MPM+MPR triple → AlignmentBundle.

This module tests ``MpmLoader`` against the two MPM-Toolbox projects in
the ``mpm_toolbox`` corpus:

- **Beethoven** — ``MPRproject_1971Curzon_VariationXIV/`` (Beethoven's
  Eroica Variations op. 35, Var. XIV; 251 notes; a 44100 Hz recording);
- **Reger** — ``Max Reger - Moment Musical (MPM Toolbox Tutorial)/``
  (Reger's Moment Musical op. 13 no. 4; 92 notes; a 48000 Hz recording).

It verifies, parametrized over both specimens unless noted:

- the score (PPQ, MSM note count) and the two logical score timelines
  (``score:clt_1`` notes; ``score:dlt_1`` notes + per-type MPM markup);
- the ``TicksToQuarters`` and modelled quarters→seconds ``TableMap``
  C-Maps;
- the two physical performance timelines and an onset spot-check;
- the spectrogram graphical timeline (``perf:dgt_1``): its pixel length
  (= the spectrogram ``.png``'s frame-column width), its 0 events, and
  its px→seconds ``ScalarMap``;
- the cross-group claims (count, all synchronous, no NOMATCH) and the
  ``ref`` ↔ ``xml:id`` bijection (0 orphans, 0 unaligned);
- the bundle shape (5 timelines, 2 groups) and that its timeline units
  span all three domains (logical / physical / graphical);
- style-resolution spot-checks (Beethoven / Reger);
- the ``performance=`` selector reaching a non-default performance; and
- a note pitch / spelling spot-check.

All counts and coordinates are exact per the Zero Tolerance Validation
Policy.  Validation logic is documented in ``tests/loader/README.md``.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from lxml import etree

from timetoalign.core import Domain, TimeUnit
from timetoalign.loader.alignment import MpmLoader
from timetoalign.maps.linear import ScalarMap
from timetoalign.maps.table import TableMap
from timetoalign.testdata import ensure_data

ROOT = ensure_data("mpm_toolbox")

BEETHOVEN_DIR = ROOT / "MPRproject_1971Curzon_VariationXIV"
BEETHOVEN_MPR = BEETHOVEN_DIR / "Beethoven_op35_1971Curzon_Var14only.mpr"

REGER_DIR = ROOT / "Max Reger - Moment Musical (MPM Toolbox Tutorial)"
REGER_MPR = REGER_DIR / "Reger - Moment Musical op 13 no 4.mpr"

SCORE_CLT_ID = "score:clt_1"
SCORE_DLT_ID = "score:dlt_1"
PERF_CPT_ID = "perf:cpt_1"
PERF_DPT_ID = "perf:dpt_1"
PERF_DGT_ID = "perf:dgt_1"

_XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


#: Per-specimen expectations.  ``markup`` is the per-type Counter of MPM
#: markup events on ``score:dlt1`` for the DEFAULT (first) performance.
SPECIMENS = {
    "beethoven": {
        "mpr": BEETHOVEN_MPR,
        "msm_notes": 251,
        "markup": {"Tempo": 7, "Dynamics": 34, "Articulation": 207},
        "dlt_total": 499,  # 251 notes + 7 + 34 + 207
        "alignment_notes": 251,
        "claims": 251,
        "tempo_anchors": 8,
        "sample_rate": 44100,
        # Spectrogram: x-axis = frame columns = the .png's pixel width.
        "spectrogram_columns": 26469,
        "hop_size": 128,
    },
    "reger": {
        "mpr": REGER_MPR,
        "msm_notes": 92,
        "markup": {"Tempo": 1, "Dynamics": 11, "Articulation": 80, "Ornament": 1},
        "dlt_total": 185,  # 92 notes + 1 + 11 + 80 + 1
        "alignment_notes": 92,
        "claims": 92,
        "tempo_anchors": 2,
        "sample_rate": 48000,
        "spectrogram_columns": 1587,
        "hop_size": 512,
    },
}

SPECIMEN_IDS = list(SPECIMENS)


# region Fixtures / helpers


@pytest.fixture(params=SPECIMEN_IDS)
def specimen(request: pytest.FixtureRequest) -> dict[str, Any]:
    """A per-specimen expectation dict (parametrized over both specimens)."""
    return SPECIMENS[request.param]


def _loaded(mpr: Path, *, performance: str | None = None) -> MpmLoader:
    """Load a project from its ``.mpr`` (optionally selecting a performance)."""
    return MpmLoader().load(mpr, performance=performance)


def _event_rows(timeline: Any) -> list[dict[str, Any]]:
    """Return a timeline's events as a list of row dicts."""
    return timeline.events._table.to_pylist()


def _event_type_counts(timeline: Any) -> Counter:
    """Return a Counter of ``event_type`` over a timeline's events."""
    return Counter(timeline.events._table.column("event_type").to_pylist())


def _parse(path: Path) -> Any:
    """Parse XML the same way the loader does (id-collection disabled)."""
    parser = etree.XMLParser(recover=True, collect_ids=False)
    return etree.parse(str(path), parser).getroot()


def _localname(element: Any) -> str:
    return etree.QName(element).localname


# endregion


# region Loading + score


def test_loader_repr_and_ppq(specimen: dict[str, Any]) -> None:
    """The loader exposes PPQ 720 and the default performance in its repr."""
    loader = _loaded(specimen["mpr"])
    assert loader.ppq == 720
    assert loader.performance_name == "MEI export performance"
    assert repr(loader) == (
        f"MpmLoader(performance='MEI export performance', "
        f"claims={specimen['claims']})"
    )


def test_repr_html_shows_claims_not_zero_events(specimen: dict[str, Any]) -> None:
    """The HTML card reports the real claim count, never the base Events: 0."""
    loader = _loaded(specimen["mpr"])
    html = loader._repr_html_()
    assert f"<tr><td><b>Claims</b></td><td>{specimen['claims']}</td></tr>" in html
    assert "Events" not in html
    assert "<b>Project</b>" in html
    assert "<b>Performance</b>" in html
    assert "in 2 group(s) (score, perf)" in html


def test_from_file_matches_load(specimen: dict[str, Any]) -> None:
    """``from_file`` is equivalent to ``load`` for the default performance."""
    via_from_file = MpmLoader.from_file(specimen["mpr"])
    assert via_from_file.performance_name == "MEI export performance"
    assert len(via_from_file) == specimen["claims"]


def test_score_clt_holds_only_notes(specimen: dict[str, Any]) -> None:
    """``score:clt_1`` holds exactly the MSM notes (no markup)."""
    loader = _loaded(specimen["mpr"])
    clt = loader.create_timeline(SCORE_CLT_ID)
    counts = _event_type_counts(clt)
    assert clt.events._table.num_rows == specimen["msm_notes"]
    assert dict(counts) == {"Note": specimen["msm_notes"]}


def test_score_dlt_markup_counts(specimen: dict[str, Any]) -> None:
    """``score:dlt_1`` holds the notes plus the per-type MPM markup events."""
    loader = _loaded(specimen["mpr"])
    dlt = loader.create_timeline(SCORE_DLT_ID)
    counts = _event_type_counts(dlt)

    expected = {"Note": specimen["msm_notes"], **specimen["markup"]}
    assert dict(counts) == expected
    assert dlt.events._table.num_rows == specimen["dlt_total"]
    # Asynchrony is absent from both default performances.
    assert counts["Asynchrony"] == 0


# endregion


# region C-Maps


def test_ticks_to_quarters_cmap(specimen: dict[str, Any]) -> None:
    """The ticks→quarters C-Map on ``score:dlt_1`` resolves 360→0.5, 720→1.0."""
    loader = _loaded(specimen["mpr"])
    dlt = loader.create_timeline(SCORE_DLT_ID)
    assert dlt.get_timestamp(360).get_unit(TimeUnit.quarters, format="float") == 0.5
    assert dlt.get_timestamp(720).get_unit(TimeUnit.quarters, format="float") == 1.0


def test_tempo_table_map_basics(specimen: dict[str, Any]) -> None:
    """The quarters→seconds TableMap starts at 0 and is monotonic increasing."""
    loader = _loaded(specimen["mpr"])
    tempo_map = loader.tempo_map
    assert isinstance(tempo_map, TableMap)
    assert tempo_map(0) == 0.0
    assert len(tempo_map.x_values) == specimen["tempo_anchors"]
    ys = list(tempo_map.y_values)
    assert all(ys[i] < ys[i + 1] for i in range(len(ys) - 1))


def test_reger_tempo_map_is_three_quarters_per_second() -> None:
    """Reger's single-tempo map is ``s = 3·q`` (beatLength 0.0625, bpm 80)."""
    loader = _loaded(REGER_MPR)
    tempo_map = loader.tempo_map
    assert tempo_map(1) == 3.0
    assert tempo_map(2) == 6.0


def test_beethoven_tempo_map_anchor() -> None:
    """Beethoven's first two segments pin ``map(39.5)=23.7``, ``map(40)=24.9``.

    The first tempoMap entry resolves to bpm 100 / beatLength 0.25, i.e.
    spq = (0.25/0.25)*(60/100) = 0.6, so 39.5 quarters → 23.7 s.  The
    second entry (bpm 25, spq 2.4) advances the next 0.5 quarter by 1.2 s
    to 24.9 s.
    """
    loader = _loaded(BEETHOVEN_MPR)
    tempo_map = loader.tempo_map
    assert tempo_map(39.5) == 23.7
    assert tempo_map(40.0) == 24.9


# endregion


# region Performance timelines


def test_perf_timeline_event_counts(specimen: dict[str, Any]) -> None:
    """Both physical timelines hold one Note per observed onset."""
    loader = _loaded(specimen["mpr"])
    cpt = loader.create_timeline(PERF_CPT_ID)
    dpt = loader.create_timeline(PERF_DPT_ID)
    assert cpt.events._table.num_rows == specimen["alignment_notes"]
    assert dpt.events._table.num_rows == specimen["alignment_notes"]
    assert dict(_event_type_counts(cpt)) == {"Note": specimen["alignment_notes"]}
    assert dict(_event_type_counts(dpt)) == {"Note": specimen["alignment_notes"]}


def test_samples_to_seconds_cmap(specimen: dict[str, Any]) -> None:
    """``perf:dpt_1`` carries a SamplesToSeconds map at the recording's rate."""
    loader = _loaded(specimen["mpr"])
    dpt = loader.create_timeline(PERF_DPT_ID)
    rate = specimen["sample_rate"]
    assert dpt.get_timestamp(rate).get_unit(TimeUnit.seconds, format="float") == 1.0


def test_beethoven_onset_spot_check() -> None:
    """Beethoven ``nbwxzb1`` onset is ``828.0190259247195 / 1000`` seconds."""
    loader = _loaded(BEETHOVEN_MPR)
    cpt = loader.create_timeline(PERF_CPT_ID)
    expected = 828.0190259247195 / 1000.0
    onset = next(r for r in _event_rows(cpt) if r["id"] == "nbwxzb1")
    assert onset["start"]["value"] == expected


def test_beethoven_note_pitch_spot_check() -> None:
    """Beethoven ``nbwxzb1`` carries the MSM number + verbatim spelling on dlt_1.

    A number-only-plus-inconsistent-spelling source represents pitch as
    its most-expressive faithful type, EnharmonicPitch, carried as a
    ``{midi_number}`` struct; the keystone preserves it as a real struct
    column.  The verbatim ``pitchname`` / ``accidentals`` / ``octave``
    survive as non-default raw columns keeping their native PyArrow types
    (a string and two ints), no longer JSON-stringified.  ``start`` is a
    coordinate, so it round-trips numerically.
    """
    loader = _loaded(BEETHOVEN_MPR)
    dlt = loader.create_timeline(SCORE_DLT_ID)
    note = next(r for r in _event_rows(dlt) if r["id"] == "nbwxzb1")
    assert note["pitch"] == {"midi_number": 75}
    assert note["pitchname"] == "e"
    assert note["accidentals"] == -1
    assert note["octave"] == 4
    assert note["start"]["value"] == 360.0


# endregion


# region Spectrogram graphical timeline


def test_spectrogram_timeline_is_pixels_axis(specimen: dict[str, Any]) -> None:
    """``perf:dgt_1`` is a pixels axis of the .png's frame-column width, 0 events.

    The spectrogram x-axis is time in frame columns; the column count is the
    pixel width of the spectrogram ``.png`` (read from its IHDR header).  The
    timeline carries no events — it is a graphical axis, not an event
    timeline.
    """
    loader = _loaded(specimen["mpr"])
    dgt = loader.create_timeline(PERF_DGT_ID)
    assert dgt.unit == TimeUnit.pixels
    assert dgt.length.value == specimen["spectrogram_columns"]
    assert dgt.events._table.num_rows == 0


def test_spectrogram_px_to_seconds_cmap(specimen: dict[str, Any]) -> None:
    """``perf:dgt_1`` carries a px→seconds ``ScalarMap`` of ``hopSize/rate``.

    Each frame column advances ``hopSize`` audio samples, so
    ``seconds = px * hopSize / sample_rate``.  The map is a ``ScalarMap``
    whose scalar is exactly ``hop_size / sample_rate`` (the same Python
    expression the loader evaluates, so the IEEE-754 double matches).
    """
    loader = _loaded(specimen["mpr"])
    dgt = loader.create_timeline(PERF_DGT_ID)
    expected_scalar = specimen["hop_size"] / specimen["sample_rate"]

    cmap = dgt.get_conversion_map(TimeUnit.seconds)
    assert isinstance(cmap, ScalarMap)
    assert cmap.scalar == expected_scalar
    # The axis resolves px -> seconds: map(0)=0, map(1)=scalar.
    assert dgt.get_timestamp(0).get_unit(TimeUnit.seconds, format="float") == 0.0
    assert (
        dgt.get_timestamp(1).get_unit(TimeUnit.seconds, format="float")
        == expected_scalar
    )


def test_beethoven_spectrogram_duration() -> None:
    """Beethoven's last spectrogram column maps to the audio duration.

    ``map(26469) == 26469 * 128 / 44100`` seconds (the rightmost frame
    column's onset = the spectrogram's total time span).
    """
    loader = _loaded(BEETHOVEN_MPR)
    dgt = loader.create_timeline(PERF_DGT_ID)
    expected = 26469 * 128 / 44100
    assert (
        dgt.get_timestamp(26469).get_unit(TimeUnit.seconds, format="float") == expected
    )


def test_reger_spectrogram_first_column() -> None:
    """Reger's first spectrogram column maps to ``512 / 48000`` seconds."""
    loader = _loaded(REGER_MPR)
    dgt = loader.create_timeline(PERF_DGT_ID)
    assert (
        dgt.get_timestamp(1).get_unit(TimeUnit.seconds, format="float") == 512 / 48000
    )


def test_bundle_spans_three_domains(specimen: dict[str, Any]) -> None:
    """The bundle's timeline units span logical, physical, and graphical.

    Adding ``perf:dgt_1`` (pixels) completes the third domain: the bundle now
    carries a logical member (quarters / ticks), a physical member (seconds
    / samples), and a graphical member (pixels).
    """
    loader = _loaded(specimen["mpr"])
    bundle = loader.create_bundle()
    domains = {bundle.get_timeline(uid).unit.domain for uid in bundle.timeline_ids}
    assert domains == {Domain.logical, Domain.physical, Domain.graphical}


# endregion


# region Claims + bijection


def test_bundle_shape(specimen: dict[str, Any]) -> None:
    """The bundle holds 5 timelines in 2 groups (score, perf).

    The ``perf`` group holds three timelines (``perf:cpt_1`` / ``perf:dpt_1``
    / ``perf:dgt_1``); both specimens carry a spectrogram, so the graphical
    axis is always present.
    """
    loader = _loaded(specimen["mpr"])
    bundle = loader.create_bundle()
    assert bundle.n_timelines == 5
    assert bundle.n_groups == 2
    assert bundle.group_ids == ["score", "perf"]
    assert bundle.timeline_ids == [
        SCORE_CLT_ID,
        SCORE_DLT_ID,
        PERF_CPT_ID,
        PERF_DPT_ID,
        PERF_DGT_ID,
    ]
    # The perf group now holds three timelines.
    perf_group = bundle.get_group("perf")
    assert set(perf_group.timeline_ids) == {PERF_CPT_ID, PERF_DPT_ID, PERF_DGT_ID}


def test_claims_all_synchronous(specimen: dict[str, Any]) -> None:
    """Every score note yields one synchronous claim; no NOMATCH claims."""
    loader = _loaded(specimen["mpr"])
    bundle = loader.create_bundle()
    claims = bundle.get_match_claims()
    assert len(claims) == specimen["claims"]
    assert len(bundle.get_match_claims(synchronous_only=True)) == specimen["claims"]
    assert len(bundle.get_match_claims(nomatch_only=True)) == 0
    # All claims connect score:clt1 (quarters) -> perf:cpt1 (seconds).
    for claim in claims:
        assert claim.is_synchronous
        assert claim.connects_both(SCORE_CLT_ID, PERF_CPT_ID)


def test_ref_xmlid_bijection(specimen: dict[str, Any]) -> None:
    """Alignment ``ref`` ↔ MSM ``xml:id`` is a perfect bijection.

    The bijection is the loader's central join, so the test also pins the
    loader's claim count to the (raw-file) bijection size.
    """
    loader = _loaded(specimen["mpr"])

    # MSM note ids.
    mpr_root = _parse(specimen["mpr"])
    msm_name = next(c.get("file") for c in mpr_root if _localname(c) == "msm")
    msm_root = _parse(specimen["mpr"].parent / Path(msm_name).name)
    msm_ids = [n.get(_XML_ID) for n in msm_root.iter() if _localname(n) == "note"]

    # Alignment refs.
    alignment = next(el for el in mpr_root.iter() if _localname(el) == "alignment")
    refs = [n.get("ref") for n in alignment.iter() if _localname(n) == "note"]

    assert len(msm_ids) == specimen["msm_notes"]
    assert len(refs) == specimen["alignment_notes"]
    assert len(set(msm_ids)) == len(msm_ids)  # unique
    assert len(set(refs)) == len(refs)  # unique
    # Zero orphans, zero unaligned.
    assert set(refs) - set(msm_ids) == set()
    assert set(msm_ids) - set(refs) == set()
    # Every bijection pair becomes exactly one claim.
    assert len(loader) == len(set(refs))


def test_beethoven_claim_coordinates() -> None:
    """Beethoven ``nbwxzb1``'s claim projects 0.5 q → 0.828019… s."""
    loader = _loaded(BEETHOVEN_MPR)
    bundle = loader.create_bundle()
    expected_seconds = 828.0190259247195 / 1000.0
    matches = [
        c
        for c in bundle.get_match_claims()
        if c.start_anchor.coordinate_a.value == Fraction(360, 720)
        and c.start_anchor.coordinate_a.unit is TimeUnit.quarters
        and c.start_anchor.coordinate_b.value == expected_seconds
        and c.start_anchor.coordinate_b.unit is TimeUnit.seconds
    ]
    assert len(matches) == 1


# endregion


# region Style resolution


def test_beethoven_style_resolution() -> None:
    """Beethoven dynamics / tempo / articulation styles resolve to numbers.

    Carried attributes round-trip as strings through the timeline's generic
    ``add_events``, so resolved values are coerced back to float for the
    exact comparison (the convention shared with the other loader tests).
    """
    loader = _loaded(BEETHOVEN_MPR)
    rows = _event_rows(loader.create_timeline(SCORE_DLT_ID))

    p_dynamics = [
        r for r in rows if r["event_type"] == "Dynamics" and r["volume_label"] == "p"
    ]
    assert p_dynamics, "expected at least one Dynamics with volume='p'"
    assert all(float(r["volume"]) == 48.0 for r in p_dynamics)

    meno = [
        r
        for r in rows
        if r["event_type"] == "Tempo" and r["bpm_label"] == "Meno mosso."
    ]
    assert meno, "expected at least one Tempo with bpm='Meno mosso.'"
    assert all(float(r["bpm"]) == 100.0 for r in meno)

    staccato = [
        r for r in rows if r["event_type"] == "Articulation" and r["name"] == "staccato"
    ]
    assert staccato, "expected at least one staccato articulation"
    assert all(float(r["absolute_duration_ms"]) == 160.0 for r in staccato)
    # noteid has its leading '#' stripped.
    assert all(not r["noteid"].startswith("#") for r in staccato)


def test_reger_style_resolution() -> None:
    """Reger dynamics / tempo styles resolve to numbers."""
    loader = _loaded(REGER_MPR)
    rows = _event_rows(loader.create_timeline(SCORE_DLT_ID))

    dolciss = [
        r
        for r in rows
        if r["event_type"] == "Dynamics" and r["volume_label"] == "dolciss."
    ]
    assert dolciss, "expected at least one Dynamics with volume='dolciss.'"
    assert all(float(r["volume"]) == 74.0 for r in dolciss)

    andantino = [
        r for r in rows if r["event_type"] == "Tempo" and r["bpm_label"] == "Andantino"
    ]
    assert andantino, "expected at least one Tempo with bpm='Andantino'"
    assert all(float(r["bpm"]) == 80.0 for r in andantino)


def test_reger_ornament_emitted_generically() -> None:
    """Reger's lone Ornament entry is emitted with its raw attributes."""
    loader = _loaded(REGER_MPR)
    rows = _event_rows(loader.create_timeline(SCORE_DLT_ID))
    ornaments = [r for r in rows if r["event_type"] == "Ornament"]
    assert len(ornaments) == 1
    ornament = ornaments[0]
    assert ornament["name_ref"] == "arpeggio"
    assert ornament["note_order"] == "#n96 #n97 #n98"


# endregion


# region Performance selector


def test_performance_selector_reaches_audio_performance() -> None:
    """Selecting the audio performance reaches different markup counts.

    The ``Curzon_1971_DECCA-SXL6523_audio`` performance stores inline
    numeric bpm values; its dlt_1 carries 25 Asynchrony events (vs 0 in the
    default) and 123 Tempo events.
    """
    loader = _loaded(BEETHOVEN_MPR, performance="Curzon_1971_DECCA-SXL6523_audio")
    assert loader.performance_name == "Curzon_1971_DECCA-SXL6523_audio"
    counts = _event_type_counts(loader.create_timeline(SCORE_DLT_ID))
    assert counts["Asynchrony"] == 25
    assert counts["Tempo"] == 123
    # The audio performance's inline bpm resolves to a numeric value.
    rows = _event_rows(loader.create_timeline(SCORE_DLT_ID))
    tempos = [r for r in rows if r["event_type"] == "Tempo"]
    assert all(r["bpm"] is not None for r in tempos)


def test_unknown_performance_raises() -> None:
    """An unknown performance name raises ValueError."""
    with pytest.raises(ValueError, match="No performance named"):
        _loaded(BEETHOVEN_MPR, performance="nonexistent")


# endregion


# region Loader API guards


def test_create_timeline_unknown_uid_raises() -> None:
    """An unknown timeline uid raises KeyError."""
    loader = _loaded(REGER_MPR)
    with pytest.raises(KeyError, match="No timeline with uid"):
        loader.create_timeline("score:bogus")


def test_create_timelines_returns_all(specimen: dict[str, Any]) -> None:
    """``create_timelines`` returns the five timelines in canonical order."""
    loader = _loaded(specimen["mpr"])
    timelines = loader.create_timelines()
    assert [tl.id for tl in timelines] == [
        SCORE_CLT_ID,
        SCORE_DLT_ID,
        PERF_CPT_ID,
        PERF_DPT_ID,
        PERF_DGT_ID,
    ]


def test_create_timelines_id_pattern_filters(specimen: dict[str, Any]) -> None:
    """The pinned MPM shape filters to one score CLT and three perf timelines."""
    loader = _loaded(specimen["mpr"])
    assert len(loader.create_timelines(id_pattern="clt")) == 1
    assert len(loader.create_timelines(id_pattern=r"^perf:")) == 3


def test_create_timelines_id_pattern_no_match(specimen: dict[str, Any]) -> None:
    """An unmatched ID pattern returns no timelines."""
    loader = _loaded(specimen["mpr"])
    assert loader.create_timelines(id_pattern=r"^missing:") == []


# endregion


# region No-spectrogram fallback

# When a project carries no ``<spectrogram>`` (or its ``.png`` is missing),
# ``_parse_spectrogram`` returns ``None`` and ``perf:dgt1`` is never built:
# the bundle omits the graphical domain and holds 4 timelines.  Both corpus
# specimens ship a complete spectrogram, so the fallback is forced by
# stubbing the static parser to return ``None`` (a per-test monkeypatch, so
# the substitution is local and parallel-safe) before loading the
# unmodified specimen ``.mpr``.


def test_no_spectrogram_perf_dgt_is_none(
    specimen: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``_parse_spectrogram`` stubbed to ``None``, ``perf:dgt_1`` is absent."""
    monkeypatch.setattr(
        MpmLoader, "_parse_spectrogram", staticmethod(lambda *a, **k: None)
    )
    loader = _loaded(specimen["mpr"])
    assert loader._perf_dgt is None


def test_no_spectrogram_bundle_shape(
    specimen: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a spectrogram the bundle holds 4 timelines, 2 groups, no graphical.

    The ``perf`` group drops to two timelines (``perf:cpt_1`` / ``perf:dpt_1``);
    the claim count is unchanged (one synchronous claim per note); and the
    bundle's timeline-unit domains are exactly logical + physical — the
    graphical domain is absent (the converse of
    ``test_bundle_spans_three_domains``).
    """
    monkeypatch.setattr(
        MpmLoader, "_parse_spectrogram", staticmethod(lambda *a, **k: None)
    )
    loader = _loaded(specimen["mpr"])
    bundle = loader.create_bundle()

    assert bundle.n_timelines == 4
    assert bundle.n_groups == 2
    assert bundle.group_ids == ["score", "perf"]
    assert bundle.timeline_ids == [
        SCORE_CLT_ID,
        SCORE_DLT_ID,
        PERF_CPT_ID,
        PERF_DPT_ID,
    ]
    perf_group = bundle.get_group("perf")
    assert set(perf_group.timeline_ids) == {PERF_CPT_ID, PERF_DPT_ID}

    assert len(bundle.get_match_claims()) == specimen["claims"]

    domains = {bundle.get_timeline(uid).unit.domain for uid in bundle.timeline_ids}
    assert domains == {Domain.logical, Domain.physical}


def test_no_spectrogram_create_timelines_omits_graphical(
    specimen: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``create_timelines`` returns only the four non-graphical timelines."""
    monkeypatch.setattr(
        MpmLoader, "_parse_spectrogram", staticmethod(lambda *a, **k: None)
    )
    loader = _loaded(specimen["mpr"])
    timelines = loader.create_timelines()
    assert [tl.id for tl in timelines] == [
        SCORE_CLT_ID,
        SCORE_DLT_ID,
        PERF_CPT_ID,
        PERF_DPT_ID,
    ]


def test_no_spectrogram_create_timeline_dgt_raises(
    specimen: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``create_timeline('perf:dgt_1')`` raises KeyError — it was never built."""
    monkeypatch.setattr(
        MpmLoader, "_parse_spectrogram", staticmethod(lambda *a, **k: None)
    )
    loader = _loaded(specimen["mpr"])
    with pytest.raises(KeyError, match="No timeline with uid"):
        loader.create_timeline(PERF_DGT_ID)


def test_no_spectrogram_repr_html_reports_four(
    specimen: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HTML card reports ``4 in 2 group(s)``, never ``5 in 2 group(s)``."""
    monkeypatch.setattr(
        MpmLoader, "_parse_spectrogram", staticmethod(lambda *a, **k: None)
    )
    loader = _loaded(specimen["mpr"])
    html = loader._repr_html_()
    assert "4 in 2 group(s) (score, perf)" in html
    assert "5 in 2 group(s)" not in html


# endregion
