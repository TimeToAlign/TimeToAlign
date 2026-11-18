"""MapsLoader on a MEI score and a MAPS file the tests write (validation logic: README.md)."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("verovio")

from timetoalign import MapsLoader  # noqa: E402
from timetoalign.alignment.bundle import AlignmentBundle  # noqa: E402
from timetoalign.alignment.claims import Agent, MatchMetadata  # noqa: E402
from timetoalign.core import AgentType, TimeUnit  # noqa: E402
from timetoalign.loader.score import VerovioLoader  # noqa: E402
from timetoalign.maps import SamplesToSeconds  # noqa: E402
from timetoalign.timelines import (  # noqa: E402
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscretePhysicalTimeline,
)
from timetoalign.timelines.base import Timeline  # noqa: E402

SAMPLE_RATE = 44100

MEI = """<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.1">
<meiHead><fileDesc><titleStmt><title>Fixture</title></titleStmt><pubStmt/></fileDesc>
</meiHead><music><body><mdiv xml:id="mdiv1"><score>
<scoreDef xml:id="sd" meter.count="9" meter.unit="8"><staffGrp xml:id="sg">
  <staffDef xml:id="sdef1" n="1" lines="5" clef.shape="G" clef.line="2"/>
  <staffDef xml:id="sdef2" n="2" lines="5" clef.shape="F" clef.line="4"/>
</staffGrp></scoreDef>
<section xml:id="s1">
<measure xml:id="m1" n="1">
  <staff xml:id="m1s1" n="1"><layer xml:id="m1s1l1" n="1">
    <rest xml:id="r1" dur="8"/>
    <chord xml:id="ch1" dur="4">
      <note xml:id="ch1a" pname="c" oct="4"/>
      <note xml:id="ch1b" pname="e" oct="4"/>
      <note xml:id="ch1c" pname="g" oct="4"/>
    </chord>
    <note xml:id="n2" pname="a" oct="4" dur="4" dots="1"/>
    <note xml:id="n3" pname="b" oct="4" dur="4" dots="1"/>
  </layer></staff>
  <staff xml:id="m1s2" n="2"><layer xml:id="m1s2l1" n="1">
    <rest xml:id="r2" dur="8"/>
    <note xml:id="b1" pname="c" oct="3" dur="4"/>
    <note xml:id="b2" pname="f" oct="3" dur="4" dots="1"/>
    <note xml:id="b3" pname="g" oct="3" dur="4" dots="1"/>
  </layer></staff>
  <tie xml:id="tie1" startid="#n3" endid="#n4"/>
</measure>
<measure xml:id="m2" n="2">
  <staff xml:id="m2s1" n="1"><layer xml:id="m2s1l1" n="1">
    <note xml:id="n4" pname="b" oct="4" dur="4" dots="1"/>
    <note xml:id="n5" pname="c" oct="5" dur="8"/>
    <note xml:id="n6" pname="d" oct="5" dur="4"/>
    <note xml:id="n7" pname="c" oct="5" dur="4" dots="1"/>
  </layer></staff>
  <staff xml:id="m2s2" n="2"><layer xml:id="m2s2l1" n="1">
    <note xml:id="b4" pname="a" oct="2" dur="2" dots="1"/>
    <note xml:id="b5" pname="c" oct="3" dur="4" dots="1"/>
  </layer></staff>
</measure>
</section></score></mdiv></body></music></mei>
"""

#: (obs_mean_onset as written, xml_ids, score quarter, sample at 44100 Hz).
ROWS: list[tuple[float | int, list[str], Fraction, int]] = [
    (0.5, ["ch1a", "ch1b", "ch1c", "b1"], Fraction(1, 2), 22050),
    (1.25, ["n2", "b2"], Fraction(3, 2), 55125),
    (2.0, ["n3", "b3"], Fraction(3), 88200),
    (3.5, ["b4"], Fraction(9, 2), 154350),
    (4.25, ["n5"], Fraction(6), 187425),
    (4.5, ["n6"], Fraction(13, 2), 198450),
    (5, ["n7", "b5"], Fraction(15, 2), 220500),
]


def _maps_rows() -> list[dict[str, Any]]:
    """The MAPS rows of the fixture, as a file states them."""
    return [
        {"obs_mean_onset": onset, "xml_id": ids, "obs_num": n}
        for n, (onset, ids, _, _) in enumerate(ROWS, start=1)
    ]


def _write(path: Path, rows: Any) -> Path:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


@pytest.fixture
def score(tmp_path: Path) -> Timeline:
    mei = tmp_path / "score.mei"
    mei.write_text(MEI, encoding="utf-8")
    return VerovioLoader.from_file(mei).create_timeline()


@pytest.fixture
def maps_path(tmp_path: Path) -> Path:
    return _write(tmp_path / "alignment.json", _maps_rows())


def _samples_timeline() -> DiscretePhysicalTimeline:
    timeline = DiscretePhysicalTimeline(length=6 * SAMPLE_RATE, unit=TimeUnit.samples)
    timeline.add_conversion_map(SamplesToSeconds(sample_rate=SAMPLE_RATE))
    return timeline


def _seconds_timeline() -> ContinuousPhysicalTimeline:
    return ContinuousPhysicalTimeline(length=6.0, unit=TimeUnit.seconds)


def _bundle(score: Timeline, performance: Timeline, maps_path: Path) -> AlignmentBundle:
    claims = MapsLoader.from_file(maps_path).create_match_claims(score, performance)
    bundle = AlignmentBundle()
    bundle.add_timeline(score, as_group="score")
    bundle.add_timeline(performance, as_group="performance")
    bundle.add_match_claims(claims)
    return bundle


# region Rows and claims


def test_rows_are_kept_verbatim(maps_path: Path) -> None:
    loader = MapsLoader.from_file(maps_path)
    assert loader.rows == _maps_rows()
    assert len(loader) == 7
    assert loader.sources == [maps_path]


@pytest.mark.parametrize("make_performance", [_samples_timeline, _seconds_timeline])
def test_one_claim_per_listed_id(
    score: Timeline, maps_path: Path, make_performance
) -> None:
    performance = make_performance()
    claims = MapsLoader.from_file(maps_path).create_match_claims(score, performance)
    expected = [
        (xml_id, quarter, float(onset))
        for onset, ids, quarter, _ in ROWS
        for xml_id in ids
    ]
    assert len(claims) == 13
    for claim, (xml_id, quarter, seconds) in zip(claims, expected, strict=True):
        anchor = claim.start_anchor
        assert claim.is_synchronous
        assert claim.end_anchor is None
        assert claim.event_a_id == xml_id
        assert claim.event_b_id is None
        assert (claim.timeline_a_id, claim.timeline_b_id) == (score.id, performance.id)
        assert anchor.coordinate_a.unit == TimeUnit.quarters
        assert type(anchor.coordinate_a.value) is Fraction
        assert anchor.coordinate_a.value == quarter
        assert anchor.coordinate_b.unit == TimeUnit.seconds
        assert type(anchor.coordinate_b.value) is float
        assert anchor.coordinate_b.value == seconds


def test_metadata_is_recorded_on_every_claim(score: Timeline, maps_path: Path) -> None:
    metadata = MatchMetadata(
        agent=Agent(name="aligner", type=AgentType.software, identifier="1.0"),
        certainty=0.5,
    )
    claims = MapsLoader.from_file(maps_path).create_match_claims(
        score, _seconds_timeline(), metadata=metadata
    )
    assert [claim.metadata for claim in claims] == [metadata] * 13


# endregion

# region Retrieval through a bundle


def test_samples_timeline_through_the_bundle(score: Timeline, maps_path: Path) -> None:
    performance = _samples_timeline()
    bundle = _bundle(score, performance, maps_path)
    for _, _, quarter, sample in ROWS:
        stamp = bundle.get_matchstamp_at(quarter, score.id)
        value = stamp.get_coordinate_for(performance.id, format="coordinate")
        assert value.unit == TimeUnit.samples
        assert type(value.value) is int
        assert value.value == sample
        assert stamp.is_interpolated is False
    between = bundle.get_matchstamp_at(Fraction(15, 4), score.id)
    value = between.get_coordinate_for(performance.id, format="coordinate")
    assert type(value.value) is int
    assert value.value == 121275
    assert between.is_interpolated is True


def test_seconds_timeline_through_the_bundle(score: Timeline, maps_path: Path) -> None:
    performance = _seconds_timeline()
    bundle = _bundle(score, performance, maps_path)
    for onset, _, quarter, _ in ROWS:
        stamp = bundle.get_matchstamp_at(quarter, score.id)
        value = stamp.get_coordinate_for(performance.id, format="coordinate")
        assert value.unit == TimeUnit.seconds
        assert value.value == float(onset)
        assert stamp.is_interpolated is False
    between = bundle.get_matchstamp_at(Fraction(15, 4), score.id)
    assert between.get_coordinate_for(performance.id, format="coordinate").value == 2.75
    assert between.is_interpolated is True


@pytest.mark.parametrize("make_performance", [_samples_timeline, _seconds_timeline])
def test_quarter_before_the_first_anchor_does_not_resolve(
    score: Timeline, maps_path: Path, make_performance
) -> None:
    performance = make_performance()
    stamp = _bundle(score, performance, maps_path).get_matchstamp_at(
        Fraction(1, 4), score.id
    )
    assert stamp.present_timelines == [score.id]
    with pytest.raises(KeyError):
        stamp.get_coordinate_for(performance.id)


# endregion

# region Refusals


def test_unknown_id_is_refused(score: Timeline, tmp_path: Path) -> None:
    rows = _maps_rows()
    rows[3]["xml_id"] = ["zz9"]
    loader = MapsLoader.from_file(_write(tmp_path / "unknown.json", rows))
    with pytest.raises(KeyError, match=r"MAPS row 4 names xml:id 'zz9'"):
        loader.create_match_claims(score, _seconds_timeline())


@pytest.mark.parametrize(
    "performance",
    [
        DiscretePhysicalTimeline(length=6 * SAMPLE_RATE, unit=TimeUnit.samples),
        ContinuousLogicalTimeline(length=9),
    ],
    ids=["samples", "quarters"],
)
def test_performance_without_a_path_to_seconds_is_refused(
    score: Timeline, maps_path: Path, performance: Timeline
) -> None:
    loader = MapsLoader.from_file(maps_path)
    with pytest.raises(ValueError, match=r"has no C-Map to seconds"):
        loader.create_match_claims(score, performance)


def _changed(change) -> list[dict[str, Any]]:
    rows = _maps_rows()
    change(rows)
    return rows


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (_changed(lambda r: r[2].update(obs_num=4)), r"row 3 has obs_num 4"),
        (_changed(lambda r: r[1].pop("xml_id")), r"MAPS row 2 .* must be an object"),
        (
            _changed(lambda r: r[0].update(obs_mean_onset="0.5")),
            r"MAPS row 1 .*obs_mean_onset must be a number",
        ),
        (
            _changed(lambda r: r[0].update(obs_mean_onset=True)),
            r"MAPS row 1 .*obs_mean_onset must be a number",
        ),
        (_changed(lambda r: r[4].update(xml_id=[])), r"MAPS row 5 .*xml_id must be"),
        (_changed(lambda r: r[4].update(xml_id=[""])), r"MAPS row 5 .*xml_id must be"),
        ({"rows": _maps_rows()}, r"holds a JSON list of rows"),
        (
            _changed(lambda r: r[5].update(xml_id=["n6", "n2"])),
            r"MAPS rows 2 and 6 .* both name xml:id 'n2'",
        ),
    ],
    ids=[
        "obs_num-gap",
        "missing-key",
        "string-onset",
        "boolean-onset",
        "no-ids",
        "empty-id",
        "not-a-list",
        "id-in-two-rows",
    ],
)
def test_malformed_rows_are_refused(tmp_path: Path, rows: Any, message: str) -> None:
    path = _write(tmp_path / "malformed.json", rows)
    with pytest.raises(ValueError, match=message):
        MapsLoader.from_file(path)


def test_negative_onsets_and_trompa_ids_are_refused(tmp_path: Path) -> None:
    rows = _maps_rows()
    rows[2]["obs_mean_onset"] = -1
    rows.append(
        {"obs_mean_onset": 6.0, "xml_id": ["trompa-x1", "trompa-x2"], "obs_num": 8}
    )
    path = _write(tmp_path / "unscored.json", rows)
    with pytest.raises(ValueError) as raised:
        MapsLoader.from_file(path)
    assert str(raised.value).endswith(
        "Remove them to read the file: row 3 (onset -1); row 8 (ids trompa-x1, trompa-x2)."
    )


def test_a_second_source_is_refused(tmp_path: Path, maps_path: Path) -> None:
    other = _write(tmp_path / "other.json", _maps_rows())
    loader = MapsLoader.from_file(maps_path)
    with pytest.raises(ValueError, match=r"reads one MAPS file"):
        loader.load(other)
    assert loader.rows == _maps_rows()
    loader.clear()
    assert len(loader) == 0
    assert loader.load(other).sources == [other]


def test_claims_before_load_are_refused(score: Timeline) -> None:
    with pytest.raises(RuntimeError, match=r"No MAPS file loaded"):
        MapsLoader().create_match_claims(score, _seconds_timeline())


# endregion
