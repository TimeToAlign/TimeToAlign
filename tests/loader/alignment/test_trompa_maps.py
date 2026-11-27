"""MapsLoader on a MEI score and a MAPS file the tests write (validation logic: README.md)."""

from __future__ import annotations

import json
import logging
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

pytest.importorskip("verovio")

from tests.loader.minting import shape, shapes  # noqa: E402
from timetoalign import MapsLoader  # noqa: E402
from timetoalign.alignment.bundle import AlignmentBundle  # noqa: E402
from timetoalign.alignment.claims import MatchClaim  # noqa: E402
from timetoalign.core import (  # noqa: E402
    NOMATCH,
    Coordinate,
    Projection,
    Provenance,
    TimeUnit,
)
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
</meiHead><music><body><mdiv xml:id="mdiv1" label="I"><score>
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
    return MapsLoader.from_file(maps_path).create_bundle(score, performance)


INSERTED = "trompa-align_inserted_Cs4"


def _stated_rows() -> list[dict[str, Any]]:
    """The rows as trompa-align writes them: unperformed, repeated, inserted."""
    rows: list[dict[str, Any]] = [
        {"obs_mean_onset": -1, "xml_id": ["n4"], "obs_num": 1}
    ]
    rows += [{**row, "obs_num": row["obs_num"] + 1} for row in _maps_rows()]
    rows += [
        {"obs_mean_onset": 5.5, "xml_id": ["n7"], "obs_num": 9},
        {"obs_mean_onset": 2.75, "xml_id": INSERTED, "obs_num": 10},
        {"obs_mean_onset": 4.75, "xml_id": [INSERTED], "obs_num": 11},
    ]
    return rows


@pytest.fixture
def stated_path(tmp_path: Path) -> Path:
    return _write(tmp_path / "stated.json", _stated_rows())


# region Rows and claims


def test_rows_are_kept_verbatim(maps_path: Path) -> None:
    loader = MapsLoader.from_file(maps_path)
    assert loader.rows == _maps_rows()
    assert len(loader) == 7
    assert loader.sources == [maps_path]


def _on_performance(performance: Timeline, onset: float | int, sample: int) -> Any:
    """The onset of a row as the performance timeline's own axis states it."""
    return sample if performance.unit == TimeUnit.samples else float(onset)


@pytest.mark.parametrize("make_performance", [_samples_timeline, _seconds_timeline])
def test_one_claim_per_listed_id(
    score: Timeline, maps_path: Path, make_performance
) -> None:
    performance = make_performance()
    claims = MapsLoader.from_file(maps_path).create_match_claims(score, performance)
    expected = [
        (xml_id, quarter, _on_performance(performance, onset, sample))
        for onset, ids, quarter, sample in ROWS
        for xml_id in ids
    ]
    assert len(claims) == 13
    projections = performance.events.table.column("id").to_pylist()
    assert shapes(projections) == ["projection"] * 13
    for n, (claim, (xml_id, quarter, position)) in enumerate(
        zip(claims, expected, strict=True), start=1
    ):
        anchor = claim.start_anchor
        assert claim.is_synchronous
        assert claim.end_anchor is None
        assert claim.event_a_id == xml_id
        assert claim.event_b_id == projections[n - 1]
        assert (claim.timeline_a_id, claim.timeline_b_id) == (score.id, performance.id)
        assert anchor.coordinate_a.unit == TimeUnit.quarters
        assert type(anchor.coordinate_a.value) is Fraction
        assert anchor.coordinate_a.value == quarter
        assert anchor.coordinate_b.unit == performance.unit
        assert type(anchor.coordinate_b.value) is type(position)
        assert anchor.coordinate_b.value == position


@pytest.mark.parametrize("make_performance", [_samples_timeline, _seconds_timeline])
def test_projections_stand_for_the_score_notes(
    score: Timeline, maps_path: Path, make_performance
) -> None:
    """Each onset is a Projection of its score note on the performance."""
    performance = make_performance()
    MapsLoader.from_file(maps_path).create_match_claims(score, performance)
    rows = performance.events.table.to_pylist()
    assert [
        (shape(row["id"]), row["event_type"], row["projection_of"]) for row in rows
    ] == [
        ("projection", "Projection", {"timeline_id": score.id, "event_id": xml_id})
        for _, ids, _, _ in ROWS
        for xml_id in ids
    ]
    starts = [
        (
            row["start"]["numerator"]
            if performance.unit == TimeUnit.samples
            else row["start"]["value"]
        )
        for row in rows
    ]
    assert starts == [
        _on_performance(performance, onset, sample)
        for onset, ids, _, sample in ROWS
        for _ in ids
    ]


def test_provenance_is_recorded_on_every_claim(
    score: Timeline, maps_path: Path
) -> None:
    provenance = Provenance(method="inferred", note="aligner 1.0")
    claims = MapsLoader.from_file(maps_path).create_match_claims(
        score, _seconds_timeline(), provenance=provenance
    )
    assert [claim.provenance for claim in claims] == [provenance] * 13


def test_stated_rows_are_kept_verbatim(stated_path: Path) -> None:
    loader = MapsLoader.from_file(stated_path)
    assert loader.rows == _stated_rows()
    assert len(loader) == 11


def test_every_stated_row_gives_its_claims(score: Timeline, stated_path: Path) -> None:
    performance = _seconds_timeline()
    claims = MapsLoader.from_file(stated_path).create_match_claims(score, performance)
    assert len(claims) == 17

    unperformed = claims[0]
    assert unperformed.is_synchronous is False
    assert (unperformed.start_anchor, unperformed.end_anchor) == (None, None)
    assert (unperformed.timeline_a_id, unperformed.timeline_b_id) == (
        score.id,
        performance.id,
    )
    assert (unperformed.event_a_id, unperformed.event_b_id) == ("n4", NOMATCH)

    aligned = [
        (xml_id, quarter, float(onset))
        for onset, ids, quarter, _ in ROWS
        for xml_id in ids
    ]
    aligned.append(("n7", Fraction(15, 2), 5.5))
    projections = performance.events.table.to_pylist()
    ids = [row["id"] for row in projections]
    assert shapes(ids) == ["projection"] * 16
    for n, (claim, (xml_id, quarter, seconds)) in enumerate(
        zip(claims[1:15], aligned, strict=True), start=1
    ):
        assert claim.is_synchronous
        assert (claim.event_a_id, claim.event_b_id) == (xml_id, ids[n - 1])
        assert claim.start_anchor.coordinate_a == Coordinate(quarter, TimeUnit.quarters)
        assert claim.start_anchor.coordinate_b == Coordinate(seconds, TimeUnit.seconds)

    # The file gives both inserted notes one id, so each Projection is minted
    # its own and the claim keeps the file's id as the event's name.
    for n, claim in enumerate(claims[15:], start=15):
        assert claim.is_synchronous is False
        assert (claim.start_anchor, claim.end_anchor) == (None, None)
        assert (claim.timeline_a_id, claim.timeline_b_id) == (performance.id, score.id)
        assert (claim.event_a_id, claim.event_b_id) == (ids[n - 1], NOMATCH)
        assert claim.event_a_name == INSERTED

    # One Projection per performed note: 14 of score notes, 2 of none.
    assert len(projections) == 16
    assert [row["projection_of"] for row in projections[14:]] == [None, None]
    assert [row["start"]["value"] for row in projections[14:]] == [2.75, 4.75]


def test_a_single_inserted_note_keeps_its_id(score: Timeline, tmp_path: Path) -> None:
    """An inserted id the file gives once is the Projection's id."""
    rows = _maps_rows() + [{"obs_mean_onset": 2.75, "xml_id": INSERTED, "obs_num": 8}]
    performance = _samples_timeline()
    claims = MapsLoader.from_file(
        _write(tmp_path / "one.json", rows)
    ).create_match_claims(score, performance)
    inserted = claims[-1]
    assert (inserted.event_a_id, inserted.event_a_name) == (INSERTED, None)
    row = performance.events.table.to_pylist()[-1]
    assert (row["id"], row["projection_of"], row["start"]["numerator"]) == (
        INSERTED,
        None,
        121275,
    )


@pytest.mark.parametrize("single", [False, True], ids=["repeated-id", "kept-id"])
def test_a_second_call_reuses_every_projection(
    score: Timeline, tmp_path: Path, single: bool
) -> None:
    rows = _stated_rows()
    if single:
        rows = rows[:-1]
    loader = MapsLoader.from_file(_write(tmp_path / "stated.json", rows))
    performance = _samples_timeline()

    def stated(claims: list[MatchClaim]) -> list[tuple[Any, ...]]:
        return [(*_stated(claim), claim.event_b_id) for claim in claims]

    first = loader.create_match_claims(score, performance)
    held = performance.events.table.to_pylist()
    again = loader.create_match_claims(score, performance)
    assert stated(again) == stated(first)
    assert performance.events.table.to_pylist() == held
    assert len(held) == (15 if single else 16)
    if single:
        assert [first[-1].event_a_id, again[-1].event_a_id] == [INSERTED, INSERTED]


def test_a_second_reading_mints_the_same_ids(
    score: Timeline, stated_path: Path
) -> None:
    def read() -> tuple[list[str], list[str]]:
        performance = _seconds_timeline()
        claims = MapsLoader.from_file(stated_path).create_match_claims(
            score, performance
        )
        return [claim.id for claim in claims], performance.events.table.column(
            "id"
        ).to_pylist()

    first_claims, first_projections = read()
    again_claims, again_projections = read()
    assert again_claims == first_claims
    assert again_projections == first_projections
    assert len(set(first_claims)) == 17
    assert len(first_projections) == 16


def test_kinds_are_logged(stated_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="timetoalign"):
        MapsLoader.from_file(stated_path)
    assert [(record.levelname, record.getMessage()) for record in caplog.records] == [
        (
            "INFO",
            "stated.json: 14 score note(s) at a time of the recording, 1 score "
            "note(s) not performed (negative onset), 2 performed note(s) not in "
            "the score",
        ),
        (
            "WARNING",
            "stated.json lists 1 xml:id(s) in more than one row; each row's claim "
            "is kept: 'n7' in rows 8, 9",
        ),
    ]


def _stated(claim: MatchClaim) -> tuple[Any, ...]:
    """What a claim states, without its generated id."""
    anchor = claim.start_anchor
    return (
        claim.timeline_a_id,
        claim.timeline_b_id,
        claim.event_a_id,
        claim.is_synchronous,
        None if anchor is None else (anchor.coordinate_a, anchor.coordinate_b),
    )


def test_create_bundle(score: Timeline, stated_path: Path) -> None:
    performance = _samples_timeline()
    loader = MapsLoader.from_file(stated_path)
    bundle = loader.create_bundle(score, performance)
    assert bundle.name == "stated"
    assert bundle.get_group_for_timeline(score.id).name == "score"
    assert bundle.get_group_for_timeline(performance.id).name == "performance"
    # Claims carry generated ids, so they are compared by what they state;
    # made again on the same recording, they name the same Projections.
    assert [_stated(claim) for claim in bundle.get_match_claims()] == [
        _stated(claim) for claim in loader.create_match_claims(score, performance)
    ]
    for quarter, sample, interpolated in [
        (Fraction(3, 2), 55125, False),
        (Fraction(15, 4), 121275, True),
    ]:
        stamp = bundle.get_matchstamp_at(quarter, score.id)
        assert stamp.get_coordinate_for(
            performance.id, format="coordinate"
        ) == Coordinate(sample, TimeUnit.samples)
        assert stamp.is_interpolated is interpolated


def test_the_bundle_s_package_is_written_and_read_back(
    score: Timeline, stated_path: Path, tmp_path: Path
) -> None:
    """The performance's Projections hold IDs the score's own does not."""
    (own,) = score.add_projections(
        Projection(start=Coordinate(Fraction(3, 2), TimeUnit.quarters))
    )
    performance = _samples_timeline()
    bundle = MapsLoader.from_file(stated_path).create_bundle(score, performance)
    claims = bundle.get_match_claims()
    projected = [claim.event_b_id for claim in claims[1:15]] + [
        claim.event_a_id for claim in claims[15:]
    ]
    assert len(set(projected)) == 16
    assert own not in projected
    assert set(shapes([own, *projected])) == {"projection"}

    bundle.to_file(tmp_path / "package" / "package.yaml")
    table = pq.read_table(tmp_path / "package" / "projections.parquet")
    written = list(
        zip(table.column("timeline_id").to_pylist(), table.column("id").to_pylist())
    )
    assert sorted(written) == sorted(
        [(performance.id, event_id) for event_id in projected] + [(score.id, own)]
    )
    read = AlignmentBundle.from_file(tmp_path / "package" / "package.yaml")
    assert [_stated(claim) for claim in read.get_match_claims()] == [
        _stated(claim) for claim in claims
    ]


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
    (movement,) = score.list_children()
    assert stamp.present_timelines == [score.id, movement]
    assert stamp.get_coordinate_for(movement).value == Fraction(1, 4)
    with pytest.raises(KeyError):
        stamp.get_coordinate_for(performance.id)


# endregion

# region Refusals


def test_unknown_id_is_refused(score: Timeline, tmp_path: Path) -> None:
    rows = _maps_rows()
    rows[3]["xml_id"] = ["zz9"]
    loader = MapsLoader.from_file(_write(tmp_path / "unknown.json", rows))
    performance = _seconds_timeline()
    with pytest.raises(KeyError) as raised:
        loader.create_match_claims(score, performance)
    assert raised.value.args[0] == (
        f"Pair 8: no event with id 'zz9' on timeline '{score.id}' or its children "
        "(0 events of its own)"
    )
    # Rows 1-3 were read before the refusal; none of their Projections stays.
    assert len(performance.events) == 0


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
        (_changed(lambda r: r[4].update(xml_id="")), r"MAPS row 5 .*xml_id must be"),
        (_changed(lambda r: r[4].update(xml_id=7)), r"MAPS row 5 .*xml_id must be"),
        ({"rows": _maps_rows()}, r"holds a JSON list of rows"),
        (
            _changed(lambda r: r[5].update(obs_mean_onset=-1, xml_id=[INSERTED])),
            r"MAPS row 6 .* states the inserted note\(s\) trompa-align_inserted_Cs4 "
            r"at the negative onset -1",
        ),
    ],
    ids=[
        "obs_num-gap",
        "missing-key",
        "string-onset",
        "boolean-onset",
        "no-ids",
        "empty-id",
        "empty-string-id",
        "number-id",
        "not-a-list",
        "inserted-at-negative-onset",
    ],
)
def test_malformed_rows_are_refused(tmp_path: Path, rows: Any, message: str) -> None:
    path = _write(tmp_path / "malformed.json", rows)
    with pytest.raises(ValueError, match=message):
        MapsLoader.from_file(path)


def test_a_second_source_is_refused(tmp_path: Path, maps_path: Path) -> None:
    other = _write(tmp_path / "other.json", _maps_rows())
    loader = MapsLoader.from_file(maps_path)
    with pytest.raises(ValueError) as raised:
        loader.load(other)
    assert str(raised.value) == (
        f"MapsLoader reads one source and already holds {maps_path}; use a new "
        f"loader for {other}, or call clear() first."
    )
    assert loader.rows == _maps_rows()
    loader.clear()
    assert len(loader) == 0
    assert loader.load(other).sources == [other]


def test_claims_before_load_are_refused(score: Timeline) -> None:
    with pytest.raises(RuntimeError, match=r"No MAPS file loaded"):
        MapsLoader().create_match_claims(score, _seconds_timeline())


# endregion
