"""VerovioLoader on MEI fixtures and specimens (validation logic: README.md)."""

from __future__ import annotations

import csv
import importlib
import logging
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from .conftest import FLOW_CONTROL_DIR, PARANGONAR_DIR, SCORE_DIR, VIENNA_DIR

verovio = pytest.importorskip("verovio")

from timetoalign.core.enums import FlowMode  # noqa: E402
from timetoalign.core.time import struct_to_rational  # noqa: E402
from timetoalign.loader._xml import XML_ID  # noqa: E402
from timetoalign.loader.score import Music21Loader, VerovioLoader  # noqa: E402
from timetoalign.timelines import (  # noqa: E402
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
)

PINNED_VERSION = "6.4.1"
on_pinned_version = pytest.mark.skipif(
    verovio.toolkit().getVersion().split("-")[0] != PINNED_VERSION,
    reason=f"golden value established with Verovio {PINNED_VERSION}",
)

_MEI_NS = "{http://www.music-encoding.org/ns/mei}"
_STEP_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.1">
<meiHead><fileDesc><titleStmt><title>Fixture</title></titleStmt><pubStmt/></fileDesc>
</meiHead><music><body>"""
_TAIL = "</body></music></mei>"


def _score(meter: str, measures: str, staves: int = 1, mdiv: str = "mdiv1") -> str:
    """Wrap measures in one movement with one or two staves."""
    count, unit = meter.split("/")
    staff_defs = "".join(
        f'<staffDef xml:id="sdef{n}" n="{n}" lines="5" clef.shape="{shape}" '
        f'clef.line="{line}"/>'
        for n, shape, line in ((1, "G", 2), (2, "F", 4))[:staves]
    )
    return (
        f'<mdiv xml:id="{mdiv}"><score><scoreDef xml:id="sd_{mdiv}" '
        f'meter.count="{count}" meter.unit="{unit}"><staffGrp xml:id="sg_{mdiv}">'
        f'{staff_defs}</staffGrp></scoreDef><section xml:id="s_{mdiv}">{measures}'
        "</section></score></mdiv>"
    )


def _bar(xml_id: str, n: int, pname: str, attrs: str = "", extra: str = "") -> str:
    """One 2/4 measure holding one half note ``{xml_id}n``."""
    return (
        f'<measure xml:id="{xml_id}" n="{n}" {attrs}><staff xml:id="{xml_id}s" n="1">'
        f'<layer xml:id="{xml_id}l" n="1"><note xml:id="{xml_id}n" pname="{pname}" '
        f'oct="4" dur="2"/></layer></staff>{extra}</measure>'
    )


BASIC = (
    _HEAD
    + _score(
        "9/8",
        """
<measure xml:id="m1" n="1">
  <staff xml:id="m1s1" n="1"><layer xml:id="m1s1l1" n="1">
    <rest xml:id="r1" dur="8"/>
    <chord xml:id="c1" dur="4" staff="2">
      <note xml:id="c1a" pname="c" oct="4"/>
      <note xml:id="c1b" pname="a" oct="3"><accid xml:id="c1bacc" accid.ges="f"/></note>
    </chord>
    <tuplet xml:id="tu1" num="2" numbase="3">
      <note xml:id="d1" pname="e" oct="5" dur="8"/>
      <note xml:id="d2" pname="f" oct="5" dur="8"/>
    </tuplet>
    <note xml:id="t1" pname="g" oct="4" dur="4" dots="1"/>
  </layer></staff>
  <staff xml:id="m1s2" n="2"><layer xml:id="m1s2l1" n="1">
    <note xml:id="b1" pname="c" oct="3" dur="2" dots="1"/>
    <note xml:id="b2" pname="g" oct="2" dur="4" dots="1"/>
  </layer></staff>
  <tempo xml:id="tempo1" staff="1" tstamp="1">Andante très expressif</tempo>
  <tie xml:id="tie1" startid="#t1" endid="#t2"/>
</measure>
<measure xml:id="m2" n="2">
  <staff xml:id="m2s1" n="1"><layer xml:id="m2s1l1" n="1">
    <note xml:id="t2" pname="g" oct="4" dur="4" dots="1"/>
    <note xml:id="t3" pname="a" oct="4" dur="2" dots="1"/>
  </layer></staff>
  <staff xml:id="m2s2" n="2"><layer xml:id="m2s2l1" n="1">
    <rest xml:id="r2" dur="4" dots="1"/>
    <note pname="f" oct="3" dur="2" dots="1" accid="s"/>
  </layer></staff>
  <tie xml:id="tie2" startid="#t3" endid="#outside"/>
</measure>""",
        staves=2,
    )
    + _TAIL
)

GRACE = (
    _HEAD
    + _score(
        "4/4",
        """
<measure xml:id="m1" n="1"><staff xml:id="m1s1" n="1"><layer xml:id="m1s1l1" n="1">
  <note xml:id="a1" pname="c" oct="4" dur="4"/>
  <graceGrp xml:id="gg1" grace="unacc"><note xml:id="g1" pname="e" oct="4" dur="8"/></graceGrp>
  <note xml:id="p1" pname="d" oct="4" dur="4"/>
  <graceGrp xml:id="gg2" grace="acc"><note xml:id="g2" pname="g" oct="4" dur="8"/></graceGrp>
  <note xml:id="p2" pname="f" oct="4" dur="2"/>
</layer></staff></measure>
<measure xml:id="m2" n="2"><staff xml:id="m2s1" n="1"><layer xml:id="m2s1l1" n="1">
  <graceGrp xml:id="gg3" grace="unacc"><note xml:id="g3" pname="b" oct="4" dur="16"/></graceGrp>
  <note xml:id="p3" pname="c" oct="5" dur="2"/>
  <chord xml:id="ch1" dur="2">
    <note xml:id="ch1a" pname="e" oct="4"/>
    <note xml:id="ch1b" pname="g" oct="4"/>
    <note xml:id="ch1c" pname="c" oct="5"/>
  </chord>
</layer></staff><arpeg xml:id="arp1" plist="#ch1a #ch1b #ch1c"/></measure>""",
    )
    + _TAIL
)

_TEMPO_BEAT_2 = '<tempo xml:id="tp1" staff="1" tstamp="2">Allegro</tempo>'
REPEAT = (
    _HEAD
    + _score(
        "2/4",
        _bar("m1", 1, "c", 'left="rptstart"', _TEMPO_BEAT_2)
        + _bar("m2", 2, "d", 'right="rptend"')
        + _bar("m3", 3, "e", 'right="end"'),
    )
    + _TAIL
)

REPEAT_ENDINGS = (
    _HEAD
    + _score(
        "2/4",
        _bar("m1", 1, "c", 'left="rptstart"')
        + _bar("m2", 2, "d", 'right="rptend"')
        + _bar("m3", 3, "e", 'left="rptstart"')
        + '<ending xml:id="e1" n="1">'
        + _bar("m4", 4, "f", 'right="rptend"')
        + '</ending><ending xml:id="e2" n="2">'
        + _bar("m5", 5, "g", 'right="end"')
        + "</ending>",
    )
    + _TAIL
)

TWO_MOVEMENTS = _HEAD + """
<mdiv xml:id="mov1" n="1" label="I"><score>
  <scoreDef xml:id="sd1" meter.count="2" meter.unit="4"><staffGrp xml:id="sg1">
    <staffDef xml:id="sdef1" n="1" lines="5" clef.shape="G" clef.line="2"/>
  </staffGrp></scoreDef>
  <section xml:id="s1">
    <measure xml:id="m1" n="1"><staff xml:id="m1s" n="1"><layer xml:id="m1l" n="1">
      <note xml:id="n1" pname="c" oct="4" dur="4"/><note xml:id="n2" pname="d" oct="4" dur="4"/>
    </layer></staff></measure>
    <measure xml:id="m2" n="2"><staff xml:id="m2s" n="1"><layer xml:id="m2l" n="1">
      <note xml:id="n3" pname="e" oct="4" dur="2"/>
    </layer></staff></measure>
  </section></score></mdiv>
<mdiv xml:id="mov2" n="2" label="II"><score>
  <scoreDef xml:id="sd2" meter.count="3" meter.unit="4"><staffGrp xml:id="sg2">
    <staffDef xml:id="sdef2" n="1" lines="5" clef.shape="G" clef.line="2"/>
  </staffGrp></scoreDef>
  <section xml:id="s2">
    <measure xml:id="m3" n="1"><staff xml:id="m3s" n="1"><layer xml:id="m3l" n="1">
      <note xml:id="n4" pname="f" oct="4" dur="2" dots="1"/>
    </layer></staff>
    <tempo xml:id="t2" staff="1" tstamp="1" midi.bpm="60">Adagio</tempo></measure>
  </section></score></mdiv>""" + _TAIL

FAST = (
    _HEAD
    + _score(
        "2/4",
        """
<measure xml:id="m1" n="1"><staff xml:id="m1s" n="1"><layer xml:id="m1l" n="1">
  <note xml:id="f1" pname="c" oct="4" dur="256"/>
  <note xml:id="f2" pname="d" oct="4" dur="256"/>
  <note xml:id="f3" pname="e" oct="4" dur="256"/>
  <rest xml:id="rf" dur="4"/>
  <rest xml:id="rg" dur="8"/>
  <rest xml:id="rh" dur="16"/>
  <rest xml:id="ri" dur="32"/>
  <rest xml:id="rj" dur="64"/>
  <rest xml:id="rk" dur="256"/>
</layer></staff>
<tempo xml:id="tp" staff="1" tstamp="1" midi.bpm="2000">Presto</tempo></measure>""",
    )
    + _TAIL
)

STARTID = (
    _HEAD
    + _score(
        "2/4",
        """
<measure xml:id="m1" n="1"><staff xml:id="m1s" n="1"><layer xml:id="m1l" n="1">
  <note xml:id="a1" pname="c" oct="4" dur="4"/>
  <chord xml:id="ch1" dur="4">
    <note xml:id="ch1a" pname="e" oct="4"/><note xml:id="ch1b" pname="g" oct="4"/>
  </chord>
</layer></staff>
<tempo xml:id="tc" staff="1" startid="#ch1">Allegro</tempo></measure>
<measure xml:id="m2" n="2"><staff xml:id="m2s" n="1"><layer xml:id="m2l" n="1">
  <tuplet xml:id="tu" num="3" numbase="2">
    <note xml:id="u1" pname="c" oct="5" dur="8"/>
    <note xml:id="u2" pname="d" oct="5" dur="8"/>
    <note xml:id="u3" pname="e" oct="5" dur="8"/>
  </tuplet>
  <note xml:id="a2" pname="f" oct="4" dur="4"/>
</layer></staff>
<tempo xml:id="tn" staff="1" startid="#u2">Meno mosso</tempo></measure>""",
    )
    + _TAIL
)


def _leaf(xml_id: str, bars: str, label: str | None = None) -> str:
    """One 2/4 ``<mdiv>`` holding *bars*, every id unique to it, named by *label*."""
    name = f' label="{label}"' if label is not None else ""
    return (
        f'<mdiv xml:id="{xml_id}"{name}><score><scoreDef xml:id="sd_{xml_id}" '
        f'meter.count="2" meter.unit="4"><staffGrp xml:id="sg_{xml_id}"><staffDef '
        f'xml:id="sdef_{xml_id}" n="1" lines="5" clef.shape="G" clef.line="2"/>'
        f'</staffGrp></scoreDef><section xml:id="s_{xml_id}">{bars}</section>'
        "</score></mdiv>"
    )


def _div(xml_id: str, *parts: str, label: str | None = None) -> str:
    """One ``<mdiv>`` holding the ``<mdiv>``s *parts*, named by *label*."""
    name = f' label="{label}"' if label is not None else ""
    return f'<mdiv xml:id="{xml_id}"{name}>{"".join(parts)}</mdiv>'


NAMED = _HEAD + _leaf("solo", _bar("m1", 1, "c") + _bar("m2", 2, "d"), "Rondo") + _TAIL
SIBLINGS = (
    _HEAD
    + _leaf("first", _bar("m1", 1, "c") + _bar("m2", 2, "d"), "I")
    + _leaf("second", _bar("m3", 3, "e"))
    + _TAIL
)
NESTED = (
    _HEAD
    + _div(
        "work",
        _div(
            "set",
            _div(
                "group",
                _leaf("v1", _bar("m1", 1, "c") + _bar("m2", 2, "d"), "Var. 1"),
                _leaf("v2", _bar("m3", 3, "e")),
            ),
            label="Variationen",
        ),
        _leaf("coda", _bar("m4", 4, "f")),
    )
    + _TAIL
)

DUPLICATE_DYNAM = REPEAT.replace(
    _TEMPO_BEAT_2,
    '<dynam xml:id="dd" staff="1" tstamp="1">p</dynam>'
    '<dynam xml:id="dd" staff="1" tstamp="2">f</dynam>' + _TEMPO_BEAT_2,
)
DUPLICATE_NOTE = REPEAT.replace('<note xml:id="m2n"', '<note xml:id="m1n"')

FIXTURES = {
    "basic": BASIC,
    "grace": GRACE,
    "repeat": REPEAT,
    "repeat_endings": REPEAT_ENDINGS,
    "two_movements": TWO_MOVEMENTS,
    "named": NAMED,
    "siblings": SIBLINGS,
    "nested": NESTED,
    "fast": FAST,
    "startid": STARTID,
    "duplicate_dynam": DUPLICATE_DYNAM,
    "duplicate_note": DUPLICATE_NOTE,
}

GENERATED_ID = "s3clx4n"
BASIC_SPANS = {
    "r1": (Fraction(0), Fraction(1, 2)),
    "c1a": (Fraction(1, 2), Fraction(3, 2)),
    "c1b": (Fraction(1, 2), Fraction(3, 2)),
    "d1": (Fraction(3, 2), Fraction(9, 4)),
    "d2": (Fraction(9, 4), Fraction(3)),
    "t1": (Fraction(3), Fraction(9, 2)),
    "b1": (Fraction(0), Fraction(3)),
    "b2": (Fraction(3), Fraction(9, 2)),
    "t2": (Fraction(9, 2), Fraction(6)),
    "t3": (Fraction(6), Fraction(9)),
    "r2": (Fraction(9, 2), Fraction(6)),
}


@pytest.fixture(scope="module")
def mei(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("verovio")
    paths = {}
    for name, text in FIXTURES.items():
        paths[name] = folder / f"{name}.mei"
        paths[name].write_text(text, encoding="utf-8")
    return paths


@pytest.fixture(scope="module")
def loader(mei: dict[str, Path]) -> VerovioLoader:
    return VerovioLoader.from_file(mei["basic"])


@pytest.fixture(scope="module")
def timeline(loader: VerovioLoader) -> ContinuousLogicalTimeline:
    return loader.create_timeline()


@pytest.fixture(scope="module")
def events(timeline: ContinuousLogicalTimeline) -> dict[str, dict]:
    return {row["id"]: row for row in timeline.get_events(include_children=True)}


def _q(cell: dict | None) -> Fraction:
    return Fraction(struct_to_rational(cell))


def _span(row: dict) -> tuple[Fraction, Fraction]:
    return _q(row["start"]), _q(row["end"])


def _fixture_ids(path: Path, tag: str) -> set[str]:
    root = ET.parse(path).getroot()
    return {el.get(XML_ID) for el in root.iter(f"{_MEI_NS}{tag}") if el.get(XML_ID)}


def _generated_id(events: dict[str, dict], mei: dict[str, Path]) -> str:
    notes = {i for i, row in events.items() if row["event_type"] == "Note"}
    (generated,) = notes - _fixture_ids(mei["basic"], "note")
    return generated


class TestTimeline:
    def test_unnamed_movement_makes_no_child(
        self, timeline: ContinuousLogicalTimeline, events: dict[str, dict]
    ) -> None:
        assert type(timeline) is ContinuousLogicalTimeline
        assert timeline.id == "score:clt_1"
        assert str(timeline.unit) == "quarters"
        assert str(timeline.number_type) == "fraction"
        assert timeline.length.value == Fraction(9)
        assert timeline.list_children() == []
        own = [row["id"] for row in timeline.get_events(include_children=False)]
        assert len(own) == 13
        assert set(own) == set(events)

    def test_event_types(self, events: dict[str, dict]) -> None:
        assert Counter(row["event_type"] for row in events.values()) == {
            "Note": 10,
            "Rest": 2,
            "tempo indication": 1,
        }

    def test_uid_and_name_kept(self, mei: dict[str, Path]) -> None:
        timeline = VerovioLoader.from_file(mei["basic"]).create_timeline(
            uid="clt_1", name="Fixture"
        )
        assert (timeline.id, timeline.name) == ("clt_1", "Fixture")
        assert timeline.list_children() == []

    def test_options_recorded(self, loader: VerovioLoader) -> None:
        metadata = loader.store.metadata
        assert metadata["verovio_version"] == verovio.toolkit().getVersion()
        assert metadata["verovio_options"] == {"xmlIdChecksum": True, "mdivAll": True}

    def test_from_file_equals_load(
        self, mei: dict[str, Path], events: dict[str, dict]
    ) -> None:
        loaded = VerovioLoader().load(mei["basic"]).create_timeline()
        assert {r["id"]: r for r in loaded.get_events(include_children=True)} == events

    def test_one_score_per_loader(self, mei: dict[str, Path]) -> None:
        with pytest.raises(ValueError, match="reads one source"):
            VerovioLoader().load(mei["basic"], mei["grace"])


class TestMeasuresAreStructure:
    def test_measure_map(self, timeline: ContinuousLogicalTimeline) -> None:
        measures = timeline.measure_map.measures
        assert [m.id for m in measures] == ["m1", "m2"]
        assert [m.name for m in measures] == ["1", "2"]
        assert [m.qstamp for m in measures] == [Fraction(0), Fraction(9, 2)]
        assert [m.actual_length for m in measures] == [Fraction(9, 2)] * 2
        assert {m.time_signature for m in measures} == {"9/8"}

    def test_measures_are_no_events(self, events: dict[str, dict]) -> None:
        assert {"m1", "m2"}.isdisjoint(events)

    def test_no_flow_control(self, timeline: ContinuousLogicalTimeline) -> None:
        registry = timeline.flow_control
        assert (len(registry.breaks), len(registry.jumps), len(registry.markers)) == (
            0,
            0,
            0,
        )
        assert timeline.list_flow_maps() == []


class TestIdsAndOnsets:
    def test_note_ids_verbatim(
        self, events: dict[str, dict], mei: dict[str, Path]
    ) -> None:
        notes = {i for i, row in events.items() if row["event_type"] == "Note"}
        assert notes == _fixture_ids(mei["basic"], "note") | {
            _generated_id(events, mei)
        }
        rests = {i for i, row in events.items() if row["event_type"] == "Rest"}
        assert rests == {"r1", "r2"}
        assert all(row["notated_id"] == i for i, row in events.items())

    def test_generated_id_reproducible(
        self, events: dict[str, dict], mei: dict[str, Path]
    ) -> None:
        again = VerovioLoader.from_file(mei["basic"]).create_timeline()
        rows = {row["id"]: row for row in again.get_events(include_children=True)}
        assert _generated_id(rows, mei) == _generated_id(events, mei)

    @on_pinned_version
    def test_generated_id_pinned(
        self, events: dict[str, dict], mei: dict[str, Path]
    ) -> None:
        assert _generated_id(events, mei) == GENERATED_ID

    def test_spans_exact(self, events: dict[str, dict], mei: dict[str, Path]) -> None:
        spans = {**BASIC_SPANS, _generated_id(events, mei): (Fraction(6), Fraction(9))}
        assert {
            i: _span(row)
            for i, row in events.items()
            if row["temporal_type"] == "interval"
        } == spans

    def test_duplet_cell(self, events: dict[str, dict]) -> None:
        cell = events["d2"]["start"]
        assert (cell["numerator"], cell["denominator"]) == (9, 4)

    def test_measure_offsets(self, events: dict[str, dict]) -> None:
        assert _q(events["t2"]["mc_onset"]) == Fraction(0)
        assert _q(events["t3"]["mc_onset"]) == Fraction(3, 2)
        assert {row.get("mn_onset") for row in events.values()} == {None}

    def test_tempo_indication(self, events: dict[str, dict]) -> None:
        tempo = events["tempo1"]
        assert tempo["event_type"] == "tempo indication"
        assert tempo["text"] == "Andante très expressif"
        assert tempo["temporal_type"] == "instant"
        assert _q(tempo["start"]) == Fraction(0)

    def test_key_query_reaches_the_child(
        self, timeline: ContinuousLogicalTimeline
    ) -> None:
        coordinate = timeline.get_coordinate_for("d2")
        assert coordinate.value == Fraction(9, 4)
        assert isinstance(coordinate.value, Fraction)


class TestTies:
    def test_tie_marks(self, events: dict[str, dict]) -> None:
        marks = {
            i: row["tie"] for i, row in events.items() if row["event_type"] == "Note"
        }
        assert {i: mark for i, mark in marks.items() if mark} == {
            "t1": "i",
            "t2": "t",
            "t3": "i",
        }

    def test_dangling_tie(self, events: dict[str, dict], loader: VerovioLoader) -> None:
        assert _q(events["t3"]["end"]) == Fraction(9)
        assert "outside" not in events
        assert loader.store.metadata["dangling_ties"] == ["tie2"]

    def test_dangling_tie_logged(
        self, mei: dict[str, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(
            logging.WARNING, logger="timetoalign.loader.score.verovio"
        ):
            VerovioLoader.from_file(mei["basic"])
        assert [r.getMessage().split(" in ")[0] for r in caplog.records] == [
            "Tie 'tie2'"
        ]


class TestStartIdAndTriplet:
    @pytest.fixture(scope="class")
    def startid(self, mei: dict[str, Path]) -> dict[str, dict]:
        timeline = VerovioLoader.from_file(mei["startid"]).create_timeline()
        return {row["id"]: row for row in timeline.get_events(include_children=True)}

    def test_triplet_thirds(self, startid: dict[str, dict]) -> None:
        assert {i: _span(startid[i]) for i in ("u1", "u2", "u3")} == {
            "u1": (Fraction(2), Fraction(7, 3)),
            "u2": (Fraction(7, 3), Fraction(8, 3)),
            "u3": (Fraction(8, 3), Fraction(3)),
        }
        start, offset = startid["u2"]["start"], startid["u2"]["mc_onset"]
        assert (start["numerator"], start["denominator"]) == (7, 3)
        assert (offset["numerator"], offset["denominator"]) == (1, 3)

    def test_tempo_on_a_chord(self, startid: dict[str, dict]) -> None:
        assert startid["tc"]["text"] == "Allegro"
        assert _q(startid["tc"]["start"]) == _q(startid["ch1a"]["start"]) == 1

    def test_tempo_on_a_note(self, startid: dict[str, dict]) -> None:
        assert startid["tn"]["text"] == "Meno mosso"
        assert _q(startid["tn"]["start"]) == Fraction(7, 3)
        assert _q(startid["tn"]["mc_onset"]) == Fraction(1, 3)

    def test_chord_ids(self, startid: dict[str, dict]) -> None:
        chords = {i: row["chord_id"] for i, row in startid.items() if row["chord_id"]}
        assert chords == {"ch1a": "ch1", "ch1b": "ch1"}


class TestStaffAndPitch:
    def test_cross_staff_chord(self, events: dict[str, dict]) -> None:
        for xml_id in ("c1a", "c1b"):
            row = events[xml_id]
            assert (row["staff"], row["voice"], row["chord_id"]) == (2, 1, "c1")
        assert (events["r1"]["staff"], events["r1"]["voice"]) == (1, 1)
        chords = {i: row["chord_id"] for i, row in events.items() if row["chord_id"]}
        assert chords == {"c1a": "c1", "c1b": "c1"}

    def test_spelling(self, events: dict[str, dict], mei: dict[str, Path]) -> None:
        flat = events["c1b"]
        assert flat["name"] == "A♭3"
        assert flat["specific_pitch"] == {
            "step": "A",
            "alter": -1,
            "octave": 3,
            "cents": 0.0,
        }
        assert flat["midi"] == 56
        sharp = events[_generated_id(events, mei)]
        assert (sharp["name"], sharp["midi"]) == ("F♯3", 54)

    def test_midi_matches_spelling(self, events: dict[str, dict]) -> None:
        notes = [row for row in events.values() if row["event_type"] == "Note"]
        for row in notes:
            pitch = row["specific_pitch"]
            spelled = 12 * (pitch["octave"] + 1) + _STEP_SEMITONES[pitch["step"]]
            assert spelled + pitch["alter"] == row["midi"], row["id"]
        assert set("".join(row["name"] for row in notes)) <= set("ABCDEFG0123456789♯♭")


class TestPhysicalTimeline:
    @pytest.fixture(scope="class")
    def physical(self, loader: VerovioLoader) -> ContinuousPhysicalTimeline:
        return loader.create_physical_timeline()

    def test_root(self, physical: ContinuousPhysicalTimeline) -> None:
        assert type(physical) is ContinuousPhysicalTimeline
        assert physical.id == "rendered:cpt_1"
        assert str(physical.unit) == "seconds"
        assert str(physical.number_type) == "fraction"
        assert physical.length.value == Fraction(9, 2)
        assert physical.meta == {"provenance": "verovio_rendered"}
        assert physical.list_children() == []
        assert physical.measure_map is None

    def test_events_at_half_their_quarters(
        self,
        physical: ContinuousPhysicalTimeline,
        events: dict[str, dict],
    ) -> None:
        seconds = {row["id"]: row for row in physical.get_events(include_children=True)}
        assert set(seconds) == set(events)
        assert {row.get("mc_onset") for row in seconds.values()} == {None}
        assert {row["mc"] for row in seconds.values()} == {1, 2}
        for xml_id, row in events.items():
            assert _q(seconds[xml_id]["start"]) == _q(row["start"]) / 2, xml_id
            if row["temporal_type"] == "interval":
                assert _q(seconds[xml_id]["end"]) == _q(row["end"]) / 2, xml_id
        assert _span(seconds["d2"]) == (Fraction(9, 8), Fraction(3, 2))

    def test_conversion_maps(
        self,
        timeline: ContinuousLogicalTimeline,
        physical: ContinuousPhysicalTimeline,
    ) -> None:
        forward = timeline.get_conversion_map("seconds")
        inverse = physical.get_conversion_map("quarters")
        assert (timeline.n_conversion_maps, physical.n_conversion_maps) == (1, 1)
        for value, expected in (
            (forward(Fraction(9, 4)), Fraction(9, 8)),
            (inverse(Fraction(9, 8)), Fraction(9, 4)),
        ):
            assert (type(value), value) == (Fraction, expected)

    def test_conversion_maps_exact_between_anchors(self, mei: dict[str, Path]) -> None:
        loader = VerovioLoader.from_file(mei["two_movements"])
        forward = loader.create_timeline().get_conversion_map("seconds")
        inverse = loader.create_physical_timeline().get_conversion_map("quarters")
        pairs = [
            (Fraction(4), Fraction(2)),
            (Fraction(10, 3), Fraction(5, 3)),
            (Fraction(13, 3), Fraction(7, 3)),
        ]
        for quarters, seconds in pairs:
            assert (type(forward(quarters)), forward(quarters)) == (Fraction, seconds)
            assert (type(inverse(seconds)), inverse(seconds)) == (Fraction, quarters)

    def test_attach_cmap_false(self, mei: dict[str, Path]) -> None:
        loader = VerovioLoader.from_file(mei["basic"])
        assert loader.create_timeline(attach_cmap=False).n_conversion_maps == 0
        assert loader.create_physical_timeline(attach_cmap=False).n_conversion_maps == 0

    def test_coinciding_tstamps(self, mei: dict[str, Path]) -> None:
        fast = VerovioLoader.from_file(mei["fast"])
        assert fast.create_timeline().length.value == Fraction(2)
        with pytest.raises(ValueError, match="one tstamp") as raised:
            fast.create_physical_timeline()
        assert "('0', '1/64', '0')" in str(raised.value)
        assert "('1/32', '3/64', '1/1000')" in str(raised.value)


class TestGraceNotesAsVerovioPlacesThem:
    @pytest.fixture(scope="class")
    def grace(self, mei: dict[str, Path]) -> dict[str, dict]:
        timeline = VerovioLoader.from_file(mei["grace"]).create_timeline()
        return {row["id"]: row for row in timeline.get_events(include_children=True)}

    def test_spans(self, grace: dict[str, dict]) -> None:
        third = Fraction(15, 256)
        assert {i: _span(row) for i, row in grace.items()} == {
            "a1": (Fraction(0), Fraction(1)),
            "g1": (1 - third, Fraction(1)),
            "p1": (Fraction(1), Fraction(2)),
            "g2": (Fraction(2), Fraction(3)),
            "p2": (Fraction(3), Fraction(4)),
            "g3": (Fraction(4), 4 + third),
            "p3": (4 + third, Fraction(6)),
            "ch1a": (Fraction(6), Fraction(8)),
            "ch1b": (6 + third, Fraction(8)),
            "ch1c": (6 + 2 * third, Fraction(8)),
        }

    def test_seconds_exact(self, mei: dict[str, Path]) -> None:
        physical = VerovioLoader.from_file(mei["grace"]).create_physical_timeline()
        rows = {row["id"]: row for row in physical.get_events(include_children=True)}
        # 241/256 quarter at 500 ms per quarter is 470.7 ms, which Verovio rounds.
        assert _span(rows["g1"]) == (Fraction(471, 1000), Fraction(1, 2))
        assert _q(rows["p3"]["start"]) == Fraction(2029, 1000)

    def test_grace_field(self, grace: dict[str, dict]) -> None:
        assert {i: row["grace"] for i, row in grace.items() if row["grace"]} == {
            "g1": "unacc",
            "g2": "acc",
            "g3": "unacc",
        }


class TestRepeats:
    @pytest.fixture(scope="class")
    def repeat(self, mei: dict[str, Path]) -> VerovioLoader:
        return VerovioLoader.from_file(mei["repeat"])

    @pytest.fixture(scope="class")
    def rendered(self, repeat: VerovioLoader) -> ContinuousLogicalTimeline:
        return repeat.create_timeline()

    def test_rendered_events(self, rendered: ContinuousLogicalTimeline) -> None:
        rows = rendered.get_events(include_children=True)
        assert rendered.length.value == Fraction(10)
        assert {row["id"]: (_q(row["start"]), row["notated_id"]) for row in rows} == {
            "m1n": (Fraction(0), "m1n"),
            "m2n": (Fraction(2), "m2n"),
            "m1n-rend2": (Fraction(4), "m1n"),
            "m2n-rend2": (Fraction(6), "m2n"),
            "m3n": (Fraction(8), "m3n"),
            "tp1": (Fraction(1), "tp1"),
            "tp1-rend2": (Fraction(5), "tp1"),
        }

    def test_rendered_measure_map(self, rendered: ContinuousLogicalTimeline) -> None:
        measures = rendered.measure_map.measures
        assert [m.id for m in measures] == ["m1", "m2", "m1-rend2", "m2-rend2", "m3"]
        assert [m.name for m in measures] == ["1", "2", "1", "2", "3"]
        assert {m.time_signature for m in measures} == {"2/4"}
        assert [m.qstamp for m in measures] == [Fraction(2 * i) for i in range(5)]
        registry = rendered.flow_control
        assert (len(registry.breaks), len(registry.jumps), len(registry.markers)) == (
            0,
            0,
            0,
        )

    def test_flow_maps(self, rendered: ContinuousLogicalTimeline) -> None:
        assert sorted(rendered.list_flow_maps()) == ["forward_verovio", "source"]
        forward = rendered.get_flow_map("forward_verovio")
        assert [
            (s.source_start, s.source_end, s.target_start) for s in forward._sections
        ] == [
            (Fraction(0), Fraction(4), Fraction(0)),
            (Fraction(0), Fraction(6), Fraction(4)),
        ]
        assert forward.unfold_coordinate(1) == [Fraction(1), Fraction(5)]
        assert rendered.get_flow_map("source").unfold_coordinate(5) == [Fraction(1)]

    def test_printed_table_unfolds_to_verovio_order(
        self, repeat: VerovioLoader
    ) -> None:
        rows = list(repeat.store.measures)
        assert [(r["id"], _q(r["start"])) for r in rows] == [
            ("m1", Fraction(0)),
            ("m2", Fraction(2)),
            ("m3", Fraction(4)),
        ]
        assert [(r["start_repeat"], r["end_repeat"]) for r in rows] == [
            (True, False),
            (False, True),
            (False, False),
        ]
        assert _flow(repeat) == [1, 2, 1, 2, 3]

    def test_expand_never(self, mei: dict[str, Path]) -> None:
        loader = VerovioLoader.from_file(mei["repeat"], options={"expandNever": True})
        timeline = loader.create_timeline()
        rows = timeline.get_events(include_children=True)
        assert sorted(row["id"] for row in rows) == ["m1n", "m2n", "m3n", "tp1"]
        assert [m.id for m in timeline.measure_map.measures] == ["m1", "m2", "m3"]
        assert len(timeline.flow_control.jumps) == 1
        assert timeline.list_flow_maps() == []
        assert loader.store.metadata["verovio_options"] == {
            "expandNever": True,
            "xmlIdChecksum": True,
            "mdivAll": True,
        }

    def test_omitted_endings_read_unexpanded(
        self, mei: dict[str, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(
            logging.WARNING, logger="timetoalign.loader.score.verovio"
        ):
            loader = VerovioLoader.from_file(mei["repeat_endings"])
        assert loader.store.metadata["omitted_measures"] == ["m4", "m5"]
        assert ["leaves out 2 printed" in r.getMessage() for r in caplog.records] == [
            True
        ]
        rows = list(loader.store.measures)
        assert [(r["id"], _q(r["start"]), _q(r["duration"])) for r in rows] == [
            (f"m{mc}", Fraction(2 * mc - 2), Fraction(2)) for mc in range(1, 6)
        ]
        timeline = loader.create_timeline()
        assert timeline.length.value == Fraction(10)
        assert [m.id for m in timeline.measure_map.measures] == [
            "m1",
            "m2",
            "m1-rend2",
            "m2-rend2",
            "m3",
        ]
        events = timeline.get_events(include_children=True)
        assert sorted(row["id"] for row in events) == [
            "m1n",
            "m1n-rend2",
            "m2n",
            "m2n-rend2",
            "m3n",
        ]
        assert _flow(loader) == [1, 2, 1, 2, 3, 4, 3, 5]
        notated = VerovioLoader.from_file(
            mei["repeat_endings"], options={"expandNever": True}
        )
        assert notated.store.metadata["omitted_measures"] == []
        assert list(notated.store.measures) == rows

    def test_expand_always_refused(self, mei: dict[str, Path]) -> None:
        with pytest.raises(ValueError, match="expandAlways"):
            VerovioLoader.from_file(mei["repeat"], options={"expandAlways": True})


def _flow(loader: VerovioLoader) -> list[int]:
    """Return the MC sequence of the printed table's default flow."""
    flow = loader.create_flow_controller().compute_flow(FlowMode.default)
    return [
        mc
        for section in flow.sections
        for mc in range(section.mc_start, section.mc_end)
    ]


class TestMovements:
    @pytest.fixture
    def movements(self, mei: dict[str, Path]) -> VerovioLoader:
        return VerovioLoader.from_file(mei["two_movements"])

    def test_children(self, movements: VerovioLoader) -> None:
        root = movements.create_timeline()
        assert root.length.value == Fraction(7)
        assert root.list_children() == ["score:clt_2", "score:clt_3"]
        first, second = (root.get_child(c) for c in root.list_children())
        assert (first.name, first.length.value) == ("I", Fraction(4))
        assert (second.name, second.length.value) == ("II", Fraction(3))
        assert first.meta == {
            "count": 1,
            "xml_id": "mov1",
            "n": "1",
            "label": "I",
            "element": "mdiv",
        }
        assert second.meta == {
            "count": 2,
            "xml_id": "mov2",
            "n": "2",
            "label": "II",
            "element": "mdiv",
        }
        assert root.get_coordinate_for("n4").value == Fraction(4)
        assert second.get_coordinate_for("n4").value == Fraction(0)
        assert {row["id"] for row in second.get_events(include_children=False)} == {
            "n4",
            "t2",
        }

    def test_root_structure_serves_the_children(self, movements: VerovioLoader) -> None:
        root = movements.create_timeline(uid="score")
        assert [m.qstamp for m in root.measure_map.measures] == [
            Fraction(0),
            Fraction(2),
            Fraction(4),
        ]
        table = root.get_timestamp_table(format="dataframe")
        assert table.loc["n4", "quarters_to_seconds (seconds)"] == 2
        assert root.list_children() == ["score:clt_2", "score:clt_3"]
        child = root.get_child("score:clt_3")
        assert (child.measure_map, child.n_conversion_maps) == (None, 0)

    def test_physical_children(self, movements: VerovioLoader) -> None:
        physical = movements.create_physical_timeline(uid="rendered")
        assert physical.length.value == Fraction(5)
        assert physical.list_children() == ["rendered:cpt_2", "rendered:cpt_3"]
        second = physical.get_child("rendered:cpt_3")
        assert physical.get_coordinate_for("n4").value == Fraction(2)
        assert second.length.value == Fraction(3)

    def test_selected_movement(self, mei: dict[str, Path]) -> None:
        loader = VerovioLoader.from_file(
            mei["two_movements"], options={"mdivXPathQuery": "./mdiv[2]"}
        )
        root = loader.create_timeline()
        assert root.list_children() == ["score:clt_2"]
        assert root.get_child("score:clt_2").meta["count"] == 2
        assert root.length.value == Fraction(3)
        assert root.get_coordinate_for("n4").value == Fraction(0)

    def test_nested_movements(self) -> None:
        path = PARANGONAR_DIR / "Beethoven_Eroica_op35-cpjku" / "score.mei"
        root = VerovioLoader.from_file(path).create_timeline()
        assert root.length.value == Fraction(129, 2)
        assert root.list_children() == ["score:clt_2"]
        inner = root.get_child("score:clt_2")
        assert (inner.name, inner.list_children()) == ("Var. XIV.", [])
        assert inner.meta == {
            "count": 2,
            "xml_id": "Var_14",
            "n": "18",
            "label": "Var. XIV.",
            "element": "mdiv",
        }
        assert root.get_child_offset("score:clt_2").value == 0
        assert inner.length.value == Fraction(129, 2)
        assert len(list(root.get_events(include_children=False))) == 0
        assert Counter(
            row["event_type"] for row in inner.get_events(include_children=False)
        ) == {"Note": 297, "Rest": 36, "tempo indication": 2}
        assert len(root.measure_map.measures) == 33


def _tree(timeline: ContinuousLogicalTimeline) -> dict:
    """Return each timeline of the tree with its name, meta, offset, length and events.

    Events are ``{id: position on that timeline}``; children nest under ``children``.
    """
    return {
        "name": timeline.name,
        "length": timeline.length.value,
        "events": {
            row["id"]: _q(row["start"])
            for row in timeline.get_events(include_children=False)
        },
        "children": {
            child_id: {
                "offset": timeline.get_child_offset(child_id).value,
                "meta": timeline.get_child(child_id).meta,
                **_tree(timeline.get_child(child_id)),
            }
            for child_id in timeline.list_children()
        },
    }


def _root_positions(timeline: ContinuousLogicalTimeline) -> dict[str, Fraction]:
    """Return every event of the tree at its position on the root."""
    return {
        row["id"]: timeline.get_coordinate_for(row["id"]).value
        for row in timeline.get_events(include_children=True)
    }


class TestNamedMovements:
    """Only a named ``<mdiv>`` (``@label``) is a child (validation logic: README.md)."""

    def test_named_single(self, mei: dict[str, Path]) -> None:
        loader = VerovioLoader.from_file(mei["named"])
        root = loader.create_timeline()
        assert _tree(root) == {
            "name": "score:clt_1",
            "length": Fraction(4),
            "events": {},
            "children": {
                "score:clt_2": {
                    "offset": Fraction(0),
                    "meta": {
                        "count": 1,
                        "xml_id": "solo",
                        "label": "Rondo",
                        "element": "mdiv",
                    },
                    "name": "Rondo",
                    "length": Fraction(4),
                    "events": {"m1n": Fraction(0), "m2n": Fraction(2)},
                    "children": {},
                }
            },
        }
        physical = loader.create_physical_timeline()
        assert physical.list_children() == ["rendered:cpt_2"]
        child = physical.get_child("rendered:cpt_2")
        assert (child.name, child.length.value) == ("Rondo", Fraction(2))
        assert child.meta["element"] == "mdiv"

    def test_named_and_unnamed_siblings(self, mei: dict[str, Path]) -> None:
        root = VerovioLoader.from_file(mei["siblings"]).create_timeline()
        assert _tree(root) == {
            "name": "score:clt_1",
            "length": Fraction(6),
            "events": {"m3n": Fraction(4)},
            "children": {
                "score:clt_2": {
                    "offset": Fraction(0),
                    "meta": {
                        "count": 1,
                        "xml_id": "first",
                        "label": "I",
                        "element": "mdiv",
                    },
                    "name": "I",
                    "length": Fraction(4),
                    "events": {"m1n": Fraction(0), "m2n": Fraction(2)},
                    "children": {},
                }
            },
        }
        assert _root_positions(root) == {
            "m1n": Fraction(0),
            "m2n": Fraction(2),
            "m3n": Fraction(4),
        }

    def test_named_nested_in_unnamed(self, mei: dict[str, Path]) -> None:
        root = VerovioLoader.from_file(mei["nested"]).create_timeline()
        assert _tree(root) == {
            "name": "score:clt_1",
            "length": Fraction(8),
            "events": {"m4n": Fraction(6)},
            "children": {
                "score:clt_2": {
                    "offset": Fraction(0),
                    "meta": {
                        "count": 2,
                        "xml_id": "set",
                        "label": "Variationen",
                        "element": "mdiv",
                    },
                    "name": "Variationen",
                    "length": Fraction(6),
                    "events": {"m3n": Fraction(4)},
                    "children": {
                        "score:clt_3": {
                            "offset": Fraction(0),
                            "meta": {
                                "count": 4,
                                "xml_id": "v1",
                                "label": "Var. 1",
                                "element": "mdiv",
                            },
                            "name": "Var. 1",
                            "length": Fraction(4),
                            "events": {"m1n": Fraction(0), "m2n": Fraction(2)},
                            "children": {},
                        }
                    },
                }
            },
        }
        assert _root_positions(root) == {
            "m1n": Fraction(0),
            "m2n": Fraction(2),
            "m3n": Fraction(4),
            "m4n": Fraction(6),
        }


class TestIdCheck:
    def test_duplicate_outside_the_join_keys_listed(self, mei: dict[str, Path]) -> None:
        loader = VerovioLoader.from_file(mei["duplicate_dynam"])
        assert loader.store.metadata["duplicate_ids"] == ["dd"]

    def test_duplicate_join_key_refused(self, mei: dict[str, Path]) -> None:
        with pytest.raises(ValueError, match=r"\['m1n'\]"):
            VerovioLoader.from_file(mei["duplicate_note"])


_NAVIGATION = ("markers", "jump_bwd", "jump_fwd", "play_until", "next")


def _navigation(loader: VerovioLoader) -> dict[int, tuple]:
    """Return the navigation columns of the measures that have one."""
    return {
        row["mc"]: tuple(row[column] for column in _NAVIGATION)
        for row in loader.store.measures
        if any(row[column] for column in _NAVIGATION[:-1])
    }


class TestRepeatFacts:
    @pytest.fixture(scope="class")
    def flow_only(self) -> VerovioLoader:
        return VerovioLoader.from_file(
            FLOW_CONTROL_DIR / "out_of_the_flow_experience-flow_only.mei",
            options={"expandNever": True},
        )

    def test_flow_only_measure_rows_match_table(self, flow_only: VerovioLoader) -> None:
        tsv = (
            FLOW_CONTROL_DIR
            / "flow_only"
            / "out_of_the_flow_experience-flow_only-canonical.measures.tsv"
        )
        with tsv.open(newline="") as handle:
            table = list(csv.DictReader(handle, delimiter="\t"))
        rows = list(flow_only.store.measures)
        assert len(rows) == len(table) == 15
        facts = [
            (r["mc"], r["start_repeat"], r["end_repeat"], r["volta"]) for r in rows
        ]
        expected = [
            (
                mc,
                t["repeats"] in ("start", "startend"),
                t["repeats"] in ("end", "startend"),
                int(t["volta"]) if t["volta"] else None,
            )
            for mc, t in enumerate(table, start=1)
        ]
        assert facts == expected
        assert [mc for mc, start, _, _ in facts if start] == [4, 9, 10]
        assert [mc for mc, _, end, _ in facts if end] == [3, 5, 8, 10, 11, 14]

    def test_flow_only_navigation(self, flow_only: VerovioLoader) -> None:
        assert _navigation(flow_only) == {
            7: ("fine", None, None, None, "8, -1"),
            9: ("segno & coda", None, None, None, None),
            12: (None, "segno", "coda2", "coda", "9"),
            13: ("coda2", None, None, None, "14, 15"),
            15: (None, "start", None, "fine", "1"),
        }

    def test_jumps_play_until_the_fine(self) -> None:
        loader = VerovioLoader.from_file(
            SCORE_DIR / "couperin_concerts" / "c05n05_musete.mei",
            options={"expandNever": True},
        )
        assert _navigation(loader) == {
            6: ("segno", None, None, None, None),
            16: ("fine", None, None, None, "1, 17, -1"),
            17: ("segno2", None, None, None, None),
            31: (None, "start", None, "fine", "17, 1"),
            32: ("segno3", None, None, None, None),
            58: (None, "segno", None, "fine", "32, 6"),
        }


def _principal_onsets(rows: list[dict]) -> Counter:
    """Return Verovio's non-grace notes with each principal at its notated onset.

    A non-grace note starting where a grace note of its staff and voice ends
    is a principal; after an accented group its notated onset is the onset of
    the group's first grace note, after an unaccented one its own onset.
    """
    graces = {(r["staff"], r["voice"], _q(r["end"])): r for r in rows if r["grace"]}
    keys: Counter = Counter()
    for row in rows:
        if row["grace"]:
            continue
        onset = _q(row["start"])
        lead = graces.get((row["staff"], row["voice"], onset))
        if lead is not None and lead["grace"] == "acc":
            while lead is not None:
                onset = _q(lead["start"])
                lead = graces.get((row["staff"], row["voice"], onset))
        keys[(onset, row["midi"])] += 1
    return keys


def test_music21_parity() -> None:
    path = VIENNA_DIR / "Chopin_op10_no3.musicxml"
    rows = [
        row
        for row in VerovioLoader.from_file(path)
        .create_timeline()
        .get_events(include_children=True)
        if row["event_type"] == "Note"
    ]
    reference = [
        row
        for row in Music21Loader.from_file(path)
        .create_timeline()
        .get_events(include_children=True)
        if row["event_type"] == "Note"
    ]
    assert sum(1 for row in rows if row["grace"]) == 4
    assert sum(1 for row in reference if row["end"] is None) == 4
    expected = Counter(
        (_q(row["start"]), row["midi"]) for row in reference if row["end"] is not None
    )
    assert _principal_onsets(rows) == expected


def _expansion_xfail(reason: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(raises=AssertionError, strict=True, reason=reason)


HARDENING = [
    pytest.param(
        "couperin_concerts/c05n05_musete.mei",
        id="c05n05_musete",
        marks=_expansion_xfail(
            "Verovio's expansion does not follow the D.C. at MC 31 and the D.S. at "
            "MC 58 that the measure table states"
        ),
    ),
    pytest.param(
        "couperin_concerts/c11n08_Rondeau.mei",
        id="c11n08_Rondeau",
        marks=_expansion_xfail(
            "Verovio's expansion does not follow the D.C.s at MCs 18 and 27 that "
            "the measure table states"
        ),
    ),
    pytest.param(
        "beethoven_woo71/WoO71.mei",
        id="WoO71",
        marks=_expansion_xfail(
            "Verovio's expansion leaves out the endings at MCs 260-261 and the "
            "repeat of their passage"
        ),
    ),
    pytest.param(
        "beethoven_op18-4iv_multimodal/ABC/n04op18-4_04.mei",
        id="n04op18-4_04",
        marks=_expansion_xfail(
            "Verovio's expansion leaves out the endings at MCs 44-45, 93-94, "
            "102-103 and the repeats of their passages"
        ),
    ),
    pytest.param(
        "beethoven_op18-4iv_multimodal/recordings/Beethoven_Op018No4-04.mei",
        id="Beethoven_Op018No4-04",
        marks=_expansion_xfail(
            "the source leaves its first ending at MC 44 open to the end"
        ),
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("relative", HARDENING)
def test_expansion_matches_flow_controller(relative: str) -> None:
    loader = VerovioLoader.from_file(SCORE_DIR / relative)
    timeline = loader.create_timeline()
    printed = {_q(row["start"]): row["mc"] for row in loader.store.measures}
    source = timeline.get_flow_map("source")
    rendered = [
        printed[source.unfold_coordinate(m.qstamp)[0] if source else m.qstamp]
        for m in timeline.measure_map.measures
    ]
    assert _flow(loader) == rendered


@pytest.mark.slow
@on_pinned_version
@pytest.mark.parametrize(
    "relative,omitted,rendered",
    [
        ("beethoven_woo71/WoO71.mei", [260, 261], 495),
        (
            "beethoven_op18-4iv_multimodal/ABC/n04op18-4_04.mei",
            [44, 45, 93, 94, 102, 103],
            254,
        ),
    ],
    ids=["WoO71", "n04op18-4_04"],
)
def test_omitted_endings_in_provisioned_scores(
    relative: str, omitted: list[int], rendered: int
) -> None:
    loader = VerovioLoader.from_file(SCORE_DIR / relative)
    ids = set(loader.store.metadata["omitted_measures"])
    assert [row["mc"] for row in loader.store.measures if row["id"] in ids] == omitted
    assert len(loader.create_timeline().measure_map.measures) == rendered


class TestGuards:
    def test_missing_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        package = importlib.import_module("timetoalign.loader.score")
        monkeypatch.setitem(sys.modules, "verovio", None)
        monkeypatch.delitem(sys.modules, "timetoalign.loader.score.verovio")
        with pytest.raises(ImportError, match=r"timetoalign\[verovio\]"):
            package.__getattr__("VerovioLoader")

    def test_old_verovio_refused(
        self, monkeypatch: pytest.MonkeyPatch, mei: dict[str, Path]
    ) -> None:
        module = importlib.import_module("timetoalign.loader.score.verovio")

        class OldToolkit:
            def getVersion(self) -> str:
                return "4.3.1-3b8cc17"

        monkeypatch.setattr(module.verovio, "toolkit", OldToolkit)
        with pytest.raises(ImportError, match="found 4.3.1-3b8cc17"):
            VerovioLoader.from_file(mei["basic"])

    def test_qstamp_cross_check(self) -> None:
        module = importlib.import_module("timetoalign.loader.score.verovio")
        module._check_against_qstamps([{"qfrac": [45, 4]}], [{"qstamp": 11.25}])
        with pytest.raises(ValueError, match="does not round to qstamp"):
            module._check_against_qstamps([{"qfrac": [1, 3]}], [{"qstamp": 0.333}])

    def test_nothing_loaded(self) -> None:
        for create in ("create_timeline", "create_physical_timeline"):
            with pytest.raises(ValueError, match="No score loaded"):
                getattr(VerovioLoader(), create)()
