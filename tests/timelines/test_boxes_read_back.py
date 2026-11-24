"""A Box is read back by the calls that read any event type.

Validation strategy and exact values: ``tests/timelines/README.md`` (the
section "``test_boxes_read_back.py`` - A Box is read back by the calls that
read any event").
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from pydantic import BaseModel

from timetoalign.alignment import AlignmentBundle
from timetoalign.core import (
    Beat,
    BeatField,
    BoundingBox,
    Box,
    BoxField,
    Coordinate,
    Duration,
    EnharmonicPitch,
    Event,
    Measure,
    MeasureField,
    Note,
    NoteField,
    NumberType,
    Projection,
    ProjectionField,
    Quadrilateral,
    ScoreNote,
    ScoreNoteField,
    SpecificPitch,
    TimeUnit,
)
from timetoalign.core.events import canonical_pitch_name
from timetoalign.loader.alignment.ieee1599 import Ieee1599Loader
from timetoalign.loader.alignment.tilia import TiliaJsonLoader
from timetoalign.loader.graphical import DmosLoader, MeiFacsimileLoader
from timetoalign.loader.physical.eep_notes import EepNotesLoader
from timetoalign.loader.score import VerovioLoader
from timetoalign.loader.score.ms3 import Ms3Loader
from timetoalign.testdata import ensure_data
from timetoalign.timelines import (
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    DiscreteGraphicalTimeline,
    Timeline,
)

#: Thoresen's ``rect_h2`` on its 875 x 270 image, and its second crop (x0 = 7).
RECT_H2 = BoundingBox.from_xywh(385, 46, 139, 20, unit="pixels")
CROP_2 = {"x": 7, "y": 0, "width": 867, "height": 270}
#: Where the crop stands on the page built of the crops: after the first (866).
CROP_2_OFFSET = 866

#: Two measure zones of the Anhalt chorale facsimile: xml:id, ulx, uly, lrx, lry.
MEI_ZONES = [
    ("zone_01M7DF2D8RNCKHKF6SMCWB94Z5", 490, 344, 973, 877),
    ("zone_01M7DF2D8R00HP8JYGXRTE13G3", 944, 368, 1523, 864),
]
#: *Aimons-nous* (DMOS), page 1, system 1 and its first two measures, as DMOS
#: writes them (systems TL, BL, TR, BR; measures TL, TR, BL, BR).
SCORE_URL = "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688"
DMOS_SYSTEM = [[1221, 1589], [1221, 2266], [4097, 1589], [4101, 2266]]
DMOS_MEASURES = [
    [[1221, 1589], [2151, 1589], [1221, 2266], [2161, 2266]],
    [[2151, 1589], [2766, 1589], [2161, 2266], [2775, 2266]],
]
#: The thirds box of a user-units timeline.
THIRDS = BoundingBox.from_xywh(
    Fraction(1, 3),
    0,
    Fraction(2, 3),
    1,
    unit="user_units",
    number_type="fraction",
)


def _projection() -> tuple[Projection, Projection]:
    projection = Projection(id="p1", start=Coordinate(378, "pixels"))
    return projection, projection


def _box() -> tuple[Box, Box]:
    box = Box.from_shape(RECT_H2, origin=7, id="rect_h2")
    return box, box


#: A Note on ``seg_2`` as a row: its ID, position and pitch.
NOTE_ROW = {
    "event_type": "Note",
    "id": "n_378",
    "start": 378,
    "duration": 4,
    "pitch": {"midi_number": 60},
}
#: The members of that Note, on ``seg_2``'s axis.
NOTE_MEMBERS = {
    "start": Coordinate(378, "pixels"),
    "end": Coordinate(382, "pixels"),
    "duration": Duration(4, "pixels"),
    "pitch": EnharmonicPitch(midi_number=60),
}


def _note() -> tuple[dict[str, Any], Note]:
    return NOTE_ROW, Note(**NOTE_MEMBERS)


def _score_note() -> tuple[dict[str, Any], ScoreNote]:
    return {**NOTE_ROW, "staff": 1, "voice": 2}, ScoreNote(
        **NOTE_MEMBERS, voice=2, staff=1
    )


#: Every event type, each with how to make the event added (an event or its
#: row) and the event it is read back as.
EVENT_TYPES = [
    pytest.param(Box, BoxField, _box, id="Box"),
    pytest.param(Projection, ProjectionField, _projection, id="Projection"),
    pytest.param(Note, NoteField, _note, id="Note"),
    pytest.param(ScoreNote, ScoreNoteField, _score_note, id="ScoreNote"),
]


def _page_with(event: Any) -> tuple[Timeline, Timeline]:
    """A page holding a Beat, and *event* on its child ``seg_2`` at 866."""
    page = DiscreteGraphicalTimeline(length=2000, uid="page")
    seg_2 = DiscreteGraphicalTimeline(length=867, uid="seg_2", meta={"box": CROP_2})
    page.add_child(seg_2, CROP_2_OFFSET)
    page.add_events([{"event_type": "Beat", "id": "b1", "start": 0}])
    seg_2.add_events([event] if isinstance(event, dict) else event)
    return page, seg_2


def _shifted(event: Any, by: int) -> dict[str, Coordinate]:
    """The positions of *event* moved by *by* pixels: ``start``, and ``end``
    where the type holds it as a member."""
    return {
        name: Coordinate(getattr(event, name).value + by, "pixels")
        for name in ("start", "end")
        if name in type(event).model_fields
    }


class TestTheSameCalls:
    """Every assertion is written once and run for each event type."""

    @pytest.mark.parametrize(("event_type", "field_class", "make"), EVENT_TYPES)
    def test_the_events_of_a_type(
        self, event_type: type[BaseModel], field_class: type, make: Any
    ) -> None:
        added, expected = make()
        _, seg_2 = _page_with(added)
        kind = added["event_type"] if isinstance(added, dict) else type(added).__name__
        field = seg_2.get_events(event_type=kind).get_field(event_type)
        assert type(field) is field_class
        assert list(field) == [expected]
        assert field[0].start.value == 378
        assert list(seg_2.events.get_field(event_type)) == [expected]

    @pytest.mark.parametrize(("event_type", "field_class", "make"), EVENT_TYPES)
    def test_a_child_s_event_on_its_parent(
        self, event_type: type[BaseModel], field_class: type, make: Any
    ) -> None:
        added, expected = make()
        page, _ = _page_with(added)
        events = page.get_events()
        event_id = events.column_values("id")[1]
        assert events.column_values("id") == ["b1", event_id]
        field = events.get_field(event_type)
        assert field[0] is None
        read = field[1]
        assert type(read) is event_type
        assert read.start == Coordinate(CROP_2_OFFSET + 378, "pixels")
        assert read == expected.model_copy(update=_shifted(expected, CROP_2_OFFSET))

    @pytest.mark.parametrize(("event_type", "field_class", "make"), EVENT_TYPES)
    def test_a_field_filter_and_the_field_class(
        self, event_type: type[BaseModel], field_class: type, make: Any
    ) -> None:
        added, expected = make()
        page, _ = _page_with(added)
        event_id = added["id"] if isinstance(added, dict) else added.id
        events = page.get_events(id=event_id)
        field = events.get_field(event_type)
        assert len(field) == 1
        assert events.column_values("id") == [event_id]
        assert field[0].start == Coordinate(CROP_2_OFFSET + 378, "pixels")
        assert events.get_field(field_class) is field


def _mei(tmp_path: Path) -> Path:
    zones = "".join(
        f'<zone xml:id="{zone}" type="measure" ulx="{ulx}" uly="{uly}" '
        f'lrx="{lrx}" lry="{lry}"/>'
        for zone, ulx, uly, lrx, lry in MEI_ZONES
    )
    text = f"""<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.1">
  <meiHead><fileDesc><titleStmt><title>Ach bleib bei uns, Herr Jesu Christ</title>
  </titleStmt><pubStmt/></fileDesc></meiHead>
  <music><facsimile>
    <surface xml:id="surface_1" n="15" ulx="0" uly="0" lrx="3999" lry="2999">
      <graphic xml:id="graphic_1" target="facsimile/20251230_180023.jpg"
        width="4000" height="3000"/>{zones}
    </surface>
  </facsimile></music>
</mei>
"""
    path = tmp_path / "chorale_facsimile.mei"
    path.write_text(text, encoding="utf-8")
    return path


def _dmos(tmp_path: Path) -> Path:
    def measure(region: list[list[int]]) -> dict[str, Any]:
        return {
            "region": region,
            "headers": [{"no_staff": 1, "region": region}],
            "voices": [{"id": "1", "id_part": "P1", "elements": [{"no_group": 1}]}],
        }

    system = {
        "id": 1,
        "region": DMOS_SYSTEM,
        "headers": [
            {"id_part": "P1", "no_staff": 1, "name": "Voix", "region": DMOS_SYSTEM}
        ],
        "measures": [measure(region) for region in DMOS_MEASURES],
    }
    document = {
        "id": "C055_0",
        "date": "2025-06-10-11h",
        "score_image_url": SCORE_URL,
        "pages": [
            {
                "page_url": f"{SCORE_URL}/f2",
                "no_page": 1,
                "header_systems": {},
                "systems": [system],
            }
        ],
    }
    path = tmp_path / "iiif.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _mei_page(tmp_path: Path) -> tuple[Timeline, list[Box]]:
    """The MEI page, and its Boxes built from the zones' own numbers."""
    (page,) = MeiFacsimileLoader.from_file(_mei(tmp_path)).create_timelines()
    x0 = MEI_ZONES[0][1]
    expected = [
        Box.from_shape(
            BoundingBox.from_corners(ulx, uly, lrx, lry, unit="pixels"),
            origin=x0,
            id=zone,
        )
        for zone, ulx, uly, lrx, lry in MEI_ZONES
    ]
    assert [(box.start.value, box.duration.value) for box in expected] == [
        (0, 483),
        (454, 579),
    ]
    return page, expected


def _dmos_page(tmp_path: Path) -> tuple[Timeline, list[Box]]:
    """The DMOS page, and its Boxes built from the regions' own numbers."""
    (page,) = DmosLoader.from_file(_dmos(tmp_path)).create_timelines()
    order = ("top_left", "top_right", "bottom_left", "bottom_right")
    expected = [
        Box.from_shape(
            Quadrilateral.from_points(region, order=order, unit="pixels"),
            origin=DMOS_SYSTEM[0][0],
            id=f"box_{n}",
        )
        for n, region in enumerate(DMOS_MEASURES, start=1)
    ]
    assert [(box.start.value, box.duration.value) for box in expected] == [
        (0, 940),
        (930, 624),
    ]
    return page, expected


def _thirds_page(tmp_path: Path) -> tuple[Timeline, list[Box]]:
    """A user-units timeline written as fractions, holding the thirds Box."""
    page = ContinuousGraphicalTimeline(
        length=2, unit="user_units", number_type="fraction", uid="uu"
    )
    accolade = ContinuousGraphicalTimeline(
        length=1, unit="user_units", number_type="fraction", uid="uu_1"
    )
    page.add_child(accolade, 0)
    box = Box.from_shape(THIRDS, id="thirds")
    accolade.add_events(box)
    return page, [box]


PAGES = [
    pytest.param(_mei_page, int, id="MEI zones"),
    pytest.param(_dmos_page, int, id="DMOS measures"),
    pytest.param(_thirds_page, Fraction, id="thirds"),
]


def _boxes(page: Timeline) -> list[Box]:
    """The Boxes of the page's one accolade, read by the ordinary calls."""
    (accolade_id,) = page.list_children()
    accolade = page.get_child(accolade_id)
    return list(accolade.get_events(event_type="Box").get_field(Box))


def _numbers(box: Box) -> list[Any]:
    """Every number of a Box: its interval, then every number of its shape."""
    shape = box.shape
    if isinstance(shape, BoundingBox):
        members = [shape.x, shape.y, shape.width, shape.height]
    else:
        members = [c for corner in shape.corners for c in (corner.x, corner.y)]
    return [box.start.value, box.duration.value, *(m.value for m in members)]


class TestRoundTrips:
    """A Box added is the Box read back, in memory and through every written form."""

    @pytest.mark.parametrize(("make", "number_type"), PAGES)
    def test_in_memory(self, tmp_path: Path, make: Any, number_type: type) -> None:
        page, expected = make(tmp_path)
        read = _boxes(page)
        assert read == expected
        assert {type(value) for box in read for value in _numbers(box)} == {number_type}

    @pytest.mark.parametrize(("make", "number_type"), PAGES)
    def test_through_the_dict_form(
        self, tmp_path: Path, make: Any, number_type: type
    ) -> None:
        page, expected = make(tmp_path)
        read = _boxes(Timeline.from_dict(page.to_dict(events=True)))
        assert read == expected
        assert {type(value) for box in read for value in _numbers(box)} == {number_type}

    @pytest.mark.parametrize(("make", "number_type"), PAGES)
    def test_through_the_package(
        self, tmp_path: Path, make: Any, number_type: type
    ) -> None:
        page, expected = make(tmp_path)
        bundle = AlignmentBundle(id="boxes")
        bundle.add_timeline(page)
        directory = tmp_path / "package"
        directory.mkdir()
        bundle.to_file(directory / "package.yaml")
        assert sorted(path.name for path in directory.iterdir()) == [
            "boxes.parquet",
            "package.yaml",
        ]
        read_page = AlignmentBundle.from_file(directory / "package.yaml").get_timeline(
            page.id
        )
        read = _boxes(read_page)
        assert read == expected
        assert {type(value) for box in read for value in _numbers(box)} == {number_type}
        assert read_page.number_type is page.number_type is NumberType(number_type)


#: The ms3 notes of Chopin's Op. 10 No. 3, and Reger's Op. 13 No. 4 in MEI.
CHOPIN_NOTES = ensure_data("vienna_1x22") / "ms3" / "chopin_op10_no3.notes.tsv"
REGER_MEI = (
    ensure_data("mpm_toolbox")
    / "Max Reger - Moment Musical (MPM Toolbox Tutorial)"
    / "Reger - Moment Musical op 13 no 4.mei"
)
MEI = "{http://www.music-encoding.org/ns/mei}"
#: An MEI accidental as the alteration it writes.
ALTER = {"f": -1, "s": 1, "n": 0, "ff": -2, "ss": 2}


def _tsv_score_notes() -> list[ScoreNote]:
    """The notes of the ms3 TSV, built from its own columns (``cents`` 0, as
    the score's pitch column writes an equal-tempered spelled pitch)."""
    with CHOPIN_NOTES.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    notes = []
    for row in rows:
        start = Fraction(row["quarterbeats"])
        duration = Fraction(row["duration"]) * 4
        accidentals = row["name"][1:].removesuffix(row["octave"])
        # A grace note (duration 0) is an instant: no end, no duration.
        span = duration != 0
        notes.append(
            ScoreNote(
                start=Coordinate(start, "quarters"),
                end=Coordinate(start + duration, "quarters") if span else None,
                duration=Duration(duration, "quarters") if span else None,
                pitch=SpecificPitch(
                    step=row["name"][0],
                    alter=accidentals.count("#") - accidentals.count("b"),
                    octave=int(row["octave"]),
                    cents=0.0,
                ),
                voice=int(row["voice"]),
                staff=int(row["staff"]),
            )
        )
    return notes


def _mei_pitches() -> Counter[tuple[str, int, int, int, int]]:
    """Each note of the MEI file: step, alter, octave, staff, layer."""
    found: Counter[tuple[str, int, int, int, int]] = Counter()
    for staff in ET.parse(REGER_MEI).getroot().iter(f"{MEI}staff"):
        for layer in staff.iter(f"{MEI}layer"):
            for note in layer.iter(f"{MEI}note"):
                accid = note.get("accid", note.get("accid.ges"))
                found[
                    (
                        note.get("pname").upper(),
                        ALTER.get(accid, 0),
                        int(note.get("oct")),
                        int(staff.get("n")),
                        int(layer.get("n")),
                    )
                ] += 1
    return found


class TestScoresReadBack:
    """A loaded score's notes come back as the notes its source writes."""

    def test_ms3(self) -> None:
        expected = _tsv_score_notes()
        assert len(expected) == 498
        events = (
            Ms3Loader.from_file(CHOPIN_NOTES)
            .create_timeline()
            .get_events(event_type="Note")
        )
        read = list(events.get_field(ScoreNote))
        assert read == expected
        assert all(note.pitch is not None for note in read)
        assert {type(note.start.value) for note in read} == {Fraction}
        assert (read[0].pitch, read[0].start.value, read[0].staff, read[0].voice) == (
            SpecificPitch(step="B", alter=0, octave=3, cents=0.0),
            0,
            1,
            1,
        )
        plain = list(events.get_field(Note))
        assert {type(note) for note in plain} == {Note}
        assert plain == [
            Note(**{name: getattr(note, name) for name in Note.model_fields})
            for note in expected
        ]

    def test_mei_through_verovio(self) -> None:
        events = VerovioLoader.from_file(REGER_MEI).create_timeline().get_events()
        field = events.get_field(ScoreNote)
        read = [note for note in field if note is not None]
        assert (len(field), len(read)) == (107, 106)
        assert all(note.pitch is not None for note in read)
        assert (
            Counter(
                (
                    note.pitch.step,
                    note.pitch.alter,
                    note.pitch.octave,
                    note.staff,
                    note.voice,
                )
                for note in read
            )
            == _mei_pitches()
        )
        assert {type(n.start.value) for n in read} == {Fraction}
        assert {n.start.unit for n in read} == {TimeUnit.quarters}
        by_id = dict(zip(events.column_values("id"), field, strict=True))
        assert by_id["n1"] == ScoreNote(
            start=Coordinate(Fraction(0), "quarters"),
            end=Coordinate(Fraction(1, 4), "quarters"),
            duration=Duration(Fraction(1, 4), "quarters"),
            pitch=SpecificPitch(step="E", alter=-1, octave=5, cents=0.0),
            voice=1,
            staff=1,
        )
        assert by_id["n4"] == ScoreNote(
            start=Coordinate(Fraction(0), "quarters"),
            end=Coordinate(Fraction(3, 4), "quarters"),
            duration=Duration(Fraction(3, 4), "quarters"),
            pitch=SpecificPitch(step="A", alter=-1, octave=4, cents=0.0),
            voice=2,
            staff=1,
        )


#: The EEP note file of the first violin, String Quartet Op. 18 No. 4: 1,266
#: notes, each pitch a name (``Eb5``).
EEP_VLN1 = (
    ensure_data("score")
    / "beethoven_op18-4iv_multimodal"
    / "StringQuartetEEP_I_Normal"
    / "StringQuartetEEP_I_Normal_align_vln1.notes"
)
#: Bach, The Art of Fugue, Contrapunctus 1, in IEEE 1599; its timeline
#: ``los:dlt_2`` spells each of its 1,038 notes in raw columns.
IEEE_BACH = ensure_data("ieee1599") / "bach_artefuga_01.xml"

#: TiLiA's export of Bruckner 5, Scherzo; its timeline ``cpt_4`` ("Takte")
#: holds 1,146 beats as rows typed ``Beat`` and no ``index`` column.
TILIA_BRUCKNER = (
    ensure_data("score")
    / "bruckner5_scherzo"
    / "harnoncourt"
    / "Bruckner5_Scherzo.json"
)


def _no_field(field_class: type) -> str:
    """The start of the message ``get_field`` raises where no field matches."""
    return f"No field matching '{field_class.__name__}' found in table"


class TestTypesNotHeld:
    """A field only where the table holds the type's data; a field's own
    defect is no absence."""

    def test_tilia_beats_hold_no_beat_index(self) -> None:
        loader = TiliaJsonLoader()
        loader.load(TILIA_BRUCKNER)
        events = loader.create_timeline("cpt_4").get_events()
        assert len(events) == 1146
        assert set(events.column_values("event_type")) == {"Beat"}
        assert "index" not in events.schema.names
        with pytest.raises(KeyError) as raised:
            events.get_field(Beat)
        assert raised.value.args[0].startswith(_no_field(BeatField))
        assert events.has_field(BeatField) is False

    def test_a_measure_row_holds_no_measure(self) -> None:
        timeline = ContinuousLogicalTimeline(length=8, uid="score")
        timeline.add_events(
            [{"event_type": "Measure", "id": "m1", "start": 0, "end": 4}]
        )
        events = timeline.get_events()
        with pytest.raises(KeyError) as raised:
            events.get_field(Measure)
        assert raised.value.args[0].startswith(_no_field(MeasureField))

    def test_side_by_side(self) -> None:
        box, _ = _box()
        projection, _ = _projection()
        event = Event(
            id="e1", start=Coordinate(10, "pixels"), duration=Duration(5, "pixels")
        )
        _, note = _note()
        page = DiscreteGraphicalTimeline(length=2000, uid="page", meta={"box": CROP_2})
        page.add_events([{"event_type": "Beat", "id": "b1", "start": 0}])
        page.add_events([box])
        page.add_events([projection])
        page.add_events(
            [{"event_type": "Event", "id": "e1", "start": 10, "duration": 5}]
        )
        page.add_events([NOTE_ROW])
        page.add_events([{"event_type": "Measure", "id": "m1", "start": 30}])
        events = page.get_events()
        assert events.column_values("id") == [
            "b1",
            "rect_h2",
            "p1",
            "e1",
            "n_378",
            "m1",
        ]
        served = {Box: box, Projection: projection, Event: event, Note: note}
        for row, (event_type, expected) in enumerate(served.items(), start=1):
            nulls: list[Any] = [None] * 6
            nulls[row] = expected
            assert list(events.get_field(event_type)) == nulls, event_type.__name__
        assert box.quadrilateral is None
        assert projection.projection_of is None
        for event_type, field_class in ((Beat, BeatField), (Measure, MeasureField)):
            with pytest.raises(KeyError) as raised:
                events.get_field(event_type)
            assert raised.value.args[0].startswith(_no_field(field_class))

    def test_a_defect_of_the_field_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(cls: type, source: Any, **kw: Any) -> None:
            raise ValueError("broken")

        box, _ = _box()
        _, seg_2 = _page_with([box])
        monkeypatch.setattr(BoxField, "from_field", classmethod(broken))
        with pytest.raises(ValueError, match="^broken$"):
            seg_2.get_events().get_field(Box)

    @staticmethod
    def _eep_notes() -> Timeline:
        return EepNotesLoader.from_file(EEP_VLN1).create_timeline()

    @staticmethod
    def _ieee_notes() -> Timeline:
        (timeline,) = [
            timeline
            for timeline in Ieee1599Loader.from_file(IEEE_BACH).create_timelines()
            if timeline.id == "los:dlt_2"
        ]
        return timeline

    @pytest.mark.parametrize(
        ("load", "rows"),
        [
            pytest.param("_eep_notes", 1266, id="EEP pitch names"),
            pytest.param("_ieee_notes", 1038, id="IEEE 1599 raw spelling"),
        ],
    )
    def test_notes_whose_pitch_the_scalar_cannot_read(
        self, load: str, rows: int
    ) -> None:
        events = getattr(self, load)().get_events(event_type="Note")
        assert len(events) == rows
        assert not {"specific_pitch", "midi"} & set(events.schema.names)
        if load == "_eep_notes":
            assert events.schema.field("pitch").type == pa.string()
            first = EEP_VLN1.read_text(encoding="utf-8").split()[2]
            assert events.column_values("pitch")[0] == canonical_pitch_name(first)
        else:
            assert "pitch" not in events.schema.names
            assert {"step", "octave", "actual_accidental"} <= set(events.schema.names)
        for scalar, field_class in ((Note, NoteField), (ScoreNote, ScoreNoteField)):
            with pytest.raises(KeyError) as raised:
                events.get_field(scalar)
            assert raised.value.args[0].startswith(_no_field(field_class))
            assert events.has_field(field_class) is False
