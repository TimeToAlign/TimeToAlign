"""Tests for the score scalars: MidiPitch, SpecificPitch, Note, ScoreNote, Measure, DcmlHarmony."""

from __future__ import annotations

from fractions import Fraction

import pyarrow as pa
import pytest
from pydantic import ValidationError

from timetoalign.core.enums import TimeUnit
from timetoalign.core.events import (
    DcmlHarmony,
    Measure,
    MeasureConstituentField,
    MeasureField,
    MidiPitch,
    Note,
    NoteField,
    ScoreNote,
    ScoreNoteField,
    SpecificPitch,
    SpecificPitchClass,
)
from timetoalign.core.fields import build_struct_array, join_one_or_several
from timetoalign.core.protocols import (
    DcmlHarmonyLike,
    EnharmonicPitchLike,
    GenericPitchLike,
    HarmonyLabelLike,
    MeasureLike,
    NoteLike,
    ScoreNoteLike,
    SemanticTypeLike,
)
from timetoalign.core.time import Coordinate

# ---------------------------------------------------------------------------
# MidiPitch
# ---------------------------------------------------------------------------


class TestMidiPitch:
    """Tests for MidiPitch scalar construction and protocol conformance."""

    def test_construction_basic(self) -> None:
        """Construct MidiPitch with midi_number only (pitch_class auto-derived)."""
        p = MidiPitch(midi_number=60)
        assert p.midi_number == 60
        assert p.pitch_class == 0

    def test_construction_b3(self) -> None:
        """Construct MidiPitch for B3 (midi 59, pitch_class auto-derived to 11)."""
        p = MidiPitch(midi_number=59)
        assert p.midi_number == 59
        assert p.pitch_class == 11

    def test_generic_pitch_like_conformance(self) -> None:
        """MidiPitch satisfies the root pitch protocol."""
        p = MidiPitch(midi_number=64)
        assert isinstance(p, GenericPitchLike)

    def test_enharmonic_pitch_like_conformance(self) -> None:
        """MidiPitch satisfies EnharmonicPitchLike protocol."""
        p = MidiPitch(midi_number=64)
        assert isinstance(p, EnharmonicPitchLike)

    def test_semantic_type_like_conformance(self) -> None:
        """MidiPitch satisfies SemanticTypeLike protocol."""
        p = MidiPitch(midi_number=60)
        assert isinstance(p, SemanticTypeLike)

    def test_semantic_type(self) -> None:
        """MidiPitch.semantic_type == 'MidiPitch'."""
        p = MidiPitch(midi_number=60)
        assert p.semantic_type == "MidiPitch"

    def test_metadata_dict(self) -> None:
        """metadata_dict returns correct keys/values."""
        p = MidiPitch(midi_number=59)
        md = p.metadata_dict()
        assert md["field_type"] == "MidiPitchField"
        assert md["pitch_type"] == "ep"

    def test_frozen_immutable(self) -> None:
        """MidiPitch is frozen (immutable).

        With the pydantic v2 BaseModel implementation, assignment on a
        frozen instance raises ``ValidationError`` (not the dataclass
        ``AttributeError``).
        """
        p = MidiPitch(midi_number=60)
        with pytest.raises(ValidationError) as exc_info:
            p.midi_number = 61  # type: ignore[misc]
        assert "frozen" in str(exc_info.value).lower() or isinstance(
            exc_info.value, AttributeError
        )

    def test_octave_property(self) -> None:
        """MidiPitch.octave computes correctly."""
        p = MidiPitch(midi_number=60)
        assert p.octave == 4  # C4 = 60


# ---------------------------------------------------------------------------
# SpecificPitch
# ---------------------------------------------------------------------------


class TestSpecificPitch:
    """Tests for SpecificPitch scalar construction and protocol conformance."""

    def test_construction_basic(self) -> None:
        """Construct SpecificPitch for C4 (fifths auto-derived).

        ``cents`` defaults to ``None`` (the field is
        ``float | None = None``) instead of ``0.0``.  The pa.Schema
        treats ``cents`` as a nullable float64 column.
        """
        p = SpecificPitch(step="C", alter=0, octave=4)
        assert p.step == "C"
        assert p.alter == 0
        assert p.octave == 4
        assert p.fifths == 0  # auto-derived: C=0
        assert p.cents is None  # canonical default for the SpecificPitch schema

    def test_construction_gsharp(self) -> None:
        """Construct SpecificPitch for G#3 (fifths auto-derived)."""
        p = SpecificPitch(step="G", alter=1, octave=3)
        assert p.step == "G"
        assert p.alter == 1
        assert p.octave == 3
        assert p.fifths == 8  # auto-derived: G=1 + 7*1=8

    def test_generic_pitch_like_conformance(self) -> None:
        """SpecificPitch satisfies the root pitch protocol."""
        p = SpecificPitch(step="C", alter=0, octave=4)
        assert isinstance(p, GenericPitchLike)

    def test_from_string(self) -> None:
        """Construct a spelled pitch from its string representation."""
        assert SpecificPitch.from_string("C♯4") == SpecificPitch(
            step="C", alter=1, octave=4
        )

    def test_pitch_from_label_is_removed(self) -> None:
        """Pitch construction exposes no harmony-label constructor."""
        assert not hasattr(SpecificPitch, "from_label")

    def test_pitch_class_from_string(self) -> None:
        """Construct a spelled pitch class from its string representation."""
        assert SpecificPitchClass.from_string("D♭") == SpecificPitchClass(
            step="D", alter=-1
        )

    def test_pitch_class_from_label_is_removed(self) -> None:
        """Pitch-class construction exposes no harmony-label constructor."""
        assert not hasattr(SpecificPitchClass, "from_label")

    def test_midi_number_computation_c4(self) -> None:
        """SpecificPitch for C4 computes midi_number == 60."""
        p = SpecificPitch(step="C", alter=0, octave=4)
        assert p.midi_number == 60

    def test_midi_number_computation_b3(self) -> None:
        """SpecificPitch for B3 computes midi_number == 59."""
        p = SpecificPitch(step="B", alter=0, octave=3)
        assert p.midi_number == 59

    def test_midi_number_computation_gsharp3(self) -> None:
        """SpecificPitch for G#3 computes midi_number == 56."""
        p = SpecificPitch(step="G", alter=1, octave=3)
        assert p.midi_number == 56

    def test_pitch_class_computation_c(self) -> None:
        """SpecificPitch for C4 has pitch_class == 0."""
        p = SpecificPitch(step="C", alter=0, octave=4)
        assert p.pitch_class == 0

    def test_pitch_class_computation_b(self) -> None:
        """SpecificPitch for B3 has pitch_class == 11."""
        p = SpecificPitch(step="B", alter=0, octave=3)
        assert p.pitch_class == 11

    def test_semantic_type(self) -> None:
        """SpecificPitch.semantic_type == 'SpecificPitch'."""
        p = SpecificPitch(step="C", alter=0, octave=4)
        assert p.semantic_type == "SpecificPitch"

    def test_frozen_immutable(self) -> None:
        """SpecificPitch is frozen (immutable).

        With the pydantic v2 BaseModel implementation, assignment on a
        frozen instance raises ``ValidationError`` (not the dataclass
        ``AttributeError``).
        """
        p = SpecificPitch(step="C", alter=0, octave=4)
        with pytest.raises(ValidationError) as exc_info:
            p.step = "D"  # type: ignore[misc]
        assert "frozen" in str(exc_info.value).lower() or isinstance(
            exc_info.value, AttributeError
        )


# ---------------------------------------------------------------------------
# Note
# ---------------------------------------------------------------------------


class TestNote:
    """Tests for Note scalar construction and protocol conformance."""

    def test_construction_with_pitch(self) -> None:
        """Construct Note with start, end, duration, and a MidiPitch."""
        start = Coordinate(Fraction(0), TimeUnit.quarters)
        pitch = MidiPitch(midi_number=59)
        note = Note(
            start=start,
            end=Coordinate(Fraction(1, 2), TimeUnit.quarters),
            duration=Coordinate(Fraction(1, 2), TimeUnit.quarters),
            pitch=pitch,
            velocity=None,
        )
        assert note.start == start
        assert note.pitch is not None
        assert note.pitch.midi_number == 59
        assert note.duration.value == Fraction(1, 2)

    def test_construction_rest_no_pitch(self) -> None:
        """Construct a rest (Note with pitch=None)."""
        start = Coordinate(Fraction(1), TimeUnit.quarters)
        rest = Note(
            start=start,
            end=Coordinate(Fraction(2), TimeUnit.quarters),
            duration=Coordinate(Fraction(1), TimeUnit.quarters),
            pitch=None,
            velocity=None,
        )
        assert rest.pitch is None
        assert rest.duration.value == Fraction(1)

    def test_notelike_conformance(self) -> None:
        """Note satisfies NoteLike protocol."""
        start = Coordinate(Fraction(0), TimeUnit.quarters)
        note = Note(
            start=start,
            end=Coordinate(Fraction(1, 8), TimeUnit.quarters),
            duration=Coordinate(Fraction(1, 8), TimeUnit.quarters),
            pitch=MidiPitch(midi_number=60),
            velocity=80,
        )
        assert isinstance(note, NoteLike)

    def test_semantic_type(self) -> None:
        """Note.semantic_type == 'Note'."""
        start = Coordinate(Fraction(0), TimeUnit.quarters)
        note = Note(
            start=start,
            end=Coordinate(Fraction(1, 2), TimeUnit.quarters),
            duration=Coordinate(Fraction(1, 2), TimeUnit.quarters),
            pitch=MidiPitch(midi_number=60),
            velocity=None,
        )
        assert note.semantic_type == "Note"

    def test_frozen_immutable(self) -> None:
        """Note is frozen (immutable)."""
        start = Coordinate(Fraction(0), TimeUnit.quarters)
        note = Note(
            start=start,
            end=Coordinate(Fraction(1, 2), TimeUnit.quarters),
            duration=Coordinate(Fraction(1, 2), TimeUnit.quarters),
            pitch=MidiPitch(midi_number=60),
            velocity=None,
        )
        with pytest.raises(ValidationError) as exc_info:
            note.duration = Coordinate(1.0, TimeUnit.quarters)  # type: ignore[misc]
        assert "frozen" in str(exc_info.value).lower() or isinstance(
            exc_info.value, AttributeError
        )

    def test_instrument_field(self) -> None:
        """Note supports optional instrument field."""
        start = Coordinate(Fraction(0), TimeUnit.quarters)
        note = Note(
            start=start,
            end=Coordinate(Fraction(1, 2), TimeUnit.quarters),
            duration=Coordinate(Fraction(1, 2), TimeUnit.quarters),
            pitch=MidiPitch(midi_number=60),
            velocity=80,
            instrument="violin",
        )
        assert note.instrument == "violin"

    def test_instrument_default_none(self) -> None:
        """Note.instrument defaults to None."""
        start = Coordinate(Fraction(0), TimeUnit.quarters)
        note = Note(
            start=start,
            end=None,
            duration=None,
            pitch=None,
            velocity=None,
        )
        assert note.instrument is None

    def test_to_dict_embeds_exact_coordinates(self) -> None:
        note = Note(
            start=Coordinate(Fraction(1, 3), TimeUnit.quarters),
            duration=Coordinate(Fraction(2, 3), TimeUnit.quarters),
        )
        data = note.to_dict()
        assert data["start"] == {
            "unit": "quarters",
            "number_type": "fraction",
            "value": [1, 3],
        }
        assert data["duration"] == {
            "unit": "quarters",
            "number_type": "fraction",
            "value": [2, 3],
        }
        restored = Note.from_row(data, unit="quarters", number_type="fraction")
        assert restored is not None
        assert restored.start.value == Fraction(1, 3)
        assert restored.duration is not None
        assert restored.duration.value == Fraction(2, 3)


# ---------------------------------------------------------------------------
# ScoreNote
# ---------------------------------------------------------------------------

NOTE_MEMBERS = ["start", "end", "duration", "pitch", "velocity", "instrument"]


def _score_note() -> ScoreNote:
    return ScoreNote(
        start=Coordinate(Fraction(1, 3), TimeUnit.quarters),
        end=Coordinate(Fraction(1, 2), TimeUnit.quarters),
        duration=Coordinate(Fraction(1, 6), TimeUnit.quarters),
        pitch=MidiPitch(midi_number=59),
        velocity=64,
        instrument="piano",
        voice=2,
        staff=1,
    )


class TestNoteHoldsNoStaffOrVoice:
    """A plain Note has what every note has; staff and voice are a score's."""

    def test_members(self) -> None:
        assert list(Note.model_fields) == NOTE_MEMBERS

    @pytest.mark.parametrize("key", ["staff", "voice"])
    def test_refused(self, key: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Note(start=Coordinate(Fraction(0), TimeUnit.quarters), **{key: 1})
        errors = exc_info.value.errors()
        assert [(e["type"], e["loc"], e["input"]) for e in errors] == [
            ("extra_forbidden", (key,), 1)
        ]

    def test_field_struct(self) -> None:
        assert NoteField.pa_schema.names == [
            "start",
            "end",
            "duration",
            "velocity",
            "instrument",
        ]

    def test_protocol(self) -> None:
        note = Note(start=Coordinate(Fraction(0), TimeUnit.quarters))
        assert isinstance(note, NoteLike)
        assert not isinstance(note, ScoreNoteLike)
        assert "staff" not in vars(NoteLike)
        assert "voice" not in vars(NoteLike)


class TestScoreNote:
    """ScoreNote is a Note written on a staff, in a voice."""

    def test_subtype_and_members(self) -> None:
        assert issubclass(ScoreNote, Note)
        assert list(ScoreNote.model_fields) == [*NOTE_MEMBERS, "voice", "staff"]
        note = _score_note()
        assert (note.voice, note.staff) == (2, 1)
        assert note.start == Coordinate(Fraction(1, 3), TimeUnit.quarters)

    def test_field_struct(self) -> None:
        schema = ScoreNoteField.pa_schema
        assert schema.names == [*NoteField.pa_schema.names, "voice", "staff"]
        assert schema.field("voice").type == pa.int64()
        assert schema.field("staff").type == pa.int64()
        for name in NoteField.pa_schema.names:
            assert schema.field(name) == NoteField.pa_schema.field(name)

    def test_field_hierarchy(self) -> None:
        assert issubclass(ScoreNoteField, NoteField)
        assert ScoreNoteField.scalar_cls is ScoreNote
        assert ScoreNoteField.field_type() == "ScoreNoteField"
        assert ScoreNoteField.is_rest is NoteField.is_rest
        with pytest.raises(AttributeError) as exc_info:
            ScoreNoteField.from_field(
                pa.field("notes", ScoreNoteField.pa_schema), unit="quarters"
            ).is_rest
        assert str(exc_info.value) == "'StructField' object has no attribute 'is_rest'"

    def test_protocols(self) -> None:
        note = _score_note()
        assert isinstance(note, ScoreNoteLike)
        assert isinstance(note, NoteLike)
        assert isinstance(note, SemanticTypeLike)

    def test_semantic_type(self) -> None:
        assert _score_note().semantic_type == "ScoreNote"
        assert (
            Note(start=Coordinate(Fraction(0), TimeUnit.quarters)).semantic_type
            == "Note"
        )

    def test_frozen(self) -> None:
        with pytest.raises(ValidationError):
            _score_note().staff = 2  # type: ignore[misc]

    def test_to_dict_is_the_notes_plus_staff_and_voice(self) -> None:
        note = _score_note()
        plain = Note(**{name: getattr(note, name) for name in NOTE_MEMBERS})
        assert note.to_dict() == {**plain.to_dict(), "voice": 2, "staff": 1}

    def test_row_round_trip(self) -> None:
        note = _score_note()
        row = {**note.to_dict(), "midi": 59}
        assert ScoreNote.from_row(
            row, unit="quarters", number_type="fraction"
        ) == ScoreNote(
            start=note.start,
            end=note.end,
            duration=note.duration,
            pitch=note.pitch,
            velocity=64,
            instrument="piano",
            voice=2,
            staff=1,
        )

    def test_note_reads_the_same_row(self) -> None:
        note = _score_note()
        row = {**note.to_dict(), "midi": 59}
        plain = Note.from_row(row, unit="quarters", number_type="fraction")
        score = ScoreNote.from_row(row, unit="quarters", number_type="fraction")
        assert type(plain) is Note
        assert type(score) is ScoreNote
        assert {n: getattr(plain, n) for n in NOTE_MEMBERS} == {
            n: getattr(score, n) for n in NOTE_MEMBERS
        }

    def test_row_without_start(self) -> None:
        assert (
            ScoreNote.from_row(
                {"voice": 1, "staff": 1}, unit="quarters", number_type="fraction"
            )
            is None
        )


# ---------------------------------------------------------------------------
# Measure
# ---------------------------------------------------------------------------


class TestMeasure:
    """Tests for Measure scalar construction and protocol conformance."""

    def test_construction_basic(self) -> None:
        """Construct a Measure Map scalar with exact lengths."""
        m = Measure(
            id="m1",
            count=1,
            qstamp=Fraction(0),
            number=1,
            time_signature="2/4",
            nominal_length=Fraction(2),
            actual_length=Fraction(2),
        )
        assert m.id == "m1"
        assert m.name == "1"
        assert m.qstamp == 0
        assert m.time_signature == "2/4"

    def test_construction_anacrusis(self) -> None:
        """Construct an incomplete first measure without event coordinates."""
        m = Measure(
            id="m1",
            number=0,
            actual_length=Fraction(1, 2),
            nominal_length=Fraction(2),
        )
        assert m.id == "m1"
        assert m.name == "0"

    def test_measurelike_conformance(self) -> None:
        """Measure satisfies MeasureLike protocol."""
        m = Measure(id="m1", number=1)
        assert isinstance(m, MeasureLike)

    def test_time_signature_spelling_is_lossless(self) -> None:
        m = Measure(id="m3", time_signature="6/8")
        assert m.time_signature == "6/8"

    def test_semantic_type(self) -> None:
        """Measure.semantic_type == 'Measure'."""
        m = Measure()
        assert m.semantic_type == "Measure"

    def test_serialized_id_uses_measure_map_name(self) -> None:
        measure = Measure.model_validate({"ID": "m1", "number": 1})
        assert measure.id == "m1"
        assert measure.model_dump(by_alias=True)["ID"] == "m1"

    def test_frozen_immutable(self) -> None:
        """Measure is frozen (immutable)."""
        m = Measure(id="m1")
        with pytest.raises(ValidationError) as exc_info:
            m.id = "m2"  # type: ignore[misc]
        assert "frozen" in str(exc_info.value).lower() or isinstance(
            exc_info.value, AttributeError
        )

    def test_flow_control_defaults(self) -> None:
        """Flow control fields default to False/None."""
        m = Measure(id="m1")
        assert m.start_repeat is False
        assert m.end_repeat is False
        assert m.next is None
        assert m.volta is None


VOLTAS = [(2, 2), ((1, 3), (1, 3)), ([1, 3], (1, 3)), ([1], (1,)), ((3, 1), (3, 1))]
"""A volta as given and as the measure holds it."""


class TestMeasureVolta:
    """A volta is one pass or several, held as its source states it."""

    @pytest.mark.parametrize(("given", "held"), VOLTAS, ids=repr)
    def test_the_form_given_is_the_form_held(
        self, given: object, held: int | tuple[int, ...]
    ) -> None:
        volta = Measure(volta=given).volta
        assert volta == held
        assert type(volta) is type(held)

    def test_one_pass_and_the_list_of_that_pass_stay_apart(self) -> None:
        assert Measure(volta=1).volta == 1
        assert Measure(volta=[1]).volta == (1,)
        assert Measure(volta=1) != Measure(volta=[1])

    def test_volta_numbers_gives_the_numbers_of_either_form(self) -> None:
        assert Measure(volta=2).volta_numbers == (2,)
        assert Measure(volta=(1, 3)).volta_numbers == (1, 3)
        assert Measure(volta=[1]).volta_numbers == (1,)
        assert Measure(volta=(3, 1)).volta_numbers == (3, 1)
        assert Measure().volta_numbers == ()

    @pytest.mark.parametrize("numbers", [[], [1, 1]], ids=repr)
    def test_a_list_names_at_least_one_pass_each_once(self, numbers: list) -> None:
        with pytest.raises(ValidationError) as raised:
            Measure(volta=numbers)
        (error,) = raised.value.errors()
        assert error["loc"] == ("volta",)
        assert error["msg"] == (
            "Value error, a volta naming several passes holds at least one number, "
            f"each once, not {numbers!r}"
        )

    def test_a_dictionary_keeps_each_form(self) -> None:
        for given, written in ((2, 2), ((1, 3), [1, 3]), ((1,), [1])):
            data = Measure(id="m1", volta=given).to_dict()
            assert data["volta"] == written
            assert type(data["volta"]) is type(written)
            assert Measure.from_dict(data).volta == given

    def test_a_measure_with_a_volta_list_is_measure_like(self) -> None:
        assert isinstance(Measure(volta=(1, 3)), MeasureLike)


class TestMeasureFieldVolta:
    """The struct keeps the two forms of a volta apart in two members."""

    MEASURES = [
        Measure(id="1", volta=2),
        Measure(id="2", volta=(1, 3)),
        Measure(id="3", volta=(1,)),
        Measure(id="4"),
    ]

    def test_the_struct_holds_a_number_member_and_a_list_member(self) -> None:
        struct = MeasureField.struct_schema()
        assert struct.field("volta").type == pa.int64()
        assert struct.field("volta_list").type == pa.list_(pa.int64())
        assert struct.names[-2:] == ["volta", "volta_list"]
        assert MeasureConstituentField.struct_schema().names[-3:] == [
            "volta",
            "volta_list",
            "offset_within_measure",
        ]

    def test_each_form_fills_its_own_member(self) -> None:
        array = build_struct_array(Measure, self.MEASURES)
        assert array.type == MeasureField.struct_schema()
        assert array.field("volta").to_pylist() == [2, None, None, None]
        assert array.field("volta_list").to_pylist() == [None, [1, 3], [1], None]

    def test_a_parquet_file_keeps_the_two_forms_apart(self, tmp_path) -> None:
        import pyarrow.parquet as pq

        array = build_struct_array(Measure, self.MEASURES)
        field = MeasureField.from_field((array, pa.field("measure", array.type)))
        path = tmp_path / "measures.parquet"
        pq.write_table(pa.table({"measure": field.to_pyarrow()}), path)
        read = pq.read_table(path).column("measure").combine_chunks()
        assert read.type == MeasureField.struct_schema()
        assert read.field("volta").to_pylist() == [2, None, None, None]
        assert read.field("volta_list").to_pylist() == [None, [1, 3], [1], None]
        stored = MeasureField.from_field((read, pa.field("measure", read.type)))
        assert [
            join_one_or_several(row["volta"], row["volta_list"])
            for row in (stored[index] for index in range(4))
        ] == [2, (1, 3), (1,), None]


# ---------------------------------------------------------------------------
# DcmlHarmony (was Harmony)
# ---------------------------------------------------------------------------


class TestDcmlHarmony:
    """Tests for DcmlHarmony scalar construction and protocol conformance."""

    def test_construction_basic(self) -> None:
        """Construct DcmlHarmony with DCML-standard fields."""
        h = DcmlHarmony(
            label="c.i",
            globalkey="c",
            localkey="i",
            numeral="i",
            chord_type="m",
            root=0,
            bass=0,
        )
        assert h.label == "c.i"
        assert h.globalkey == "c"
        assert h.localkey == "i"
        assert h.numeral == "i"
        assert h.chord_type == "m"
        assert h.root == 0
        assert h.bass == 0

    def test_construction_dominant_seventh(self) -> None:
        """Construct DcmlHarmony for V65."""
        h = DcmlHarmony(
            label="V65",
            globalkey="c",
            localkey="i",
            numeral="V",
            chord_type="Mm7",
            inversion=1,
            root=1,
            bass=5,
        )
        assert h.label == "V65"
        assert h.numeral == "V"
        assert h.inversion == 1
        assert h.chord_type == "Mm7"

    def test_harmonylabellike_conformance(self) -> None:
        """DcmlHarmony satisfies HarmonyLabelLike protocol."""
        h = DcmlHarmony(
            label="i",
            globalkey="c",
            localkey="i",
            numeral="i",
            chord_type="m",
            root=0,
            bass=0,
        )
        assert isinstance(h, HarmonyLabelLike)

    def test_dcmllabellike_conformance(self) -> None:
        """DcmlHarmony satisfies DcmlHarmonyLike protocol."""
        h = DcmlHarmony(
            label="i",
            globalkey="c",
            localkey="i",
            numeral="i",
            chord_type="m",
            root=0,
            bass=0,
        )
        assert isinstance(h, DcmlHarmonyLike)

    @pytest.mark.parametrize(("label", "form"), [("VM6", "M"), ("V+M6", "+M")])
    def test_from_label_refuses_a_label_without_chord_type(
        self, label: str, form: str
    ) -> None:
        """A seventh-chord form with a triad's figured bass names no chord type."""
        with pytest.raises(ValueError) as exc_info:
            DcmlHarmony.from_label(label)
        assert exc_info.type is ValueError
        assert str(exc_info.value) == (
            f"Cannot parse DCML label: {label!r} (form {form!r} names a seventh "
            "chord, which cannot have figured bass '6')"
        )

    def test_semantic_type(self) -> None:
        """DcmlHarmony.semantic_type == 'DcmlHarmony'."""
        h = DcmlHarmony(label="V")
        assert h.semantic_type == "DcmlHarmony"

    def test_metadata_dict(self) -> None:
        """metadata_dict returns correct structure."""
        h = DcmlHarmony(label="c.i")
        md = h.metadata_dict()
        assert md["field_type"] == "DcmlHarmonyField"
        assert md["standard"] == "dcml"

    def test_frozen_immutable(self) -> None:
        """DcmlHarmony is frozen (immutable).

        With the pydantic v2 BaseModel implementation, assignment on a
        frozen instance raises ``ValidationError`` (not the dataclass
        ``AttributeError``).
        """
        h = DcmlHarmony(label="V")
        with pytest.raises(ValidationError) as exc_info:
            h.label = "i"  # type: ignore[misc]
        assert "frozen" in str(exc_info.value).lower() or isinstance(
            exc_info.value, AttributeError
        )
