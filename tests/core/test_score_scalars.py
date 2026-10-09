"""Tests for Circle 1 score scalars: MidiPitch, SpecificPitch, Note, Measure, DcmlHarmony."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from timetoalign.core.enums import TimeUnit
from timetoalign.core.events import (
    DcmlHarmony,
    Measure,
    MidiPitch,
    Note,
    SpecificPitch,
    SpecificPitchClass,
)
from timetoalign.core.protocols import (
    DcmlHarmonyLike,
    EnharmonicPitchLike,
    GenericPitchLike,
    HarmonyLabelLike,
    MeasureLike,
    NoteLike,
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
            voice=1,
            staff=1,
            velocity=None,
        )
        assert note.start == start
        assert note.pitch is not None
        assert note.pitch.midi_number == 59
        assert note.duration is not None
        assert note.voice == 1
        assert note.staff == 1

    def test_construction_rest_no_pitch(self) -> None:
        """Construct a rest (Note with pitch=None)."""
        start = Coordinate(Fraction(1), TimeUnit.quarters)
        rest = Note(
            start=start,
            end=Coordinate(Fraction(2), TimeUnit.quarters),
            duration=Coordinate(Fraction(1), TimeUnit.quarters),
            pitch=None,
            voice=1,
            staff=1,
            velocity=None,
        )
        assert rest.pitch is None
        assert rest.duration is not None

    def test_notelike_conformance(self) -> None:
        """Note satisfies NoteLike protocol."""
        start = Coordinate(Fraction(0), TimeUnit.quarters)
        note = Note(
            start=start,
            end=Coordinate(Fraction(1, 8), TimeUnit.quarters),
            duration=Coordinate(Fraction(1, 8), TimeUnit.quarters),
            pitch=MidiPitch(midi_number=60),
            voice=1,
            staff=1,
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
            voice=1,
            staff=1,
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
            voice=1,
            staff=1,
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
            voice=1,
            staff=1,
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
            voice=None,
            staff=None,
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
        restored = Note.from_row(data)
        assert restored is not None
        assert restored.start.value == Fraction(1, 3)
        assert restored.duration is not None
        assert restored.duration.value == Fraction(2, 3)


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
