"""Three-regimes tests for each pydantic-migrated scalar.

Each migrated scalar must demonstrate, at minimum:

1. **Column-builder bulk construction** — the canonical bulk path
   (per ``T.model_fields``, no ``model_dump`` row-wise).
2. **Internal round-trip** — ``T.model_construct(**fields)`` reconstructs
   without re-validating.
3. **Trust boundary** — ``T.model_validate({...})`` enforces validators.

For scalars whose pa.Schema diverges from ``T.model_fields`` (Coordinate,
Duration via projector; Note with the dropped ``pitch`` field; scalars
nesting ``BaseModel``), this module exercises the **column-builder
shape** and ``model_construct`` directly without going through
``build_struct_array`` (which assumes a 1:1 field<->column mapping).
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign.core.enums import TimeUnit
from timetoalign.core.events import (
    DcmlHarmony,
    EnharmonicPitch,
    EnharmonicPitchClass,
    GenericPitch,
    GenericPitchClass,
    HarmonyLabel,
    Measure,
    MidiPitch,
    Note,
    PitchBasedHarmony,
    RomanNumeralHarmony,
    SpecificPitch,
    SpecificPitchClass,
    WesternTertianHarmony,
)
from timetoalign.core.fields import build_struct_array
from timetoalign.core.time import Coordinate, Duration

# ---------------------------------------------------------------------------
# Atomic-shape scalars: column-builder works through build_struct_array.
# ---------------------------------------------------------------------------


class TestEnharmonicPitchRegimes:
    def test_bulk_column_builder(self) -> None:
        objs = [EnharmonicPitch(60), EnharmonicPitch(72), None, EnharmonicPitch(48)]
        arr = build_struct_array(EnharmonicPitch, objs)
        assert len(arr) == 4
        assert arr.field("midi_number").to_pylist() == [60, 72, None, 48]
        # Null at position 2 (the None entry); other rows valid.
        assert arr.is_valid().to_pylist() == [True, True, False, True]

    def test_model_construct(self) -> None:
        # regime: internal round-trip — bypasses validators.
        p = EnharmonicPitch.model_construct(midi_number=60)
        assert p.midi_number == 60
        assert p.pitch_class == 0

    def test_model_validate(self) -> None:
        # regime: trust boundary — full validators.
        p = EnharmonicPitch.model_validate({"midi_number": 60})
        assert p.midi_number == 60


class TestEnharmonicPitchClassRegimes:
    def test_bulk(self) -> None:
        arr = build_struct_array(
            EnharmonicPitchClass,
            [EnharmonicPitchClass(0), EnharmonicPitchClass(11)],
        )
        assert arr.field("pitch_class").to_pylist() == [0, 11]

    def test_model_construct(self) -> None:
        # regime: internal round-trip
        p = EnharmonicPitchClass.model_construct(pitch_class=7)
        assert p.pitch_class == 7

    def test_model_validate(self) -> None:
        # regime: trust boundary
        p = EnharmonicPitchClass.model_validate({"pitch_class": 3})
        assert p.pitch_class == 3

    def test_eq_supports_int(self) -> None:
        assert EnharmonicPitchClass(0) == 0
        assert EnharmonicPitchClass(7) == 7


class TestGenericPitchRegimes:
    def test_bulk(self) -> None:
        arr = build_struct_array(GenericPitch, [GenericPitch(0, 4), GenericPitch(6, 3)])
        assert arr.field("step").to_pylist() == [0, 6]
        assert arr.field("octave").to_pylist() == [4, 3]

    def test_model_construct(self) -> None:
        # regime: internal round-trip
        p = GenericPitch.model_construct(step=2, octave=5)
        assert p.step == 2
        assert p.octave == 5

    def test_model_validate(self) -> None:
        # regime: trust boundary
        p = GenericPitch.model_validate({"step": 4, "octave": 3})
        assert p.step == 4
        assert p.octave == 3


class TestGenericPitchClassRegimes:
    def test_bulk(self) -> None:
        arr = build_struct_array(
            GenericPitchClass, [GenericPitchClass(0), GenericPitchClass(6)]
        )
        assert arr.field("step").to_pylist() == [0, 6]

    def test_model_construct(self) -> None:
        # regime: internal round-trip
        p = GenericPitchClass.model_construct(step=4)
        assert p.step == 4

    def test_model_validate(self) -> None:
        # regime: trust boundary
        p = GenericPitchClass.model_validate({"step": 2})
        assert p.step == 2


class TestSpecificPitchClassRegimes:
    def test_bulk(self) -> None:
        arr = build_struct_array(
            SpecificPitchClass,
            [SpecificPitchClass("C", 1), SpecificPitchClass("E", -1)],
        )
        assert arr.field("step").to_pylist() == ["C", "E"]
        assert arr.field("alter").to_pylist() == [1, -1]

    def test_model_construct(self) -> None:
        # regime: internal round-trip — bypasses step normalisation
        p = SpecificPitchClass.model_construct(step="F", alter=2)
        assert p.step == "F"
        assert p.alter == 2

    def test_model_validate(self) -> None:
        # regime: trust boundary
        p = SpecificPitchClass.model_validate({"step": "G", "alter": -1})
        assert p.step == "G"
        assert p.alter == -1

    def test_model_validate_rejects_bad_step(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SpecificPitchClass.model_validate({"step": "Z", "alter": 0})


class TestMidiPitchRegimes:
    """``MidiPitch`` is a thin subclass of ``EnharmonicPitch``."""

    def test_bulk(self) -> None:
        arr = build_struct_array(MidiPitch, [MidiPitch(60), MidiPitch(72)])
        assert arr.field("midi_number").to_pylist() == [60, 72]

    def test_model_construct(self) -> None:
        # regime: internal round-trip
        p = MidiPitch.model_construct(midi_number=60)
        assert p.midi_number == 60
        assert p.semantic_type == "MidiPitch"

    def test_model_validate(self) -> None:
        # regime: trust boundary
        p = MidiPitch.model_validate({"midi_number": 72})
        assert p.midi_number == 72

    def test_construct_via_subclass(self) -> None:
        p = MidiPitch(60)
        assert p.midi_number == 60
        assert p.semantic_type == "MidiPitch"
        assert repr(p) == "MP(60)"


class TestHarmonyLabelRegimes:
    def test_bulk(self) -> None:
        arr = build_struct_array(
            HarmonyLabel,
            [
                HarmonyLabel(label="V", standard="roman"),
                HarmonyLabel(label="I", standard="roman"),
            ],
        )
        assert arr.field("label").to_pylist() == ["V", "I"]

    def test_model_construct(self) -> None:
        # regime: internal round-trip
        h = HarmonyLabel.model_construct(label="V65", standard="dcml")
        assert h.label == "V65"
        assert h.standard == "dcml"

    def test_model_validate(self) -> None:
        # regime: trust boundary
        h = HarmonyLabel.model_validate({"label": "I", "standard": "roman"})
        assert h.label == "I"


class TestPitchBasedHarmonyRegimes:
    def test_bulk(self) -> None:
        arr = build_struct_array(
            PitchBasedHarmony,
            [
                PitchBasedHarmony(label="C", standard="rn", root=0, bass=0),
                PitchBasedHarmony(label="G", standard="rn", root=7, bass=7),
            ],
        )
        assert arr.field("root").to_pylist() == [0, 7]

    def test_model_construct(self) -> None:
        # regime: internal round-trip
        h = PitchBasedHarmony.model_construct(label="C", standard="rn", root=0, bass=0)
        assert h.root == 0

    def test_model_validate(self) -> None:
        # regime: trust boundary
        h = PitchBasedHarmony.model_validate(
            {"label": "G", "standard": "rn", "root": 7, "bass": 7}
        )
        assert h.bass == 7


class TestWesternTertianHarmonyRegimes:
    def test_bulk(self) -> None:
        arr = build_struct_array(
            WesternTertianHarmony,
            [
                WesternTertianHarmony(
                    label="Cmaj", standard="cs", chord_type="M", inversion=0
                ),
                WesternTertianHarmony(
                    label="Em", standard="cs", chord_type="m", inversion=0
                ),
            ],
        )
        assert arr.field("chord_type").to_pylist() == ["M", "m"]

    def test_model_construct(self) -> None:
        # regime: internal round-trip
        h = WesternTertianHarmony.model_construct(
            label="Cmaj", standard="cs", chord_type="M", inversion=0
        )
        assert h.chord_type == "M"

    def test_model_validate(self) -> None:
        # regime: trust boundary
        h = WesternTertianHarmony.model_validate(
            {"label": "Em", "standard": "cs", "chord_type": "m", "inversion": 1}
        )
        assert h.inversion == 1


class TestRomanNumeralHarmonyRegimes:
    def test_bulk(self) -> None:
        arr = build_struct_array(
            RomanNumeralHarmony,
            [
                RomanNumeralHarmony(
                    label="V",
                    standard="rn",
                    numeral="V",
                    localkey="I",
                    globalkey="C",
                ),
                RomanNumeralHarmony(
                    label="i",
                    standard="rn",
                    numeral="i",
                    localkey="i",
                    globalkey="c",
                ),
            ],
        )
        assert arr.field("numeral").to_pylist() == ["V", "i"]

    def test_model_construct(self) -> None:
        # regime: internal round-trip
        h = RomanNumeralHarmony.model_construct(
            label="V",
            standard="rn",
            numeral="V",
            localkey="I",
            globalkey="C",
        )
        assert h.numeral == "V"

    def test_model_validate(self) -> None:
        # regime: trust boundary
        h = RomanNumeralHarmony.model_validate(
            {
                "label": "i",
                "standard": "rn",
                "numeral": "i",
                "localkey": "i",
                "globalkey": "c",
            }
        )
        assert h.globalkey == "c"


class TestDcmlHarmonyRegimes:
    def test_bulk(self) -> None:
        arr = build_struct_array(
            DcmlHarmony,
            [
                DcmlHarmony(label="V"),
                DcmlHarmony(label="i", numeral="i", chord_type="m", root=0, bass=0),
            ],
        )
        assert arr.field("label").to_pylist() == ["V", "i"]
        assert arr.field("standard").to_pylist() == ["dcml", "dcml"]

    def test_model_construct(self) -> None:
        # regime: internal round-trip
        h = DcmlHarmony.model_construct(label="V65/IV", standard="dcml")
        assert h.label == "V65/IV"
        assert h.standard == "dcml"

    def test_model_validate_rejects_non_dcml_standard(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DcmlHarmony.model_validate({"label": "V", "standard": "roman"})


# ---------------------------------------------------------------------------
# Duration — Coordinate-shaped scalar (uses value-projector).
# ---------------------------------------------------------------------------


class TestDurationRegimes:
    def test_construction_positional(self) -> None:
        d = Duration(0.5, TimeUnit.quarters)
        assert d.value == 0.5
        assert d.unit is TimeUnit.quarters

    def test_negative_value_allowed(self) -> None:
        """Duration accepts negative values (produced by Coordinate - Coordinate)."""
        d = Duration(-1.0, TimeUnit.quarters)
        assert d.value == -1.0
        assert d.is_negative() is True

    def test_fraction_value(self) -> None:
        d = Duration(Fraction(1, 4), TimeUnit.quarters)
        assert d.value == Fraction(1, 4)

    def test_model_construct(self) -> None:
        # regime: internal round-trip — bypasses validators.
        d = Duration.model_construct(value=0.0, unit=TimeUnit.quarters)
        assert d.value == 0.0

    def test_model_validate(self) -> None:
        # regime: trust boundary — runs the full validator stack.
        d = Duration.model_validate({"value": 0.5, "unit": "quarters"})
        assert d.value == 0.5

    def test_bulk_via_coordinate_struct_array(self) -> None:
        # regime: bulk construction — Duration shares Coordinate's storage
        # struct and routes through ``build_coordinate_struct_array``.
        from timetoalign.core.time import (
            build_coordinate_struct_array,
        )

        arr = build_coordinate_struct_array(
            [Duration(0.5, TimeUnit.quarters), Duration(1.0, TimeUnit.quarters), None]
        )
        # Parent-struct mask hides the third row; sub-field values for the
        # live rows are exact.
        assert arr.is_valid().to_pylist() == [True, True, False]
        assert arr.field("value").to_pylist()[:2] == [0.5, 1.0]

    def test_semantic_type(self) -> None:
        d = Duration(1.0, TimeUnit.seconds)
        assert d.semantic_type == "Duration"
        md = d.metadata_dict()
        assert md["field_type"] == "DurationField"


# ---------------------------------------------------------------------------
# Note — nested scalars + dropped pitch column.
# ---------------------------------------------------------------------------


class TestNoteRegimes:
    def test_construction_with_pitch(self) -> None:
        c = Coordinate(0.0, TimeUnit.quarters)
        d = Duration(0.5, TimeUnit.quarters)
        n = Note(start=c, duration=d, pitch=SpecificPitch(step="C", octave=4))
        assert n.is_rest is False
        assert n.pitch is not None
        assert n.pitch.midi_number == 60

    def test_construction_rest(self) -> None:
        c = Coordinate(0.0, TimeUnit.quarters)
        n = Note(start=c, pitch=None)
        assert n.is_rest is True

    def test_model_construct(self) -> None:
        # regime: internal round-trip
        c = Coordinate(0.0, TimeUnit.quarters)
        n = Note.model_construct(start=c, pitch=None)
        assert n.start is c

    def test_model_validate(self) -> None:
        # regime: trust boundary — nested Coordinate validates too.
        n = Note.model_validate(
            {"start": {"value": 0.0, "unit": "quarters"}, "pitch": None}
        )
        assert n.start.value == 0.0
        assert n.is_rest is True

    def test_accepts_both_pitch_types(self) -> None:
        c = Coordinate(0.0, TimeUnit.quarters)
        n1 = Note(start=c, pitch=EnharmonicPitch(60))
        n2 = Note(start=c, pitch=SpecificPitch(step="C", octave=4))
        assert n1.pitch is not None
        assert n2.pitch is not None
        assert n1.pitch.midi_number == n2.pitch.midi_number == 60


# ---------------------------------------------------------------------------
# Measure — nested + variadic tuple + fixed tuple.
# ---------------------------------------------------------------------------


class TestMeasureRegimes:
    def test_construction(self) -> None:
        m = Measure(id="m1", number=1, time_signature="4/4")
        assert m.id == "m1"
        assert m.time_signature == "4/4"
        assert m.next is None

    def test_next_ids_are_stringified(self) -> None:
        m = Measure(id="m1", next=("m2", "m3"))
        assert m.next == ("m2", "m3")

    def test_next_ids_strings_pass_through(self) -> None:
        m = Measure(id="m1", next=("a", "b"))
        assert m.next == ("a", "b")

    def test_model_construct(self) -> None:
        # regime: internal round-trip
        m = Measure.model_construct(id="m1", number=1, time_signature="4/4", next=None)
        assert m.id == "m1"

    def test_model_validate(self) -> None:
        # regime: trust boundary — nested Coordinate + tuple validation.
        m = Measure.model_validate(
            {
                "ID": "m2",
                "number": 2,
                "qstamp": "4",
                "time_signature": "3/8",
                "next": ["m3"],
            }
        )
        assert m.id == "m2"
        assert m.qstamp == Fraction(4)
        assert m.time_signature == "3/8"
        assert m.next == ("m3",)


# ---------------------------------------------------------------------------
# Field-name preservation for round-trips through pa.Schema.
# ---------------------------------------------------------------------------


class TestNestedBuild:
    """``build_struct_array`` recurses into nested ``BaseModel`` fields."""

    def test_note_bulk_with_coordinate_start(self) -> None:
        c0 = Coordinate(0.0, TimeUnit.quarters)
        c1 = Coordinate(Fraction(1, 2), TimeUnit.quarters)
        n0 = Note(
            start=c0, end=c1, duration=Duration(0.5, TimeUnit.quarters), pitch=None
        )
        n1 = Note(start=c1, pitch=EnharmonicPitch(60))
        arr = build_struct_array(Note, [n0, n1, None])
        assert len(arr) == 3
        # Null mask on the parent struct hides the third row.
        assert arr.is_valid().to_pylist() == [True, True, False]
        # start column is itself a struct
        start_arr = arr.field("start")
        # Sub-field values for the first two (live) rows.
        assert start_arr.field("value").to_pylist()[:2] == [0.0, 0.5]
        # pitch field MUST be absent (columnar separation)
        names = {arr.type.field(i).name for i in range(arr.type.num_fields)}
        assert "pitch" not in names

    def test_measure_bulk_with_time_signature_and_next_ids(self) -> None:
        m0 = Measure(id="m1", number=1, time_signature="4/4", next=("a", "b"))
        m1 = Measure(id="m2", number=2, time_signature="3/8", next=None)
        arr = build_struct_array(Measure, [m0, m1])
        assert len(arr) == 2
        assert arr.field("time_signature").to_pylist() == ["4/4", "3/8"]
        assert arr.field("next").to_pylist() == [["a", "b"], None]


class TestSchemaFieldNameCoverage:
    """Pin: every migrated scalar's pa.Schema field names ⊆ its model_fields."""

    @pytest.mark.parametrize(
        "cls",
        [
            EnharmonicPitch,
            EnharmonicPitchClass,
            GenericPitch,
            GenericPitchClass,
            SpecificPitchClass,
            SpecificPitch,
            HarmonyLabel,
            PitchBasedHarmony,
            WesternTertianHarmony,
            RomanNumeralHarmony,
            DcmlHarmony,
        ],
    )
    def test_pa_fields_subset_of_model_fields(self, cls: type) -> None:
        """The pa.Schema must not introduce fields the scalar doesn't declare."""
        from timetoalign.core.fields import derive_arrow_struct

        struct = derive_arrow_struct(cls)
        pa_names = {struct.field(i).name for i in range(struct.num_fields)}
        model_names = set(cls.model_fields.keys())
        assert (
            pa_names <= model_names
        ), f"{cls.__name__} pa.Schema has extra fields: {pa_names - model_names}"
