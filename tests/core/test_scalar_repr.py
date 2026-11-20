"""Uniform scalar ``repr()`` / ``str()`` behaviour.

``repr()`` is the SHORT typed form ``ABBR(token)``; ``str()`` is the PRETTY
human token.  The pitch scalars consolidate the rule onto the shared
``TwelveTETPitchMixin``; the MIDI-event scalars share one rendered repr via a
``_repr_parts()`` hook.  Every assertion uses exact expected strings with the
canonical ♯ (U+266F) / ♭ (U+266D) characters.
"""

from __future__ import annotations

import pandas as pd
import pytest

from timetoalign.core.enums import TimeUnit
from timetoalign.core.events import (
    DcmlHarmony,
    EnharmonicPitch,
    EnharmonicPitchClass,
    GenericPitch,
    GenericPitchClass,
    HarmonyLabel,
    Id,
    Measure,
    MeasureNumber,
    MidiEvent,
    MidiPitch,
    Note,
    PitchBasedHarmony,
    RomanNumeralHarmony,
    ScoreMidiEvent,
    SpecificPitch,
    SpecificPitchClass,
    WesternTertianHarmony,
)
from timetoalign.core.time import Coordinate, Duration


class TestPitchScalarRepr:
    """Seven pitch scalars: short typed repr, pretty str."""

    @pytest.mark.parametrize(
        ("scalar", "expected_repr", "expected_str"),
        [
            (EnharmonicPitchClass(0), "EPC(C)", "C"),
            (EnharmonicPitchClass(1), "EPC(C♯/D♭)", "C♯/D♭"),
            (GenericPitchClass(0), "GPC(C)", "C"),
            (GenericPitch(0, 4), "GP(C4)", "C4"),
            (SpecificPitchClass(step="C", alter=1), "SPC(C♯)", "C♯"),
            (EnharmonicPitch(56), "EP(G♯/A♭3)", "G♯/A♭3"),
            (EnharmonicPitch(60), "EP(C4)", "C4"),
            (MidiPitch(60), "MP(60)", "60"),
            (SpecificPitch(step="C", alter=1, octave=4), "SP(C♯4)", "C♯4"),
        ],
    )
    def test_repr_and_str(self, scalar, expected_repr, expected_str):
        assert repr(scalar) == expected_repr
        assert str(scalar) == expected_str

    def test_enharmonic_pitch_class_str_is_pretty_label(self):
        # EPC is the only pitch scalar whose get() returns the bare
        # integer; its __str__ override mirrors the repr's inner label.
        assert str(EnharmonicPitchClass(0)) == "C"
        assert str(EnharmonicPitchClass(1)) == "C♯/D♭"


def test_enharmonic_pitch_string_preserves_ambiguity() -> None:
    assert str(EnharmonicPitch(61)) == "C♯/D♭4"
    assert str(EnharmonicPitch(60)) == "C4"


def test_enharmonic_and_specific_pitch_strings_are_distinguishable() -> None:
    enharmonic = EnharmonicPitch(61)
    specific = SpecificPitch.from_string("C♯4")

    assert str(enharmonic) == "C♯/D♭4"
    assert str(specific) == "C♯4"
    assert str(enharmonic) != str(specific)


def test_dataframe_distinguishes_enharmonic_and_specific_pitch() -> None:
    frame = pd.DataFrame(
        {
            "ep": [EnharmonicPitch(61)],
            "sp": [SpecificPitch.from_string("C♯4")],
        }
    )

    assert frame.to_string(index=False) == "    ep  sp\nC♯/D♭4 C♯4"


@pytest.mark.parametrize(
    ("scalar", "expected_str", "expected_repr"),
    [
        (EnharmonicPitchClass(1), "C♯/D♭", "EPC(C♯/D♭)"),
        (EnharmonicPitch(61), "C♯/D♭4", "EP(C♯/D♭4)"),
        (MidiPitch(61), "61", "MP(61)"),
    ],
)
def test_enharmonic_scalar_strings_match_repr_pretty_tokens(
    scalar: EnharmonicPitchClass | EnharmonicPitch,
    expected_str: str,
    expected_repr: str,
) -> None:
    assert str(scalar) == expected_str
    assert repr(scalar) == expected_repr
    assert str(scalar) == repr(scalar).partition("(")[2][:-1]


class TestShallowScalarRepr:
    """MeasureNumber and Id: short repr, pretty str."""

    def test_measure_number(self):
        mn = MeasureNumber(mn="16")
        assert repr(mn) == "MeasureNumber(mn='16')"
        assert str(mn) == "16"

    def test_measure_number_carries_its_source_facts(self):
        """A split-bar label keeps its suffix, its count and its volta."""
        mn = MeasureNumber(mn="237b", mc=261, volta=2)
        assert repr(mn) == "MeasureNumber(mn='237b', mc=261, volta=2)"
        assert str(mn) == "237b"

    def test_id(self):
        i = Id(value="n0")
        assert repr(i) == "Id('n0')"
        assert str(i) == "n0"


class TestMidiEventRepr:
    """MidiEvent / ScoreMidiEvent: one rendered repr via _repr_parts()."""

    def test_note_event(self):
        ev = MidiEvent(pitch=EnharmonicPitch(60), velocity=80, channel=0)
        assert repr(ev) == "MidiEvent(pitch=EP(C4), velocity=80, channel=0)"

    def test_control_change_no_pitch(self):
        ev = MidiEvent(control=64, value=127, channel=0)
        assert repr(ev) == "MidiEvent(channel=0, control=64, value=127)"

    def test_score_event_with_voice_staff_part(self):
        ev = ScoreMidiEvent(
            pitch=EnharmonicPitch(60),
            velocity=80,
            voice=1,
            staff=2,
            part_id="P1",
        )
        assert repr(ev) == (
            "ScoreMidiEvent(pitch=EP(C4), velocity=80, "
            "voice=1, staff=2, part_id='P1')"
        )

    def test_score_event_repr_has_no_string_surgery_artifact(self):
        # A no-pitch ScoreMidiEvent must NOT leave a stray leading comma
        # or empty segment from slicing the base repr string.
        ev = ScoreMidiEvent(voice=3)
        assert repr(ev) == "ScoreMidiEvent(voice=3)"


class TestHarmonyScalarStr:
    """Every harmony scalar's str() returns the bare label."""

    def test_harmony_label(self):
        assert str(HarmonyLabel(label="V7", standard="dcml")) == "V7"

    def test_pitch_based_harmony(self):
        assert str(PitchBasedHarmony(label="Cmaj", standard="ohr")) == "Cmaj"

    def test_western_tertian_harmony(self):
        assert str(WesternTertianHarmony(label="Cmaj7", standard="wt")) == "Cmaj7"

    def test_roman_numeral_harmony(self):
        assert str(RomanNumeralHarmony(label="V7", standard="rn")) == "V7"

    def test_dcml_harmony(self):
        assert str(DcmlHarmony(label="V(64)")) == "V(64)"


class TestNoteAndMeasureStr:
    """Note and Measure gain a pretty str()."""

    def test_note(self):
        n = Note(
            start=Coordinate(0.0, TimeUnit.quarters),
            duration=Duration(1.0, TimeUnit.quarters),
            pitch=EnharmonicPitch(61),
        )
        assert str(n) == "C♯/D♭4 @0 quarters+1 quarters"

    def test_rest(self):
        rest = Note(
            start=Coordinate(0.0, TimeUnit.quarters),
            duration=Duration(1.0, TimeUnit.quarters),
        )
        assert str(rest) == "rest @0 quarters+1 quarters"

    def test_measure(self):
        m = Measure(id="m16", number=16)
        assert str(m) == "16"
