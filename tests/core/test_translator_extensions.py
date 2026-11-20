"""Tests for the bulk translator extensions.

The pilot translator handled atomic types + ``Literal`` + the projector
registry.  The full translator extends it with nested ``BaseModel``,
variable-length ``tuple[T, ...]``, fixed-length ``tuple[T1, T2, ...]``,
and a strict error on unions of ``BaseModel`` subclasses (which would
otherwise translate to forbidden Arrow ``dense_union``).  This module
pins each extension against a minimal pydantic model.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from pydantic import BaseModel, ConfigDict

from timetoalign.core.events import (
    DcmlHarmony,
    EnharmonicPitch,
    EnharmonicPitchClass,
    GenericPitch,
    GenericPitchClass,
    HarmonyLabel,
    Measure,
    Note,
    PitchBasedHarmony,
    RomanNumeralHarmony,
    SpecificPitchClass,
    WesternTertianHarmony,
)
from timetoalign.core.fields import derive_arrow_struct
from timetoalign.core.time import Duration


class TestNestedBaseModel:
    """Nested ``BaseModel`` fields translate to ``pa.struct(...)``."""

    def test_nested_basemodel_emits_struct(self) -> None:
        class Inner(BaseModel):
            model_config = ConfigDict(frozen=True)
            a: int
            b: str

        class Outer(BaseModel):
            model_config = ConfigDict(frozen=True)
            inner: Inner

        derived = derive_arrow_struct(Outer)
        assert derived.num_fields == 1
        inner_type = derived.field("inner").type
        assert pa.types.is_struct(inner_type)
        names = [inner_type.field(i).name for i in range(inner_type.num_fields)]
        assert names == ["a", "b"]
        assert inner_type.field("a").type == pa.int64()
        assert inner_type.field("b").type == pa.string()

    def test_optional_nested_basemodel_is_nullable(self) -> None:
        class Inner(BaseModel):
            model_config = ConfigDict(frozen=True)
            x: int

        class Outer(BaseModel):
            model_config = ConfigDict(frozen=True)
            inner: Inner | None = None

        derived = derive_arrow_struct(Outer)
        assert derived.field("inner").nullable is True
        assert pa.types.is_struct(derived.field("inner").type)


class TestVariadicTuple:
    """``tuple[T, ...]`` translates to ``pa.list_(T)``."""

    def test_variadic_tuple_of_string(self) -> None:
        class M(BaseModel):
            model_config = ConfigDict(frozen=True)
            tags: tuple[str, ...]

        derived = derive_arrow_struct(M)
        tags_type = derived.field("tags").type
        assert pa.types.is_list(tags_type)
        assert tags_type.value_type == pa.string()

    def test_variadic_tuple_of_int(self) -> None:
        class M(BaseModel):
            model_config = ConfigDict(frozen=True)
            scores: tuple[int, ...]

        derived = derive_arrow_struct(M)
        scores_type = derived.field("scores").type
        assert pa.types.is_list(scores_type)
        assert scores_type.value_type == pa.int64()


class TestFixedLengthTuple:
    """``tuple[T1, T2, ...]`` translates to a positional ``pa.struct``.

    The positional struct shape (``{_0, _1, ...}``) is the canonical
    round-trip choice: storing as ``pa.list_`` of 2 elements is lossless
    but indistinguishable from variadic lists; the struct shape keeps
    the fixed-length semantics in the pa.Schema itself.
    """

    def test_time_signature_pair(self) -> None:
        class M(BaseModel):
            model_config = ConfigDict(frozen=True)
            ts: tuple[int, int]

        derived = derive_arrow_struct(M)
        ts_type = derived.field("ts").type
        assert pa.types.is_struct(ts_type)
        assert ts_type.num_fields == 2
        names = [ts_type.field(i).name for i in range(ts_type.num_fields)]
        assert names == ["_0", "_1"]
        assert ts_type.field("_0").type == pa.int64()
        assert ts_type.field("_1").type == pa.int64()

    def test_mixed_pair(self) -> None:
        class M(BaseModel):
            model_config = ConfigDict(frozen=True)
            kv: tuple[str, int]

        derived = derive_arrow_struct(M)
        kv_type = derived.field("kv").type
        assert pa.types.is_struct(kv_type)
        assert kv_type.field("_0").type == pa.string()
        assert kv_type.field("_1").type == pa.int64()


class TestBool:
    """``bool`` translates to ``pa.bool_()``."""

    def test_bool_field(self) -> None:
        class M(BaseModel):
            model_config = ConfigDict(frozen=True)
            flag: bool

        derived = derive_arrow_struct(M)
        assert derived.field("flag").type == pa.bool_()


class TestLiteralStringWithDefault:
    """``Literal[str, ...]`` with default stays ``pa.string()`` (not dictionary)."""

    def test_literal_str_with_default(self) -> None:
        from typing import Literal

        class M(BaseModel):
            model_config = ConfigDict(frozen=True)
            standard: Literal["dcml"] = "dcml"

        derived = derive_arrow_struct(M)
        assert derived.field("standard").type == pa.string()
        assert not pa.types.is_dictionary(derived.field("standard").type)


class TestForbiddenBaseModelUnion:
    """Union of ``BaseModel`` subclasses is rejected — distinct scalars live in distinct fields."""

    def test_union_of_basemodels_rejected(self) -> None:
        class A(BaseModel):
            model_config = ConfigDict(frozen=True)
            x: int

        class B(BaseModel):
            model_config = ConfigDict(frozen=True)
            y: int

        class M(BaseModel):
            model_config = ConfigDict(frozen=True)
            polymorphic: A | B

        with pytest.raises(TypeError, match="distinct fields"):
            derive_arrow_struct(M)


class TestCacheInvalidationAfterRegister:
    """``register_value_projector`` invalidates the cache (already covered, regression pin)."""

    def test_cache_clears_on_register(self) -> None:
        from timetoalign.core.fields import register_value_projector

        class M(BaseModel):
            model_config = ConfigDict(frozen=True)
            a: int
            b: int

        first = derive_arrow_struct(M)
        assert first.num_fields == 2
        register_value_projector(M, "b", lambda *_: [])
        second = derive_arrow_struct(M)
        assert second.num_fields == 1


# ---------------------------------------------------------------------------
# Per-scalar schema pins for the migrated scalars.
# ---------------------------------------------------------------------------


class TestDurationSchema:
    def test_storage_struct_matches_coordinate_shape(self) -> None:
        derived = derive_arrow_struct(Duration)
        names = [derived.field(i).name for i in range(derived.num_fields)]
        assert names == ["value", "numerator", "denominator"]


class TestEnharmonicPitchClassSchema:
    def test_single_int_field(self) -> None:
        derived = derive_arrow_struct(EnharmonicPitchClass)
        assert derived.num_fields == 1
        assert derived.field("pitch_class").type == pa.int64()


class TestGenericPitchClassSchema:
    def test_single_step_field(self) -> None:
        derived = derive_arrow_struct(GenericPitchClass)
        assert derived.num_fields == 1
        assert derived.field("step").type == pa.int64()


class TestGenericPitchSchema:
    def test_step_octave(self) -> None:
        derived = derive_arrow_struct(GenericPitch)
        names = [derived.field(i).name for i in range(derived.num_fields)]
        assert names == ["step", "octave"]
        assert derived.field("step").type == pa.int64()
        assert derived.field("octave").type == pa.int64()


class TestSpecificPitchClassSchema:
    def test_minimal_two_field(self) -> None:
        derived = derive_arrow_struct(SpecificPitchClass)
        names = {derived.field(i).name for i in range(derived.num_fields)}
        assert names == {"step", "alter"}
        assert derived.field("step").type == pa.string()
        assert derived.field("alter").type == pa.int64()
        assert "fifths" not in names  # computed field


class TestEnharmonicPitchSchema:
    def test_collapsed_single_field(self) -> None:
        derived = derive_arrow_struct(EnharmonicPitch)
        assert derived.num_fields == 1
        assert derived.field("midi_number").type == pa.int64()


class TestHarmonyLabelSchema:
    def test_two_field(self) -> None:
        derived = derive_arrow_struct(HarmonyLabel)
        names = [derived.field(i).name for i in range(derived.num_fields)]
        assert names == ["label", "standard"]
        assert derived.field("label").type == pa.string()
        assert derived.field("standard").type == pa.string()


class TestPitchBasedHarmonySchema:
    def test_root_bass(self) -> None:
        derived = derive_arrow_struct(PitchBasedHarmony)
        names = [derived.field(i).name for i in range(derived.num_fields)]
        assert names == ["label", "standard", "root", "bass"]


class TestWesternTertianHarmonySchema:
    def test_chord_type_inversion(self) -> None:
        derived = derive_arrow_struct(WesternTertianHarmony)
        names = [derived.field(i).name for i in range(derived.num_fields)]
        assert names == ["label", "standard", "root", "bass", "chord_type", "inversion"]


class TestRomanNumeralHarmonySchema:
    def test_full_roman_numeral(self) -> None:
        derived = derive_arrow_struct(RomanNumeralHarmony)
        names = [derived.field(i).name for i in range(derived.num_fields)]
        assert names == [
            "label",
            "standard",
            "root",
            "bass",
            "chord_type",
            "inversion",
            "numeral",
            "localkey",
            "globalkey",
        ]


class TestDcmlHarmonySchema:
    def test_eleven_field(self) -> None:
        derived = derive_arrow_struct(DcmlHarmony)
        names = [derived.field(i).name for i in range(derived.num_fields)]
        assert names == [
            "label",
            "standard",
            "root",
            "bass",
            "chord_type",
            "inversion",
            "numeral",
            "localkey",
            "globalkey",
            "tonicized_key",
            "pedal",
        ]
        # Literal["dcml"] stays as pa.string(), not dictionary-encoded.
        assert derived.field("standard").type == pa.string()


class TestNoteSchemaDropsPitch:
    """``Note.pitch`` MUST be absent from the pa.Schema (columnar separation)."""

    def test_pitch_absent(self) -> None:
        derived = derive_arrow_struct(Note)
        names = {derived.field(i).name for i in range(derived.num_fields)}
        assert "pitch" not in names

    def test_nested_coordinate_struct(self) -> None:
        derived = derive_arrow_struct(Note)
        # start is nested Coordinate -> nested struct
        start_type = derived.field("start").type
        assert pa.types.is_struct(start_type)
        sub_names = [start_type.field(i).name for i in range(start_type.num_fields)]
        assert sub_names == ["value", "numerator", "denominator"]


class TestMeasureSchema:
    def test_time_signature_is_string(self) -> None:
        derived = derive_arrow_struct(Measure)
        ts_type = derived.field("time_signature").type
        assert ts_type == pa.string()

    def test_next_is_list_of_string(self) -> None:
        derived = derive_arrow_struct(Measure)
        nxt = derived.field("next").type
        assert pa.types.is_list(nxt)
        assert nxt.value_type == pa.string()

    def test_repeat_flags_are_bool(self) -> None:
        derived = derive_arrow_struct(Measure)
        assert derived.field("start_repeat").type == pa.bool_()
        assert derived.field("end_repeat").type == pa.bool_()

    def test_exact_positions_are_rational_structs(self) -> None:
        derived = derive_arrow_struct(Measure)
        qstamp_type = derived.field("qstamp").type
        assert pa.types.is_struct(qstamp_type)
        sub_names = [qstamp_type.field(i).name for i in range(qstamp_type.num_fields)]
        assert sub_names == ["value", "numerator", "denominator"]
