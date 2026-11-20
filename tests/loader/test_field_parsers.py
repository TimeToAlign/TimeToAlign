"""Tests for the :class:`FieldParser` hierarchy and the resolver.

Two domains are exercised:

* **DataField blueprint mechanism** — ``IntField``, ``FloatField``,
  ``StringField``, ``RedundantNumberField``, ``DenominateNumberField``, and
  paired :class:`SemanticField` subclasses all accept ``name=`` for
  blueprint construction and expose a uniform
  ``from_array(source, name=...)`` materialisation entry point.
* **Resolver** — :func:`resolve_field_parser` maps user-facing inputs
  (Python types, ``pa.DataType``, raw / paired ``DataField``
  subclasses, blueprint instances, ``FieldParser`` instances,
  callables) to a producer (DataField blueprint or FieldParser).

Composite splitting (separator + regex strategies) and the callable
escape hatch live in :class:`CompositeFieldParser` and
:class:`CallableFieldParser` respectively; both are exercised here.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pyarrow as pa
import pytest

from timetoalign.core import (
    DenominateNumberField,
    EnharmonicPitch,
    EnharmonicPitchField,
    FloatField,
    Id,
    IdField,
    IntField,
    MeasureNumber,
    MeasureNumberField,
    NumericField,
    RedundantNumberField,
    StringField,
    StructField,
    TimeUnit,
)
from timetoalign.loader.tabular.field_parsers import (
    CallableFieldParser,
    CompositeFieldParser,
    resolve_field_parser,
)

# ═══════════════════════════════════════════════════════════════════════════
# DataField blueprint mechanism
# ═══════════════════════════════════════════════════════════════════════════


class TestIntFieldBlueprint:

    def test_blueprint_has_name_and_type(self) -> None:
        bp = IntField(name="channel")
        assert bp.is_empty is True
        assert bp.name == "channel"
        assert bp.pa_type == pa.int64()

    def test_from_array_casts_strings_to_int64(self) -> None:
        bp = IntField(name="channel")
        result = bp.from_array(pa.array(["1", "2", "3"]))
        assert isinstance(result, IntField)
        assert isinstance(result, NumericField)
        assert result.pa_type == pa.int64()
        assert result.data.to_pylist() == [1, 2, 3]
        assert result.name == "channel"

    def test_from_array_with_name_override(self) -> None:
        bp = IntField(name="channel")
        result = bp.from_array(pa.array([10, 20]), name="velocity")
        assert result.name == "velocity"

    def test_from_array_on_live_field_raises(self) -> None:
        bp = IntField(name="x")
        live = bp.from_array(pa.array([1, 2, 3]))
        with pytest.raises(TypeError, match="blueprint"):
            live.from_array(pa.array([1]))


class TestFloatFieldBlueprint:

    def test_blueprint_has_float64_type(self) -> None:
        bp = FloatField(name="x")
        assert bp.is_empty is True
        assert bp.pa_type == pa.float64()

    def test_from_array_casts_strings_to_float(self) -> None:
        bp = FloatField(name="x")
        result = bp.from_array(pa.array(["0.5", "1.5", "2.5"]))
        assert isinstance(result, FloatField)
        assert result.pa_type == pa.float64()
        assert result.data.to_pylist() == [0.5, 1.5, 2.5]


class TestStringFieldBlueprint:

    def test_blueprint_has_string_type(self) -> None:
        bp = StringField(name="note_id")
        assert bp.is_empty is True
        assert bp.pa_type == pa.string()

    def test_from_array_passes_string_through(self) -> None:
        bp = StringField(name="note_id")
        result = bp.from_array(pa.array(["n1", "n2", "n3"]))
        assert isinstance(result, StringField)
        assert result.data.to_pylist() == ["n1", "n2", "n3"]

    def test_from_array_casts_non_string_input(self) -> None:
        bp = StringField(name="x")
        result = bp.from_array(pa.array([1, 2, 3]))
        assert isinstance(result, StringField)
        assert result.data.to_pylist() == ["1", "2", "3"]


class TestRedundantNumberFieldBlueprint:

    def test_blueprint_has_rational_struct(self) -> None:
        bp = RedundantNumberField(name="ratio")
        assert bp.is_empty is True
        assert pa.types.is_struct(bp.pa_type)

    def test_from_array_parses_fraction_strings(self) -> None:
        bp = RedundantNumberField(name="ratio")
        result = bp.from_array(pa.array(["3/4", "1/2", "7/8"]))
        assert isinstance(result, RedundantNumberField)
        py = result.data.to_pylist()
        assert py[0] == {"value": 0.75, "numerator": 3, "denominator": 4}
        assert py[1] == {"value": 0.5, "numerator": 1, "denominator": 2}
        assert py[2] == {"value": 0.875, "numerator": 7, "denominator": 8}

    def test_from_array_handles_null_input(self) -> None:
        bp = RedundantNumberField(name="ratio")
        result = bp.from_array(pa.array(["3/4", None, "1/2"]))
        assert result.data.is_null().to_pylist() == [False, True, False]

    def test_from_array_zero_denominator_becomes_null(self) -> None:
        bp = RedundantNumberField(name="ratio")
        result = bp.from_array(pa.array(["3/4", "3/0"]))
        assert result.data.is_null().to_pylist() == [False, True]


class TestDenominateNumberFieldBlueprint:

    def test_blueprint_requires_unit(self) -> None:
        # ``unit=`` is mandatory; constructing without it raises.
        with pytest.raises(TypeError):
            DenominateNumberField(name="duration")  # type: ignore[call-arg]

    def test_blueprint_with_unit(self) -> None:
        bp = DenominateNumberField(name="duration", unit=TimeUnit.quarters)
        assert bp.is_empty is True
        assert bp.unit == TimeUnit.quarters

    def test_from_array_parses_fractions_and_stamps_unit(self) -> None:
        bp = DenominateNumberField(name="duration", unit=TimeUnit.quarters)
        result = bp.from_array(pa.array(["3/4", "1/2"]))
        assert isinstance(result, DenominateNumberField)
        assert result.unit == TimeUnit.quarters
        py = result.data.to_pylist()
        assert py[0] == {"value": 0.75, "numerator": 3, "denominator": 4}
        # Unit lives in the emitted field's metadata blob.
        meta = result.field.metadata
        assert meta is not None
        assert b"timetoalign" in meta
        assert b"quarters" in meta[b"timetoalign"]


class TestSemanticFieldBlueprint:

    def test_measure_number_blueprint_emits_struct(self) -> None:
        bp = MeasureNumberField(name="measure_number")
        assert bp.is_empty is True
        result = bp.from_array(pa.array([1, 2, 16]))
        assert isinstance(result, MeasureNumberField)
        assert result[0] == MeasureNumber(mn="1")
        assert result[2] == MeasureNumber(mn="16")

    def test_id_blueprint_packs_string(self) -> None:
        bp = IdField(name="note_id")
        result = bp.from_array(pa.array(["abc", "def"]))
        assert isinstance(result, IdField)
        assert result[0] == Id(value="abc")
        assert result[1] == Id(value="def")

    def test_enharmonic_pitch_blueprint_packs_int(self) -> None:
        bp = EnharmonicPitchField(name="midi_pitch")
        result = bp.from_array(pa.array([60, 62, 64]))
        assert isinstance(result, EnharmonicPitchField)
        assert result[0] == EnharmonicPitch(midi_number=60)
        assert result[2] == EnharmonicPitch(midi_number=64)


# ═══════════════════════════════════════════════════════════════════════════
# resolve_field_parser — universal resolution table
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveFieldParser:
    """Every documented input shape maps to the correct producer."""

    def test_python_int_type(self) -> None:
        assert isinstance(resolve_field_parser(int, default_name="x"), IntField)

    def test_python_float_type(self) -> None:
        assert isinstance(resolve_field_parser(float, default_name="x"), FloatField)

    def test_python_str_type(self) -> None:
        assert isinstance(resolve_field_parser(str, default_name="x"), StringField)

    def test_python_fraction_type(self) -> None:
        assert isinstance(
            resolve_field_parser(Fraction, default_name="x"), RedundantNumberField
        )

    def test_pa_int64(self) -> None:
        assert isinstance(resolve_field_parser(pa.int64(), default_name="x"), IntField)

    def test_pa_int32_routes_to_int_field(self) -> None:
        # All integer widths route to IntField.
        assert isinstance(resolve_field_parser(pa.int32(), default_name="x"), IntField)

    def test_pa_float64(self) -> None:
        assert isinstance(
            resolve_field_parser(pa.float64(), default_name="x"), FloatField
        )

    def test_pa_string(self) -> None:
        assert isinstance(
            resolve_field_parser(pa.string(), default_name="x"), StringField
        )

    def test_unsupported_pa_type_raises(self) -> None:
        with pytest.raises(TypeError):
            resolve_field_parser(pa.binary(), default_name="x")

    def test_unsupported_value_raises(self) -> None:
        with pytest.raises(TypeError):
            resolve_field_parser(object(), default_name="x")

    def test_data_field_class_returns_blueprint(self) -> None:
        producer = resolve_field_parser(IntField, default_name="x")
        assert isinstance(producer, IntField)
        assert producer.is_empty is True
        assert producer.name == "x"

    def test_data_field_blueprint_instance_passes_through(self) -> None:
        bp = IntField(name="velocity")
        assert resolve_field_parser(bp) is bp

    def test_live_data_field_rejected(self) -> None:
        # A live DataField must not appear in column_specs.
        live = IntField(name="x").from_array(pa.array([1, 2]))
        with pytest.raises(TypeError, match="blueprint"):
            resolve_field_parser(live)

    def test_semantic_field_subclass_returns_blueprint(self) -> None:
        producer = resolve_field_parser(EnharmonicPitchField, default_name="pitch")
        assert isinstance(producer, EnharmonicPitchField)
        assert producer.is_empty is True
        assert producer.name == "pitch"

    def test_field_parser_instance_passes_through(self) -> None:
        parser = CompositeFieldParser(separator="+", parts=[int, int])
        assert resolve_field_parser(parser) is parser

    def test_callable_wrapped(self) -> None:
        def my_fn(arr: pa.Array, *, name: str = "x") -> IntField:  # pragma: no cover
            return IntField(arr.cast(pa.int64()), pa.field(name, pa.int64()))

        producer = resolve_field_parser(my_fn)
        assert isinstance(producer, CallableFieldParser)


# ═══════════════════════════════════════════════════════════════════════════
# CompositeFieldParser
# ═══════════════════════════════════════════════════════════════════════════


class TestCompositeFieldParser:

    def test_separator_with_iterable_parts(self) -> None:
        parser = CompositeFieldParser(
            separator="+",
            parts=[IntField(name="measure"), RedundantNumberField(name="onset")],
            name="position",
        )
        result = parser.from_array(
            pa.array(["1+3/8", "2+1/4", "16+0/1"]), name="position"
        )
        assert isinstance(result, StructField)
        assert result.field.name == "position"
        assert result.field_names == ["measure", "onset"]
        measures = result.get_sub_field("measure").data.to_pylist()
        assert measures == [1, 2, 16]

    def test_separator_with_dict_parts(self) -> None:
        parser = CompositeFieldParser(
            separator="+",
            parts={"measure": int, "onset": Fraction},
            name="position",
        )
        result = parser.from_array(pa.array(["1+3/8", "2+1/4"]), name="position")
        assert result.field_names == ["measure", "onset"]
        assert result.get_sub_field("measure").data.to_pylist() == [1, 2]

    def test_default_part_name_from_semantic_field_class(self) -> None:
        parser = CompositeFieldParser(
            separator="+",
            parts=[MeasureNumberField, RedundantNumberField(name="mn_onset")],
            name="position",
        )
        # MeasureNumberField → "measure_number" (snake_case, sans Field suffix).
        # RedundantNumberField(name="mn_onset") supplies its own name.
        assert parser.part_keys == ["measure_number", "mn_onset"]

    def test_regex_named_groups(self) -> None:
        pattern = re.compile(r"(?P<measure>\d+)\+(?P<onset>\d+/\d+)")
        parser = CompositeFieldParser(
            pattern=pattern,
            parts={"measure": int, "onset": Fraction},
            name="position",
        )
        result = parser.from_array(pa.array(["1+3/8", "2+1/4"]), name="position")
        assert result.get_sub_field("measure").data.to_pylist() == [1, 2]

    def test_regex_positional_groups(self) -> None:
        pattern = r"(\d+)\+(\d+/\d+)"
        parser = CompositeFieldParser(
            pattern=pattern,
            parts=[IntField(name="measure"), RedundantNumberField(name="onset")],
            name="position",
        )
        result = parser.from_array(pa.array(["1+3/8", "2+1/4"]), name="position")
        assert result.get_sub_field("measure").data.to_pylist() == [1, 2]

    def test_separator_mismatch_produces_null_row(self) -> None:
        parser = CompositeFieldParser(
            separator="+",
            parts=[IntField(name="measure"), RedundantNumberField(name="onset")],
        )
        # Second row has no separator → its parts are null.
        result = parser.from_array(pa.array(["1+3/8", "no_separator"]), name="x")
        measures = result.get_sub_field("measure").data.to_pylist()
        assert measures == [1, None]

    def test_separator_and_pattern_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError):
            CompositeFieldParser(parts=[int, int])
        with pytest.raises(ValueError):
            CompositeFieldParser(separator="+", pattern=r"\d+", parts=[int])


# ═══════════════════════════════════════════════════════════════════════════
# CallableFieldParser
# ═══════════════════════════════════════════════════════════════════════════


class TestCallableFieldParser:

    def test_forwards_to_callable(self) -> None:
        def my_fn(arr: pa.Array, *, name: str = "x") -> NumericField:
            return NumericField(arr.cast(pa.int64()), pa.field(name, pa.int64()))

        parser = CallableFieldParser(my_fn, name="forwarded")
        result = parser.from_array(pa.array(["1", "2"]), name="ignored")
        assert isinstance(result, NumericField)
        assert result.field.name == "forwarded"

    def test_callable_without_name_kwarg(self) -> None:
        def my_fn(arr: pa.Array) -> NumericField:
            return NumericField(arr.cast(pa.int64()), pa.field("x", pa.int64()))

        parser = CallableFieldParser(my_fn)
        result = parser.from_array(pa.array(["10", "20"]), name="ignored")
        assert isinstance(result, NumericField)


# ═══════════════════════════════════════════════════════════════════════════
# Cross-resolver smoke test
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveAndFromArray:
    """The producer returned by the resolver exposes a working from_array()."""

    def test_python_type_then_emit(self) -> None:
        producer = resolve_field_parser(int, default_name="velocity")
        result = producer.from_array(pa.array(["1", "2", "3"]), name="velocity")
        assert isinstance(result, IntField)
        assert result.data.to_pylist() == [1, 2, 3]

    def test_pa_type_then_emit(self) -> None:
        producer = resolve_field_parser(pa.float64(), default_name="x")
        result = producer.from_array(pa.array([0.5, 1.5]), name="x")
        assert isinstance(result, FloatField)

    def test_semantic_class_then_emit_packs_struct(self) -> None:
        producer = resolve_field_parser(IdField, default_name="note_id")
        result = producer.from_array(pa.array(["abc", "def"]), name="note_id")
        assert isinstance(result, IdField)
        assert result[0] == Id(value="abc")
