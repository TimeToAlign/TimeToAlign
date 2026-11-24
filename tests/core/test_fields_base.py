"""Tests for fields/base.py — DataField hierarchy and protocol conformance."""

from __future__ import annotations

import copy
import pickle
import warnings
from fractions import Fraction
from typing import Any

import pyarrow as pa
import pytest

from timetoalign.alignment import AlignmentBundle, MatchClaimField
from timetoalign.core.enums import Domain, NumberType, TimeUnit
from timetoalign.core.events import EnharmonicPitchField
from timetoalign.core.fields import (
    MapField,
    NumericField,
    SemanticField,
    StringField,
    StructField,
)
from timetoalign.core.protocols import CoordinateLike, SemanticTypeLike
from timetoalign.core.time import (
    Coordinate,
    CoordinateField,
    Duration,
    IdCoordinateField,
    Interval,
    IntervalField,
)

# ---------------------------------------------------------------------------
# Protocol Conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Verify that Coordinate satisfies SemanticTypeLike and CoordinateLike."""

    def test_coordinate_satisfies_semantic_type_like(self) -> None:
        """isinstance(Coordinate(...), SemanticTypeLike) is True."""
        coord = Coordinate(1.5, TimeUnit.seconds)
        assert isinstance(coord, SemanticTypeLike)

    def test_coordinate_satisfies_coordinate_like(self) -> None:
        """isinstance(Coordinate(...), CoordinateLike) is True."""
        coord = Coordinate(120, TimeUnit.ticks)
        assert isinstance(coord, CoordinateLike)

    def test_coordinate_semantic_type(self) -> None:
        """Coordinate.semantic_type == 'Coordinate'."""
        coord = Coordinate(1.5, TimeUnit.seconds)
        assert coord.semantic_type == "Coordinate"

    def test_coordinate_metadata_dict(self) -> None:
        """metadata_dict returns correct keys/values."""
        coord = Coordinate(120, TimeUnit.ticks)
        md = coord.metadata_dict()
        assert md["field_type"] == "CoordinateField"
        assert md["unit"] == "ticks"
        assert md["domain"] == "logical"
        assert md["number_type"] == "int"


# ---------------------------------------------------------------------------
# NumericField
# ---------------------------------------------------------------------------


class TestNumericField:
    """Tests for NumericField construction and access."""

    def test_numeric_field_from_int_array(self) -> None:
        """Construct from pa.array([1, 2, 3], type=pa.int64()), verify len and __getitem__."""
        arr = pa.array([1, 2, 3], type=pa.int64())
        field = pa.field("count", pa.int64())
        nf = NumericField(arr, field)
        assert len(nf) == 3
        assert nf[0] == 1
        assert nf[1] == 2
        assert nf[2] == 3

    def test_numeric_field_from_float_array(self) -> None:
        """Construct from pa.array([1.1, 2.2, 3.3], type=pa.float64())."""
        arr = pa.array([1.1, 2.2, 3.3], type=pa.float64())
        field = pa.field("vals", pa.float64())
        nf = NumericField(arr, field)
        assert len(nf) == 3
        assert nf[0] == 1.1
        assert nf[2] == 3.3

    def test_numeric_field_rejects_string(self) -> None:
        """Passing string array raises TypeError."""
        arr = pa.array(["a", "b"], type=pa.string())
        field = pa.field("bad", pa.string())
        with pytest.raises(TypeError, match="NumericField requires a numeric type"):
            NumericField(arr, field)

    def test_numeric_field_schema_only(self) -> None:
        """Construct with data=None, verify is_empty and __len__ raises."""
        field = pa.field("empty", pa.int64())
        nf = NumericField(None, field)
        assert nf.is_empty is True
        with pytest.raises(TypeError, match="Cannot compute length"):
            len(nf)


# ---------------------------------------------------------------------------
# StringField
# ---------------------------------------------------------------------------


class TestStringField:
    """Tests for StringField construction and access."""

    def test_string_field_basic(self) -> None:
        """Construct and verify __getitem__ returns str."""
        arr = pa.array(["hello", "world"], type=pa.string())
        field = pa.field("text", pa.string())
        sf = StringField(arr, field)
        assert len(sf) == 2
        assert sf[0] == "hello"
        assert sf[1] == "world"
        assert isinstance(sf[0], str)

    def test_string_field_with_nulls(self) -> None:
        """Verify None for null entries."""
        arr = pa.array(["a", None, "c"], type=pa.string())
        field = pa.field("text", pa.string())
        sf = StringField(arr, field)
        assert sf[0] == "a"
        assert sf[1] is None
        assert sf[2] == "c"

    def test_string_field_rejects_numeric(self) -> None:
        """Passing numeric array raises TypeError."""
        arr = pa.array([1, 2], type=pa.int64())
        field = pa.field("bad", pa.int64())
        with pytest.raises(TypeError, match="StringField requires a string type"):
            StringField(arr, field)


# ---------------------------------------------------------------------------
# StructField
# ---------------------------------------------------------------------------


class TestStructField:
    """Tests for StructField construction and access."""

    @staticmethod
    def _make_struct_field() -> StructField:
        """Helper to build a simple StructField with two sub-fields."""
        struct_type = pa.struct(
            [
                pa.field("value", pa.float64()),
                pa.field("numerator", pa.int64()),
            ]
        )
        arr = pa.array(
            [{"value": 1.5, "numerator": 3}, {"value": 2.0, "numerator": 4}],
            type=struct_type,
        )
        pa_field = pa.field("coord", struct_type)
        return StructField(arr, pa_field)

    def test_struct_field_basic(self) -> None:
        """Construct from pa.StructArray, verify field_names and __getitem__ returns dict."""
        sf = self._make_struct_field()
        assert sf.field_names == ["value", "numerator"]
        row = sf[0]
        assert isinstance(row, dict)
        assert row["value"] == 1.5
        assert row["numerator"] == 3

    def test_struct_field_get_sub_field(self) -> None:
        """get_sub_field returns correctly typed DataField."""
        sf = self._make_struct_field()
        val_field = sf.get_sub_field("value")
        assert isinstance(val_field, NumericField)
        assert val_field[0] == 1.5
        assert val_field[1] == 2.0

        num_field = sf.get_sub_field("numerator")
        assert isinstance(num_field, NumericField)
        assert num_field[0] == 3

    def test_struct_field_get_sub_field_unknown(self) -> None:
        """KeyError for nonexistent sub-field."""
        sf = self._make_struct_field()
        with pytest.raises(KeyError, match="no sub-field"):
            sf.get_sub_field("nonexistent")

    def test_struct_field_rejects_non_struct(self) -> None:
        """Passing non-struct type raises TypeError."""
        arr = pa.array([1, 2], type=pa.int64())
        field = pa.field("bad", pa.int64())
        with pytest.raises(TypeError, match="StructField requires a struct type"):
            StructField(arr, field)


# ---------------------------------------------------------------------------
# MapField
# ---------------------------------------------------------------------------


class TestMapField:
    """Tests for MapField construction and access."""

    def test_map_field_basic(self) -> None:
        """Construct a MapField, verify __getitem__ returns dict."""
        map_type = pa.map_(pa.string(), pa.int64())
        arr = pa.array(
            [[("a", 1), ("b", 2)], [("c", 3)]],
            type=map_type,
        )
        field = pa.field("attrs", map_type)
        mf = MapField(arr, field)
        assert len(mf) == 2
        row0 = mf[0]
        assert isinstance(row0, dict)
        assert row0 == {"a": 1, "b": 2}
        assert mf[1] == {"c": 3}


# ---------------------------------------------------------------------------
# SemanticField
# ---------------------------------------------------------------------------


class _TestSemanticField(SemanticField[StructField]):
    """Minimal concrete SemanticField for testing (SemanticField is abstract)."""

    @property
    def semantic_type(self) -> str:
        return "TestSemantic"

    def metadata_dict(self) -> dict[str, str]:
        return {"field_type": "TestSemantic"}

    @classmethod
    def from_field(cls, source: Any, **kw: Any) -> "_TestSemanticField":
        data, field = source
        raw = StructField(data, field)
        return cls(raw)


class TestSemanticField:
    """Tests for SemanticField delegation and value access."""

    @staticmethod
    def _make_semantic_field() -> _TestSemanticField:
        struct_type = pa.struct(
            [
                pa.field("value", pa.float64()),
                pa.field("numerator", pa.int64()),
            ]
        )
        arr = pa.array(
            [{"value": 1.0, "numerator": 10}],
            type=struct_type,
        )
        pa_field = pa.field("sem", struct_type)
        return _TestSemanticField(StructField(arr, pa_field))

    def test_semantic_field_delegation(self) -> None:
        """__getattr__ delegates to raw field (e.g., field_names)."""
        sf = self._make_semantic_field()
        # field_names is defined on StructField, not SemanticField
        assert sf.field_names == ["value", "numerator"]

    def test_semantic_field_value_returns_raw(self) -> None:
        """.value returns the raw StructField."""
        sf = self._make_semantic_field()
        raw = sf.value
        assert isinstance(raw, StructField)
        assert raw.field_names == ["value", "numerator"]

    @pytest.mark.parametrize("name", ["_raw", "__setstate__", "__deepcopy__"])
    def test_uninitialised_instance_delegates_nothing(self, name: str) -> None:
        """``_raw`` and dunder names are not delegated before ``__init__`` runs."""
        bare = _TestSemanticField.__new__(_TestSemanticField)
        with pytest.raises(AttributeError, match=f"^{name}$"):
            getattr(bare, name)

    @pytest.mark.parametrize(
        "duplicate",
        [copy.deepcopy, lambda field: pickle.loads(pickle.dumps(field))],
        ids=["deepcopy", "pickle"],
    )
    def test_copy_and_pickle(self, duplicate: Any) -> None:
        """A copied field is its class, equal row for row, with its own raw field."""
        sf = self._make_semantic_field()
        copied = duplicate(sf)
        assert type(copied) is _TestSemanticField
        assert copied.to_pyarrow().to_pylist() == [{"value": 1.0, "numerator": 10}]
        assert copied.field_names == ["value", "numerator"]
        assert copied.value is not sf.value

    def test_deepcopy_of_a_bundle(self) -> None:
        """A bundle holding its MatchClaim table as a field deep-copies."""
        bundle = AlignmentBundle(id="x")
        copied = copy.deepcopy(bundle)
        assert copied == bundle
        assert copied is not bundle
        assert type(copied.match_claim_table) is MatchClaimField
        assert len(copied.match_claim_table) == 0
        assert copied.match_claim_table is not bundle.match_claim_table


# ---------------------------------------------------------------------------
# DataField general behaviour
# ---------------------------------------------------------------------------


class TestDataFieldGeneral:
    """Tests for DataField base behaviour: to_pyarrow, chunked arrays, metadata, repr."""

    def test_datafield_to_pyarrow(self) -> None:
        """to_pyarrow returns a contiguous pa.Array."""
        arr = pa.array([10, 20, 30], type=pa.int64())
        field = pa.field("x", pa.int64())
        nf = NumericField(arr, field)
        result = nf.to_pyarrow()
        assert isinstance(result, pa.Array)
        assert result.to_pylist() == [10, 20, 30]

    def test_datafield_chunked_array(self) -> None:
        """ChunkedArray is handled correctly in __getitem__ and to_pyarrow."""
        chunk1 = pa.array([1, 2], type=pa.int64())
        chunk2 = pa.array([3, 4], type=pa.int64())
        chunked = pa.chunked_array([chunk1, chunk2])
        field = pa.field("x", pa.int64())
        nf = NumericField(chunked, field)
        assert len(nf) == 4
        assert nf[0] == 1
        assert nf[3] == 4
        result = nf.to_pyarrow()
        assert isinstance(result, pa.Array)
        assert not isinstance(result, pa.ChunkedArray)
        assert result.to_pylist() == [1, 2, 3, 4]

    def test_datafield_metadata(self) -> None:
        """metadata property decodes pa.Field metadata."""
        field = pa.field(
            "x", pa.int64(), metadata={b"unit": b"seconds", b"domain": b"physical"}
        )
        nf = NumericField(pa.array([1], type=pa.int64()), field)
        md = nf.metadata
        assert md == {"unit": "seconds", "domain": "physical"}

    def test_datafield_metadata_empty(self) -> None:
        """metadata returns empty dict when no metadata present."""
        field = pa.field("x", pa.int64())
        nf = NumericField(pa.array([1], type=pa.int64()), field)
        assert nf.metadata == {}

    def test_datafield_repr(self) -> None:
        """repr includes class name, field name, and length."""
        arr = pa.array([1, 2], type=pa.int64())
        field = pa.field("myfield", pa.int64())
        nf = NumericField(arr, field)
        r = repr(nf)
        assert "NumericField" in r
        assert "myfield" in r
        assert "len=2" in r

    def test_datafield_repr_schema_only(self) -> None:
        """repr of schema-only field shows len=0."""
        field = pa.field("empty", pa.int64())
        nf = NumericField(None, field)
        r = repr(nf)
        assert "len=0" in r

    def test_datafield_schema_only_getitem_raises(self) -> None:
        """__getitem__ on schema-only field raises TypeError."""
        field = pa.field("empty", pa.int64())
        nf = NumericField(None, field)
        with pytest.raises(TypeError, match="Cannot index schema-only"):
            nf[0]

    def test_datafield_schema_only_to_pyarrow_raises(self) -> None:
        """to_pyarrow on schema-only field raises TypeError."""
        field = pa.field("empty", pa.int64())
        nf = NumericField(None, field)
        with pytest.raises(TypeError, match="Cannot convert schema-only"):
            nf.to_pyarrow()


# ---------------------------------------------------------------------------
# Fields declared without data
# ---------------------------------------------------------------------------


def _coordinate_cells(*values: int) -> pa.Array:
    return pa.array(
        [{"value": float(v), "numerator": v, "denominator": 1} for v in values],
        type=CoordinateField.pa_schema,
    )


def _bound_coordinate_field() -> CoordinateField:
    return CoordinateField.from_field(
        _coordinate_cells(0, 1),
        unit=TimeUnit.quarters,
        number_type=NumberType.fraction,
        name="onset",
    )


def _bound_interval_field() -> IntervalField:
    half = Coordinate(Fraction(1, 2), TimeUnit.quarters, number_type="fraction")
    whole = Coordinate(Fraction(1), TimeUnit.quarters, number_type="fraction")
    return IntervalField.from_intervals([Interval(start=half, end=whole)], name="iv")


class TestUnboundAxisFields:
    """A field declared by ``source_fields=`` answers and prints as unbound."""

    @pytest.mark.parametrize(
        ("field", "text"),
        [
            (
                CoordinateField(source_fields="onset"),
                "CoordinateField(name='onset', unbound, source_fields='onset')",
            ),
            (
                IntervalField(source_fields="iv"),
                "IntervalField(name='iv', unbound, source_fields='iv')",
            ),
        ],
        ids=["coordinate", "interval"],
    )
    def test_unbound_field_answers_none_and_prints_unbound(
        self, field: Any, text: str
    ) -> None:
        assert field.unit is None
        assert field.domain is None
        assert hasattr(field, "unit") is True
        assert hasattr(field, "domain") is True
        assert repr(field) == text
        assert str(field) == text
        name = type(field).__name__
        assert field.metadata_dict() == {"field_type": name}
        source = field.name
        assert (
            "<tr><td><b>Axis</b></td>"
            f"<td>unbound, source_fields=&#x27;{source}&#x27;</td></tr>"
        ) in field._repr_html_()

    def test_unbound_coordinate_field_refuses_its_rows(self) -> None:
        field = CoordinateField(source_fields="onset")
        message = (
            "CoordinateField 'onset' is unbound, source_fields='onset'; "
            "it has no axis until it is read from a table"
        )
        with pytest.raises(TypeError) as indexed:
            field[0]
        assert str(indexed.value) == message
        with pytest.raises(TypeError) as subtracted:
            field - Duration(1, TimeUnit.quarters)
        assert str(subtracted.value) == message

    def test_bound_coordinate_field_answers_as_before(self) -> None:
        field = _bound_coordinate_field()
        assert field.unit is TimeUnit.quarters
        assert field.domain is Domain.logical
        assert repr(field) == (
            "CoordinateField(name='onset', unit=quarters, "
            "number_type=fraction, len=2)"
        )
        assert field.metadata_dict() == {
            "field_type": "CoordinateField",
            "unit": "quarters",
            "domain": "logical",
            "number_type": "fraction",
        }
        assert "<b>Axis</b>" not in field._repr_html_()

    def test_bound_interval_field_answers_as_before(self) -> None:
        field = _bound_interval_field()
        assert field.unit is TimeUnit.quarters
        assert field.domain is Domain.logical
        assert repr(field) == "IntervalField(name='iv', raw=StructField, len=1)"
        assert field.metadata_dict() == {
            "field_type": "IntervalField",
            "unit": "quarters",
            "domain": "logical",
            "number_type": "fraction",
        }
        assert "<b>Axis</b>" not in field._repr_html_()


# ---------------------------------------------------------------------------
# Keywords from_field() does not use
# ---------------------------------------------------------------------------


def _pitch_cells() -> tuple[pa.Array, pa.Field]:
    schema = EnharmonicPitchField.pa_schema
    return (
        pa.array([{"midi_number": 60}], type=schema),
        pa.field("pitch", schema),
    )


def _interval_cells() -> tuple[pa.Array, pa.Field]:
    field = _bound_interval_field()
    return field.to_pyarrow(), pa.field("iv", field.pa_type)


def _same_field(left: Any, right: Any) -> None:
    assert type(left) is type(right)
    assert left.name == right.name
    assert left.to_pyarrow().to_pylist() == right.to_pyarrow().to_pylist()
    if isinstance(left, SemanticField):
        assert left.metadata_dict() == right.metadata_dict()


_INT_CELLS = (pa.array([1, 2], type=pa.int64()), pa.field("n", pa.int64()))
_QUARTERS = {"unit": TimeUnit.quarters, "number_type": NumberType.fraction}

_BUILDS: dict[str, tuple[Any, Any, dict[str, Any]]] = {
    "raw": (NumericField, lambda: _INT_CELLS, {}),
    "semantic": (EnharmonicPitchField, _pitch_cells, {}),
    "time_scalar": (CoordinateField, lambda: _coordinate_cells(0, 1), _QUARTERS),
    "id_time_scalar": (
        IdCoordinateField,
        lambda: _coordinate_cells(0, 1),
        {**_QUARTERS, "timeline_id": "clt_1"},
    ),
    "shared_axis": (IntervalField, _interval_cells, _QUARTERS),
}


class TestFromFieldIgnoredKeywords:
    """Every ``from_field()`` warns once, naming the keywords it ignores."""

    @pytest.mark.parametrize("family", list(_BUILDS))
    def test_one_warning_naming_the_ignored_keywords(self, family: str) -> None:
        field_cls, source, used = _BUILDS[family]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            built = field_cls.from_field(source(), foo=1, bar=None, **used)
        assert len(caught) == 1
        assert caught[0].category is UserWarning
        assert str(caught[0].message) == (
            f"{field_cls.__name__}.from_field() ignores unknown keyword "
            "argument(s): bar, foo"
        )
        assert caught[0].filename == __file__
        _same_field(built, field_cls.from_field(source(), **used))

    @pytest.mark.parametrize("family", list(_BUILDS))
    def test_no_warning_for_the_keywords_used(self, family: str) -> None:
        field_cls, source, used = _BUILDS[family]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            field_cls.from_field(source(), **used)

    def test_no_warning_through_from_table(self) -> None:
        table = pa.table({"onset": _coordinate_cells(0, 1)})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            field = CoordinateField.from_table(table, "onset", **_QUARTERS)
        assert repr(field) == (
            "CoordinateField(name='onset', unit=quarters, "
            "number_type=fraction, len=2)"
        )

    def test_no_warning_through_the_interval_endpoints(self) -> None:
        field = _bound_interval_field()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            start = field.start
        assert repr(start) == (
            "CoordinateField(name='start', unit=quarters, "
            "number_type=fraction, len=1)"
        )
