"""Tests for struct column support in TabularLoader.

Tests the Field and ComputedField struct features that enable:
- JSON string to struct column parsing
- Struct field access for start/end columns
- Computed fields from expressions
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from timetoalign.core import NumberType, TimeUnit
from timetoalign.loader import (
    ComputedField,
    Field,
    parse_json_to_struct,
)
from timetoalign.loader.tabular import TsvLoader  # noqa: F401 — used by sample fixture

# region Test Fixtures


@pytest.fixture
def sample_tsv_with_json(tmp_path: Path) -> Path:
    """Create a sample TSV file with JSON column."""
    content = """id\tstart_time_sec\tduration_sec\trect_coords_json\tlabel
e001\t0.0\t1.5\t{"x": 10, "y": 90, "width": 148, "height": 55}\tregion_1
e002\t1.5\t2.0\t{"x": 158, "y": 90, "width": 200, "height": 55}\tregion_2
e003\t3.5\t1.0\t{"x": 358, "y": 90, "width": 100, "height": 55}\tregion_3
"""
    tsv_path = tmp_path / "events_with_json.tsv"
    tsv_path.write_text(content)
    return tsv_path


# endregion


# region Field Tests


class TestField:
    """Tests for the Field class."""

    def test_field_creation(self):
        """Test basic Field construction."""
        f = Field("rect_coords", "x")
        assert f.column == "rect_coords"
        assert f.fields == ("x",)

    def test_field_nested(self):
        """Test nested field access."""
        f = Field("metadata", "timing", "offset")
        assert f.column == "metadata"
        assert f.fields == ("timing", "offset")

    def test_field_requires_field_names(self):
        """Test that Field requires at least one field name."""
        with pytest.raises(ValueError, match="at least one field name"):
            Field("column")

    def test_field_resolve(self):
        """Test Field resolution against a PyArrow table."""
        # Create a table with a struct column
        struct_type = pa.struct([("x", pa.int64()), ("y", pa.int64())])
        struct_arr = pa.array(
            [{"x": 10, "y": 20}, {"x": 30, "y": 40}], type=struct_type
        )
        table = pa.table({"rect": struct_arr})

        # Resolve field
        f = Field("rect", "x")
        result = f.resolve(table)

        assert result.to_pylist() == [10, 30]

    def test_field_repr(self):
        """Test Field string representation."""
        f = Field("rect", "x")
        assert repr(f) == "Field('rect', 'x')"

    def test_field_equality(self):
        """Test Field equality comparison."""
        f1 = Field("rect", "x")
        f2 = Field("rect", "x")
        f3 = Field("rect", "y")

        assert f1 == f2
        assert f1 != f3


# endregion


# region ComputedField Tests


class TestComputedField:
    """Tests for the ComputedField class."""

    def test_computed_field_formula(self):
        """Test ComputedField with formula."""
        cf = ComputedField("end", formula="rect.x + rect.width")
        assert cf.name == "end"
        assert cf.formula == "rect.x + rect.width"
        assert cf.expr is None

    def test_computed_field_expr(self):
        """Test ComputedField with callable expression."""
        import pyarrow.compute as pc

        def compute_end(table):
            x = pc.struct_field(table["rect"], "x")
            w = pc.struct_field(table["rect"], "width")
            return pc.add(x, w)

        cf = ComputedField("end", expr=compute_end)
        assert cf.name == "end"
        assert cf.formula is None
        assert cf.expr is compute_end

    def test_computed_field_requires_formula_or_expr(self):
        """Test that ComputedField requires either formula or expr."""
        with pytest.raises(ValueError, match="either 'formula' or 'expr'"):
            ComputedField("end")

    def test_computed_field_cannot_have_both(self):
        """Test that ComputedField cannot have both formula and expr."""
        with pytest.raises(ValueError, match="cannot have both"):
            ComputedField("end", formula="a + b", expr=lambda t: t["a"])

    def test_computed_field_compute_formula(self):
        """Test formula evaluation."""
        struct_type = pa.struct([("x", pa.int64()), ("width", pa.int64())])
        struct_arr = pa.array(
            [{"x": 10, "width": 100}, {"x": 200, "width": 50}], type=struct_type
        )
        table = pa.table({"rect": struct_arr})

        cf = ComputedField("end", formula="rect.x + rect.width")
        result = cf.compute(table)

        assert result.to_pylist() == [110, 250]

    def test_computed_field_compute_expr(self):
        """Test callable expression evaluation."""
        import pyarrow.compute as pc

        struct_type = pa.struct([("x", pa.int64()), ("width", pa.int64())])
        struct_arr = pa.array(
            [{"x": 10, "width": 100}, {"x": 200, "width": 50}], type=struct_type
        )
        table = pa.table({"rect": struct_arr})

        def compute_end(t):
            x = pc.struct_field(t["rect"], "x")
            w = pc.struct_field(t["rect"], "width")
            return pc.add(x, w)

        cf = ComputedField("end", expr=compute_end)
        result = cf.compute(table)

        assert result.to_pylist() == [110, 250]

    def test_computed_field_repr(self):
        """Test ComputedField string representation."""
        cf1 = ComputedField("end", formula="a + b")
        cf2 = ComputedField("end", expr=lambda t: t["a"])

        assert "formula='a + b'" in repr(cf1)
        assert "expr=..." in repr(cf2)


def _number_table() -> pa.Table:
    return pa.table(
        {
            "n": pa.array([3, 4], pa.int64()),
            "x": pa.array([1.0, 2.0], pa.float64()),
            "rect": pa.array([{"x": 10}, {"x": 20}], pa.struct([("x", pa.int64())])),
            "2nd": pa.array([5, 6], pa.int64()),
        }
    )


class TestComputedFieldNumbers:
    """A numeric literal in a formula is a number, never a column name."""

    @pytest.mark.parametrize(
        ("formula", "values", "arrow_type"),
        [
            ("n * 2", [6, 8], pa.int64()),
            ("n / 2.5", [1.2, 1.6], pa.float64()),
            ("n / 2", [1, 2], pa.int64()),
            ("n / 2.0", [1.5, 2.0], pa.float64()),
            ("n * 1.5", [4.5, 6.0], pa.float64()),
            ("2 * n", [6, 8], pa.int64()),
            ("rect.x - 1", [9, 19], pa.int64()),
            ("n * n", [9, 16], pa.int64()),
            ("n + x", [4.0, 6.0], pa.float64()),
            ("2nd + n", [8, 10], pa.int64()),
        ],
    )
    def test_formula_values(self, formula, values, arrow_type):
        result = ComputedField("c", formula=formula).compute(_number_table())
        assert isinstance(result, pa.Array)
        assert result.type == arrow_type
        assert result.to_pylist() == values

    @pytest.mark.parametrize(
        "formula", ["n % 2", "n % n", "n 2", "- n", "n * * 2", "n +"]
    )
    def test_formula_not_alternating_is_refused(self, formula):
        with pytest.raises(ValueError) as caught:
            ComputedField("c", formula=formula).compute(_number_table())
        assert str(caught.value) == (
            f"A formula alternates operands and the operators + - * /: {formula!r}"
        )

    @pytest.mark.parametrize("formula", ["2", "2 * 3"])
    def test_formula_without_a_column_is_refused(self, formula):
        with pytest.raises(ValueError) as caught:
            ComputedField("c", formula=formula).compute(_number_table())
        assert str(caught.value) == f"A formula names at least one column: {formula!r}"


# endregion


# region parse_json_to_struct Tests


class TestParseJsonToStruct:
    """Tests for the parse_json_to_struct function."""

    def test_parse_basic_json(self):
        """Test parsing basic JSON strings."""
        json_strings = [
            '{"x": 10, "y": 20}',
            '{"x": 30, "y": 40}',
            '{"x": 50, "y": 60}',
        ]
        result = parse_json_to_struct(json_strings)

        assert pa.types.is_struct(result.type)
        assert result.type.num_fields == 2

        # Check values
        x_values = result.field("x").to_pylist()
        y_values = result.field("y").to_pylist()
        assert x_values == [10, 30, 50]
        assert y_values == [20, 40, 60]

    def test_parse_with_explicit_schema(self):
        """Test parsing with explicit schema."""
        json_strings = ['{"a": 1, "b": 2.5}', '{"a": 3, "b": 4.5}']
        result = parse_json_to_struct(
            json_strings, struct_schema={"a": int, "b": float}
        )

        assert result.type.num_fields == 2
        assert result.field("a").to_pylist() == [1, 3]
        assert result.field("b").to_pylist() == [2.5, 4.5]

    def test_parse_with_nulls(self):
        """Test parsing JSON with null values."""
        json_strings = ['{"x": 10}', None, '{"x": 30}']
        result = parse_json_to_struct(json_strings)

        # Check null mask
        assert result.is_null().to_pylist() == [False, True, False]

    def test_parse_invalid_json(self):
        """Test that invalid JSON produces null."""
        json_strings = ['{"x": 10}', "not valid json", '{"x": 30}']
        result = parse_json_to_struct(json_strings)

        assert result.is_null().to_pylist() == [False, True, False]


# endregion


# region TabularLoader Integration Tests


class TestTabularLoaderStructIntegration:
    """Integration tests for TabularLoader with struct columns."""

    def test_physical_loader_direct_columns(self, sample_tsv_with_json: Path):
        """Test loader using direct column names (traditional approach)."""
        import pyarrow.compute as pc

        class ThoresenPhysicalLoader(TsvLoader):
            """Loader for Thoresen TSV using physical time coordinates."""

            start_column = "start_time_sec"
            duration_column = "duration_sec"
            _default_unit = TimeUnit.seconds
            coordinate_type = NumberType.float
            default_event_type = "ThoresenSegment"

        loader = ThoresenPhysicalLoader()
        loader.load(sample_tsv_with_json)

        events = loader.events
        assert events is not None
        assert events.count == 3

        # Check coordinates are in seconds
        # Access start column and extract value field
        table = events.table
        start_col = table.column("start")
        starts = pc.struct_field(start_col, "value").to_pylist()
        # Values are parsed from JSON integers; exactly representable as float64
        assert starts[0] == 0.0
        assert starts[1] == 1.5
        assert starts[2] == 3.5


# JSON-to-struct integration tests for the legacy ``extra_columns``
# path were removed alongside the path itself.  The ``Field`` /
# ``ComputedField`` / ``parse_json_to_struct`` helpers continue to be
# exercised by the unit-test classes above.


# endregion
