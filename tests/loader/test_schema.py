"""Tests for loader/schema.py."""

from __future__ import annotations

from fractions import Fraction

import pyarrow as pa
import pytest

from timetoalign.core import NumberType, TimeUnit
from timetoalign.core.fields import field_metadata
from timetoalign.core.time import (
    coordinate_to_struct,
    struct_to_coordinate,
)
from timetoalign.loader import (
    TEMPORAL_TYPE_INSTANT,
    TEMPORAL_TYPE_INTERVAL,
    extend_schema,
    get_base_field_names,
    get_unit_from_schema,
    make_base_schema,
    make_coordinate_field,
    make_coordinate_type,
    make_table_metadata,
    parse_table_metadata,
)


class TestMakeCoordinateType:
    """Tests for make_coordinate_type."""

    def test_creates_struct_type(self) -> None:
        """Returns a struct type."""
        coord_type = make_coordinate_type(TimeUnit.ticks)
        assert isinstance(coord_type, pa.StructType)

    def test_has_required_fields(self) -> None:
        """Struct has value, numerator, denominator fields."""
        coord_type = make_coordinate_type(TimeUnit.ticks)
        field_names = [coord_type.field(i).name for i in range(len(coord_type))]
        assert "value" in field_names
        assert "numerator" in field_names
        assert "denominator" in field_names

    def test_value_is_float64(self) -> None:
        """The value field is float64."""
        coord_type = make_coordinate_type(TimeUnit.ticks)
        value_field = coord_type.field("value")
        assert value_field.type == pa.float64()


class TestMakeCoordinateField:
    """Tests for make_coordinate_field."""

    def test_creates_field(self) -> None:
        """Returns a PyArrow field."""
        field = make_coordinate_field("instant", TimeUnit.ticks)
        assert isinstance(field, pa.Field)

    def test_has_struct_type(self) -> None:
        """Field has struct type."""
        field = make_coordinate_field("instant", TimeUnit.ticks)
        assert isinstance(field.type, pa.StructType)

    def test_has_unit_metadata(self) -> None:
        """Unit is stored in field metadata."""
        field = make_coordinate_field("instant", TimeUnit.seconds)
        assert field.metadata is not None
        assert "unit" in field_metadata(field)
        assert field_metadata(field)["unit"] == "seconds"

    def test_nullable_default_true(self) -> None:
        """Field is nullable by default."""
        field = make_coordinate_field("instant", TimeUnit.ticks)
        assert field.nullable is True

    def test_nullable_false(self) -> None:
        """Can create non-nullable field."""
        field = make_coordinate_field("instant", TimeUnit.ticks, nullable=False)
        assert field.nullable is False


class TestCoordinateToStruct:
    """Tests for coordinate_to_struct."""

    def test_int_coordinate(self) -> None:
        """Int coordinates store exact numerator/denominator."""
        result = coordinate_to_struct(120)
        assert result["value"] == 120.0
        assert result["numerator"] == 120
        assert result["denominator"] == 1

    def test_float_coordinate(self) -> None:
        """A float is canonical; the ratio mirrors it exactly."""
        result = coordinate_to_struct(1.5)
        assert result["value"] == 1.5
        assert result["numerator"] == 3
        assert result["denominator"] == 2

    def test_fraction_coordinate(self) -> None:
        """Fraction coordinates store exact numerator/denominator."""
        result = coordinate_to_struct(Fraction(3, 4))
        assert result["value"] == 0.75
        assert result["numerator"] == 3
        assert result["denominator"] == 4


class TestStructToCoordinate:
    """Tests for struct_to_coordinate."""

    def test_to_fraction(self) -> None:
        """Converts struct back to Fraction."""
        struct = {"value": 0.75, "numerator": 3, "denominator": 4}
        result = struct_to_coordinate(struct, NumberType.fraction)
        assert result == Fraction(3, 4)

    def test_to_fraction_without_numerator(self) -> None:
        """Falls back to float-to-Fraction conversion."""
        struct = {"value": 0.5, "numerator": None, "denominator": None}
        result = struct_to_coordinate(struct, NumberType.fraction)
        assert result == Fraction(1, 2)

    def test_to_int(self) -> None:
        """Converts struct to int."""
        struct = {"value": 120.0, "numerator": 120, "denominator": 1}
        result = struct_to_coordinate(struct, NumberType.int)
        assert result == 120
        assert isinstance(result, int)

    def test_to_float(self) -> None:
        """Converts struct to float."""
        struct = {"value": 1.5, "numerator": None, "denominator": None}
        result = struct_to_coordinate(struct, NumberType.float)
        assert result == 1.5
        assert isinstance(result, float)


class TestMakeBaseSchema:
    """Tests for make_base_schema."""

    def test_creates_schema(self) -> None:
        """Returns a PyArrow schema."""
        schema = make_base_schema(TimeUnit.ticks)
        assert isinstance(schema, pa.Schema)

    def test_has_required_columns(self) -> None:
        """Schema has all base fields.

        Note: The schema no longer has a separate 'instant' field.
        InstantEvents use 'start' with 'end' being null or equal to 'start'.
        IntervalEvents use 'start' and 'end' with 'duration' computed.
        """
        schema = make_base_schema(TimeUnit.ticks)
        names = schema.names
        assert "id" in names
        assert "name" in names
        assert "temporal_type" in names
        assert "event_type" in names
        assert "start" in names
        assert "end" in names
        assert "duration" in names

    def test_id_is_not_nullable(self) -> None:
        """The id field is not nullable."""
        schema = make_base_schema(TimeUnit.ticks)
        assert not schema.field("id").nullable

    def test_name_is_nullable(self) -> None:
        """The name field is nullable."""
        schema = make_base_schema(TimeUnit.ticks)
        assert schema.field("name").nullable


class TestGetBaseColumnNames:
    """Tests for get_base_field_names."""

    def test_returns_list(self) -> None:
        """Returns a list of strings."""
        names = get_base_field_names()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_has_all_columns(self) -> None:
        """Returns all 7 base fields.

        Note: The schema no longer has a separate 'instant' field.
        Base fields are: id, name, temporal_type, event_type, start, end, duration.
        """
        names = get_base_field_names()
        assert len(names) == 7
        assert "id" in names
        assert "start" in names


class TestExtendSchema:
    """Tests for extend_schema."""

    def test_adds_fields(self) -> None:
        """Extended schema has new fields."""
        base = make_base_schema(TimeUnit.ticks)
        extra = [pa.field("pitch", pa.int8())]
        extended = extend_schema(base, extra)

        assert "pitch" in extended.names
        assert len(extended) == len(base) + 1


class TestGetUnitFromSchema:
    """Tests for get_unit_from_schema."""

    def test_extracts_unit_from_instant(self) -> None:
        """Extracts unit from instant field metadata."""
        schema = make_base_schema(TimeUnit.seconds)
        unit = get_unit_from_schema(schema)
        assert unit == TimeUnit.seconds

    def test_returns_none_for_no_metadata(self) -> None:
        """Returns None if no coordinate fields exist."""
        schema = pa.schema([pa.field("id", pa.string())])
        unit = get_unit_from_schema(schema)
        assert unit is None

    def test_get_unit_field_exists_no_metadata(self) -> None:
        """Returns None if coordinate field exists but has no metadata."""
        schema = pa.schema([pa.field("instant", pa.float64())])
        unit = get_unit_from_schema(schema)
        assert unit is None


class TestMakeTableMetadata:
    """Tests for make_table_metadata."""

    def test_creates_metadata_dict(self) -> None:
        """Returns a bytes dict."""
        metadata = make_table_metadata(TimeUnit.ticks, NumberType.int)
        assert isinstance(metadata, dict)
        assert b"timetoalign" in metadata

    def test_includes_version(self) -> None:
        """Metadata includes timetoalign version."""
        import json

        metadata = make_table_metadata(TimeUnit.ticks, NumberType.int)
        parsed = json.loads(metadata[b"timetoalign"])
        assert "timetoalign_version" in parsed

    def test_includes_sources(self) -> None:
        """Metadata includes sources list."""
        import json

        sources = [{"path": "file.mid"}]
        metadata = make_table_metadata(TimeUnit.ticks, NumberType.int, sources=sources)
        parsed = json.loads(metadata[b"timetoalign"])
        assert parsed["sources"] == sources

    def test_make_table_metadata_with_extra(self) -> None:
        """Metadata includes extra fields."""
        import json

        metadata = make_table_metadata(
            TimeUnit.ticks, NumberType.int, extra={"custom": "value"}
        )
        parsed = json.loads(metadata[b"timetoalign"])
        assert parsed["custom"] == "value"


class TestParseTableMetadata:
    """Tests for parse_table_metadata."""

    def test_parses_metadata(self) -> None:
        """Parses metadata from schema."""
        base_schema = make_base_schema(TimeUnit.ticks)
        metadata = make_table_metadata(TimeUnit.ticks, NumberType.int)
        schema = base_schema.with_metadata(metadata)

        parsed = parse_table_metadata(schema)
        assert parsed["unit"] == "ticks"
        assert parsed["number_type"] == "int"

    def test_returns_empty_for_no_metadata(self) -> None:
        """Returns empty dict if no timetoalign metadata."""
        schema = pa.schema([pa.field("id", pa.string())])
        parsed = parse_table_metadata(schema)
        assert parsed == {}


class TestScoreStoreMetadata:
    """A score store's table states the number type its cells are in."""

    @staticmethod
    def _stores() -> list[type]:
        from timetoalign.loader.score.stores.annotations import AnnotationEventData
        from timetoalign.loader.score.stores.controls import ControlEventData
        from timetoalign.loader.score.stores.measures import MeasureData
        from timetoalign.loader.score.stores.notes import NoteEventData

        return [NoteEventData, AnnotationEventData, ControlEventData, MeasureData]

    @pytest.mark.parametrize(
        ("unit", "number_type", "stated"),
        [
            (TimeUnit.quarters, None, "fraction"),
            (TimeUnit.ticks, None, "int"),
            (TimeUnit.seconds, None, "float"),
            (TimeUnit.quarters, NumberType.float, "float"),
        ],
    )
    def test_from_dicts_states_the_resolved_number_type(
        self, unit: TimeUnit, number_type: NumberType | None, stated: str
    ) -> None:
        for store in self._stores():
            data = store.from_dicts(
                [{"event_type": "X", "start": 0, "duration": 1}], unit, number_type
            )
            assert parse_table_metadata(data.table.schema)["number_type"] == stated
            assert str(data.number_type) == stated


class TestTemporalTypeConstants:
    """Tests for temporal type constants."""

    def test_instant_constant(self) -> None:
        """TEMPORAL_TYPE_INSTANT is 'instant'."""
        assert TEMPORAL_TYPE_INSTANT == "instant"

    def test_interval_constant(self) -> None:
        """TEMPORAL_TYPE_INTERVAL is 'interval'."""
        assert TEMPORAL_TYPE_INTERVAL == "interval"
