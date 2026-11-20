"""Tests for fields/coordinate.py — CoordinateField construction, access, and serialization."""

from __future__ import annotations

from fractions import Fraction

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timetoalign.core.enums import Domain, NumberType, TimeUnit
from timetoalign.core.fields import (
    TIMETOALIGN_METADATA_KEY,
    StructField,
    metadata_blob_from_dict,
    parse_metadata_blob,
)
from timetoalign.core.protocols import CoordinateLike, SemanticTypeLike
from timetoalign.core.time import (
    Coordinate,
    CoordinateField,
    coordinate_to_struct,
)
from timetoalign.storage.schema import make_coordinate_type

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_coord_array(
    unit: TimeUnit = TimeUnit.quarters,
    values: list | None = None,
) -> tuple[pa.Array, pa.StructType]:
    """Build a coordinate struct array from simple values."""
    coord_type = make_coordinate_type(unit)
    if values is None:
        values = [Fraction(3, 4), 1.5, 2]
    data = [coordinate_to_struct(v) for v in values]
    arr = pa.array(data, type=coord_type)
    return arr, coord_type


# ---------------------------------------------------------------------------
# Protocol Conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Verify CoordinateField satisfies CoordinateLike and SemanticTypeLike."""

    def test_coordinate_field_satisfies_coordinate_like(self) -> None:
        arr, _ = _make_coord_array()
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.quarters, number_type=NumberType.float
        )
        assert isinstance(cf, CoordinateLike)

    def test_coordinate_field_satisfies_semantic_type_like(self) -> None:
        arr, _ = _make_coord_array()
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.quarters, number_type=NumberType.float
        )
        assert isinstance(cf, SemanticTypeLike)

    def test_coordinate_field_semantic_type(self) -> None:
        arr, _ = _make_coord_array()
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.seconds, number_type=NumberType.float
        )
        assert cf.semantic_type == "Coordinate"


# ---------------------------------------------------------------------------
# Construction (from_field)
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for CoordinateField.from_field() with various source types."""

    def test_from_pa_array(self) -> None:
        """Construct from pa.Array with unit + number_type kwargs."""
        arr, _ = _make_coord_array(TimeUnit.seconds, [1.0, 2.0, 3.0])
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.seconds, number_type=NumberType.float
        )
        assert len(cf) == 3
        assert cf.unit == TimeUnit.seconds
        assert cf.number_type == NumberType.float

    def test_from_struct_field(self) -> None:
        """Construct from an existing StructField."""
        arr, coord_type = _make_coord_array(TimeUnit.quarters)
        pa_field = pa.field("start", coord_type)
        sf = StructField(arr, pa_field)
        cf = CoordinateField.from_field(
            sf, unit=TimeUnit.quarters, number_type=NumberType.fraction
        )
        assert len(cf) == 3
        assert cf.unit == TimeUnit.quarters
        assert cf.number_type == NumberType.fraction

    def test_from_pa_field_schema_only(self) -> None:
        """Construct from pa.Field (no data), verify is_empty."""
        coord_type = make_coordinate_type(TimeUnit.ticks)
        pa_field = pa.field(
            "onset",
            coord_type,
            metadata={
                TIMETOALIGN_METADATA_KEY: metadata_blob_from_dict(
                    {"unit": "ticks", "number_type": "int"}
                )
            },
        )
        cf = CoordinateField.from_field(pa_field)
        assert cf.is_empty is True
        assert cf.unit == TimeUnit.ticks
        assert cf.number_type == NumberType.int

    def test_from_tuple(self) -> None:
        """Construct from (pa.Array, pa.Field) tuple."""
        arr, coord_type = _make_coord_array(TimeUnit.seconds, [0.5, 1.0])
        pa_field = pa.field(
            "time",
            coord_type,
            metadata={
                TIMETOALIGN_METADATA_KEY: metadata_blob_from_dict(
                    {"unit": "seconds", "number_type": "float"}
                )
            },
        )
        cf = CoordinateField.from_field((arr, pa_field))
        assert len(cf) == 2
        assert cf.unit == TimeUnit.seconds
        assert cf.number_type == NumberType.float

    def test_from_pa_field_with_tta_metadata(self) -> None:
        """Reconstruct from a pa.Field carrying the TTA blob (Parquet round-trip)."""
        meta_dict = {
            "field_type": "CoordinateField",
            "unit": "quarters",
            "domain": "logical",
            "number_type": "fraction",
        }
        coord_type = make_coordinate_type(TimeUnit.quarters)
        pa_field = pa.field(
            "onset",
            coord_type,
            metadata={TIMETOALIGN_METADATA_KEY: metadata_blob_from_dict(meta_dict)},
        )
        cf = CoordinateField.from_field(pa_field)
        assert cf.unit == TimeUnit.quarters
        assert cf.number_type == NumberType.fraction
        assert cf.is_empty is True

    def test_from_field_missing_unit_raises(self) -> None:
        """ValueError when unit is missing from both kwargs and metadata."""
        arr, _ = _make_coord_array()
        with pytest.raises(ValueError, match="unit"):
            CoordinateField.from_field(arr, number_type=NumberType.float)

    def test_from_field_missing_number_type_raises(self) -> None:
        """ValueError when number_type is missing from both kwargs and metadata."""
        arr, _ = _make_coord_array()
        with pytest.raises(ValueError, match="number_type"):
            CoordinateField.from_field(arr, unit=TimeUnit.quarters)


# ---------------------------------------------------------------------------
# Element Access (__getitem__)
# ---------------------------------------------------------------------------


class TestElementAccess:
    """Tests for CoordinateField.__getitem__."""

    def test_getitem_returns_coordinate(self) -> None:
        """Verify returns Coordinate instance with correct value and unit."""
        arr, _ = _make_coord_array(TimeUnit.seconds, [1.5])
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.seconds, number_type=NumberType.float
        )
        coord = cf[0]
        assert isinstance(coord, Coordinate)
        assert coord.value == 1.5
        assert coord.unit == TimeUnit.seconds

    def test_getitem_fraction(self) -> None:
        """Verify Fraction precision is preserved when number_type=fraction."""
        arr, _ = _make_coord_array(TimeUnit.quarters, [Fraction(3, 4)])
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.quarters, number_type=NumberType.fraction
        )
        coord = cf[0]
        assert isinstance(coord, Coordinate)
        assert isinstance(coord.value, Fraction)
        assert coord.value == Fraction(3, 4)

    def test_getitem_int(self) -> None:
        """Verify int values when number_type=int."""
        arr, _ = _make_coord_array(TimeUnit.ticks, [120])
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.ticks, number_type=NumberType.int
        )
        coord = cf[0]
        assert isinstance(coord, Coordinate)
        assert isinstance(coord.value, int)
        assert coord.value == 120

    def test_getitem_null_returns_none(self) -> None:
        """Verify None for null struct entries."""
        coord_type = make_coordinate_type(TimeUnit.seconds)
        data = [coordinate_to_struct(1.0), None, coordinate_to_struct(3.0)]
        arr = pa.array(data, type=coord_type)
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.seconds, number_type=NumberType.float
        )
        assert cf[0] is not None
        assert cf[1] is None
        assert cf[2] is not None

    def test_getitem_multiple_elements(self) -> None:
        """Iterate several elements, verify each."""
        values = [Fraction(1, 2), Fraction(3, 4), Fraction(7, 8)]
        arr, _ = _make_coord_array(TimeUnit.quarters, values)
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.quarters, number_type=NumberType.fraction
        )
        for i, expected in enumerate(values):
            coord = cf[i]
            assert coord is not None
            assert coord.value == expected
            assert coord.unit == TimeUnit.quarters


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Tests for CoordinateField properties."""

    def test_unit_property(self) -> None:
        arr, _ = _make_coord_array(TimeUnit.seconds, [1.0])
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.seconds, number_type=NumberType.float
        )
        assert cf.unit == TimeUnit.seconds

    def test_domain_property(self) -> None:
        """Verify .domain is derived from unit."""
        arr, _ = _make_coord_array(TimeUnit.quarters, [1.0])
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.quarters, number_type=NumberType.float
        )
        assert cf.domain == Domain.logical

        arr2, _ = _make_coord_array(TimeUnit.seconds, [1.0])
        cf2 = CoordinateField.from_field(
            arr2, unit=TimeUnit.seconds, number_type=NumberType.float
        )
        assert cf2.domain == Domain.physical

        arr3, _ = _make_coord_array(TimeUnit.pixels, [1.0])
        cf3 = CoordinateField.from_field(
            arr3, unit=TimeUnit.pixels, number_type=NumberType.int
        )
        assert cf3.domain == Domain.graphical

    def test_number_type_property(self) -> None:
        arr, _ = _make_coord_array(TimeUnit.quarters, [1.0])
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.quarters, number_type=NumberType.fraction
        )
        assert cf.number_type == NumberType.fraction

    def test_metadata_dict(self) -> None:
        """Verify returns dict with field_type, unit, domain, number_type."""
        arr, _ = _make_coord_array(TimeUnit.ticks, [120])
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.ticks, number_type=NumberType.int
        )
        md = cf.metadata_dict()
        assert md == {
            "field_type": "CoordinateField",
            "unit": "ticks",
            "domain": "logical",
            "number_type": "int",
        }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """Tests for CoordinateField serialization."""

    def test_to_field_injects_metadata(self) -> None:
        """Verify to_field() produces a pa.Field carrying the TTA JSON blob."""
        arr, _ = _make_coord_array(TimeUnit.quarters, [Fraction(3, 4)])
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.quarters, number_type=NumberType.fraction
        )
        pa_field = cf.to_field()

        assert isinstance(pa_field, pa.Field)
        raw_meta = pa_field.metadata
        assert TIMETOALIGN_METADATA_KEY in raw_meta
        blob = parse_metadata_blob(raw_meta[TIMETOALIGN_METADATA_KEY])
        assert blob["field_type"] == "CoordinateField"
        assert blob["unit"] == "quarters"
        assert blob["domain"] == "logical"
        assert blob["number_type"] == "fraction"

    def test_parquet_round_trip(self, tmp_path: object) -> None:
        """Write pa.Table with CoordinateField column, read back, verify data + metadata."""
        from pathlib import Path

        tmp_dir = Path(str(tmp_path))
        parquet_path = tmp_dir / "coords.parquet"

        # Build CoordinateField
        values = [Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)]
        arr, _ = _make_coord_array(TimeUnit.quarters, values)
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.quarters, number_type=NumberType.fraction
        )

        # Build table using the enriched pa.Field
        enriched_field = cf.to_field()
        table = pa.table(
            {"coordinate": arr},
            schema=pa.schema([enriched_field.with_name("coordinate")]),
        )

        # Write and read back
        pq.write_table(table, str(parquet_path))
        table_back = pq.read_table(str(parquet_path))

        # Reconstruct CoordinateField from the read-back table
        cf2 = CoordinateField.from_table(table_back, "coordinate")

        # Verify metadata survived
        assert cf2.unit == TimeUnit.quarters
        assert cf2.number_type == NumberType.fraction
        assert cf2.domain == Domain.logical

        # Verify data survived
        assert len(cf2) == 3
        for i, expected in enumerate(values):
            coord = cf2[i]
            assert coord is not None
            assert coord.value == expected
            assert coord.unit == TimeUnit.quarters


# ---------------------------------------------------------------------------
# from_table
# ---------------------------------------------------------------------------


class TestFromTable:
    """Tests for CoordinateField.from_table()."""

    def _make_table(self) -> pa.Table:
        """Build a table with a single coordinate column carrying TTA metadata."""
        values = [Fraction(1, 4), Fraction(1, 2)]
        arr, _ = _make_coord_array(TimeUnit.quarters, values)
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.quarters, number_type=NumberType.fraction
        )
        enriched = cf.to_field()
        return pa.table(
            {"coordinate": arr},
            schema=pa.schema([enriched.with_name("coordinate")]),
        )

    def test_from_table_explicit_column(self) -> None:
        table = self._make_table()
        cf = CoordinateField.from_table(table, "coordinate")
        assert len(cf) == 2
        assert cf.unit == TimeUnit.quarters
        assert cf.number_type == NumberType.fraction
        assert cf[0] is not None
        assert cf[0].value == Fraction(1, 4)

    def test_from_table_auto_detect(self) -> None:
        table = self._make_table()
        cf = CoordinateField.from_table(table)
        assert len(cf) == 2
        assert cf.unit == TimeUnit.quarters

    def test_from_table_no_candidate_raises(self) -> None:
        table = pa.table({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="No struct field"):
            CoordinateField.from_table(table)

    def test_from_table_multiple_candidates_raises(self) -> None:
        values = [Fraction(1, 4)]
        arr, _ = _make_coord_array(TimeUnit.quarters, values)
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.quarters, number_type=NumberType.fraction
        )
        enriched = cf.to_field()
        table = pa.table(
            {"onset": arr, "offset": arr},
            schema=pa.schema(
                [
                    enriched.with_name("onset"),
                    enriched.with_name("offset"),
                ]
            ),
        )
        with pytest.raises(ValueError, match="Multiple candidate"):
            CoordinateField.from_table(table)


# ---------------------------------------------------------------------------
# Copy-on-write
# ---------------------------------------------------------------------------


class TestCopyOnWrite:
    """Tests for CoordinateField.with_unit."""

    def test_with_unit(self) -> None:
        """Verify new CoordinateField has different unit, same data."""
        arr, _ = _make_coord_array(TimeUnit.quarters, [Fraction(1, 2), Fraction(3, 4)])
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.quarters, number_type=NumberType.fraction
        )
        cf2 = cf.with_unit(TimeUnit.number)

        # Different unit
        assert cf2.unit == TimeUnit.number
        assert cf.unit == TimeUnit.quarters

        # Same data
        assert len(cf2) == 2
        coord = cf2[0]
        assert coord is not None
        assert coord.value == Fraction(1, 2)
        assert coord.unit == TimeUnit.number  # unit changed in returned Coordinate too

        # Same number_type
        assert cf2.number_type == NumberType.fraction


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------


class TestDelegation:
    """Tests for SemanticField delegation to the inner StructField."""

    def test_value_returns_struct_field(self) -> None:
        """Verify .value returns the inner StructField."""
        arr, _ = _make_coord_array(TimeUnit.seconds, [1.0])
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.seconds, number_type=NumberType.float
        )
        raw = cf.value
        assert isinstance(raw, StructField)

    def test_delegation_field_names(self) -> None:
        """Verify .field_names works via __getattr__ delegation."""
        arr, _ = _make_coord_array(TimeUnit.seconds, [1.0])
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.seconds, number_type=NumberType.float
        )
        assert cf.field_names == ["value", "numerator", "denominator"]

    def test_delegation_get_sub_field(self) -> None:
        """Verify .get_sub_field("value") works via delegation."""
        arr, _ = _make_coord_array(TimeUnit.seconds, [1.5])
        cf = CoordinateField.from_field(
            arr, unit=TimeUnit.seconds, number_type=NumberType.float
        )
        sub = cf.get_sub_field("value")
        assert sub[0] == 1.5
