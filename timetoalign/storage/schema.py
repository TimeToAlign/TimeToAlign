"""Schema definitions for TimeToAlign! EventStore.

This module defines PyArrow schemas for event storage, including:
- Coordinate struct types (with float + optional Fraction representation)
- Base event schema fields
- Schema construction utilities

The coordinate storage strategy preserves original precision (Fraction numerator/denominator)
while providing a float64 representation for fast queries. Unit metadata is stored at
the field level, not per-row.
"""

from __future__ import annotations

from typing import Any, Callable

import pyarrow as pa
import pyarrow.compute as pc

from timetoalign.core import NumberType, TimeUnit
from timetoalign.core.fields import (
    TIMETOALIGN_METADATA_KEY,
    field_metadata,
    metadata_blob_from_dict,
    parse_metadata_blob,
)
from timetoalign.core.time import (
    RATIONAL_STRUCT_TYPE,
)

# region Coordinate Schema


def make_coordinate_type(unit: TimeUnit) -> pa.StructType:
    """Create a coordinate struct type.

    The struct contains:
    - value: float64 representation (nullable when struct is null)
    - numerator: int64 (for Fraction, nullable)
    - denominator: int64 (for Fraction, nullable)

    Note: Unit metadata is stored at the Field level, not the StructType level.
    Use make_coordinate_field() to create a field with unit metadata.

    Note: All fields are nullable to allow for null struct values (e.g., missing
    end coordinates for instant events).

    Args:
        unit: The time unit (unused here, kept for API consistency).

    Returns:
        A PyArrow struct type.
    """
    return RATIONAL_STRUCT_TYPE


def make_rational_field(
    name: str,
    nullable: bool = True,
    metadata: dict[str, str] | None = None,
) -> pa.Field:
    """Create a rational-valued field with optional metadata.

    The column type is the canonical
    :data:`~timetoalign.core.time.RATIONAL_STRUCT_TYPE`; build its row
    values with :func:`~timetoalign.core.time.rational_to_struct`. The unit
    and representation go into the versioned metadata blob every other
    field in the library uses, so a reader resolves them the same way here
    as anywhere else.

    Args:
        name: The field name.
        nullable: Whether the field is nullable.
        metadata: Additional metadata (e.g., unit, number_type).

    Returns:
        A PyArrow field with RATIONAL_STRUCT_TYPE.
    """
    payload: dict[str, str] = {"number_type": NumberType.fraction.name}
    if metadata:
        payload.update(metadata)
    return pa.field(
        name,
        RATIONAL_STRUCT_TYPE,
        nullable=nullable,
        metadata={TIMETOALIGN_METADATA_KEY: metadata_blob_from_dict(payload)},
    )


def make_coordinate_field(
    name: str,
    unit: TimeUnit,
    nullable: bool = True,
    number_type: NumberType | None = None,
) -> pa.Field:
    """Create a coordinate field with unit and number_type in metadata.

    Args:
        name: The field name.
        unit: The time unit to store in metadata.
        nullable: Whether the field is nullable.
        number_type: The number type to store in metadata.

    Returns:
        A PyArrow field with struct type and unit/number_type metadata.
    """
    coord_type = make_coordinate_type(unit)
    payload: dict[str, str] = {"unit": str(unit)}
    payload["number_type"] = unit.resolve_number_type(number_type).name
    return pa.field(
        name,
        coord_type,
        nullable=nullable,
        metadata={TIMETOALIGN_METADATA_KEY: metadata_blob_from_dict(payload)},
    )


def is_coordinate_type(dtype: pa.DataType) -> bool:
    """Check if a PyArrow type is a coordinate struct type.

    Coordinate structs have three specific fields: value, numerator, denominator.
    This is used to detect coordinate fields regardless of where they appear
    in the schema.

    Args:
        dtype: The PyArrow data type to check.

    Returns:
        True if the type is a coordinate struct, False otherwise.
    """
    if not pa.types.is_struct(dtype):
        return False

    field_names = {f.name for f in dtype}
    return field_names == {"value", "numerator", "denominator"}


# endregion


# region Base Event Schema

# Temporal type values (not an enum in PyArrow, just string literals)
TEMPORAL_TYPE_INSTANT = "instant"
TEMPORAL_TYPE_INTERVAL = "interval"


def make_base_schema(
    unit: TimeUnit, number_type: NumberType | None = None
) -> pa.Schema:
    """Create the base event schema with coordinate fields.

    The base schema includes fields that are always present:
    - id: unique identifier (required)
    - name: human-readable label (optional)
    - temporal_type: "instant" or "interval"
    - event_type: class name (e.g., "Note", "Beat")
    - instant: coordinate for InstantEvents (nullable)
    - start: start coordinate for IntervalEvents (nullable)
    - end: end coordinate for IntervalEvents (nullable)
    - duration: computed duration for IntervalEvents (nullable)

    Args:
        unit: The time unit for coordinate fields.
        number_type: The number type for coordinate fields.

    Returns:
        A PyArrow schema with base event fields.
    """
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("name", pa.string(), nullable=True),
            pa.field("temporal_type", pa.string(), nullable=False),
            pa.field("event_type", pa.string(), nullable=False),
            make_coordinate_field(
                "start", unit, nullable=True, number_type=number_type
            ),
            make_coordinate_field("end", unit, nullable=True, number_type=number_type),
            make_coordinate_field(
                "duration", unit, nullable=True, number_type=number_type
            ),
        ]
    )


def get_base_field_names() -> list[str]:
    """Return the list of base field names.

    Returns:
        List of field names in the base schema.
    """
    return [
        "id",
        "name",
        "temporal_type",
        "event_type",
        "start",
        "end",
        "duration",
    ]


def extend_schema(base: pa.Schema, extra_fields: list[pa.Field]) -> pa.Schema:
    """Extend a base schema with additional fields.

    Args:
        base: The base PyArrow schema.
        extra_fields: Additional fields to append.

    Returns:
        A new schema with the extra fields appended.
    """
    fields = list(base)
    fields.extend(extra_fields)
    metadata: dict[bytes | str, bytes | str] | None = (
        None if base.metadata is None else {**base.metadata}
    )
    return pa.schema(fields, metadata=metadata)


def get_unit_from_schema(schema: pa.Schema) -> TimeUnit | None:
    """Extract the time unit from a schema's coordinate field metadata.

    Args:
        schema: A PyArrow schema with coordinate fields.

    Returns:
        The TimeUnit if found, None otherwise.
    """
    # Check the 'start' field
    try:
        unit = field_metadata(schema.field("start")).get("unit")
    except KeyError:
        return None
    return TimeUnit(unit) if unit else None


# endregion


# region Schema Metadata


def make_table_metadata(
    unit: TimeUnit,
    number_type: NumberType,
    sources: list[dict[str, Any]] | None = None,
    loader_class: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[bytes, bytes]:
    """Create metadata dict for a PyArrow table schema.

    Table-level counterpart of the per-field stamp: the payload is
    encoded by :func:`~timetoalign.core.fields.metadata_blob_from_dict`,
    so it lands under the same key and carries the same version stamp.

    Args:
        unit: The time unit for coordinates.
        number_type: The number type (int64, float64, fraction).
        sources: List of source file metadata dicts.
        loader_class: Name of the Loader subclass.
        extra: Additional metadata to include.

    Returns:
        A dict suitable for PyArrow schema metadata.
    """
    metadata: dict[str, Any] = {
        "timetoalign_version": "0.1.0",
        "unit": str(unit),
        "number_type": str(number_type),
        "sources": sources or [],
        "loader_class": loader_class,
    }
    if extra:
        metadata.update(extra)

    return {TIMETOALIGN_METADATA_KEY: metadata_blob_from_dict(metadata)}


def parse_table_metadata(schema: pa.Schema) -> dict[str, Any]:
    """Parse TimeToAlign! metadata from a PyArrow schema.

    Args:
        schema: A PyArrow schema with timetoalign metadata.

    Returns:
        The parsed metadata dict, or empty dict if not found.

    Raises:
        ValueError: If the blob is present but unversioned or newer than
            this build understands.
    """
    if schema.metadata:
        return parse_metadata_blob(schema.metadata.get(TIMETOALIGN_METADATA_KEY))
    return {}


# endregion


# region LoaderSchema


# Type mappings for simple type specifications
_TYPE_MAP: dict[str | type, pa.DataType] = {
    "int": pa.int64(),
    "int64": pa.int64(),
    "int32": pa.int32(),
    "float": pa.float64(),
    "float64": pa.float64(),
    "float32": pa.float32(),
    "str": pa.string(),
    "string": pa.string(),
    "bool": pa.bool_(),
    "boolean": pa.bool_(),
    int: pa.int64(),
    float: pa.float64(),
    str: pa.string(),
    bool: pa.bool_(),
}


# region Field and ComputedField


class Field:
    """Reference to a struct field for coordinate access.

    Field provides a way to specify nested struct field access in TabularLoader
    configurations. It supports multiple input formats:

    - Tuple: ("rect_coords", "x")
    - Field object: Field("rect_coords", "x")
    - PyArrow expression (for advanced use)

    The Field is resolved during column extraction using PyArrow's struct_field
    compute function for vectorized access.

    Attributes:
        column: The parent column name containing the struct.
        fields: Nested field path within the struct.

    Examples:
        >>> # Access x field from rect_coords struct
        >>> start = Field("rect_coords", "x")

        >>> # Deeply nested access
        >>> value = Field("metadata", "timing", "offset")

        >>> # In loader configuration — combine with explicit JSON-to-struct
        >>> # preprocessing on the loader side, then point start_column at the
        >>> # nested field.
        >>> class MyLoader(TsvLoader):
        ...     start_column = Field("rect_coords", "x")
    """

    def __init__(self, column: str, *fields: str) -> None:
        """Initialize Field.

        Args:
            column: The parent column name.
            *fields: One or more nested field names to access.

        Raises:
            ValueError: If no field names are provided.
        """
        if not fields:
            raise ValueError("Field requires at least one field name after the column")
        self.column = column
        self.fields = fields

    def resolve(self, table: pa.Table) -> pa.Array:
        """Resolve the field reference to a PyArrow array.

        Uses PyArrow compute functions for vectorized struct field access.
        If the source column is a string (not already a struct), automatically
        parses it as JSON to extract nested fields.

        Args:
            table: The PyArrow table containing the struct column.

        Returns:
            The extracted field as a PyArrow array.

        Raises:
            KeyError: If the column doesn't exist.
            ValueError: If the field path is invalid or JSON parsing fails.
        """
        if self.column not in table.column_names:
            raise KeyError(
                f"Column '{self.column}' not found in table. "
                f"Available columns: {table.column_names}"
            )

        array = table.column(self.column).combine_chunks()

        # Auto-detect JSON string columns and parse them
        if pa.types.is_string(array.type) or pa.types.is_large_string(array.type):
            # Column is a string - assume it's JSON and parse it
            array = parse_json_to_struct(array.to_pylist())

        # Navigate through nested fields
        for field_name in self.fields:
            if not pa.types.is_struct(array.type):
                raise ValueError(
                    f"Cannot access field '{field_name}' on non-struct type {array.type}. "
                    f"Column '{self.column}' may not contain valid JSON."
                )
            array = pc.struct_field(array, field_name)

        return array

    def __repr__(self) -> str:
        fields_str = ", ".join(repr(f) for f in self.fields)
        return f"Field({self.column!r}, {fields_str})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Field):
            return NotImplemented
        return self.column == other.column and self.fields == other.fields

    def __hash__(self) -> int:
        return hash((self.column, self.fields))


class ComputedField:
    """Computed column derived from an expression or function.

    ComputedField allows defining derived columns using either:
    - A formula string (e.g., "rect_coords.x + rect_coords.width")
    - A callable that operates on PyArrow arrays

    The computation is fully vectorized using PyArrow compute functions.

    Attributes:
        name: The output column name.
        formula: Optional formula string for simple expressions.
        expr: Optional callable for complex computations.

    Formula Syntax:
        - Struct field access: "column.field" or "column.nested.field"
        - Arithmetic: +, -, *, /
        - Column references: plain column names

    Examples:
        >>> # Simple addition using formula
        >>> end = ComputedField("end", formula="rect_coords.x + rect_coords.width")

        >>> # Using a callable for complex logic
        >>> def compute_end(table):
        ...     x = pc.struct_field(table["rect_coords"], "x")
        ...     w = pc.struct_field(table["rect_coords"], "width")
        ...     return pc.add(x, w)
        >>> end = ComputedField("end", expr=compute_end)

        >>> # In loader configuration
        >>> class MyLoader(TsvLoader):
        ...     end_column = ComputedField("end", formula="rect_coords.x + rect_coords.width")
    """

    def __init__(
        self,
        name: str,
        formula: str | None = None,
        expr: Callable[[pa.Table], pa.Array] | None = None,
    ) -> None:
        """Initialize ComputedField.

        Args:
            name: The output column name.
            formula: A formula string (e.g., "a.x + a.width").
            expr: A callable that takes a PyArrow Table and returns an Array.

        Raises:
            ValueError: If neither formula nor expr is provided.
        """
        if formula is None and expr is None:
            raise ValueError("ComputedField requires either 'formula' or 'expr'")
        if formula is not None and expr is not None:
            raise ValueError("ComputedField cannot have both 'formula' and 'expr'")

        self.name = name
        self.formula = formula
        self.expr = expr

    def compute(self, table: pa.Table) -> pa.Array:
        """Compute the field value from the table.

        Args:
            table: The PyArrow table with source columns.

        Returns:
            The computed PyArrow array.
        """
        if self.expr is not None:
            return self.expr(table)
        else:
            return self._evaluate_formula(table)

    def _evaluate_formula(self, table: pa.Table) -> pa.Array:
        """Evaluate the formula string against the table.

        Supports simple arithmetic expressions with struct field access.
        Uses a basic expression parser for safety.

        Args:
            table: The PyArrow table.

        Returns:
            The computed array.
        """
        import re

        formula = self.formula
        assert formula is not None

        # Tokenize the formula. A number comes first and must not run on into
        # a name, so "2" and "1.5" are numbers while "2nd" and "a.x" stay
        # column references.
        token_pattern = r"(\d+(?:\.\d+)?)(?![\w.])|(\w+(?:\.\w+)*)|([+\-*/])"
        tokens = re.findall(token_pattern, formula)

        # Parse into operands and operators
        operands: list[pa.Array | pa.Scalar] = []
        operators: list[str] = []
        names_a_column = False

        for num, ref, op in tokens:
            if num:
                # An integer literal stays an integer, so it keeps an integer
                # column integral; a decimal literal is a double.
                operands.append(pa.scalar(float(num) if "." in num else int(num)))
            elif ref:
                # Column or struct field reference
                operands.append(self._resolve_reference(table, ref))
                names_a_column = True
            elif op:
                operators.append(op)

        if not operands:
            raise ValueError(f"Invalid formula: {formula}")
        if not names_a_column:
            raise ValueError(f"A formula names at least one column: {formula!r}")

        # Evaluate left-to-right (no precedence for simplicity)
        # For proper math precedence, would need a full expression parser
        result = operands[0]

        for i, op in enumerate(operators):
            right = operands[i + 1]

            if op == "+":
                result = pc.add(result, right)
            elif op == "-":
                result = pc.subtract(result, right)
            elif op == "*":
                result = pc.multiply(result, right)
            elif op == "/":
                result = pc.divide(result, right)
            else:
                raise ValueError(f"Unknown operator: {op}")

        # The formula names a column (checked above), and arithmetic with an
        # array operand gives an array.
        assert isinstance(result, pa.Array)
        return result

    def _resolve_reference(self, table: pa.Table, ref: str) -> pa.Array:
        """Resolve a column or struct field reference.

        Args:
            table: The PyArrow table.
            ref: Reference string like "column" or "column.field.subfield".

        Returns:
            The resolved PyArrow array.
        """
        parts = ref.split(".")

        if parts[0] not in table.column_names:
            raise KeyError(
                f"Column '{parts[0]}' not found in table. "
                f"Available columns: {table.column_names}"
            )

        array = table.column(parts[0]).combine_chunks()

        # Auto-detect JSON string columns and parse them (same as Field.resolve)
        if len(parts) > 1 and (
            pa.types.is_string(array.type) or pa.types.is_large_string(array.type)
        ):
            # Column is a string but we need struct access - assume it's JSON
            array = parse_json_to_struct(array.to_pylist())

        # Navigate struct fields
        for field_name in parts[1:]:
            if not pa.types.is_struct(array.type):
                raise ValueError(
                    f"Cannot access field '{field_name}' on non-struct type {array.type}. "
                    f"Column '{parts[0]}' may not contain valid JSON."
                )
            array = pc.struct_field(array, field_name)

        return array

    def __repr__(self) -> str:
        if self.formula:
            return f"ComputedField({self.name!r}, formula={self.formula!r})"
        else:
            return f"ComputedField({self.name!r}, expr=...)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ComputedField):
            return NotImplemented
        return self.name == other.name and self.formula == other.formula

    def __hash__(self) -> int:
        return hash((self.name, self.formula))


# endregion


def parse_json_to_struct(
    json_strings: Any,
    struct_schema: dict[str, type] | None = None,
) -> pa.StructArray:
    """Parse JSON strings into a PyArrow StructArray (VECTORIZED).

    Uses vectorized JSON parsing for efficiency. No row iteration.

    Args:
        json_strings: Array-like of JSON strings.
        struct_schema: Optional dict of {field: type} for explicit schema.
            If None, schema is inferred from first valid JSON object.

    Returns:
        PyArrow StructArray with parsed JSON data.

    Raises:
        ValueError: If JSON parsing fails or schema cannot be inferred.
    """
    import json

    import numpy as np

    # Convert to numpy array if needed
    if hasattr(json_strings, "to_numpy"):
        json_arr = json_strings.to_numpy()
    else:
        json_arr = np.asarray(json_strings)

    # n = len(json_arr)

    # Parse all JSON strings (vectorized via list comprehension)
    parsed = []
    for s in json_arr:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            parsed.append(None)
        elif isinstance(s, str):
            try:
                parsed.append(json.loads(s))
            except json.JSONDecodeError:
                parsed.append(None)
        elif isinstance(s, dict):
            # Already parsed
            parsed.append(s)
        else:
            parsed.append(None)

    # Infer schema from first valid object if not provided
    if struct_schema is None:
        for obj in parsed:
            if obj is not None:
                # Infer types from first object
                struct_schema = {}
                for k, v in obj.items():
                    if isinstance(v, bool):
                        struct_schema[k] = bool
                    elif isinstance(v, int):
                        struct_schema[k] = int
                    elif isinstance(v, float):
                        struct_schema[k] = float
                    else:
                        struct_schema[k] = str
                break

    if struct_schema is None:
        raise ValueError("Cannot infer struct schema: no valid JSON objects found")

    # Build field arrays (vectorized extraction)
    field_arrays = {}
    null_mask = np.array([obj is None for obj in parsed], dtype=bool)

    for field_name, field_type in struct_schema.items():
        # Extract field values (vectorized list comprehension)
        values = [obj.get(field_name) if obj is not None else None for obj in parsed]

        # Convert to appropriate type
        if field_type in (int, "int", "int64"):
            # Handle potential None values
            arr = np.array([v if v is not None else 0 for v in values], dtype=np.int64)
            field_null = np.array([v is None for v in values], dtype=bool)
            field_arrays[field_name] = pa.array(arr, mask=field_null, type=pa.int64())
        elif field_type in (float, "float", "float64"):
            arr = np.array(
                [v if v is not None else np.nan for v in values], dtype=np.float64
            )
            field_null = np.array([v is None for v in values], dtype=bool)
            field_arrays[field_name] = pa.array(arr, mask=field_null, type=pa.float64())
        elif field_type in (bool, "bool", "boolean"):
            arr = np.array([v if v is not None else False for v in values], dtype=bool)
            field_null = np.array([v is None for v in values], dtype=bool)
            field_arrays[field_name] = pa.array(arr, mask=field_null, type=pa.bool_())
        else:
            # String type
            arr = [str(v) if v is not None else None for v in values]
            field_arrays[field_name] = pa.array(arr, type=pa.string())

    # Build struct type and array
    struct_fields = []
    arrays = []
    for field_name, field_type in struct_schema.items():
        if field_type in _TYPE_MAP:
            pa_type = _TYPE_MAP[field_type]
        elif isinstance(field_type, pa.DataType):
            pa_type = field_type
        else:
            pa_type = pa.string()
        struct_fields.append(pa.field(field_name, pa_type, nullable=True))
        arrays.append(field_arrays[field_name])

    return pa.StructArray.from_arrays(
        arrays,
        fields=struct_fields,
        mask=pa.array(null_mask),
    )


# endregion
