"""EventData: PyArrow-based bulk event storage for TimeToAlign!

This module provides the EventData class which stores events in a PyArrow table
with efficient columnar operations. Events are NOT wrapped in Python objects -
they are rows in the table.

NOTE: This class was renamed from EventStore to EventData in the 2026-01 API
refactoring. EventStore now refers to the container class (formerly EventBundle)
that holds one or more EventData tables.

Design principles:
- Bulk operations are the primary API (from_dicts, from_arrays, from_dataframe)
- Schema is fixed per class, with extension points for subclasses
- Coordinates stored with both original precision and float representation
- Unit metadata at the field level (all events share same unit)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from typing_extensions import Self

from timetoalign.core import (
    Coordinate,
    CoordinateSpec,
    IntervalPolicy,
    NumberType,
    TimeUnit,
    resolve_coordinate_spec,
)
from timetoalign.core.fields import field_metadata
from timetoalign.core.time import (
    build_number_struct_array,
    coordinate_to_struct,
    number_from_json,
    struct_to_coordinate,
    struct_to_rational,
)

from .mixins import SemanticFieldAccessMixin
from .schema import (
    extend_schema,
    get_base_field_names,
    is_coordinate_type,
    make_base_schema,
    make_table_metadata,
    parse_table_metadata,
)

if TYPE_CHECKING:
    from timetoalign.timelines.base import Timeline

module_logger = logging.getLogger(__name__)

_timeline_factory: (
    Callable[["EventData", str | None, dict[str, Any] | None], "Timeline"] | None
) = None


def _register_timeline_factory(
    factory: Callable[["EventData", str | None, dict[str, Any] | None], "Timeline"],
) -> None:
    """Register the timeline-layer constructor used by ``create_timeline``."""
    global _timeline_factory
    _timeline_factory = factory


def _has_value_subfield(pa_type: pa.DataType) -> bool:
    """Return True iff *pa_type* is a struct with a ``value`` sub-field.

    Matches the canonical ``{value, numerator, denominator}`` coordinate
    struct as well as reduced shapes that carry only ``value`` (e.g. a
    column built without the fraction components), so
    :meth:`EventData.column_values` decodes any rational-shaped column,
    not just the full three-field one.
    """
    if not pa.types.is_struct(pa_type):
        return False
    return any(pa_type.field(i).name == "value" for i in range(pa_type.num_fields))


def _first_non_null(rows: list[dict[str, Any]], name: str) -> Any:
    """Return the first non-null value carried under *name*, or ``None``."""
    for row in rows:
        val = row.get(name)
        if val is not None:
            return val
    return None


def _struct_fields_match_unordered(
    actual: pa.StructType, expected: pa.StructType
) -> bool:
    """Return True iff *actual* and *expected* have the same ``{name: type}`` set.

    Order-independent counterpart to ``core.fields._struct_types_match``:
    ``pa.array`` infers struct sub-fields in alphabetical order, which may
    differ from a paired class's canonical ``pa_schema`` order, so a carried
    struct dict must be matched by name-set, then reordered to canonical.
    """
    if not pa.types.is_struct(actual):
        return False
    if actual.num_fields != expected.num_fields:
        return False
    actual_map = {
        actual.field(i).name: actual.field(i).type for i in range(actual.num_fields)
    }
    for i in range(expected.num_fields):
        e = expected.field(i)
        a_type = actual_map.get(e.name)
        if a_type is None or not a_type.equals(e.type):
            return False
    return True


def _semantic_field_type_for_struct(
    struct_type: pa.StructType,
) -> tuple[str, pa.StructType] | None:
    """Return the paired ``SemanticField`` matching *struct_type* by name-set.

    Walks the scalar→field registry and returns the ``(field_type_str,
    canonical_pa_schema)`` of the first paired ``SemanticField`` whose
    ``pa_schema`` has the same sub-field ``{name: type}`` set as
    *struct_type* (order-independent).  Returns ``None`` when no paired
    class claims the shape — the column then stays a plain struct with no
    semantic identity.

    The generic coordinate struct ``{value, numerator, denominator}`` is
    deliberately excluded: ``start`` / ``end`` / ``duration`` already live
    in the base schema, so a stray coordinate-shaped extra column should
    stay a plain struct rather than masquerade as a CoordinateField.
    """
    from timetoalign.storage.mixins import _get_field_type_map

    for field_type_str, field_cls in _get_field_type_map().items():
        schema = field_cls.pa_schema
        if schema is None:
            continue
        names = [schema.field(i).name for i in range(schema.num_fields)]
        if names == ["value", "numerator", "denominator"]:
            continue
        if _struct_fields_match_unordered(struct_type, schema):
            return field_type_str, schema
    return None


def _build_extra_field_schema(
    schema: pa.Schema,
    extra_field_names: list[str],
    processed_rows: list[dict[str, Any]],
) -> pa.Schema:
    """Append extra (non-base-schema) fields to *schema*, preserving shape.

    Each extra column is materialised under the most faithful PyArrow type
    its values support, so that the semantic-field affordance of a carried
    column survives the round-trip:

    * **struct-shaped** values (a ``dict`` sample) become a real
      ``pa.struct`` column whose type is inferred from the dicts.  When the
      inferred struct matches a paired ``SemanticField``'s ``pa_schema``,
      the field is stamped with the ``TIMETOALIGN_METADATA_KEY`` ``field_type``
      metadata blob so ``get_field()`` round-trips the semantic type
      directly (no reliance on shape discovery).
    * **list-shaped** values become a ``pa.list_`` column.
    * **scalar** values keep their native inferred type (int / float /
      bool / string).  Genuinely unrepresentable objects fall back to
      ``str`` — but a struct ``dict`` is NEVER serialised to a JSON string,
      because that would silently destroy the field's struct affordance.

    Mutates *processed_rows* in place to null-fill absent columns and to
    coerce non-native scalars to ``str``.

    Args:
        schema: The base + declared-extra schema to extend.
        extra_field_names: Sorted names of the carried columns not already
            in *schema*.
        processed_rows: The row dicts about to be turned into a table.

    Returns:
        A new ``pa.Schema`` with the extra fields appended.
    """
    from timetoalign.core.fields import (
        TIMETOALIGN_METADATA_KEY,
        metadata_blob_from_dict,
    )

    new_fields = list(schema)
    for name in extra_field_names:
        if name == "volta":
            for row in processed_rows:
                value = row.get(name)
                row[name] = int(value) if value is not None else None
            new_fields.append(pa.field(name, pa.int64(), nullable=True))
            continue

        sample = _first_non_null(processed_rows, name)

        if isinstance(sample, dict):
            # Infer the struct type from the carried dicts and keep it as a
            # real struct column.  Stamp paired-class metadata when the
            # shape is a known SemanticField so the affordance round-trips.
            values = [row.get(name) for row in processed_rows]
            try:
                arr = pa.array(values)
            except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
                # Heterogeneous dict shapes that PyArrow cannot unify: keep
                # the column out of the typed path rather than crash.  Fall
                # back to a string rendering (last resort, logged).
                module_logger.warning(
                    "Carried struct column %r has heterogeneous shapes; "
                    "storing as string.",
                    name,
                )
                for row in processed_rows:
                    val = row.get(name)
                    row[name] = None if val is None else str(val)
                new_fields.append(pa.field(name, pa.string(), nullable=True))
                continue
            metadata = None
            struct_type = arr.type
            if pa.types.is_struct(struct_type):
                match = _semantic_field_type_for_struct(struct_type)
                if match is not None:
                    field_type_str, canonical_schema = match
                    # Use the paired class's canonical sub-field order so
                    # the stored struct round-trips through from_row and is
                    # claimed by shape discovery; from_pylist coerces the
                    # carried dicts to this order.
                    struct_type = canonical_schema
                    metadata = {
                        TIMETOALIGN_METADATA_KEY: metadata_blob_from_dict(
                            {"field_type": field_type_str}
                        )
                    }
            new_fields.append(
                pa.field(name, struct_type, nullable=True, metadata=metadata)
            )
            continue

        if isinstance(sample, list):
            values = [row.get(name) for row in processed_rows]
            try:
                arr = pa.array(values)
                new_fields.append(pa.field(name, arr.type, nullable=True))
                continue
            except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
                pass  # fall through to string rendering below

        if sample is not None and not isinstance(sample, (str, bytes)):
            # Scalar of a native arrow type (int / float / bool): infer and
            # keep the native type so numeric carried columns stay numeric.
            values = [row.get(name) for row in processed_rows]
            try:
                arr = pa.array(values)
                if not pa.types.is_null(arr.type):
                    new_fields.append(pa.field(name, arr.type, nullable=True))
                    continue
            except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
                pass  # fall through to string rendering below
            # Unrepresentable scalar object: render to string.
            for row in processed_rows:
                val = row.get(name)
                if val is not None and not isinstance(val, str):
                    row[name] = str(val)
                elif name not in row:
                    row[name] = None
            new_fields.append(pa.field(name, pa.string(), nullable=True))
            continue

        # String (or all-null) column.
        for row in processed_rows:
            if name not in row:
                row[name] = None
        new_fields.append(pa.field(name, pa.string(), nullable=True))

    return pa.schema(new_fields, metadata=schema.metadata)


class EventData(SemanticFieldAccessMixin):
    """PyArrow-based storage for timeline events.

    EventData wraps a PyArrow table containing events. Events are rows in the
    table, not Python wrapper objects. The primary API is bulk operations:

    - from_dicts(): Create from list of row dictionaries
    - from_arrays(): Create from field-oriented arrays
    - from_dataframe(): Create from pandas DataFrame

    The schema is fixed at class definition time but can be extended by
    subclasses to add domain-specific fields (e.g., pitch, velocity for notes).

    NOTE: This class was renamed from EventStore to EventData in the 2026-01 API
    refactoring. EventStore now refers to the container class (formerly EventBundle)
    that holds one or more EventData tables.

    Attributes:
        table: The underlying PyArrow table.
        unit: The time unit for all coordinates.
        number_type: The number type used for coordinates.

    Examples:
        >>> data = EventData.from_dicts([
        ...     {"id": "e1", "temporal_type": "instant", "event_type": "Beat",
        ...      "instant": 0.0},
        ...     {"id": "e2", "temporal_type": "interval", "event_type": "Note",
        ...      "start": 0.0, "end": 1.0},
        ... ], unit=TimeUnit.seconds)
        >>> len(data)
        2
    """

    # Class-level schema configuration (subclasses can extend)
    _extra_fields: ClassVar[list[pa.Field]] = []

    def __init__(
        self,
        table: pa.Table,
        unit: TimeUnit,
        number_type: NumberType | None = None,
    ) -> None:
        """Initialize EventData with an existing table.

        Use class methods (from_dicts, from_arrays, etc.) to create instances.

        Args:
            table: The PyArrow table containing events.
            unit: The time unit for coordinates.
            number_type: The number type for coordinate interpretation.
                Defaults to the one the unit itself uses.
        """
        self._table = table
        self._unit = unit
        self._number_type = unit.resolve_number_type(number_type)

    # region Schema

    @property
    def schema(self) -> pa.Schema:
        """The PyArrow schema of the underlying table.

        Returns:
            The schema of the stored PyArrow table, including all base and
            extra fields with their metadata.
        """
        return self._table.schema

    @classmethod
    def get_schema(
        cls, unit: TimeUnit, number_type: NumberType | None = None
    ) -> pa.Schema:
        """Get the canonical PyArrow schema for this EventData class.

        This is a class-level method that returns the schema for a given unit,
        independent of any specific instance. Useful for constructing empty
        tables or validating incoming data.

        Args:
            unit: The time unit for coordinate fields.
            number_type: The number type for coordinate fields.

        Returns:
            The complete schema including base and extra fields.
        """
        base = make_base_schema(unit, number_type=number_type)
        if cls._extra_fields:
            return extend_schema(base, cls._extra_fields)
        return base

    @classmethod
    def field_names(cls) -> list[str]:
        """Get the list of field names for this EventData class.

        Returns:
            List of all field names (base + extra).
        """
        base_names = get_base_field_names()
        extra_names = [f.name for f in cls._extra_fields]
        return base_names + extra_names

    # endregion

    # region Interval Normalisation

    @staticmethod
    def _normalize_intervals_vectorized(
        processed: dict[str, Any],
        policy: IntervalPolicy = IntervalPolicy.warn,
    ) -> dict[str, Any]:
        """Normalise interval events in a field dict (FULLY VECTORIZED).

        Ensures that every interval event has both ``end`` and ``duration``
        populated, every instant event has both null, and that the
        ``temporal_type`` field is consistent.  Also handles the case
        where start/end/duration arrays are raw numeric (not yet parsed)
        by operating purely on the float ``value`` field of coordinate
        struct arrays.

        The *policy* parameter controls behaviour when both ``end`` and
        ``duration`` are present but inconsistent
        (``end - start != duration``).

        Args:
            processed: Mutable field dict (modified **in-place**).
                Must already contain ``"start"`` as a ``pa.StructArray``.
                ``"end"`` and ``"duration"`` may be ``pa.StructArray``,
                ``pa.NullArray``, or absent.
            policy: How to resolve inconsistencies between ``end`` and
                ``duration``.

        Returns:
            The same *processed* dict (for chaining convenience).

        Raises:
            ValueError: If *policy* is ``strict`` and an inconsistency is
                detected.
        """

        start_arr = processed.get("start")
        if start_arr is None or not isinstance(start_arr, pa.StructArray):
            return processed

        n = len(start_arr)
        start_is_null = start_arr.is_null().to_numpy(zero_copy_only=False)
        start_val = start_arr.field("value").to_numpy(zero_copy_only=False)
        start_num = start_arr.field("numerator").to_numpy(zero_copy_only=False)
        start_den = start_arr.field("denominator").to_numpy(zero_copy_only=False)

        # Determine which fields are present and non-null
        end_arr = processed.get("end")
        dur_arr = processed.get("duration")

        def _is_real(arr: Any) -> bool:
            """True when the array is a StructArray with at least one non-null."""
            if arr is None:
                return False
            if isinstance(arr, pa.StructArray):
                return arr.null_count < len(arr)
            if isinstance(arr, pa.ChunkedArray):
                return arr.null_count < len(arr)
            return False

        has_end_data = _is_real(end_arr)
        has_dur_data = _is_real(dur_arr)

        # Extract end arrays (if present)
        if has_end_data:
            if isinstance(end_arr, pa.ChunkedArray):
                end_arr = end_arr.combine_chunks()
            end_is_null = end_arr.is_null().to_numpy(zero_copy_only=False)
            end_val = end_arr.field("value").to_numpy(zero_copy_only=False)
            end_num = end_arr.field("numerator").to_numpy(zero_copy_only=False)
            end_den = end_arr.field("denominator").to_numpy(zero_copy_only=False)
        else:
            end_is_null = np.ones(n, dtype=bool)
            end_val = np.full(n, np.nan, dtype=np.float64)
            end_num = np.full(n, np.nan, dtype=np.float64)
            end_den = np.full(n, np.nan, dtype=np.float64)

        # Extract duration arrays (if present)
        if has_dur_data:
            if isinstance(dur_arr, pa.ChunkedArray):
                dur_arr = dur_arr.combine_chunks()
            dur_is_null = dur_arr.is_null().to_numpy(zero_copy_only=False)
            dur_val = dur_arr.field("value").to_numpy(zero_copy_only=False)
            dur_num = dur_arr.field("numerator").to_numpy(zero_copy_only=False)
            dur_den = dur_arr.field("denominator").to_numpy(zero_copy_only=False)
        else:
            dur_is_null = np.ones(n, dtype=bool)
            dur_val = np.full(n, np.nan, dtype=np.float64)
            dur_num = np.full(n, np.nan, dtype=np.float64)
            dur_den = np.full(n, np.nan, dtype=np.float64)

        # Boolean masks for each situation (vectorized)
        has_start = ~start_is_null
        temporal_type = processed.get("temporal_type")
        if temporal_type is None:
            instant_rows = np.zeros(n, dtype=bool)
        else:
            if isinstance(temporal_type, (pa.Array, pa.ChunkedArray)):
                temporal_values = temporal_type.to_pylist()
            else:
                temporal_values = list(temporal_type)
            instant_rows = np.array(
                [value == "instant" for value in temporal_values], dtype=bool
            )

        # Explicit instant events have no interval endpoints.  In particular,
        # do not let a populated start coordinate manufacture a zero-length
        # value-only interval during merged-table assembly.
        has_end = ~end_is_null & ~instant_rows
        has_dur = ~dur_is_null & ~instant_rows

        # Identify rows that are interval events (have start and at least end or dur)
        # is_interval = has_start & (has_end | has_dur)

        # ---- Policy-driven resolution ----

        # Rows where both end AND duration are present (potential conflict)
        both_present = has_start & has_end & has_dur
        inconsistent = np.zeros(n, dtype=bool)  # default: no inconsistencies
        if both_present.any():
            # Compute expected duration from end - start (float)
            expected_dur = end_val - start_val
            # Compare against supplied duration
            actual_dur = dur_val
            discrepancy = np.abs(expected_dur - actual_dur)
            # Use a tight tolerance for float comparison
            tol = 1e-9
            inconsistent = both_present & (discrepancy > tol)

            if inconsistent.any():
                n_bad = int(inconsistent.sum())
                # Find first inconsistent row for the error message
                first_idx = int(np.argmax(inconsistent))

                if policy == IntervalPolicy.strict:
                    raise ValueError(
                        f"Interval inconsistency in {n_bad} event(s): "
                        f"end - start != duration. First at index {first_idx}: "
                        f"start={start_val[first_idx]}, end={end_val[first_idx]}, "
                        f"duration={dur_val[first_idx]} "
                        f"(expected duration={expected_dur[first_idx]})."
                    )
                elif policy == IntervalPolicy.warn:
                    module_logger.warning(
                        "Interval inconsistency in %d event(s): end - start != "
                        "duration. Recomputing duration from end. First at index "
                        "%d: start=%s, end=%s, duration=%s (expected %s).",
                        n_bad,
                        first_idx,
                        start_val[first_idx],
                        end_val[first_idx],
                        dur_val[first_idx],
                        expected_dur[first_idx],
                    )
                # For warn, prefer_end, and strict-without-error: fall through
                # to the fill logic below (which uses the policy).

        # ---- Fill missing values ----
        #
        # After this block every interval row will have both end and
        # duration.  The policy controls which value is authoritative
        # when both are already present.

        # Allocate output arrays (copy so we don't mutate the originals)
        out_end_val = end_val.copy()
        out_end_num = end_num.copy()
        out_end_den = end_den.copy()
        out_end_null = end_is_null.copy() | instant_rows

        out_dur_val = dur_val.copy()
        out_dur_num = dur_num.copy()
        out_dur_den = dur_den.copy()
        out_dur_null = dur_is_null.copy() | instant_rows

        # Helper: vectorized fraction subtraction a/b - c/d
        def _frac_sub(a_num, a_den, b_num, b_den):
            r_num = a_num * b_den - b_num * a_den
            r_den = a_den * b_den
            g = np.gcd(np.abs(r_num).astype(np.int64), np.abs(r_den).astype(np.int64))
            g = np.where(g == 0, 1, g)
            return (r_num // g), (r_den // g)

        # Helper: vectorized fraction addition a/b + c/d
        def _frac_add(a_num, a_den, b_num, b_den):
            r_num = a_num * b_den + b_num * a_den
            r_den = a_den * b_den
            g = np.gcd(np.abs(r_num).astype(np.int64), np.abs(r_den).astype(np.int64))
            g = np.where(g == 0, 1, g)
            return (r_num // g), (r_den // g)

        def _has_frac(num_arr, den_arr):
            """Return the rows whose rational pair is complete."""
            return (~pd.isna(num_arr)) & (~pd.isna(den_arr))

        # 1) Rows with end but no duration -> compute duration = end - start
        need_dur = has_start & has_end & ~has_dur
        if policy == IntervalPolicy.prefer_end:
            # Also recompute duration for rows that already have both
            need_dur = need_dur | both_present
        elif policy in (IntervalPolicy.warn, IntervalPolicy.strict):
            # Recompute duration for inconsistent rows (prefer end)
            if inconsistent.any():
                need_dur = need_dur | (both_present & inconsistent)

        if need_dur.any():
            out_dur_val[need_dur] = out_end_val[need_dur] - start_val[need_dur]
            out_dur_null[need_dur] = False

            # Fraction arithmetic where both have fractions
            s_has = _has_frac(start_num, start_den)
            e_has = _has_frac(out_end_num, out_end_den)
            frac_mask = need_dur & s_has & e_has
            if frac_mask.any():
                s_n = np.where(pd.isna(start_num), 0, start_num).astype(np.int64)
                s_d = np.where(pd.isna(start_den), 1, start_den).astype(np.int64)
                e_n = np.where(pd.isna(out_end_num), 0, out_end_num).astype(np.int64)
                e_d = np.where(pd.isna(out_end_den), 1, out_end_den).astype(np.int64)
                r_n, r_d = _frac_sub(e_n, e_d, s_n, s_d)
                out_dur_num[frac_mask] = r_n[frac_mask]
                out_dur_den[frac_mask] = r_d[frac_mask]

        # 2) Rows with duration but no end -> compute end = start + duration
        need_end = has_start & has_dur & ~has_end
        if policy == IntervalPolicy.prefer_duration:
            # Also recompute end for rows that already have both
            need_end = need_end | both_present

        if need_end.any():
            out_end_val[need_end] = start_val[need_end] + out_dur_val[need_end]
            out_end_null[need_end] = False

            s_has = _has_frac(start_num, start_den)
            d_has = _has_frac(out_dur_num, out_dur_den)
            frac_mask = need_end & s_has & d_has
            if frac_mask.any():
                s_n = np.where(pd.isna(start_num), 0, start_num).astype(np.int64)
                s_d = np.where(pd.isna(start_den), 1, start_den).astype(np.int64)
                d_n = np.where(pd.isna(out_dur_num), 0, out_dur_num).astype(np.int64)
                d_d = np.where(pd.isna(out_dur_den), 1, out_dur_den).astype(np.int64)
                r_n, r_d = _frac_add(s_n, s_d, d_n, d_d)
                out_end_num[frac_mask] = r_n[frac_mask]
                out_end_den[frac_mask] = r_d[frac_mask]

        # ---- Build output struct arrays ----

        coord_type = pa.struct(
            [
                pa.field("value", pa.float64(), nullable=True),
                pa.field("numerator", pa.int64(), nullable=True),
                pa.field("denominator", pa.int64(), nullable=True),
            ]
        )

        # Determine fraction null mask for output arrays
        def _build_struct_array(
            val, num, den, null_mask, frac_source_num, frac_source_den
        ):
            """Build a coordinate StructArray from numpy arrays."""
            # Fraction fields are null wherever the struct is null or source had no frac
            frac_present = (~pd.isna(frac_source_num)) & (~pd.isna(frac_source_den))
            frac_null = null_mask | ~frac_present
            # Convert to safe int arrays
            safe_num = np.where(pd.isna(num), 0, num).astype(np.int64)
            safe_den = np.where(pd.isna(den), 1, den).astype(np.int64)
            return pa.StructArray.from_arrays(
                [
                    pa.array(val, mask=null_mask, type=pa.float64()),
                    pa.array(safe_num, mask=frac_null, type=pa.int64()),
                    pa.array(safe_den, mask=frac_null, type=pa.int64()),
                ],
                fields=list(coord_type),
                mask=pa.array(null_mask),
            )

        processed["end"] = _build_struct_array(
            out_end_val,
            out_end_num,
            out_end_den,
            out_end_null,
            out_end_num,
            out_end_den,
        )
        processed["duration"] = _build_struct_array(
            out_dur_val,
            out_dur_num,
            out_dur_den,
            out_dur_null,
            out_dur_num,
            out_dur_den,
        )

        # ---- Ensure temporal_type is consistent ----
        # Recompute: interval if end is now non-null, instant otherwise
        new_has_end = ~out_end_null
        inferred_tt = np.where(new_has_end, "interval", "instant")

        # Only overwrite temporal_type if it was all-null or not provided;
        # otherwise just fix the rows that changed.
        tt = processed.get("temporal_type")
        if tt is None:
            processed["temporal_type"] = inferred_tt
        elif isinstance(tt, pa.Array):
            if tt.null_count == len(tt):
                processed["temporal_type"] = inferred_tt
            else:
                # Overwrite individual rows where our inference differs
                existing = tt.to_pylist()
                for i in range(n):
                    if new_has_end[i] and existing[i] != "interval":
                        existing[i] = "interval"
                    elif not new_has_end[i] and existing[i] != "instant":
                        existing[i] = "instant"
                processed["temporal_type"] = pa.array(existing, type=pa.string())
        elif isinstance(tt, np.ndarray):
            # Update mismatches
            tt[new_has_end & (tt != "interval")] = "interval"
            tt[~new_has_end & (tt != "instant")] = "instant"

        return processed

    @staticmethod
    def _normalize_intervals_row(
        processed: dict[str, Any],
        policy: IntervalPolicy = IntervalPolicy.warn,
    ) -> dict[str, Any]:
        """Normalise interval fields in a single event row dict.

        Ensures that ``end`` and ``duration`` are both present (or both
        absent) and consistent with ``start``.  Also ensures that the
        ``temporal_type`` field is set correctly.  Coordinate fields are
        converted to struct-dict format via ``coordinate_to_struct``.

        This is the **row-based** counterpart to
        ``_normalize_intervals_vectorized`` and is called from
        ``from_dicts``.

        Args:
            processed: Mutable row dict (modified **in-place**).
            policy: How to resolve inconsistencies.

        Returns:
            The same *processed* dict (for chaining convenience).

        Raises:
            ValueError: If *policy* is ``strict`` and ``end - start !=
                duration``.
        """
        # ---- Ensure coordinate struct format ----
        for coord_col in ("start", "end", "duration"):
            val = processed.get(coord_col)
            if val is not None:
                if isinstance(val, dict):
                    if "value" not in val:
                        processed[coord_col] = coordinate_to_struct(val)
                elif isinstance(val, list):
                    processed[coord_col] = coordinate_to_struct(number_from_json(val))
                else:
                    processed[coord_col] = coordinate_to_struct(val)
            elif coord_col not in processed:
                processed[coord_col] = None

        if processed.get("temporal_type") == "instant":
            processed["end"] = None
            processed["duration"] = None
            return processed

        # ---- Extract float values ----
        def _float_of(v: Any) -> float | None:
            if v is None:
                return None
            if isinstance(v, dict):
                return v.get("value")
            return float(v)

        # ---- Extract exact Fraction, when the struct carries one ----
        def _fraction_of(v: Any) -> Fraction | None:
            if not isinstance(v, dict):
                return None
            try:
                return struct_to_rational(v)
            except (TypeError, ValueError, ZeroDivisionError):
                return None

        def _add_or_float(
            left: Any, right: Any, left_val: float, right_val: float
        ) -> Any:
            """Add exact operands as fractions and otherwise use their values."""
            left_frac = _fraction_of(left)
            right_frac = _fraction_of(right)
            if left_frac is not None and right_frac is not None:
                return left_frac + right_frac
            return left_val + right_val

        def _sub_or_float(
            left: Any, right: Any, left_val: float, right_val: float
        ) -> Any:
            """Subtract exact operands as fractions and otherwise use their values."""
            left_frac = _fraction_of(left)
            right_frac = _fraction_of(right)
            if left_frac is not None and right_frac is not None:
                return left_frac - right_frac
            return left_val - right_val

        start_val = _float_of(processed.get("start"))
        end_val = _float_of(processed.get("end"))
        dur_val = _float_of(processed.get("duration"))

        has_start = start_val is not None
        has_end = end_val is not None
        has_dur = dur_val is not None

        # ---- Consistency check when both present ----
        if has_start and has_end and has_dur:
            expected = end_val - start_val
            if abs(expected - dur_val) > 1e-9:
                if policy == IntervalPolicy.strict:
                    raise ValueError(
                        f"Interval inconsistency: start={start_val}, "
                        f"end={end_val}, duration={dur_val} "
                        f"(expected duration={expected})."
                    )
                elif policy == IntervalPolicy.warn:
                    module_logger.warning(
                        "Interval inconsistency: start=%s, end=%s, "
                        "duration=%s (expected %s). Recomputing duration "
                        "from end.",
                        start_val,
                        end_val,
                        dur_val,
                        expected,
                    )
                # After warning: fall through to fill logic

        # ---- Fill missing values ----
        if has_start:
            if policy == IntervalPolicy.prefer_end:
                if has_end:
                    processed["duration"] = coordinate_to_struct(
                        _sub_or_float(
                            processed.get("end"),
                            processed.get("start"),
                            end_val,
                            start_val,
                        )
                    )
                elif has_dur:
                    processed["end"] = coordinate_to_struct(
                        _add_or_float(
                            processed.get("start"),
                            processed.get("duration"),
                            start_val,
                            dur_val,
                        )
                    )
            elif policy == IntervalPolicy.prefer_duration:
                if has_dur:
                    processed["end"] = coordinate_to_struct(
                        _add_or_float(
                            processed.get("start"),
                            processed.get("duration"),
                            start_val,
                            dur_val,
                        )
                    )
                elif has_end:
                    processed["duration"] = coordinate_to_struct(
                        _sub_or_float(
                            processed.get("end"),
                            processed.get("start"),
                            end_val,
                            start_val,
                        )
                    )
            else:
                # warn / strict: prefer end when both present, otherwise fill
                if has_end and not has_dur:
                    processed["duration"] = coordinate_to_struct(
                        _sub_or_float(
                            processed.get("end"),
                            processed.get("start"),
                            end_val,
                            start_val,
                        )
                    )
                elif has_dur and not has_end:
                    processed["end"] = coordinate_to_struct(
                        _add_or_float(
                            processed.get("start"),
                            processed.get("duration"),
                            start_val,
                            dur_val,
                        )
                    )
                elif has_end and has_dur:
                    # Recompute duration from end (prefer end); stay exact
                    # when both endpoints carry exact ratios.
                    processed["duration"] = coordinate_to_struct(
                        _sub_or_float(
                            processed.get("end"),
                            processed.get("start"),
                            end_val,
                            start_val,
                        )
                    )

        # ---- Infer temporal_type ----
        if "temporal_type" not in processed or processed["temporal_type"] is None:
            now_has_end = processed.get("end") is not None
            now_has_dur = processed.get("duration") is not None
            if has_start and (now_has_end or now_has_dur):
                processed["temporal_type"] = "interval"
            else:
                processed["temporal_type"] = "instant"

        return processed

    # endregion

    # region Class Methods - Creation

    @classmethod
    def empty(cls, unit: TimeUnit, number_type: NumberType | None = None) -> Self:
        """Create an empty EventData.

        Args:
            unit: The time unit for coordinates.
            number_type: The number type for coordinates. Defaults to the
                one the unit itself uses.

        Returns:
            An empty EventData with the appropriate schema.
        """
        number_type = unit.resolve_number_type(number_type)
        schema = cls.get_schema(unit, number_type=number_type)
        metadata = make_table_metadata(unit, number_type, loader_class=cls.__name__)
        schema = schema.with_metadata(metadata)
        table = pa.table({name: [] for name in cls.field_names()}, schema=schema)
        return cls(table, unit, number_type)

    @classmethod
    def from_dicts(
        cls,
        rows: list[dict[str, Any]],
        unit: TimeUnit,
        number_type: NumberType | None = None,
        *,
        interval_policy: IntervalPolicy = IntervalPolicy.warn,
    ) -> Self:
        """Create EventData from a list of row dictionaries.

        Coordinate values (instant, start, end, duration) are automatically
        converted to the internal struct format. Convenience defaults are
        applied so that callers can omit boilerplate fields:

        - **id**: Auto-generated as ``{event_type}:{counter}`` if missing,
          e.g. ``note:000001``, ``rest:000001``, ``beat:000001``.
          When events are placed on a timeline, the timeline's ID is
          prepended, yielding e.g. ``clt:1:note:000001``.
        - **temporal_type**: Inferred from the keys present in the dict --
          ``"interval"`` when *both* ``start`` and ``end`` (or ``duration``)
          are given, ``"instant"`` otherwise.

        Missing ``end`` or ``duration`` values are computed automatically
        from the other (``end = start + duration`` or
        ``duration = end - start``).  Behaviour when both are present but
        inconsistent is controlled by *interval_policy*.

        Args:
            rows: List of event dictionaries. At minimum each dict needs a
                coordinate (``instant`` *or* ``start``/``end``) and an
                ``event_type``. All other fields have sensible defaults.
            unit: The time unit for coordinates.
            number_type: The number type for coordinates.
            interval_policy: How to handle end/duration inconsistencies.
                See `IntervalPolicy` for options.

        Returns:
            A new EventData containing the events.

        Examples:
            >>> data = EventData.from_dicts([
            ...     {"event_type": "Beat", "instant": 0},
            ...     {"event_type": "Note", "start": 0, "end": 0.5},
            ... ], unit=TimeUnit.seconds)
        """
        number_type = unit.resolve_number_type(number_type)
        if not rows:
            return cls.empty(unit, number_type)

        # Convert coordinate values to struct format
        processed_rows = []
        # Per-type counters for generating informative IDs
        type_counters: dict[str, int] = {}
        for i, row in enumerate(rows):
            processed = dict(row)

            # Auto-generate id if missing: use event_type prefix for informative IDs
            if "id" not in processed or processed["id"] is None:
                etype = str(processed.get("event_type", "event")).lower()
                type_counters.setdefault(etype, 0)
                type_counters[etype] += 1
                processed["id"] = f"{etype}:{type_counters[etype]:06d}"

            # Map 'instant' to 'start'
            if "instant" in processed and processed.get("start") is None:
                processed["start"] = processed.pop("instant")

            # Remove 'instant' key if it remains
            processed.pop("instant", None)

            # Unified interval normalisation: converts coordinate fields
            # to struct format, fills missing end/duration, infers
            # temporal_type, and checks consistency per policy.
            cls._normalize_intervals_row(processed, policy=interval_policy)

            # Ensure name field exists
            if "name" not in processed:
                processed["name"] = None
            processed_rows.append(processed)

        schema = cls.get_schema(unit, number_type=number_type)
        metadata = make_table_metadata(unit, number_type, loader_class=cls.__name__)
        schema = schema.with_metadata(metadata)

        # Collect extra fields not in the base schema and add them dynamically
        base_field_names = set(schema.names)
        extra_field_names: set[str] = set()
        for row in processed_rows:
            for key in row.keys():
                if key not in base_field_names:
                    extra_field_names.add(key)

        if extra_field_names:
            schema = _build_extra_field_schema(
                schema, sorted(extra_field_names), processed_rows
            )

        table = pa.Table.from_pylist(processed_rows, schema=schema)
        return cls(table, unit, number_type)

    @classmethod
    def from_arrays(
        cls,
        fields: dict[str, np.ndarray | pa.Array | list[Any]],
        unit: TimeUnit,
        number_type: NumberType | None = None,
        *,
        validate: bool = True,
        extra_fields: list[pa.Field] | None = None,
        interval_policy: IntervalPolicy = IntervalPolicy.warn,
    ) -> Self:
        """Create EventData from field-oriented arrays (VECTORIZED).

        This is the PRIMARY construction method for loaders. All operations
        are vectorized - NO row iteration occurs.

        Missing ``end`` or ``duration`` values are computed automatically
        from the other (``end = start + duration`` or
        ``duration = end - start``).  Behaviour when both are present but
        inconsistent is controlled by *interval_policy*.

        Args:
            fields: Dict mapping field names to arrays. Supports:
                - np.ndarray: NumPy arrays
                - pa.Array: PyArrow arrays (including StructArray for coords)
                - list: Python lists (converted to numpy)

                For coordinate fields (start, end, duration):
                - If pa.StructArray: used directly
                - If numeric/string array: parsed via build_number_struct_array

            unit: The time unit for coordinates.
            number_type: The number type for coordinates.
            validate: Whether to validate arrays before table construction.
            extra_fields: Optional list of PyArrow fields for extra data.
                These fields include metadata (e.g., unit for CoordinateFields).
                If not provided, fields are inferred from the data arrays.
            interval_policy: How to handle end/duration inconsistencies.
                See `IntervalPolicy` for options.

        Returns:
            A new EventData containing the events.

        Raises:
            ValueError: If validation fails (missing fields, length mismatch, etc.)

        Examples:
            >>> # Vectorized construction from arrays
            >>> data = EventData.from_arrays({
            ...     "id": np.array(["e1", "e2"]),
            ...     "temporal_type": np.array(["instant", "instant"]),
            ...     "event_type": np.array(["Beat", "Beat"]),
            ...     "start": build_number_struct_array([0, 480], number_type=NumberType.int),
            ... }, unit=TimeUnit.ticks)

            >>> # Direct from loader output (StructArrays already parsed)
            >>> data = EventData.from_arrays(loader_fields, unit=TimeUnit.quarters)
        """
        number_type = unit.resolve_number_type(number_type)
        if not fields:
            return cls.empty(unit, number_type)

        # Check if any field has data
        first_arr = next(iter(fields.values()), None)
        if first_arr is None or len(first_arr) == 0:
            return cls.empty(unit, number_type)

        n_rows = len(first_arr)

        # Helper to get field array (with instant->start mapping)
        def get_arr(name: str) -> Any:
            if name == "start" and "start" not in fields and "instant" in fields:
                return fields["instant"]
            return fields.get(name)

        # Build processed dict for PyArrow table
        processed: dict[str, Any] = {}
        schema = cls.get_schema(unit, number_type=number_type)

        for pa_field in schema:
            field_name = pa_field.name
            arr_data = get_arr(field_name)

            if field_name in ("start", "end", "duration"):
                # Coordinate fields - may be StructArray or need parsing
                if arr_data is None:
                    # Create null struct array (vectorized)
                    processed[field_name] = pa.nulls(n_rows, type=pa_field.type)
                elif isinstance(arr_data, (pa.Array, pa.ChunkedArray)):
                    arr = (
                        arr_data.combine_chunks()
                        if isinstance(arr_data, pa.ChunkedArray)
                        else arr_data
                    )
                    if isinstance(arr, pa.StructArray):
                        processed[field_name] = arr
                    elif pa.types.is_null(arr.type):
                        processed[field_name] = pa.nulls(n_rows, type=pa_field.type)
                    else:
                        null_mask = pc.is_null(arr, nan_is_null=True)
                        fill_value: str | int = (
                            "0"
                            if pa.types.is_string(arr.type)
                            or pa.types.is_large_string(arr.type)
                            else 0
                        )
                        filled = pc.if_else(
                            null_mask, pa.scalar(fill_value, type=arr.type), arr
                        )
                        parsed = build_number_struct_array(
                            filled.to_numpy(zero_copy_only=False),
                            number_type=number_type,
                        )
                        processed[field_name] = pa.StructArray.from_arrays(
                            [parsed.field(i) for i in range(parsed.type.num_fields)],
                            fields=list(parsed.type),
                            mask=null_mask,
                        )
                else:
                    # Need to build the storage cells from a raw column
                    arr = _as_numpy(arr_data)
                    # Handle None/NaN values (create mask)
                    if arr.dtype == object:
                        # Check for None values
                        mask = pd.Series(arr).isna().to_numpy()
                        if mask.any():
                            # Create valid array and null array, combine
                            valid_indices = ~mask
                            if valid_indices.any():
                                valid_arr = arr[valid_indices]
                                parsed = build_number_struct_array(
                                    valid_arr, number_type=number_type
                                )
                                # Build full array with nulls (VECTORIZED)
                                # Extract parsed struct fields
                                parsed_values = parsed.field("value").to_numpy(
                                    zero_copy_only=False
                                )
                                parsed_nums = parsed.field("numerator").to_numpy(
                                    zero_copy_only=False
                                )
                                parsed_dens = parsed.field("denominator").to_numpy(
                                    zero_copy_only=False
                                )

                                # Create full arrays with None placeholders (vectorized)
                                full_values = np.full(n_rows, np.nan, dtype=np.float64)
                                full_nums = np.full(n_rows, np.nan, dtype=np.float64)
                                full_dens = np.full(n_rows, np.nan, dtype=np.float64)

                                # Place valid values using boolean indexing (vectorized)
                                full_values[valid_indices] = parsed_values
                                full_nums[valid_indices] = parsed_nums
                                full_dens[valid_indices] = parsed_dens

                                # Convert to PyArrow with proper nulls
                                processed[field_name] = pa.StructArray.from_arrays(
                                    [
                                        pa.array(full_values),
                                        pa.array(
                                            np.where(
                                                np.isnan(full_nums), None, full_nums
                                            ).astype(object),
                                            type=pa.int64(),
                                        ),
                                        pa.array(
                                            np.where(
                                                np.isnan(full_dens), None, full_dens
                                            ).astype(object),
                                            type=pa.int64(),
                                        ),
                                    ],
                                    names=["value", "numerator", "denominator"],
                                    mask=pa.array(mask),
                                )
                            else:
                                processed[field_name] = pa.nulls(
                                    n_rows, type=pa_field.type
                                )
                        else:
                            processed[field_name] = build_number_struct_array(
                                arr, number_type=number_type
                            )
                    else:
                        processed[field_name] = build_number_struct_array(
                            arr, number_type=number_type
                        )
            elif field_name == "id":
                if arr_data is not None:
                    # Ensure string type (vectorized)
                    if isinstance(arr_data, np.ndarray):
                        processed[field_name] = pa.array(arr_data.astype(str))
                    elif isinstance(arr_data, pa.Array):
                        processed[field_name] = arr_data.cast(pa.string())
                    else:
                        processed[field_name] = pa.array(
                            [str(x) for x in arr_data], type=pa.string()
                        )
                else:
                    # Auto-generate IDs using event_type prefix if available
                    event_types = get_arr("event_type")
                    if event_types is not None:
                        # Use per-type counters for informative IDs
                        type_counters: dict[str, int] = {}
                        id_list = []
                        if isinstance(event_types, (pa.Array, pa.ChunkedArray)):
                            et_py = event_types.to_pylist()
                        elif isinstance(event_types, np.ndarray):
                            et_py = event_types.tolist()
                        else:
                            et_py = list(event_types)
                        for et in et_py:
                            etype = str(et).lower() if et else "event"
                            type_counters.setdefault(etype, 0)
                            type_counters[etype] += 1
                            id_list.append(f"{etype}:{type_counters[etype]:06d}")
                        processed[field_name] = pa.array(id_list, type=pa.string())
                    else:
                        ids = np.array([f"event:{i + 1:06d}" for i in range(n_rows)])
                        processed[field_name] = pa.array(ids)
            elif arr_data is not None:
                # Regular field - convert to PyArrow array
                if isinstance(arr_data, (pa.Array, pa.ChunkedArray)):
                    processed[field_name] = arr_data
                elif isinstance(arr_data, np.ndarray):
                    processed[field_name] = pa.array(arr_data)
                else:
                    processed[field_name] = pa.array(arr_data)
            else:
                # Field not provided - fill with nulls
                processed[field_name] = pa.nulls(n_rows, type=pa_field.type)

        # Unified interval normalisation: compute missing end/duration,
        # check consistency, and infer temporal_type -- all vectorized.
        cls._normalize_intervals_vectorized(processed, policy=interval_policy)

        # Infer event_type if not provided or all null (vectorized)
        if "event_type" in processed:
            et_arr = processed["event_type"]
            if isinstance(et_arr, pa.Array) and et_arr.null_count == len(et_arr):
                # All null - use default
                processed["event_type"] = pa.array(["Event"] * n_rows)

        # Handle extra fields not in base schema
        # These are fields passed by loaders via extra_fields configuration
        base_field_names = set(schema.names)
        extra_field_names = set(fields.keys()) - base_field_names - {"instant"}

        # Build lookup for provided extra_fields (has proper metadata)
        provided_fields: dict[str, pa.Field] = {}
        if extra_fields:
            for f in extra_fields:
                provided_fields[f.name] = f

        inferred_fields = []
        for name in extra_field_names:
            data = fields[name]
            if data is None:
                continue

            # Infer PyArrow type from the data
            if isinstance(data, (pa.Array, pa.ChunkedArray)):
                arr = data
                if isinstance(arr, pa.ChunkedArray):
                    arr = arr.combine_chunks()
            elif isinstance(data, np.ndarray):
                arr = pa.array(data)
            else:
                arr = pa.array(data)

            processed[name] = arr

            # Use provided field if available (has metadata from CoordinateField etc.)
            if name in provided_fields:
                inferred_fields.append(provided_fields[name])
            else:
                # For coordinate struct fields, add unit metadata
                # This enables dataframe conversion to properly convert them

                if is_coordinate_type(arr.type):
                    # Coordinate field - add unit metadata (use default unit)
                    inferred_fields.append(
                        pa.field(
                            name,
                            arr.type,
                            nullable=True,
                            metadata={"unit": str(unit)},
                        )
                    )
                else:
                    inferred_fields.append(pa.field(name, arr.type, nullable=True))

        # Extend schema with extra fields
        if inferred_fields:
            schema = extend_schema(schema, inferred_fields)

        # Validate arrays if requested (vectorized validation)
        if validate:
            ArrayValidator.validate_field_dict(processed, schema)

        # Build table in single operation
        metadata = make_table_metadata(unit, number_type, loader_class=cls.__name__)
        schema = schema.with_metadata(metadata)

        table = pa.table(processed, schema=schema)
        return cls(table, unit, number_type)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        unit: TimeUnit,
        number_type: NumberType | None = None,
    ) -> Self:
        """Create EventData from a pandas DataFrame.

        Args:
            df: DataFrame with event data. Field names should match the schema.
            unit: The time unit for coordinates.
            number_type: The number type for coordinates.

        Returns:
            A new EventData containing the events.
        """
        if df.empty:
            return cls.empty(unit, number_type)

        return cls.from_dicts(df.to_dict("records"), unit, number_type)

    @classmethod
    def from_parquet(cls, path: Path | str) -> Self:
        """Load EventData from a Parquet file.

        Args:
            path: Path to the Parquet file.

        Returns:
            An EventData loaded from the file.

        Raises:
            ValueError: If the file lacks required TimeToAlign! metadata.
        """
        table = pq.read_table(path)
        metadata = parse_table_metadata(table.schema)

        if not metadata:
            raise ValueError(f"File {path} lacks TimeToAlign! metadata")

        unit = TimeUnit(metadata["unit"])
        number_type = NumberType(metadata["number_type"])

        return cls(table, unit, number_type)

    # endregion

    # region Properties

    @property
    def table(self) -> pa.Table:
        """The underlying PyArrow table."""
        return self._table

    @property
    def unit(self) -> TimeUnit:
        """The time unit for coordinates."""
        return self._unit

    @property
    def number_type(self) -> NumberType:
        """The number type for coordinate interpretation."""
        return self._number_type

    @property
    def count(self) -> int:
        """The number of events in the store."""
        return self._table.num_rows

    # endregion

    # region Magic Methods

    def __len__(self) -> int:
        """Return the number of events."""
        return self.count

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over events as dictionaries."""
        for batch in self._table.to_batches():
            for row in batch.to_pylist():
                yield row

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"count={self.count}, unit={self.unit}, number_type={self.number_type})"
        )

    def _repr_html_(self) -> str:
        """Rich HTML affordance card for Jupyter notebooks.

        Lists the event count, unit, number type, and the semantic fields
        reachable via :meth:`get_field` — each as ``name : ScalarType``
        with the backing column name — followed by the typical next-step
        invocations.
        """
        from timetoalign.display.html import affordance_html, code
        from timetoalign.storage.mixins import (
            _get_field_type_map,
            _parse_field_type_metadata,
        )

        rows: list[tuple[str, str]] = [
            ("Events", str(self.count)),
            ("Unit", str(self.unit)),
            ("Number type", str(self.number_type)),
        ]

        # Enumerate reachable semantic fields best-effort from the table
        # schema's TIMETOALIGN_METADATA_KEY field_type metadata (the same source
        # get_field uses). Columns with no paired field are skipped.
        registry = _get_field_type_map()
        field_items: list[str] = []
        listed_columns: set[str] = set()
        schema = self._table.schema
        for i in range(len(schema)):
            pa_field = schema.field(i)
            ft_str = _parse_field_type_metadata(pa_field)
            if ft_str is None:
                continue
            field_cls = registry.get(ft_str)
            if field_cls is None:
                continue
            scalar_cls = field_cls.scalar_cls
            scalar_name = scalar_cls.__name__ if scalar_cls is not None else "—"
            field_items.append(f"<li>{code(pa_field.name)} : {scalar_name}</li>")
            listed_columns.add(pa_field.name)

        # Also surface fields *afforded* over a raw atomic column
        # (represent-once): the typed pitch view is reachable via
        # get_field() / get_pitch_field() even though the column itself
        # carries no semantic metadata, so an honest "reachable fields"
        # listing must include it.
        column_names = set(self._table.column_names)
        afforded: dict[str, Any] = getattr(type(self), "_afforded_fields", {})
        for column, afforded_cls in afforded.items():
            if column in listed_columns or column not in column_names:
                continue
            scalar_cls = afforded_cls.scalar_cls
            scalar_name = scalar_cls.__name__ if scalar_cls is not None else "—"
            field_items.append(f"<li>{code(column)} : {scalar_name}</li>")
            listed_columns.add(column)

        if field_items:
            fields_value = "<ul>" + "".join(field_items) + "</ul>"
        else:
            fields_value = "(none)"
        rows.append(("Fields", fields_value))

        affordances = ["get_field(<Scalar>)", "get_pitch_field()", "get_raw('<col>')"]
        return affordance_html(type(self).__name__, rows, affordances=affordances)

    # endregion

    # region Extend/Merge

    def extend(self, other: "EventData") -> None:
        """Extend this data with events from another EventData (in-place).

        Args:
            other: Another EventData with compatible schema (extra fields
                are allowed and will be merged using schema promotion).

        Raises:
            ValueError: If units don't match.
        """
        if other.unit != self.unit:
            raise ValueError(f"Unit mismatch: {self.unit} vs {other.unit}")

        # Use promote_options="default" to handle schema differences
        # (e.g., extra fields from JSON loaders). Missing fields are
        # filled with nulls.
        self._table = pa.concat_tables(
            [self._table, other._table], promote_options="default"
        )
        # The table was replaced in place; any cached SemanticField (incl.
        # an afforded pitch view over a raw column) now wraps the stale
        # pre-extend array, so drop the cache and let the affordance
        # re-attach over the concatenated table on next access.
        self._invalidate_field_cache()

    def concat(self, *others: "EventData") -> "EventData":
        """Concatenate with other EventData, returning a new EventData.

        Args:
            *others: Other EventData to concatenate (extra fields are
                allowed and will be merged using schema promotion).

        Returns:
            A new EventData containing all events.

        Raises:
            ValueError: If any units don't match.
        """
        tables = [self._table]
        for other in others:
            if other.unit != self.unit:
                raise ValueError(f"Unit mismatch: {self.unit} vs {other.unit}")
            tables.append(other._table)

        # Use promote_options="default" to handle schema differences
        new_table = pa.concat_tables(tables, promote_options="default")
        return self.__class__(new_table, self.unit, self.number_type)

    def prefix_ids(self, prefix: str) -> "EventData":
        """Return a new EventData with all event IDs prefixed.

        Prepends ``prefix:`` to every event ID. Used when events are placed
        onto a timeline so that IDs become globally unique and informative,
        e.g. ``clt_1:note:000001``.

        If the IDs already start with the prefix, they are left unchanged.

        Args:
            prefix: The prefix to prepend (without trailing colon).

        Returns:
            A new EventData with prefixed IDs.
        """
        id_arr = self._table.column("id")
        prefix_str = f"{prefix}:"
        new_ids = pa.array(
            [
                f"{prefix_str}{v}" if not v.startswith(prefix_str) else v
                for v in id_arr.to_pylist()
            ],
            type=pa.string(),
        )
        idx = self._table.schema.get_field_index("id")
        new_table = self._table.set_column(idx, self._table.schema.field(idx), new_ids)
        return self.__class__(new_table, self._unit, self._number_type)

    def _with_segment_renamed(self, old_id: str, new_id: str) -> "EventData":
        """Return a copy whose segment row for child *old_id* names *new_id*.

        A timeline records each child as one ``Segment`` row whose ``id`` and
        ``name`` are the child's id; this keeps that row in step when the
        child's id changes.
        """
        table = self._table
        names = table.schema.names
        if "event_type" not in names or "id" not in names:
            return self
        mask = pc.fill_null(
            pc.and_kleene(
                pc.equal(table.column("event_type"), "Segment"),
                pc.equal(table.column("id"), old_id),
            ),
            False,
        )
        for column in ("id", "name"):
            if column not in names:
                continue
            idx = names.index(column)
            values = table.column(idx)
            renamed = pc.if_else(mask, pa.scalar(new_id, values.type), values)
            table = table.set_column(idx, table.schema.field(idx), renamed)
        return self.__class__(table, self._unit, self._number_type)

    # endregion

    # region Query/Filter

    def filter(
        self,
        *,
        temporal_type: Literal["instant", "interval"] | None = None,
        event_type: str | None = None,
        min_coord: CoordinateSpec | None = None,
        max_coord: CoordinateSpec | None = None,
        **kwargs: Any,
    ) -> "EventData":
        """Filter events by criteria, returning a new EventData.

        All criteria are AND-ed together.

        Args:
            temporal_type: Filter by "instant" or "interval".
            event_type: Filter by event type name.
            min_coord: Minimum coordinate (inclusive), optionally with a unit.
            max_coord: Maximum coordinate (exclusive), optionally with a unit.
            **kwargs: Exact match filters for other fields (e.g. event_category="note").

        Coordinate bounds are compared against the stored float coordinate
        representation. Fraction bounds are converted to float for this
        comparison; integer and float bounds retain their respective types.
        The minimum remains inclusive and the maximum remains exclusive.

        Returns:
            A new EventData with filtered events.
        """
        mask = None

        if temporal_type is not None:
            expr = pc.equal(pc.field("temporal_type"), temporal_type)
            mask = expr if mask is None else (mask & expr)

        if event_type is not None:
            expr = pc.equal(pc.field("event_type"), event_type)
            mask = expr if mask is None else (mask & expr)

        if min_coord is not None or max_coord is not None:
            # For coordinate filtering, we use the float 'value' field
            # Check start.value
            coord_val = pc.struct_field(pc.field("start"), "value")

            if min_coord is not None:
                resolved_min = resolve_coordinate_spec(min_coord)
                if resolved_min.unit is not None and resolved_min.unit != self.unit:
                    raise ValueError(
                        f"Coordinate unit mismatch: {resolved_min.unit} vs {self.unit}"
                    )
                min_value = (
                    float(resolved_min.value)
                    if isinstance(resolved_min.value, Fraction)
                    else resolved_min.value
                )
                expr = pc.greater_equal(coord_val, min_value)
                mask = expr if mask is None else (mask & expr)

            if max_coord is not None:
                resolved_max = resolve_coordinate_spec(max_coord)
                if resolved_max.unit is not None and resolved_max.unit != self.unit:
                    raise ValueError(
                        f"Coordinate unit mismatch: {resolved_max.unit} vs {self.unit}"
                    )
                max_value = (
                    float(resolved_max.value)
                    if isinstance(resolved_max.value, Fraction)
                    else resolved_max.value
                )
                expr = pc.less(coord_val, max_value)
                mask = expr if mask is None else (mask & expr)

        # Generic kwargs filtering
        for name, val in kwargs.items():
            # Only if field exists in schema
            if name in self.field_names():
                expr = pc.equal(pc.field(name), val)
                mask = expr if mask is None else (mask & expr)

        if mask is None:
            return self

        filtered = self._table.filter(mask)
        return self.__class__(filtered, self.unit, self.number_type)

    def column_values(self, name: str, *, default: Any = None) -> list[Any]:
        """Return a column's values as plain, decoded Python objects.

        A rational-shaped column -- any struct column carrying a ``value``
        sub-field, such as the canonical ``{value, numerator, denominator}``
        coordinate struct used by ``start``, ``end``, ``duration``, and any
        extra field sharing that shape -- decodes to an exact `Fraction`
        via `struct_to_rational`. When a row's struct carries no exact
        ratio (missing or non-integral numerator/denominator), the float
        ``value`` member is used instead, wrapped in a `Fraction`. Every
        other column type is returned via ``to_pylist()`` unchanged,
        including ``None`` for a null cell -- *default* only substitutes
        for a null rational-shaped struct or a column absent from this
        table entirely.

        Args:
            name: Field name in the underlying table.
            default: Value substituted for every row when *name* is not a
                column of this table, and for individual null rational
                struct cells.

        Returns:
            A list of per-row values, one for each event.
        """
        if name not in self._table.column_names:
            return [default] * len(self)

        pa_field = self._table.schema.field(name)
        column = self._table.column(name)

        if not _has_value_subfield(pa_field.type):
            return column.to_pylist()

        values: list[Any] = []
        for cell in column.to_pylist():
            if cell is None:
                values.append(default)
                continue
            try:
                values.append(struct_to_coordinate(cell, NumberType.fraction))
            except (ValueError, TypeError, ZeroDivisionError):
                values.append(default)
        return values

    def select(self, fields: list[str]) -> pa.Table:
        """Select specific fields from the table.

        Args:
            fields: List of field names to select.

        Returns:
            A PyArrow table with only the selected fields.
        """
        return self._table.select(fields)

    def where(self, expression: pc.Expression) -> "EventData":
        """Filter with a custom PyArrow compute expression.

        Args:
            expression: A PyArrow compute expression.

        Returns:
            A new EventData with filtered events.
        """
        filtered = self._table.filter(expression)
        return self.__class__(filtered, self.unit, self.number_type)

    # endregion

    # region Stats/Overview

    def count_by(self, field: str) -> dict[str, int]:
        """Count events grouped by a field's values.

        Args:
            field: The field to group by.

        Returns:
            Dict mapping field values to counts.
        """
        result = self._table.group_by(field).aggregate([(field, "count")])
        return {row[field]: row[f"{field}_count"] for row in result.to_pylist()}

    def coordinate_range(self) -> tuple[float | Fraction, float | Fraction] | None:
        """Get the min and max coordinates across all events.

        Returns:
            Tuple of (min, max) coordinates, or None if store is empty.
            Returns Fraction values when number_type is fraction.
        """
        if self.count == 0:
            return None

        coordinate_values: list[float | Fraction] = []
        for field_name in ["start", "end"]:
            for cell in self._table.column(field_name).to_pylist():
                if cell is None or cell.get("value") is None:
                    continue

                coordinate_values.append(struct_to_coordinate(cell, self._number_type))

        if not coordinate_values:
            return None

        return min(coordinate_values), max(coordinate_values)

    def event_types(self) -> list[str]:
        """Get the list of unique event types.

        Returns:
            List of event type names.
        """
        unique = pc.unique(self._table.column("event_type"))
        return [v.as_py() for v in unique if v.as_py() is not None]

    def summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of the store.

        Returns:
            Dict with count, temporal type counts, event type counts,
            coordinate range, unit, and number type.
        """
        return {
            "count": self.count,
            "unit": str(self.unit),
            "number_type": str(self.number_type),
            "temporal_types": self.count_by("temporal_type"),
            "event_types": self.count_by("event_type"),
            "coordinate_range": self.coordinate_range(),
        }

    # endregion

    # region Timeline Creation

    def create_timeline(
        self,
        uid: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> "Timeline":
        """Create a Timeline from this EventData.

        This is a convenience method that creates a timeline with the data's
        events directly. The timeline class and number_type are inferred from
        the data's unit (e.g., ticks -> DiscreteLogicalTimeline with int).

        Args:
            uid: Unique ID for the timeline. Auto-generated if None.
            filters: Filter kwargs to apply before timeline creation.
                Example: {"event_type": "Note"} to exclude rests.

        Returns:
            A Timeline containing the (filtered) events.

        Examples:
            >>> timeline = data.create_timeline(uid="notes")
            >>> filtered = data.create_timeline(filters={"event_type": "Note"})
        """
        if _timeline_factory is None:
            raise RuntimeError("Timeline creation requires timetoalign.timelines")
        return _timeline_factory(self, uid, filters)

    # endregion

    # region Serialization

    def to_parquet(self, path: Path | str) -> None:
        """Save the EventData to a Parquet file.

        Args:
            path: Path to write the Parquet file.
        """
        pq.write_table(self._table, path)

    def to_dataframe(
        self,
        format: str = "pandas",
        *,
        raw: bool = False,
        coordinates: bool = False,
    ) -> pd.DataFrame:
        """Convert to a DataFrame in the specified format.

        Higher-level method that dispatches to format-specific implementations.
        Currently supports pandas; polars support can be added later.

        Args:
            format: DataFrame format ("pandas"). Default "pandas".
            raw: If True, return raw conversion with struct dicts for coordinates.
            coordinates: If True, wrap values in Coordinate objects with unit info.

        Returns:
            A DataFrame in the requested format.

        Raises:
            ValueError: If format is not supported.

        Examples:
            >>> df = events.to_dataframe()  # pandas DataFrame
            >>> df.iloc[0]['start']  # Fraction(1, 4) or 0.25
            >>> df = events.to_dataframe("pandas", raw=True)
            >>> df.iloc[0]['start']  # {'value': 0.25, 'numerator': 1, 'denominator': 4}
            >>> df = events.to_dataframe(coordinates=True)
            >>> df.iloc[0]['start']  # Coordinate(value=Fraction(1, 4), unit=quarters)
        """
        if format == "pandas":
            df = self._table.to_pandas()

            if raw:
                return df

            # Detect coordinate fields from schema (struct with value/num/den fields)
            # This handles both core fields (start, end, duration) and extra
            # CoordinateField fields.

            for pa_field in self._table.schema:
                field_name = pa_field.name
                if field_name not in df.columns:
                    continue

                if is_coordinate_type(pa_field.type):
                    # Get unit from field metadata, fall back to EventData unit
                    unit = self._unit
                    unit_str = field_metadata(pa_field).get("unit")
                    if unit_str:
                        try:
                            unit = TimeUnit(unit_str)
                        except ValueError:
                            pass  # Use default unit

                    if coordinates:
                        field_unit = unit
                        df[field_name] = df[field_name].apply(
                            lambda s, u=field_unit: (
                                Coordinate(
                                    struct_to_coordinate(s, self._number_type), u
                                )
                                if s is not None
                                else None
                            )
                        )
                    else:
                        df[field_name] = df[field_name].apply(
                            lambda s: (
                                struct_to_coordinate(s, self._number_type)
                                if s is not None
                                else None
                            )
                        )

            return df
        else:
            raise ValueError(
                f"Unsupported DataFrame format: {format!r}. "
                f"Supported formats: 'pandas'"
            )

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return the first ``n`` events as a pandas DataFrame.

        A pandas-style preview of the stored rows. Coordinate columns render
        as their native numbers — the same conversion :meth:`to_dataframe`
        applies — so ``events.head()`` shows the leading events directly,
        without reaching for the raw PyArrow table.

        Args:
            n: Number of leading events to include. Values larger than the
                event count return every event; ``n <= 0`` returns an empty
                frame.

        Returns:
            A pandas DataFrame of the first ``n`` events.

        Examples:
            >>> events.head()      # first five events
            >>> events.head(3)     # first three events
        """
        preview = type(self)(
            self._table.slice(0, max(n, 0)), self._unit, self._number_type
        )
        return preview.to_dataframe()

    # endregion


def _as_numpy(values: Any) -> np.ndarray:
    """Convert an array-like to a numpy array.

    Raises:
        TypeError: If conversion is not possible.
    """
    if isinstance(values, np.ndarray):
        return values
    if isinstance(values, pd.Series):
        return values.to_numpy()
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        return values.to_numpy(zero_copy_only=False)
    if isinstance(values, list):
        return np.array(values)
    raise TypeError(f"Cannot convert {type(values)} to numpy array")


class ArrayValidator:
    """Vectorized array validation before table construction.

    Validates:
    - Array lengths match
    - Required fields present
    - Temporal type consistency (instant vs interval)
    - ID uniqueness

    All checks use vectorized operations - NO row iteration.

    Examples:
        >>> fields = {
        ...     "id": np.array(["e1", "e2"]),
        ...     "temporal_type": np.array(["instant", "interval"]),
        ...     "event_type": np.array(["Beat", "Note"]),
        ...     "start": pa.array([...]),  # StructArray
        ...     "end": pa.array([None, ...]),  # StructArray with nulls
        ... }
        >>> ArrayValidator.validate_field_dict(fields, schema)
    """

    @staticmethod
    def validate_field_dict(
        fields: dict[str, np.ndarray | pd.Series | list | pa.Array],
        schema: pa.Schema,
    ) -> None:
        """Validate field dictionary before table creation (vectorized).

        Args:
            fields: Dict of {field_name: array_values}.
            schema: Expected PyArrow schema.

        Raises:
            ValueError: If validation fails.
        """
        # Check 1: All arrays have same length (vectorized check)
        lengths = {name: len(arr) for name, arr in fields.items()}
        unique_lengths = set(lengths.values())

        if len(unique_lengths) > 1:
            raise ValueError(f"Field length mismatch: {lengths}")

        # Check 2: Required fields present
        required = {"id", "temporal_type", "event_type", "start"}
        missing = required - set(fields.keys())

        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Check 3: ID uniqueness (vectorized)
        id_arr = _as_numpy(fields["id"])
        unique_ids = np.unique(id_arr)

        if len(unique_ids) != len(id_arr):
            duplicates = len(id_arr) - len(unique_ids)
            raise ValueError(
                f"Duplicate IDs found: {len(id_arr)} IDs, " f"{duplicates} duplicates"
            )

        # Check 4: Temporal type consistency (vectorized)
        ArrayValidator._validate_temporal_consistency(fields)

    @staticmethod
    def _validate_temporal_consistency(
        fields: dict[str, np.ndarray | pd.Series | list | pa.Array],
    ) -> None:
        """Validate instant vs interval consistency (vectorized).

        Strategy:
        - instant events: end must be None/null
        - interval events: end must be present
        - Use pandas boolean indexing for vectorized checks

        Args:
            fields: Field dictionary.

        Raises:
            ValueError: If temporal consistency is violated.
        """
        temporal_type = pd.Series(_as_numpy(fields["temporal_type"]))

        # Get masks for instant/interval events (vectorized)
        is_instant = temporal_type == "instant"
        is_interval = temporal_type == "interval"

        # Validate all values are either instant or interval
        valid_count = is_instant.sum() + is_interval.sum()
        if valid_count != len(temporal_type):
            invalid = temporal_type[~(is_instant | is_interval)]
            raise ValueError(
                f"Invalid temporal_type values: {invalid.unique().tolist()}"
            )

        # Validate instant events have no end coordinate (vectorized check)
        if "end" in fields:
            # Handle both PyArrow StructArray and numpy array
            end_arr = fields["end"]
            if isinstance(end_arr, pa.StructArray):
                end_is_null = end_arr.is_null().to_numpy(zero_copy_only=False)
            elif isinstance(end_arr, pa.ChunkedArray):
                end_is_null = end_arr.is_null().to_numpy(zero_copy_only=False)
            else:
                end_series = pd.Series(_as_numpy(end_arr))
                end_is_null = end_series.isna().to_numpy()

            instant_with_end = is_instant.to_numpy() & ~end_is_null

            if instant_with_end.any():
                raise ValueError(
                    f"Instant events cannot have 'end' coordinate: "
                    f"{instant_with_end.sum()} violations"
                )

        # Validate interval events have end coordinate (vectorized check)
        if "end" not in fields:
            if is_interval.any():
                raise ValueError(
                    f"Interval events require 'end' coordinate: "
                    f"{is_interval.sum()} events missing 'end'"
                )
        else:
            # Check that interval events have non-null end
            end_arr = fields["end"]
            if isinstance(end_arr, pa.StructArray):
                end_is_null = end_arr.is_null().to_numpy(zero_copy_only=False)
            elif isinstance(end_arr, pa.ChunkedArray):
                end_is_null = end_arr.is_null().to_numpy(zero_copy_only=False)
            else:
                end_series = pd.Series(_as_numpy(end_arr))
                end_is_null = end_series.isna().to_numpy()

            interval_no_end = is_interval.to_numpy() & end_is_null

            if interval_no_end.any():
                raise ValueError(
                    f"Interval events missing 'end' coordinate: "
                    f"{interval_no_end.sum()} events"
                )
