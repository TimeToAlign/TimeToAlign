"""TabularLoader: Vectorized base class for loading tabular data.

This module provides the TabularLoader class that supports:

- ZERO ROW ITERATION: All operations are vectorized
- Two-step column → field pipeline driven by ``column_specs`` (Step 1)
  and ``field_specs`` (Step 2)
- Multiple coordinate parsing strategies (float, int, fraction)
- Delimiter configuration for different formats
- Event type inference from data

Design

* Single file read: ``pd.read_csv()`` -> DataFrame.
* The source DataFrame is materialised as a faithful ``pa.Table`` and
  stored as ``self._raw_table`` (no type coercion).
* Step 1 — ``column_specs`` resolves each source column into a typed
  :class:`DataField` via :func:`field_parsers.resolve_field_parser`.
  Each emitted field carries its own name and ``pa.Field`` metadata.
* Coordinate parsing (``start`` / ``end`` / ``duration``) continues
  to flow through :func:`~timetoalign.core.time.build_number_struct_array`,
  driven by
  ``start_column`` / ``end_column`` / ``duration_column`` on the
  loader.  This produces the canonical struct-shaped coordinate
  columns the rest of the codebase expects.
* Step 2 — ``field_specs`` (optional) materialises blueprint-mode
  :class:`SemanticField` instances against the Step-1 output and
  decorates the corresponding column with paired-class metadata so
  ``EventData.get_field()`` round-trips the semantic type.

NO FOR LOOPS OVER ROWS. EVER.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from timetoalign.core import (
    NumberType,
    SemanticField,
    TimeUnit,
)
from timetoalign.core.fields import (
    TIMETOALIGN_METADATA_KEY,
    DataField,
    metadata_blob_from_dict,
    parse_metadata_blob,
)
from timetoalign.core.time import build_number_struct_array
from timetoalign.loader.base import EventLoader
from timetoalign.storage.schema import (
    ComputedField,
    Field,
)

from .field_parsers import (
    CompositeFieldParser,
    FieldParser,
    resolve_field_parser,
)

if TYPE_CHECKING:
    pass

module_logger = logging.getLogger(__name__)


def _producer_name(producer: DataField | FieldParser) -> str | None:
    """Return the producer's preferred output name, or ``None`` for fallback.

    DataField blueprints carry a name on their :attr:`DataField.name`
    attribute; FieldParser instances carry an optional override in
    :attr:`FieldParser.name`.
    """
    if isinstance(producer, FieldParser):
        return producer.name
    if isinstance(producer, DataField):
        return producer.name
    return None


def _merge_field_type_metadata(pa_field: pa.Field, field_type: str) -> pa.Field:
    """Return *pa_field* with a TTA blob carrying ``field_type``.

    If *pa_field* already carries a ``TIMETOALIGN_METADATA_KEY`` payload
    (e.g. ``DenominateNumberField.from_array()`` writes ``{"unit": ...}``), the
    existing payload is preserved and ``"field_type"`` is added /
    overwritten on top.  Metadata entries under any other key are passed
    through unchanged.
    """
    existing = dict(pa_field.metadata or {})
    payload: dict[str, Any] = {}
    blob = existing.pop(TIMETOALIGN_METADATA_KEY, None)
    if blob is not None:
        payload = parse_metadata_blob(blob)
        if not isinstance(payload, dict):
            payload = {}
    payload["field_type"] = field_type
    existing[TIMETOALIGN_METADATA_KEY] = metadata_blob_from_dict(payload)
    return pa_field.with_metadata(existing)


# region TabularLoader


class TabularLoader(EventLoader):
    """Vectorized base class for loading tabular data with column/field specs.

    TabularLoader provides a ZERO ROW ITERATION framework for loading
    CSV, TSV, and other delimited formats into EventData.  All
    operations use vectorized numpy / pandas / pyarrow operations.

    Configuration Attributes:
        delimiter: Field delimiter character (default: ``,``).
        header_row: Row index containing column headers (default: 0).
        id_column: Source column name for event IDs (auto-generated if
            ``None``).
        name_column: Source column name for event names (optional).
        start_column: Source column name for start coordinate (required).
        end_column: Source column name for end coordinate (None for
            instant events).
        duration_column: Source column name for duration (alternative to
            ``end_column``).
        event_type_column: Source column name for event type (optional).
        default_event_type: Default event type if no column provides one.
        column_specs: Mapping or sequence describing the per-column
            translation from source to :class:`DataField`.  See
            :mod:`timetoalign.loader.tabular.field_parsers`.
        field_specs: Optional sequence/mapping of blueprint-mode
            :class:`SemanticField` instances that promote raw fields to
            paired semantic fields.
        coordinate_unit: TimeUnit for coordinate values.

    Examples:
        >>> from timetoalign.core import IntField
        >>> class MyLoader(TabularLoader):
        ...     delimiter = "\\t"
        ...     start_column = "onset"
        ...     end_column = "offset"
        ...     coordinate_unit = TimeUnit.seconds
        ...     column_specs = {"velocity": IntField(name="velocity"), "channel": int}
    """

    # region Class Configuration

    # Parsing configuration
    delimiter: ClassVar[str] = ","
    header_row: ClassVar[int] = 0
    encoding: ClassVar[str] = "utf-8"

    # Column mapping - subclasses should override
    id_column: ClassVar[str | None] = None
    name_column: ClassVar[str | None] = None
    # start_column can be:
    #   - str: Direct column name
    #   - tuple: Struct field access like ("rect_coords", "x")
    #   - Field: Struct field object like Field("rect_coords", "x")
    #   - ComputedField: Computed value (not typical for start)
    start_column: ClassVar[str | tuple | Field | ComputedField] = "start"  # Required
    _fallback_start_column: ClassVar[str | None] = (
        None  # Fallback if start_column missing
    )
    # end_column can be:
    #   - str: Direct column name
    #   - tuple: Struct field access like ("rect_coords", "x")
    #   - Field: Struct field object
    #   - ComputedField: Computed value like "rect_coords.x + rect_coords.width"
    #   - None: Instant events (no end)
    end_column: ClassVar[str | tuple | Field | ComputedField | None] = None
    duration_column: ClassVar[str | None] = None  # Alternative to end_column
    event_type_column: ClassVar[str | None] = None
    default_event_type: ClassVar[str] = "Event"

    # Step 1 — column_specs.  Translates source columns to typed
    # :class:`DataField` instances.  Two shapes:
    #
    # * ``dict[str, X]`` — keys are source column names (header-based
    #   sources); the dict-key seeds the emitted field's name unless
    #   the producer carries an explicit ``name=`` override.
    # * ``Sequence[X]`` — positional, for header-less sources.  Each
    #   entry MUST carry a ``name=`` (either through an explicit
    #   constructor kwarg or, for paired SemanticField subclasses, via
    #   the snake-case default — e.g. ``MeasureNumberField`` resolves
    #   to ``measure_number``).
    #
    # ``X`` resolves via the universal table in
    # :func:`field_parsers.resolve_field_parser`.
    column_specs: ClassVar[dict[str, Any] | Sequence[Any] | None] = None

    # Step 2 — field_specs.  Sequence or dict of blueprint-mode
    # :class:`SemanticField` instances that materialise against the
    # Step-1 output table.  Optional; many loaders are complete with
    # Step 1 alone.
    field_specs: ClassVar[Sequence[SemanticField] | dict[str, SemanticField] | None] = (
        None
    )

    # Coordinate configuration
    _default_unit: ClassVar[TimeUnit] = TimeUnit.seconds

    # endregion

    def __init__(
        self,
        unit: TimeUnit | None = None,
        number_type: NumberType | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize TabularLoader.

        Args:
            unit: Override the default time unit.
            number_type: Override the default number type.
            **kwargs: Additional arguments passed to parent Loader.
        """

        super().__init__(unit=unit, number_type=number_type, **kwargs)
        self._logger = module_logger.getChild(self.__class__.__name__)
        # Extra pa.Fields contributed by column_specs / field_specs.
        # These carry the typed (and possibly semantic) metadata that
        # EventData.from_arrays propagates onto its schema.
        self._extra_schema_fields: list[pa.Field] = []
        # Store the original pa.Table for the loader-first pipeline.
        # This is the faithful representation of the source file, before
        # any field extraction or type coercion.
        self._raw_table: pa.Table | None = None

    # region Properties

    @property
    def table(self) -> pa.Table | None:
        """The original source data as a faithful PyArrow table.

        This is the raw tabular data from the source file, before any
        column extraction, coordinate parsing, or type coercion. Returns
        ``None`` if no source has been loaded yet.

        Returns:
            The original ``pa.Table``, or ``None``.
        """
        return self._raw_table

    # endregion

    # region Arrow Conversion

    @staticmethod
    def _dataframe_to_arrow(df: pd.DataFrame) -> pa.Table:
        """Convert a DataFrame to a PyArrow Table safely.

        Handles columns containing types that PyArrow cannot natively
        serialize (e.g., ``Fraction`` objects from ms3) by converting
        them to string representation.

        Args:
            df: The pandas DataFrame to convert.

        Returns:
            A ``pa.Table`` faithfully representing the source data.
        """
        try:
            return pa.Table.from_pandas(df, preserve_index=False)
        except (pa.lib.ArrowTypeError, pa.lib.ArrowInvalid):
            # Fallback: convert problematic object columns to strings
            df_safe = df.copy()
            for name in df_safe.columns:
                if df_safe[name].dtype == object:
                    df_safe[name] = df_safe[name].astype(str)
            return pa.Table.from_pandas(df_safe, preserve_index=False)

    # endregion

    # region Column / Field Spec Plumbing

    def _resolved_column_specs(
        self, df: pd.DataFrame
    ) -> list[tuple[str | int, DataField | FieldParser, str]]:
        """Resolve ``column_specs`` into ``(source_key, producer, name)``.

        Returns one entry per spec, in declaration order.  For a
        ``dict`` form, ``source_key`` is the dict key (a column name)
        and the dict-key seeds ``default_name`` for blueprint
        construction.  For a ``Sequence`` form, ``source_key`` is the
        positional index (an int) and the spec must supply its own
        name (via constructor or class-level default — the resolver is
        called with ``default_name=None``).

        Raises:
            ValueError: If a positional spec lacks a name.
        """
        specs = self.column_specs
        if specs is None:
            return []
        resolved: list[tuple[str | int, DataField | FieldParser, str]] = []
        if isinstance(specs, dict):
            for col_name, raw in specs.items():
                producer = resolve_field_parser(raw, default_name=col_name)
                emit_name = _producer_name(producer) or col_name
                resolved.append((col_name, producer, emit_name))
        else:
            for i, raw in enumerate(specs):
                producer = resolve_field_parser(raw)
                emit_name = _producer_name(producer)
                if emit_name is None:
                    raise ValueError(
                        f"column_specs[{i}] has no name; positional specs "
                        "require an explicit name= or a class-level default"
                    )
                resolved.append((i, producer, emit_name))
        return resolved

    def _source_array(self, df: pd.DataFrame, source_key: str | int) -> pa.Array | None:
        """Return the column at *source_key* as a ``pa.Array``.

        ``str`` keys are looked up by column name (header-based
        source); a missing column returns ``None`` so the caller can
        skip the spec gracefully (column_specs entries are advisory
        for sibling-file variants — a measures.tsv lacks ``staff`` /
        ``voice`` columns that notes.tsv carries).

        ``int`` keys are looked up positionally; a missing index is a
        hard error because positional schemas are fixed.
        """
        if isinstance(source_key, int):
            if source_key >= len(df.columns):
                raise IndexError(
                    f"column_specs positional index {source_key} out of range "
                    f"(source has {len(df.columns)} columns)"
                )
            col = df.iloc[:, source_key]
        else:
            if source_key not in df.columns:
                return None
            col = df[source_key]
        return pa.array(col)

    def _apply_column_specs(
        self, df: pd.DataFrame, columns: dict[str, Any]
    ) -> dict[str, DataField]:
        """Run Step 1: ``column_specs`` → emitted :class:`DataField`s.

        Each emitted field is written into *columns* (as a ``pa.Array``)
        and its ``pa.Field`` (with a ``field_type`` blob under
        ``TIMETOALIGN_METADATA_KEY``) is appended to ``self._extra_schema_fields``.  The
        metadata stamp lets downstream filtering (``get_events(properties=False)``)
        recognise column-spec emissions as *fields* rather than raw
        property columns.  Any metadata the producer's ``from_array()`` had
        already attached (e.g. ``DenominateNumberField`` writes the
        ``unit``) is preserved and merged with the ``field_type`` key.

        Side effect:
            Populates ``self._consumed_source_columns`` with the set of
            source-DataFrame column names that were consumed by
            ``column_specs`` (used by :meth:`_extract_column_arrays` to
            decide which raw source columns survive as property columns).

        Returns:
            A dict mapping output names to materialised DataField objects (for
            Step 2 lookup).
        """
        emitted: dict[str, DataField] = {}
        consumed: set[str] = set()
        for source_key, producer, emit_name in self._resolved_column_specs(df):
            source_arr = self._source_array(df, source_key)
            if source_arr is None:
                continue
            # Record the source-DataFrame column that backed this spec.
            if isinstance(source_key, int):
                if 0 <= source_key < len(df.columns):
                    consumed.add(str(df.columns[source_key]))
            else:
                consumed.add(source_key)
            data_field = producer.from_array(source_arr, name=emit_name)
            emitted[emit_name] = data_field
            data = data_field.data
            if data is None:
                continue
            columns[emit_name] = data
            pa_field_with_meta = _merge_field_type_metadata(
                data_field.field, type(data_field).__name__
            )
            self._extra_schema_fields.append(pa_field_with_meta)
        self._consumed_source_columns = consumed
        return emitted

    def _apply_field_specs(
        self, columns: dict[str, Any], emitted: dict[str, DataField]
    ) -> None:
        """Run Step 2: blueprint :class:`SemanticField` materialisation.

        For each blueprint, look up its ``source_fields`` against the
        Step-1 emitted dict (or the running columns), pack the source
        column into the target's ``pa_schema`` shape when needed, and
        decorate the resulting column with paired-class metadata so
        EventData discovery round-trips the semantic type.

        Currently supports the string shorthand:

        * ``source_fields="<name>"`` — single-source promotion.  Atomic
          source columns are packed into a single-field struct whose
          sub-field name matches the target's ``pa_schema`` (e.g.
          ``EnharmonicPitchField`` → ``{midi_number: <source>}``).
        """
        specs = self.field_specs
        if specs is None:
            return
        iterable = list(specs.values()) if isinstance(specs, dict) else list(specs)
        for blueprint in iterable:
            if not isinstance(blueprint, SemanticField):
                raise TypeError(
                    f"field_specs entries must be SemanticField instances, got "
                    f"{type(blueprint).__name__}"
                )
            if not blueprint.is_blueprint:
                raise TypeError(
                    "field_specs entries must be blueprint-mode SemanticField "
                    "(constructed via source_fields=...)"
                )
            spec_str = blueprint._blueprint_source_fields  # type: ignore[attr-defined]
            if not isinstance(spec_str, str):
                raise NotImplementedError(
                    "Step 2 currently supports the 'source_fields=<name>' "
                    "shorthand only; multi-source / dict blueprints land in "
                    "a follow-up"
                )
            source_name = spec_str
            if source_name not in columns:
                raise KeyError(
                    f"field_specs blueprint references unknown source "
                    f"{source_name!r}; available: {sorted(columns.keys())}"
                )

            blueprint_cls = type(blueprint)
            data_arr = columns[source_name]

            # Pack atomic source into a single-field struct matching the
            # target schema when needed.
            packed_arr = self._pack_field_spec_source(
                data_arr, blueprint_cls, source_name
            )
            if packed_arr is not None:
                columns[source_name] = packed_arr
                data_arr = packed_arr

            # Decorate the column's pa.Field with paired-class metadata.
            meta_blob = metadata_blob_from_dict({"field_type": blueprint_cls.__name__})
            existing_idx = next(
                (
                    i
                    for i, pf in enumerate(self._extra_schema_fields)
                    if pf.name == source_name
                ),
                None,
            )
            if existing_idx is not None:
                # The packing step may have changed the column's dtype;
                # rebuild the pa.Field from the current data array.
                arr_type = (
                    data_arr.type
                    if isinstance(data_arr, (pa.Array, pa.ChunkedArray))
                    else pa.array(data_arr).type
                )
                self._extra_schema_fields[existing_idx] = pa.field(
                    source_name,
                    arr_type,
                    metadata={TIMETOALIGN_METADATA_KEY: meta_blob},
                )
            else:
                arr_type = (
                    data_arr.type
                    if isinstance(data_arr, (pa.Array, pa.ChunkedArray))
                    else pa.array(data_arr).type
                )
                self._extra_schema_fields.append(
                    pa.field(
                        source_name,
                        arr_type,
                        metadata={TIMETOALIGN_METADATA_KEY: meta_blob},
                    )
                )

    @staticmethod
    def _pack_field_spec_source(
        data_arr: Any,
        blueprint_cls: type[SemanticField[Any]],
        source_name: str,
    ) -> pa.Array | None:
        """Pack an atomic source array into a target struct shape.

        The packing rule belongs to the target field class — it is the
        one that knows which of its sub-fields a plain source column
        fills — so this defers to that class's blueprint
        :meth:`~timetoalign.core.SemanticField.from_array`.

        Args:
            data_arr: The source column.
            blueprint_cls: The paired Field class being promoted to.
            source_name: The source column's name.

        Returns:
            The packed array, or ``None`` when no packing is required
            (the source already matches the target shape).

        Raises:
            TypeError: When the target field class states no packing
                rule for a plain source column.
        """
        if blueprint_cls.pa_schema is None:
            return None
        if not isinstance(data_arr, (pa.Array, pa.ChunkedArray)):
            data_arr = pa.array(data_arr)
        if pa.types.is_struct(data_arr.type):
            # A struct source is already in some shape of its own; a
            # mismatch surfaces downstream in EventData.get_field.
            return None
        if not blueprint_cls.supports_plain_blueprint():
            raise TypeError(
                f"field_specs cannot pack atomic source {source_name!r} into "
                f"target {blueprint_cls.__name__} ({blueprint_cls.pa_schema}): "
                f"that field "
                f"class needs more than a name to exist and states no packing "
                f"rule for a plain source column"
            )
        packed = blueprint_cls(name=source_name).from_array(data_arr)
        return packed.data

    # endregion

    # region Vectorized Loading

    def _load_source(
        self, source: Path
    ) -> tuple[dict[str, Any], dict[str, np.ndarray | pa.Array]]:
        """Load a single source file (VECTORIZED).

        Reads the tabular file using vectorized pandas operations and
        returns column arrays ready for ``EventData.from_arrays()``.  NO
        ROW ITERATION.

        Architecture:
            1. Single file read: ``pd.read_csv()``.
            2. Validate required columns exist.
            3. Extract canonical columns (id, name, start, end / duration,
               event_type, temporal_type) — vectorized.
            4. Apply ``column_specs`` (Step 1) → typed DataFields.
            5. Apply ``field_specs`` (Step 2, optional) → semantic
               decorations.
            6. Hook ``_post_process_columns`` for subclass-specific
               extensions.
            7. Return column dict (NOT row dicts).

        Args:
            source: Path to the source file.

        Returns:
            A tuple of (metadata_dict, column_dict):
            - metadata_dict: File-specific metadata
            - column_dict: Dict[str, np.ndarray | pa.Array] for
              ``EventData.from_arrays()``.

        Raises:
            FileNotFoundError: If the source file doesn't exist.
            ValueError: If required columns are missing.
        """
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        # Step 1: Single file read (vectorized I/O)
        df = self._read_dataframe(source)

        # Store the faithful original table (before any field extraction).
        self._raw_table = self._dataframe_to_arrow(df)

        # Step 2: Validate required columns exist
        self._validate_columns(df, source)

        # Step 3: Extract canonical columns + column_specs (vectorized)
        columns = self._extract_column_arrays(df)

        # Step 3b: Apply field_specs (Step 2) — semantic decorations.
        self._apply_field_specs(columns, getattr(self, "_emitted_fields", {}))

        # Step 3c: Post-process hook (subclass extension point).
        self._post_process_columns(df, columns)

        # Step 4: Build metadata
        metadata = {
            "format": self._infer_format(source),
            "delimiter": self.delimiter,
            "row_count": len(df),
            "columns": list(df.columns),
        }

        return metadata, columns

    # endregion

    # region DataFrame Processing

    def _read_dataframe(self, source: Path) -> pd.DataFrame:
        """Read source file into a DataFrame.

        Subclasses can override for custom reading logic.

        Args:
            source: Path to the source file.

        Returns:
            DataFrame containing the tabular data.
        """
        return pd.read_csv(
            source,
            sep=self.delimiter,
            header=self.header_row,
            encoding=self.encoding,
        )

    def _validate_columns(self, df: pd.DataFrame, source: Path) -> None:
        """Validate that required columns exist.

        Handles string column names, Field references, and ComputedField.
        For Field/ComputedField, validates that required source columns exist.

        ``start_column`` may name either a source column (read from
        ``df``) or a synthesised column produced by ``column_specs``;
        synthesised names are not checked here and are resolved later in
        :meth:`_extract_column_arrays`.

        Args:
            df: The loaded DataFrame.
            source: Path to source (for error messages).

        Raises:
            ValueError: If required columns are missing.
        """
        start_col = self.start_column

        # For Field, validate the parent column exists
        if isinstance(start_col, Field):
            if start_col.column not in df.columns:
                raise ValueError(
                    f"Column '{start_col.column}' for Field not found in {source}. "
                    f"Available columns: {list(df.columns)}"
                )
        elif isinstance(start_col, tuple):
            # Tuple is shorthand for Field
            if start_col[0] not in df.columns:
                raise ValueError(
                    f"Column '{start_col[0]}' for tuple field not found in {source}. "
                    f"Available columns: {list(df.columns)}"
                )
        elif isinstance(start_col, ComputedField):
            # ComputedField is validated during computation
            pass
        elif isinstance(start_col, str) and start_col not in df.columns:
            # Defer if column_specs may synthesise the column.
            if start_col in self._column_spec_output_names():
                return
            # Check fallback
            if not (
                self._fallback_start_column
                and self._fallback_start_column in df.columns
            ):
                raise ValueError(
                    f"Required column '{start_col}' not found in {source}. "
                    f"Available columns: {list(df.columns)}"
                )

    def _column_spec_output_names(self) -> set[str]:
        """Names produced by ``column_specs`` (top-level + composite parts).

        Used by :meth:`_validate_columns` to permit ``start_column``
        references to synthesised columns.  Composite parts are
        included so a loader can route ``start_column`` to a part name
        when that part should drive the canonical start coordinate.
        """
        names: set[str] = set()
        specs = self.column_specs
        if specs is None:
            return names
        # Resolve through resolve_field_parser so dict keys seed
        # blueprint names consistently with _resolved_column_specs.
        if isinstance(specs, dict):
            for key, raw in specs.items():
                producer = resolve_field_parser(raw, default_name=key)
                name = _producer_name(producer)
                names.add(name if name is not None else key)
                if isinstance(producer, CompositeFieldParser):
                    names.update(producer.part_keys)
        else:
            for raw in specs:
                producer = resolve_field_parser(raw)
                name = _producer_name(producer)
                if name is not None:
                    names.add(name)
                if isinstance(producer, CompositeFieldParser):
                    names.update(producer.part_keys)
        return names

    def _extract_column_arrays(
        self, df: pd.DataFrame
    ) -> dict[str, np.ndarray | pa.Array]:
        """Extract canonical column arrays + run Step 1 (VECTORIZED).

        NO ROW ITERATION.  All operations use vectorized
        numpy / pandas / pyarrow.

        Order:

        1. ``column_specs`` (Step 1) — emits typed DataFields into
           ``columns``.  Each emitted field's data array is also
           exposed under its part-key name so canonical
           ``start_column`` / ``duration_column`` references can route
           through synthesised columns.
        2. Canonical columns — id, name, start, end / duration,
           event_type, temporal_type.  ``start_column`` references may
           resolve against either a source-DataFrame column or a
           Step-1 synthesised column.

        Args:
            df: The loaded DataFrame.

        Returns:
            Dict of {column_name: array} ready for
            ``EventData.from_arrays()``.
        """

        n = len(df)
        columns: dict[str, Any] = {}

        # Reset extra schema fields for this load.
        self._extra_schema_fields = []

        # Step 1: column_specs — typed DataField emission.  Run first
        # so that canonical start_column / duration_column references
        # can route through synthesised columns.
        self._emitted_fields = self._apply_column_specs(df, columns)
        # Expose composite-part keys directly as columns so
        # start_column="<part_name>" works.  Only opaque struct fields
        # (CompositeFieldParser emissions, NOT RedundantNumberField / numeric
        # struct emissions which are atomic semantic units) get their
        # sub-fields surfaced as top-level columns.
        from timetoalign.core.fields import (
            StructField,
        )
        from timetoalign.core.time import (
            RedundantNumberField,
        )

        for emit_name, df_field in self._emitted_fields.items():
            if isinstance(df_field, RedundantNumberField):
                continue
            if not isinstance(df_field, StructField) or isinstance(
                df_field, RedundantNumberField
            ):
                continue
            # Skip if the struct is a SemanticField's inner storage —
            # SemanticField subclasses are atomic semantic units.
            if isinstance(df_field, SemanticField):
                continue
            for sub_name in df_field.field_names:
                if sub_name not in columns:
                    sub_field = df_field.get_sub_field(sub_name)
                    if sub_field.data is not None:
                        columns[sub_name] = sub_field.data
                        # Track the sub-field in the schema with a
                        # ``field_type`` blob so it is preserved under
                        # ``properties=False`` filtering.
                        self._extra_schema_fields.append(
                            _merge_field_type_metadata(
                                sub_field.field, type(sub_field).__name__
                            )
                        )

        # ID column (vectorized generation if not present)
        if self.id_column and self.id_column in df.columns:
            columns["id"] = df[self.id_column].astype(str).to_numpy()
        else:
            columns["id"] = np.array([f"e{i:06d}" for i in range(n)])

        # Name column (vectorized extraction)
        if self.name_column and self.name_column in df.columns:
            name_series = df[self.name_column]
            name_arr = name_series.astype(str).to_numpy()
            null_mask = name_series.isna().to_numpy()
            name_arr[null_mask] = None  # type: ignore[assignment]
            columns["name"] = name_arr
        else:
            columns["name"] = np.array([None] * n, dtype=object)

        # Extract start coordinate (may reference a synthesised column).
        start_values = self._resolve_start_or_duration(
            self.start_column, df, columns, "start"
        )
        columns["start"] = build_number_struct_array(
            start_values, number_type=self._number_type
        )

        # Extract end coordinate.
        columns["end"] = self._extract_end_column(df, self._raw_table, columns, n)

        # Temporal type (vectorized)
        if isinstance(columns["end"], pa.StructArray):
            has_end_arr = ~columns["end"].is_null().to_numpy(zero_copy_only=False)
        else:
            has_end_arr = np.zeros(n, dtype=bool)
        columns["temporal_type"] = np.where(has_end_arr, "interval", "instant")

        # Event type (vectorized extraction or default)
        if self.event_type_column and self.event_type_column in df.columns:
            event_type_series = df[self.event_type_column]
            columns["event_type"] = np.where(
                event_type_series.isna(),
                self.default_event_type,
                event_type_series.astype(str).to_numpy(),
            )
        else:
            columns["event_type"] = np.full(n, self.default_event_type, dtype=object)

        # Property columns — every source column NOT consumed by
        # column_specs or by a canonical reference is propagated as a
        # raw, semantically opaque pa.Array.  ``get_events(properties=)``
        # then controls which of these survive into the final
        # EventData table.  No metadata is attached: these are
        # unconsumed source columns, not fields.
        self._propagate_property_columns(df, columns)

        return columns

    def _propagate_property_columns(
        self, df: pd.DataFrame, columns: dict[str, Any]
    ) -> None:
        """Copy unconsumed source columns into *columns* as raw pa.Arrays.

        A source column is "consumed" iff it backed an entry in
        ``column_specs`` (tracked by :meth:`_apply_column_specs` via
        ``self._consumed_source_columns``) OR it was referenced by a
        canonical loader attribute (``id_column`` / ``name_column`` /
        ``start_column`` / ``end_column`` / ``duration_column`` /
        ``event_type_column``).  Every other source column is added to
        *columns* under its original name.

        No TTA metadata is attached — unconsumed source
        columns are property columns, not fields, and
        ``get_events(properties=False)`` is expected to drop them.
        """
        consumed: set[str] = set(getattr(self, "_consumed_source_columns", set()))
        for attr in (
            "id_column",
            "name_column",
            "event_type_column",
        ):
            value = getattr(self, attr, None)
            if isinstance(value, str):
                consumed.add(value)
        for attr in ("start_column", "end_column", "duration_column"):
            value = getattr(self, attr, None)
            if isinstance(value, str):
                consumed.add(value)
            elif isinstance(value, tuple) and value and isinstance(value[0], str):
                consumed.add(value[0])
            elif isinstance(value, Field):
                consumed.add(value.column)
            # ComputedField references are opaque; we don't attempt to
            # introspect them — any source columns it reads will simply
            # appear as property columns alongside the computed result.

        for col in df.columns:
            name = str(col)
            if name in columns:
                continue
            if name in consumed:
                continue
            columns[name] = pa.array(df[col])

    def _resolve_start_or_duration(
        self,
        col_ref: str | tuple | Field | ComputedField,
        df: pd.DataFrame,
        columns: dict[str, Any],
        context: str,
    ) -> np.ndarray:
        """Resolve a coordinate column reference.

        Like :meth:`_resolve_column_reference`, but additionally
        recognises string references to columns produced by
        ``column_specs``.  When the column is a synthesised rational
        struct, the resolver extracts a numpy float-equivalent
        (computed from numerator / denominator) so the number builder
        retains the exact rational representation.
        """
        if (
            isinstance(col_ref, str)
            and col_ref not in df.columns
            and col_ref in columns
        ):
            arr = columns[col_ref]
            if isinstance(arr, pa.StructArray) and {f.name for f in arr.type} >= {
                "value",
                "numerator",
                "denominator",
            }:
                num = arr.field("numerator").to_numpy(zero_copy_only=False)
                den = arr.field("denominator").to_numpy(zero_copy_only=False)
                from fractions import Fraction

                # Build a Python-object array of Fraction instances so
                # The builder preserves the exact rational form.
                out = np.empty(len(arr), dtype=object)
                for i in range(len(arr)):
                    if num[i] is None or den[i] is None:
                        out[i] = None
                    else:
                        out[i] = Fraction(int(num[i]), int(den[i]))
                return out
            if isinstance(arr, (pa.Array, pa.ChunkedArray)):
                return arr.to_numpy(zero_copy_only=False)
            return np.asarray(arr)
        return self._resolve_column_reference(col_ref, df, self._raw_table, context)

    def _post_process_columns(self, df: pd.DataFrame, columns: dict[str, Any]) -> None:
        """Post-process extracted columns before table construction.

        Subclass hook called after ``_extract_column_arrays()`` and
        ``_apply_field_specs()``.  Override to inject domain-specific
        fields not expressible in ``column_specs`` / ``field_specs``.

        Args:
            df: The original DataFrame.
            columns: The mutable column dict that will be passed to
                ``EventData.from_arrays()``.
        """

    def _resolve_column_reference(
        self,
        col_ref: str | tuple | Field | ComputedField,
        df: pd.DataFrame,
        temp_table: pa.Table | None,
        context: str,
    ) -> np.ndarray:
        """Resolve a column reference to a numpy array of values.

        Handles:
        - str: Direct DataFrame column lookup
        - tuple: Convert to Field and resolve
        - Field: Struct field access via PyArrow
        - ComputedField: Compute from formula/expression

        Args:
            col_ref: The column reference to resolve.
            df: The source DataFrame.
            temp_table: PyArrow table for Field/ComputedField resolution.
            context: Context name for error messages (e.g., "start", "end").

        Returns:
            Numpy array of values.
        """
        if isinstance(col_ref, str):
            if col_ref in df.columns:
                return df[col_ref].to_numpy()
            elif self._fallback_start_column and context == "start":
                if self._fallback_start_column in df.columns:
                    return df[self._fallback_start_column].to_numpy()
            raise ValueError(
                f"Column '{col_ref}' not found for {context}. "
                f"Available: {list(df.columns)}"
            )

        elif isinstance(col_ref, tuple):
            col_ref = Field(col_ref[0], *col_ref[1:])

        if isinstance(col_ref, Field):
            if temp_table is None:
                raise ValueError(
                    f"Cannot resolve Field reference for {context}: source "
                    "table not yet materialised"
                )
            array = col_ref.resolve(temp_table)
            return array.to_numpy(zero_copy_only=False)

        elif isinstance(col_ref, ComputedField):
            if temp_table is None:
                raise ValueError(
                    f"Cannot resolve ComputedField for {context}: source "
                    "table not yet materialised"
                )
            array = col_ref.compute(temp_table)
            return array.to_numpy(zero_copy_only=False)

        else:
            raise TypeError(
                f"Invalid column reference type for {context}: {type(col_ref)}"
            )

    def _extract_end_column(
        self,
        df: pd.DataFrame,
        temp_table: pa.Table | None,
        columns: dict[str, Any],
        n: int,
    ) -> pa.Array:
        """Extract the end column, handling various source types.

        Handles:
        - Field/tuple/ComputedField references
        - Direct column names
        - Duration column (computes end = start + duration)
        - None (instant events)

        Args:
            df: Source DataFrame.
            temp_table: PyArrow table for Field resolution.
            columns: Already extracted columns (includes "start").
            n: Number of rows.

        Returns:
            PyArrow array for the end column.
        """
        end_col_ref = self.end_column

        # Case 1: No end column (instant events)
        if end_col_ref is None and self.duration_column is None:
            return pa.nulls(n, type=columns["start"].type)

        # Case 2: Field, tuple, or ComputedField reference
        if isinstance(end_col_ref, (Field, tuple, ComputedField)):
            end_values = self._resolve_column_reference(
                end_col_ref, df, temp_table, "end"
            )
            return build_number_struct_array(end_values, number_type=self._number_type)

        # Case 3: Direct column name
        if isinstance(end_col_ref, str) and end_col_ref in df.columns:
            end_series = df[end_col_ref]
            has_end_values = end_series.notna()
            if bool(has_end_values.any()):
                end_values = end_series.to_numpy()
                valid_mask = ~pd.isna(end_values)
                if valid_mask.all():
                    return build_number_struct_array(
                        end_values, number_type=self._number_type
                    )
                else:
                    return self._parse_nullable_coordinates(end_values, valid_mask)
            else:
                return pa.nulls(n, type=columns["start"].type)

        # Case 4: Duration column (compute end = start + duration).
        # Also accept a synthesised duration column from column_specs.
        if self.duration_column and (
            self.duration_column in df.columns or self.duration_column in columns
        ):
            return self._compute_end_from_duration(df, columns, n)

        # Case 5: end_column specified but not found, and no duration
        if end_col_ref is not None:
            return pa.nulls(n, type=columns["start"].type)

        # Default: instant events
        return pa.nulls(n, type=columns["start"].type)

    def _compute_end_from_duration(
        self,
        df: pd.DataFrame,
        columns: dict[str, Any],
        n: int,
    ) -> pa.Array:
        """Compute end column from start + duration (FULLY VECTORIZED).

        The duration source may be a raw DataFrame column or a
        synthesised column (rational struct or scalar) produced by
        ``column_specs``.

        Args:
            df: Source DataFrame.
            columns: Already extracted columns (includes "start").
            n: Number of rows.

        Returns:
            PyArrow array for the end column.
        """
        # Resolve the duration source: DataFrame column or synthesised
        # column from column_specs.
        if self.duration_column in df.columns:
            dur_series = df[self.duration_column]
            has_dur_values = dur_series.notna()
            if not bool(has_dur_values.any()):
                return pa.nulls(n, type=columns["start"].type)
            dur_values = dur_series.to_numpy()
        else:
            synth = columns[self.duration_column]
            if isinstance(synth, pa.StructArray) and {f.name for f in synth.type} >= {
                "value",
                "numerator",
                "denominator",
            }:
                num = synth.field("numerator").to_numpy(zero_copy_only=False)
                den = synth.field("denominator").to_numpy(zero_copy_only=False)
                from fractions import Fraction

                dur_values = np.empty(n, dtype=object)
                for i in range(n):
                    if pd.isna(num[i]) or pd.isna(den[i]):
                        dur_values[i] = None
                    else:
                        dur_values[i] = Fraction(int(num[i]), int(den[i]))
            elif isinstance(synth, (pa.Array, pa.ChunkedArray)):
                dur_values = synth.to_numpy(zero_copy_only=False)
            else:
                dur_values = np.asarray(synth)
            if all(v is None for v in dur_values):
                return pa.nulls(n, type=columns["start"].type)

        # Compute valid masks (vectorized)
        start_is_null = columns["start"].is_null().to_numpy(zero_copy_only=False)
        valid_dur_mask = ~pd.isna(dur_values)
        valid_end_mask = ~start_is_null & valid_dur_mask

        # Parse duration through the one number builder
        valid_dur_values = dur_values[valid_dur_mask]
        parsed_dur = build_number_struct_array(
            valid_dur_values, number_type=self._number_type
        )

        # Build full duration arrays using scatter (vectorized)
        dur_value_full = np.full(n, np.nan, dtype=np.float64)
        dur_num_full = np.zeros(n, dtype=np.int64)
        dur_den_full = np.ones(n, dtype=np.int64)

        valid_indices = np.where(valid_dur_mask)[0]
        dur_value_full[valid_indices] = parsed_dur.field("value").to_numpy(
            zero_copy_only=False
        )

        # Extract numerator/denominator arrays from parsed (vectorized)
        parsed_num_arr = parsed_dur.field("numerator").to_numpy(zero_copy_only=False)
        parsed_den_arr = parsed_dur.field("denominator").to_numpy(zero_copy_only=False)

        num_valid = ~pd.isna(parsed_num_arr)
        den_valid = ~pd.isna(parsed_den_arr)
        pair_valid = num_valid & den_valid

        if num_valid.any():
            dur_num_full[valid_indices[num_valid]] = parsed_num_arr[num_valid].astype(
                np.int64
            )
        if den_valid.any():
            dur_den_full[valid_indices[den_valid]] = parsed_den_arr[den_valid].astype(
                np.int64
            )

        # Get start arrays (vectorized)
        start_value = columns["start"].field("value").to_numpy(zero_copy_only=False)
        start_num = columns["start"].field("numerator").to_numpy(zero_copy_only=False)
        start_den = columns["start"].field("denominator").to_numpy(zero_copy_only=False)

        # Compute end = start + duration (vectorized)
        end_value = start_value + dur_value_full

        # Vectorized fraction addition: a/b + c/d = (a*d + c*b) / (b*d)
        start_has_frac = ~pd.isna(start_num) & ~pd.isna(start_den)
        dur_has_frac = np.zeros(n, dtype=bool)
        dur_has_frac[valid_indices] = pair_valid
        both_frac_mask = start_has_frac & dur_has_frac & valid_end_mask

        s_num = np.where(pd.isna(start_num), 0, start_num).astype(np.int64)
        s_den = np.where(pd.isna(start_den), 1, start_den).astype(np.int64)
        d_num = dur_num_full
        d_den = dur_den_full

        result_num = s_num * d_den + d_num * s_den
        result_den = s_den * d_den

        end_num = np.zeros(n, dtype=np.int64)
        end_den = np.ones(n, dtype=np.int64)

        if both_frac_mask.any():
            gcd_vals = np.gcd(result_num[both_frac_mask], result_den[both_frac_mask])
            end_num[both_frac_mask] = result_num[both_frac_mask] // gcd_vals
            end_den[both_frac_mask] = result_den[both_frac_mask] // gcd_vals

        # Build coordinate struct arrays (vectorized)
        coord_type = pa.struct(
            [
                pa.field("value", pa.float64(), nullable=True),
                pa.field("numerator", pa.int64(), nullable=True),
                pa.field("denominator", pa.int64(), nullable=True),
            ]
        )

        end_null_mask = ~valid_end_mask
        end_num_null_mask = ~both_frac_mask

        # Also store duration in columns
        dur_null_mask = ~valid_dur_mask
        dur_num_has_pair = np.zeros(n, dtype=bool)
        dur_num_has_pair[valid_indices] = pair_valid
        dur_num_null_full = ~dur_num_has_pair

        columns["duration"] = pa.StructArray.from_arrays(
            [
                pa.array(dur_value_full, mask=dur_null_mask, type=pa.float64()),
                pa.array(dur_num_full, mask=dur_num_null_full, type=pa.int64()),
                pa.array(dur_den_full, mask=dur_num_null_full, type=pa.int64()),
            ],
            fields=list(coord_type),
            mask=pa.array(dur_null_mask),
        )

        return pa.StructArray.from_arrays(
            [
                pa.array(end_value, mask=end_null_mask, type=pa.float64()),
                pa.array(end_num, mask=end_num_null_mask, type=pa.int64()),
                pa.array(end_den, mask=end_num_null_mask, type=pa.int64()),
            ],
            fields=list(coord_type),
            mask=pa.array(end_null_mask),
        )

    def _parse_nullable_coordinates(
        self, values: np.ndarray, valid_mask: np.ndarray
    ) -> pa.StructArray:
        """Parse coordinate array with null values (FULLY VECTORIZED).

        NO ROW ITERATION. Uses numpy advanced indexing and PyArrow
        array construction with proper null masks.

        Args:
            values: Array of coordinate values (may contain nulls).
            valid_mask: Boolean mask indicating valid (non-null) values.

        Returns:
            PyArrow StructArray with nulls where mask is False.
        """
        n = len(values)
        coord_type = pa.struct(
            [
                pa.field("value", pa.float64(), nullable=True),
                pa.field("numerator", pa.int64(), nullable=True),
                pa.field("denominator", pa.int64(), nullable=True),
            ]
        )

        if not valid_mask.any():
            return pa.nulls(n, type=coord_type)

        # Parse valid values (vectorized)
        valid_values = values[valid_mask]
        parsed = build_number_struct_array(valid_values, number_type=self._number_type)

        # Extract parsed arrays (vectorized)
        parsed_value = parsed.field("value").to_numpy(zero_copy_only=False)
        parsed_num = parsed.field("numerator").to_numpy(zero_copy_only=False)
        parsed_den = parsed.field("denominator").to_numpy(zero_copy_only=False)

        # Build full arrays with proper defaults (vectorized)
        value_arr = np.full(n, np.nan, dtype=np.float64)
        num_arr = np.zeros(n, dtype=np.int64)
        den_arr = np.ones(n, dtype=np.int64)

        # Scatter valid values using boolean indexing (vectorized)
        valid_indices = np.where(valid_mask)[0]
        value_arr[valid_indices] = parsed_value

        num_valid_in_parsed = ~pd.isna(parsed_num)
        den_valid_in_parsed = ~pd.isna(parsed_den)

        if num_valid_in_parsed.any():
            target_indices = valid_indices[num_valid_in_parsed]
            num_arr[target_indices] = parsed_num[num_valid_in_parsed].astype(np.int64)
        if den_valid_in_parsed.any():
            target_indices = valid_indices[den_valid_in_parsed]
            den_arr[target_indices] = parsed_den[den_valid_in_parsed].astype(np.int64)

        struct_null_mask = ~valid_mask

        pair_valid = num_valid_in_parsed & den_valid_in_parsed
        num_has_pair = np.zeros(n, dtype=bool)
        num_has_pair[valid_indices] = pair_valid
        num_null_mask = ~num_has_pair

        value_pa = pa.array(value_arr, mask=struct_null_mask, type=pa.float64())
        num_pa = pa.array(num_arr, mask=num_null_mask, type=pa.int64())
        den_pa = pa.array(den_arr, mask=num_null_mask, type=pa.int64())

        return pa.StructArray.from_arrays(
            [value_pa, num_pa, den_pa],
            fields=list(coord_type),
            mask=pa.array(struct_null_mask),
        )

    # endregion

    # region Utilities

    def _infer_format(self, source: Path) -> str:
        """Infer the format name from the file extension.

        Args:
            source: Path to the source file.

        Returns:
            Format name string.
        """
        ext = source.suffix.lower()
        if ext == ".csv":
            return "csv"
        elif ext in (".tsv", ".txt"):
            return "tsv"
        elif ext == ".parquet":
            return "parquet"
        else:
            return ext.lstrip(".")

    # endregion

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"delimiter={self.delimiter!r}, "
            f"start_column={self.start_column!r}, "
            f"unit={self._unit})"
        )


# endregion
