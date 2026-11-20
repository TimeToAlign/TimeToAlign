"""Semantic field access mixins for EventData subclasses.

Provides three levels of field access on ``EventData``:

- **``events.table``** -- direct access to the underlying ``pa.Table``
  (specialist use only).
- **``events.get_raw("name")``** -- returns a raw ``DataField``
  (``StructField``, ``NumericField``, etc.) wrapping the field.
- **``events.get_field(...)``** -- returns a ``SemanticField`` wrapping
  the field with semantic identity and typed scalar access.

``get_field()`` accepts four argument forms:

1. ``get_field("start")`` -- string field name; looks up metadata or
   default-field mapping to determine the ``SemanticField`` subclass.
2. ``get_field(EnharmonicPitchField)`` -- a paired ``SemanticField``
   class; scans all fields for metadata or shape matching that class.
3. ``get_field(EnharmonicPitch)`` -- a **pydantic scalar class**;
   resolves to the matching paired ``SemanticField`` and dispatches via
   form (2).  ``get_field(ScalarClass, name="col")`` disambiguates when
   more than one column holds that scalar.
4. ``get_field(EnharmonicPitchField(source_fields="midi_pitch"))`` --
   a *blueprint* instance; resolves the named source field on the
   table (or a dict-shaped spec for multi-field blueprints) and caches
   the result.

Pitch access is uniform: ``get_pitch_field(type, format=)`` lives on the
base :class:`SemanticFieldAccessMixin` and returns the single
most-expressive pitch field an EventData affords (including fields
afforded over a raw atomic column via ``_afforded_fields``).
The remaining domain mixin adds its own convenience accessor:

- ``HarmonyAccessMixin`` -- ``get_harmony_field(type, format=)``
"""

from __future__ import annotations

import enum
import json
from typing import Any

import pyarrow as pa
from pydantic import BaseModel

from timetoalign.core.fields import (
    TIMETOALIGN_METADATA_KEY,
    DataField,
    MapField,
    NumericField,
    SemanticField,
    StringField,
    StructField,
    _GenericField,
    parse_metadata_blob,
)


class MultipleFieldsError(KeyError):
    """Raised when ``get_field(ScalarClass)`` matches more than one column.

    Carries the offending scalar class name and the list of matching
    column names in the message; the caller can resolve the ambiguity
    by passing ``name="<col>"`` to ``get_field``.
    """


def _are_interchangeable_pitch_views(
    declared: type[SemanticField[Any]], requested: type[SemanticField[Any]]
) -> bool:
    """Return whether two field classes are MIDI-number pitch views."""
    from timetoalign.core.events import EnharmonicPitchField, MidiPitchField

    return {declared, requested} == {EnharmonicPitchField, MidiPitchField}


# ---------------------------------------------------------------------------
# DefaultField -- core temporal fields present on every EventData
# ---------------------------------------------------------------------------


class DefaultField(enum.Enum):
    """Core temporal fields present on every EventData table.

    These map field names to the ``SemanticField`` subclass that
    should wrap them.  ``start`` and ``end`` are ``CoordinateField``;
    ``duration`` is ``DurationField``.
    """

    start = "start"
    end = "end"
    duration = "duration"


# ---------------------------------------------------------------------------
# Field-type string -> class mapping (lazy-loaded to avoid circular imports)
# ---------------------------------------------------------------------------

_FIELD_TYPE_MAP: dict[str, type[SemanticField[Any]]] | None = None


def _get_field_type_map() -> dict[str, type[SemanticField[Any]]]:
    """Return the field_type string -> class mapping, building it on first call.

    Each paired ``XField(SemanticField[X])`` owns exactly one entry here.
    No umbrella classes, no legacy aliases.
    """
    global _FIELD_TYPE_MAP
    if _FIELD_TYPE_MAP is not None:
        return _FIELD_TYPE_MAP

    from timetoalign.core.events import (
        BeatField,
        BeatPolicyField,
        DcmlHarmonyField,
        EnharmonicPitchClassField,
        EnharmonicPitchField,
        GenericPitchClassField,
        GenericPitchField,
        HarmonyLabelField,
        IdField,
        MeasureField,
        MeasureNumberField,
        MidiPitchField,
        NoteField,
        PitchBasedHarmonyField,
        RomanNumeralHarmonyField,
        SpecificPitchClassField,
        SpecificPitchField,
        WesternTertianHarmonyField,
    )
    from timetoalign.core.time import (
        CoordinateField,
        DurationField,
        IdCoordinateField,
        IdDurationField,
    )

    _FIELD_TYPE_MAP = {
        # Coordinate / Duration (plus Id-variants)
        "CoordinateField": CoordinateField,
        "DurationField": DurationField,
        "IdCoordinateField": IdCoordinateField,
        "IdDurationField": IdDurationField,
        # Pitch — paired classes only
        "EnharmonicPitchField": EnharmonicPitchField,
        "EnharmonicPitchClassField": EnharmonicPitchClassField,
        "GenericPitchField": GenericPitchField,
        "GenericPitchClassField": GenericPitchClassField,
        "MidiPitchField": MidiPitchField,
        "SpecificPitchField": SpecificPitchField,
        "SpecificPitchClassField": SpecificPitchClassField,
        # Harmony — paired classes only
        "DcmlHarmonyField": DcmlHarmonyField,
        "HarmonyLabelField": HarmonyLabelField,
        "PitchBasedHarmonyField": PitchBasedHarmonyField,
        "RomanNumeralHarmonyField": RomanNumeralHarmonyField,
        "WesternTertianHarmonyField": WesternTertianHarmonyField,
        # Event scalars
        "BeatField": BeatField,
        "BeatPolicyField": BeatPolicyField,
        "IdField": IdField,
        "MeasureField": MeasureField,
        "MeasureNumberField": MeasureNumberField,
        "NoteField": NoteField,
    }
    return _FIELD_TYPE_MAP


_SCALAR_TO_FIELD_MAP: dict[type[BaseModel], type[SemanticField[Any]]] | None = None


def _get_scalar_to_field_map() -> dict[type[BaseModel], type[SemanticField[Any]]]:
    """Return the pydantic-scalar-class → paired Field-class mapping.

    Derived from :func:`_get_field_type_map` on first call.  Each entry
    is ``scalar_cls`` (the pydantic ``BaseModel`` subclass) → its paired
    ``XField`` class.
    """
    global _SCALAR_TO_FIELD_MAP
    if _SCALAR_TO_FIELD_MAP is not None:
        return _SCALAR_TO_FIELD_MAP
    mapping: dict[type[BaseModel], type[SemanticField[Any]]] = {}
    for field_cls in _get_field_type_map().values():
        scalar = field_cls.scalar_cls
        if scalar is None:
            continue
        # First registration wins; later entries (for shared scalars)
        # do not overwrite.  Coordinate / IdCoordinate are distinct
        # scalars so no collision occurs there.
        mapping.setdefault(scalar, field_cls)
    # Manually connect the TimeScalar pair (CoordinateField / DurationField
    # ship via core.time and are already in the field-type map).  Also
    # connect IdCoordinate / IdDuration, which carry distinct scalar_cls
    # but share a struct shape with their non-Id parents.
    from timetoalign.core.time import (
        CoordinateField,
        DurationField,
        IdCoordinate,
        IdCoordinateField,
        IdDuration,
        IdDurationField,
    )

    mapping[IdCoordinate] = IdCoordinateField
    mapping[IdDuration] = IdDurationField
    # CoordinateField / DurationField already registered via the
    # _get_field_type_map path.  Confirm presence (defensive).
    mapping.setdefault(CoordinateField.scalar_cls, CoordinateField)
    mapping.setdefault(DurationField.scalar_cls, DurationField)
    _SCALAR_TO_FIELD_MAP = mapping
    return mapping


def _scalar_satisfies_protocol(scalar_cls: type[BaseModel], protocol: type) -> bool:
    """Return True iff *scalar_cls* satisfies the runtime-checkable *protocol*.

    Pydantic scalars cannot be probed via ``issubclass`` against
    Protocols carrying properties — Python rejects such checks at
    runtime (``"Protocols with non-method members don't support
    issubclass()"``).  Instead, instantiate a sample scalar via
    ``model_construct`` (bypasses validation) populated with field
    defaults / zero values, then run ``isinstance`` against the
    Protocol — which IS supported for runtime_checkable Protocols.
    """
    try:
        # Build a sample scalar using zero/empty defaults for each
        # pydantic field.  ``model_construct`` bypasses validation, so
        # any placeholder value is acceptable.
        sample_kwargs: dict[str, Any] = {}
        for name, info in scalar_cls.model_fields.items():
            if info.is_required():
                sample_kwargs[name] = _zero_for_annotation(info.annotation)
        sample = scalar_cls.model_construct(**sample_kwargs)
    except (TypeError, ValueError):
        return False
    try:
        return isinstance(sample, protocol)
    except TypeError:
        return False


def _zero_for_annotation(annotation: Any) -> Any:
    """Return a zero-shaped placeholder consistent with *annotation*."""
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is bool:
        return False
    if annotation is str:
        return ""
    # Fall back to None — model_construct accepts it without checking
    # the annotation.
    return None


def _get_field_class_for_scalar(
    scalar_cls: type[BaseModel],
) -> type[SemanticField[Any]]:
    """Return the paired ``XField`` class for a pydantic scalar class."""
    mapping = _get_scalar_to_field_map()
    field_cls = mapping.get(scalar_cls)
    if field_cls is not None:
        return field_cls
    raise KeyError(
        f"No paired SemanticField registered for scalar {scalar_cls.__name__!r}. "
        f"Known scalars: {sorted(c.__name__ for c in mapping)}"
    )


def _parse_field_type_metadata(pa_field: pa.Field) -> str | None:
    """Extract the ``field_type`` string from a PyArrow field's metadata.

    Returns ``None`` when the field carries no TimeToAlign! blob at
    all, or when the blob is not decodable JSON.  A blob that *is*
    readable but violates the payload contract (missing or too-new
    ``version``) propagates the ``ValueError`` raised by
    :func:`~timetoalign.core.fields.parse_metadata_blob`.
    """
    meta = pa_field.metadata
    if not meta:
        return None
    blob = meta.get(TIMETOALIGN_METADATA_KEY)
    if blob is None:
        return None
    try:
        parsed = parse_metadata_blob(blob)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        return None
    return parsed.get("field_type")


def _reconstruct_field(
    pa_field: pa.Field,
    col: pa.ChunkedArray | pa.Array | None,
    field_type_str: str,
) -> SemanticField[Any]:
    """Reconstruct a ``SemanticField`` subclass from field data and metadata."""
    registry = _get_field_type_map()
    cls = registry.get(field_type_str)
    if cls is None:
        raise KeyError(
            f"Unknown field_type {field_type_str!r} in metadata. "
            f"Known types: {sorted(registry)}"
        )
    return cls.from_field((col, pa_field))


# ---------------------------------------------------------------------------
# Schema-based field discovery (fallback when no metadata is present)
# ---------------------------------------------------------------------------


def _discover_by_schema(
    table: pa.Table,
    field_type: type[SemanticField[Any]],
    cache: dict[tuple[str, str], SemanticField[Any]],
    registry: dict[str, type[SemanticField[Any]]],
) -> list[SemanticField[Any]]:
    """Try to match *field_type* against table fields by schema structure.

    Each concrete ``SemanticField`` subclass may declare a
    ``_default_field_name`` attribute (the expected field name) and optionally
    a ``_expected_schema`` (the PyArrow struct type to match).  When the
    table contains a field with a matching name and compatible struct
    type, the field is constructed via ``from_field()`` and cached.

    This function also checks registered entries that are subclasses of
    *field_type*.

    Args:
        table: The PyArrow table to search.
        field_type: The ``SemanticField`` subclass (or parent) to match.
        cache: The field cache dict to populate.
        registry: The field type string → class mapping.

    Returns:
        A list of discovered fields (may be empty).
    """
    result: list[SemanticField[Any]] = []
    field_names = set(table.column_names)

    # Collect candidate classes: the field_type itself + any registered
    # subclasses that are subclasses of field_type.
    candidates: list[type[SemanticField[Any]]] = []
    if hasattr(field_type, "_default_field_name"):
        candidates.append(field_type)
    for cls in registry.values():
        if cls is field_type:
            continue
        if issubclass(cls, field_type) and hasattr(cls, "_default_field_name"):
            if cls not in candidates:
                candidates.append(cls)

    for cls in candidates:
        field_name = cls._default_field_name  # type: ignore[attr-defined]
        if field_name not in field_names:
            continue

        cache_key = (field_name, cls.__name__)
        cached = cache.get(cache_key)
        if cached is not None:
            result.append(cached)
            continue

        arr = table.column(field_name)
        pa_field = table.schema.field(field_name)

        # Verify the field is a struct (SemanticFields wrap struct arrays)
        if not pa.types.is_struct(pa_field.type):
            continue

        try:
            field = cls.from_field((arr, pa_field))
        except (TypeError, KeyError, ValueError):
            continue
        cache[cache_key] = field
        result.append(field)

    return result


def _discover_by_shape(
    table: pa.Table,
    field_type: type[SemanticField[Any]],
    cache: dict[tuple[str, str], SemanticField[Any]],
) -> list[SemanticField[Any]]:
    """Discover fields by delegating to ``field_type.matches_pa_field()``.

    Iterates the table schema in field order.  For each ``pa.Field`` that
    *field_type* (or one of its subclasses in the registry) claims via
    :py:meth:`SemanticField.matches_pa_field`, constructs the field via
    ``from_field()`` and caches it.

    This is the third-line discovery strategy used by
    :py:meth:`SemanticFieldAccessMixin.get_fields`, after metadata-based
    and ``_default_field_name``-based lookup both yield nothing.  It catches
    fields produced by loaders that have not (yet) injected
    TTA metadata onto raw struct fields
    (e.g. ``midi_pitch`` / ``specific_pitch`` on TSV-loaded score notes).

    Args:
        table: The PyArrow table to search.
        field_type: The ``SemanticField`` subclass to match.
        cache: The field cache dict to populate.

    Returns:
        A list of discovered fields, ordered by their position in
        the table schema.  Empty list if nothing matches.
    """
    result: list[SemanticField[Any]] = []
    schema = table.schema
    for i in range(len(schema)):
        pa_field = schema.field(i)
        if not field_type.matches_pa_field(pa_field):
            continue
        cache_key = (pa_field.name, field_type.__name__)
        cached = cache.get(cache_key)
        if cached is not None:
            result.append(cached)
            continue
        arr = table.column(i)
        try:
            field = field_type.from_field((arr, pa_field))
        except (TypeError, KeyError, ValueError):
            continue
        cache[cache_key] = field
        result.append(field)
    return result


# ---------------------------------------------------------------------------
# Raw field wrapping
# ---------------------------------------------------------------------------


def _wrap_raw(col: pa.ChunkedArray | pa.Array, pa_field: pa.Field) -> DataField:
    """Wrap a field's array in the appropriate raw ``DataField`` subclass."""
    dt = pa_field.type
    if pa.types.is_struct(dt):
        return StructField(col, pa_field)
    if pa.types.is_integer(dt) or pa.types.is_floating(dt):
        return NumericField(col, pa_field)
    if pa.types.is_string(dt) or pa.types.is_large_string(dt):
        return StringField(col, pa_field)
    if pa.types.is_map(dt):
        return MapField(col, pa_field)
    return _GenericField(col, pa_field)


# ---------------------------------------------------------------------------
# Default field resolution (start/end -> CoordinateField, duration -> DurationField)
# ---------------------------------------------------------------------------

_DEFAULT_FIELD_MAP: dict[str, str] | None = None


def _get_default_field_map() -> dict[str, str]:
    """Map default field names to semantic field class names."""
    global _DEFAULT_FIELD_MAP
    if _DEFAULT_FIELD_MAP is None:
        _DEFAULT_FIELD_MAP = {
            DefaultField.start.value: "CoordinateField",
            DefaultField.end.value: "CoordinateField",
            DefaultField.duration.value: "DurationField",
        }
    return _DEFAULT_FIELD_MAP


def _resolve_default_field(
    name: str,
    col: pa.ChunkedArray | pa.Array,
    pa_field: pa.Field,
) -> SemanticField[Any] | None:
    """Try to construct a semantic field for a default (core) field name."""
    mapping = _get_default_field_map()
    ft_str = mapping.get(name)
    if ft_str is None:
        return None
    try:
        return _reconstruct_field(pa_field, col, ft_str)
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# SemanticFieldAccessMixin
# ---------------------------------------------------------------------------


class SemanticFieldAccessMixin:
    """Base mixin for field access on EventData.

    Provides three access levels:

    - ``get_raw("name")`` -- raw ``DataField`` wrapper (no semantics).
    - ``get_field("name")`` -- ``SemanticField`` wrapper (typed scalars).
    - ``get_field(PitchField(ep="col"))`` -- blueprint resolution.

    Expects the host class to provide a ``_table`` attribute.

    **Afforded fields.**  An EventData subclass may declare
    :attr:`_afforded_fields` — a mapping of *raw atomic column name* →
    paired ``SemanticField`` subclass that the column can be promoted to
    on request.  This is how a faithful raw column (e.g. a bare MIDI
    pitch ``int`` from a MIDI / performance source) affords its
    most-expressive semantic view (``EnharmonicPitch``) without storing a
    redundant semantic struct.  The promotion is materialised lazily by
    :meth:`get_fields` (and therefore by ``get_field`` /
    ``get_pitch_field`` / ``get_fields_satisfying``) via the paired
    class's ``from_array()``, and cached.  The raw column stays raw and
    queryable; the affordance is reached only when asked for.
    """

    _table: pa.Table  # provided by EventData

    # Maps a raw atomic column name to the paired SemanticField subclass
    # the EventData affords over it (see the class docstring).  Empty on
    # the base — subclasses that carry a bare-number pitch column (or any
    # other raw column with a most-expressive semantic view) declare it.
    _afforded_fields: dict[str, type[SemanticField[Any]]] = {}

    @property
    def _field_cache(self) -> dict[tuple[str, str], SemanticField[Any]]:
        try:
            return self.__field_cache
        except AttributeError:
            self.__field_cache: dict[tuple[str, str], SemanticField[Any]] = {}
            return self.__field_cache

    def _invalidate_field_cache(self) -> None:
        """Drop every cached :class:`SemanticField` view.

        A cached field — including one *afforded* over a raw atomic column
        via :attr:`_afforded_fields` — wraps the underlying ``pa.Array`` as
        it was at first access.  Any in-place replacement of ``self._table``
        (e.g. :meth:`EventData.extend`, which reassigns the table to a
        ``pa.concat_tables`` result) therefore leaves the cache stale: a
        subsequent ``get_field`` / ``get_pitch_field`` would silently expose
        only the pre-mutation rows.  Call this after such a mutation so the
        affordance re-attaches over the new table.
        """
        self._field_cache.clear()

    # -- get_raw() -----------------------------------------------------------

    def get_raw(self, name: str) -> DataField:
        """Return a raw ``DataField`` for the named field.

        The returned field is a ``StructField``, ``NumericField``,
        ``StringField``, or ``MapField`` -- whichever matches the
        field's PyArrow type.  No semantic metadata is attached.

        Args:
            name: Field name in the underlying table.

        Returns:
            A raw ``DataField`` wrapping the field data.

        Raises:
            KeyError: If *name* is not in the table.
        """
        if name not in self._table.column_names:
            raise KeyError(
                f"Field {name!r} not found. " f"Available: {self._table.column_names}"
            )
        arr = self._table.column(name)
        pa_field = self._table.schema.field(name)
        return _wrap_raw(arr, pa_field)

    # -- get_field() ---------------------------------------------------------

    def get_field(
        self,
        selector: str | type[SemanticField[Any]] | type[BaseModel] | SemanticField[Any],
        *,
        name: str | None = None,
    ) -> SemanticField[Any]:
        """Return a semantic field, resolved from *selector*.

        Four forms are supported:

        1. **String** -- field name: ``get_field("start")``.  Checks
           field metadata for ``field_type``; falls back to
           ``DefaultField`` mapping for core temporal fields.
        2. **SemanticField class** -- ``get_field(EnharmonicPitchField)``.
           Scans all columns for metadata or shape matching the class.
        3. **Pydantic scalar class** -- ``get_field(EnharmonicPitch)``.
           Looks up the paired ``EnharmonicPitchField`` via the
           scalar→field registry, then delegates to form (2).  Pass
           ``name="col"`` to disambiguate when more than one column
           holds the same scalar.
        4. **Blueprint instance** -- ``get_field(EnharmonicPitchField(source_fields="midi_pitch"))``.
           Resolves the named field using the blueprint.

        Args:
            selector: Field name, paired ``SemanticField`` class, pydantic
                scalar class, or blueprint instance.
            name: Column name used to disambiguate when *selector* is a
                scalar class that matches more than one column.

        Returns:
            A ``SemanticField`` subclass instance (cached).

        Raises:
            KeyError: If no matching field is found.
            MultipleFieldsError: If ``selector`` is a scalar class with
                no ``name=`` and more than one column matches.
        """
        if isinstance(selector, str):
            if name is not None:
                raise TypeError(
                    "name= is only meaningful when selector is a scalar class"
                )
            return self._get_field_by_name(selector)
        if isinstance(selector, type) and issubclass(selector, SemanticField):
            if name is not None:
                return self._get_field_by_class_and_name(selector, name)
            return self._get_field_by_class(selector)
        if isinstance(selector, type) and issubclass(selector, BaseModel):
            field_cls = _get_field_class_for_scalar(selector)
            if name is not None:
                return self._get_field_by_class_and_name(field_cls, name)
            return self._get_field_by_class(field_cls)
        if isinstance(selector, SemanticField):
            if name is not None:
                raise TypeError(
                    "name= is only meaningful when selector is a scalar class"
                )
            return self._get_field_by_blueprint(selector)
        raise TypeError(
            f"get_field() expects str, SemanticField class, pydantic scalar class, "
            f"or blueprint instance, got {type(selector).__name__}"
        )

    def get_fields(
        self,
        field_type: type[SemanticField[Any]],
        *,
        strict: bool = False,
    ) -> list[SemanticField[Any]]:
        """Return ALL fields matching *field_type*'s class hierarchy.

        Discovery uses three strategies, applied in order:

        1. **Metadata-based**: fields carrying a ``field_type`` entry in
           their ``TIMETOALIGN_METADATA_KEY`` blob.
        2. **Default-field-name fallback**: tries to match *field_type*'s
           ``_default_field_name`` against table fields.
        3. **Shape-based fallback**: scans all fields and constructs the
           field via ``cls.matches_pa_field()`` and ``cls.from_field()``.
           This catches loader output that has not (yet) injected
           TTA metadata — e.g. raw ``midi_pitch`` /
           ``specific_pitch`` struct fields from the TSV loader.

        Returns the first non-empty result; never falls through to a
        later strategy once an earlier strategy finds anything.  Order
        matches field order in the table (with structural-match
        fields ordered by the table schema, returning the first
        match for the "first match" convention used by
        ``_get_field_by_class``).

        Args:
            field_type: The ``SemanticField`` subclass to search for.
            strict: If True, only return fields whose ``field_type``
                metadata equals ``field_type.__name__`` exactly — never
                a subclass.  Used by scalar-class and Field-class
                dispatch to discriminate sibling leaves (``CoordinateField``
                vs ``IdCoordinateField``) that share a struct shape.

        Returns:
            A list of matching ``SemanticField`` subclass instances
            (may be empty if nothing matches).
        """
        registry = _get_field_type_map()
        cache = self._field_cache
        result: list[SemanticField[Any]] = []

        schema = self._table.schema
        for i in range(len(schema)):
            pa_field = schema.field(i)
            ft_str = _parse_field_type_metadata(pa_field)
            if ft_str is None:
                continue
            cls = registry.get(ft_str)
            if cls is None:
                continue
            if strict:
                # Exact-class metadata match only.  Sibling leaves that
                # share a struct shape (Id-variants) are excluded.
                if cls is not field_type:
                    continue
            else:
                if not issubclass(cls, field_type):
                    continue
            cache_key = (pa_field.name, ft_str)
            cached = cache.get(cache_key)
            if cached is not None:
                result.append(cached)
            else:
                arr = self._table.column(i)
                field = _reconstruct_field(pa_field, arr, ft_str)
                cache[cache_key] = field
                result.append(field)

        if result:
            return result

        # Strategy 2: schema-based discovery via _default_field_name
        result = _discover_by_schema(self._table, field_type, cache, registry)
        if result:
            return result

        # Strategy 3: structural discovery via cls.matches_pa_field()
        result = _discover_by_shape(self._table, field_type, cache)
        if result:
            return result

        # Strategy 4: declared affordances over raw atomic columns.
        # A subclass may declare that a raw column (e.g. a bare MIDI
        # pitch int) affords a most-expressive semantic view; materialise
        # it lazily here so get_field / get_pitch_field surface it.
        return self._discover_afforded(field_type, cache, strict=strict)

    def _discover_afforded(
        self,
        field_type: type[SemanticField[Any]],
        cache: dict[tuple[str, str], SemanticField[Any]],
        *,
        strict: bool,
    ) -> list[SemanticField[Any]]:
        """Materialise declared :attr:`_afforded_fields` matching *field_type*.

        For each ``(column, afforded_cls)`` declared on this EventData
        whose ``afforded_cls`` matches *field_type* (exact class when
        *strict*, else a subclass), and whose ``column`` is present as a
        non-struct (atomic) column on the table, promote the raw column to
        a live ``afforded_cls`` via its blueprint ``from_array()`` and cache it.

        Returns the matches in declaration order (an empty list when none
        apply).
        """
        afforded: dict[str, type[SemanticField[Any]]] = getattr(
            type(self), "_afforded_fields", {}
        )
        if not afforded:
            return []
        result: list[SemanticField[Any]] = []
        column_names = set(self._table.column_names)
        for column, afforded_cls in afforded.items():
            if strict:
                if (
                    afforded_cls is not field_type
                    and not _are_interchangeable_pitch_views(afforded_cls, field_type)
                ):
                    continue
                materialized_cls = field_type
            else:
                if not issubclass(afforded_cls, field_type):
                    continue
                materialized_cls = afforded_cls
            if column not in column_names:
                continue
            field = self._materialize_afforded(
                column, materialized_cls, cache, raise_on_failure=False
            )
            if field is not None:
                result.append(field)
        return result

    def _materialize_afforded(
        self,
        column: str,
        field_type: type[SemanticField[Any]],
        cache: dict[tuple[str, str], SemanticField[Any]],
        *,
        raise_on_failure: bool,
    ) -> SemanticField[Any] | None:
        """Promote one declared raw-column affordance through ``from_array()``.

        This is the single raw-column promotion path shared by class-based
        discovery and source-field blueprints.  A ``None`` result means that
        *column* is not a declared affordance for *field_type*, or that it is
        already struct-shaped and should be handled by normal field
        reconstruction.  Class discovery treats a failed promotion as a
        non-match; blueprint resolution converts the same failure into the
        public ``TypeError`` expected for an unpromotable source.
        """
        afforded: dict[str, type[SemanticField[Any]]] = getattr(
            type(self), "_afforded_fields", {}
        )
        declared_cls = afforded.get(column)
        if declared_cls is None:
            return None
        if declared_cls is not field_type and not _are_interchangeable_pitch_views(
            declared_cls, field_type
        ):
            return None

        pa_field = self._table.schema.field(column)
        # Struct columns are handled by metadata/default/shape discovery and
        # must retain their existing field metadata and schema.
        if pa.types.is_struct(pa_field.type):
            return None

        cache_key = (column, field_type.__name__)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        arr = self._table.column(column)
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        try:
            field = field_type(name=column).from_array(arr, name=column)
        except (TypeError, KeyError, ValueError, pa.ArrowInvalid) as exc:
            if not raise_on_failure:
                return None
            if isinstance(exc, TypeError):
                raise
            raise TypeError(
                f"{field_type.__name__} cannot promote source field "
                f"{column!r} with type {pa_field.type}"
            ) from exc
        cache[cache_key] = field
        return field

    def get_fields_satisfying(self, protocol: type) -> list[SemanticField[Any]]:
        """Return all fields whose ``scalar_cls`` satisfies *protocol*.

        Discovery walks every column on the underlying table and, for
        each one that maps to a known paired ``XField``, checks whether
        ``XField.scalar_cls`` structurally satisfies *protocol*.  Uses
        :func:`isinstance` against ``runtime_checkable`` ``Protocol``
        types (attribute presence is the criterion — types are not
        enforced).

        Order of results: as encountered in the table schema.
        Deduplication: each ``(name, field_cls)`` pair appears at most
        once.

        Args:
            protocol: A ``runtime_checkable`` Protocol (e.g.
                ``GenericPitchLike``, ``TimeScalarLike``).

        Returns:
            A list of paired ``SemanticField`` instances (may be empty).
        """
        registry = _get_field_type_map()
        matching_field_classes: list[type[SemanticField[Any]]] = []
        for field_cls in registry.values():
            scalar_cls = field_cls.scalar_cls
            if scalar_cls is None:
                continue
            if _scalar_satisfies_protocol(scalar_cls, protocol):
                matching_field_classes.append(field_cls)

        out: list[SemanticField[Any]] = []
        seen: set[tuple[str, str]] = set()
        for field_cls in matching_field_classes:
            for field in self.get_fields(field_cls):
                key = (field.name, type(field).__name__)
                if key in seen:
                    continue
                seen.add(key)
                out.append(field)
        return out

    def has_field(self, field_type: type[SemanticField[Any]]) -> bool:
        """Check whether any field matches *field_type*.

        Args:
            field_type: The ``SemanticField`` subclass to search for.

        Returns:
            ``True`` if at least one matching field exists.
        """
        registry = _get_field_type_map()
        cache = self._field_cache

        for (_, cached_ft_str), _ in cache.items():
            cls = registry.get(cached_ft_str)
            if cls is not None and issubclass(cls, field_type):
                return True
        for (_, cached_ft_str), _ in cache.items():
            if cached_ft_str == field_type.__name__:
                return True

        schema = self._table.schema
        for i in range(len(schema)):
            pa_field = schema.field(i)
            ft_str = _parse_field_type_metadata(pa_field)
            if ft_str is None:
                continue
            cls = registry.get(ft_str)
            if cls is None:
                continue
            if issubclass(cls, field_type):
                return True

        field_names = set(self._table.column_names)
        candidates: list[type[SemanticField[Any]]] = []
        if hasattr(field_type, "_default_field_name"):
            candidates.append(field_type)
        for cls in registry.values():
            if cls is field_type:
                continue
            if issubclass(cls, field_type) and hasattr(cls, "_default_field_name"):
                if cls not in candidates:
                    candidates.append(cls)
        for cls in candidates:
            field_name = cls._default_field_name  # type: ignore[attr-defined]
            if field_name in field_names:
                pa_field = self._table.schema.field(field_name)
                if pa.types.is_struct(pa_field.type):
                    return True

        # Strategy 3: structural match via field_type.matches_pa_field()
        for i in range(len(schema)):
            if field_type.matches_pa_field(schema.field(i)):
                return True

        # Strategy 4: declared affordances over raw atomic columns.
        afforded: dict[str, type[SemanticField[Any]]] = getattr(
            type(self), "_afforded_fields", {}
        )
        for column, afforded_cls in afforded.items():
            if column not in field_names:
                continue
            if not issubclass(
                afforded_cls, field_type
            ) and not _are_interchangeable_pitch_views(afforded_cls, field_type):
                continue
            if not pa.types.is_struct(self._table.schema.field(column).type):
                return True

        return False

    # -- convenience accessors for default temporal fields -------------------

    def get_start_field(self) -> SemanticField[Any]:
        """Return the ``CoordinateField`` for the ``start`` field."""
        return self.get_field("start")

    def get_end_field(self) -> SemanticField[Any]:
        """Return the ``CoordinateField`` for the ``end`` field."""
        return self.get_field("end")

    def get_duration_field(self) -> SemanticField[Any]:
        """Return the ``DurationField`` for the ``duration`` field."""
        return self.get_field("duration")

    # -- pitch access (uniform across every EventData) -----------------------

    def get_pitch_field(
        self,
        pitch_field_type: type[SemanticField[Any]] | None = None,
        *,
        format: str | None = None,
    ) -> SemanticField[Any]:
        """Return the most-expressive pitch field this EventData affords.

        Pitch is **represented exactly once** on an EventData: the single
        default semantic pitch field is the most-expressive type the
        source faithfully supports.  This accessor returns that default,
        chosen from the priority order

            ``SpecificPitchField`` > ``EnharmonicPitchField`` >
            ``SpecificPitchClassField`` > ``GenericPitchClassField``

        applied over every pitch-like field discovered on the table —
        including fields *afforded* over a raw atomic column via
        :attr:`_afforded_fields` (e.g. a bare MIDI pitch int promoted to
        ``EnharmonicPitch``).  Poorer / derived views are reached on
        request (``get_field(<ScalarClass>)`` or
        ``field.convert_to(...)``), never stored a second time.

        The accessor lives on the base field-access mixin so that *every*
        EventData exposes it uniformly; an EventData with no pitch column
        raises ``KeyError`` (as it always has).

        Args:
            pitch_field_type: A specific paired ``XField`` class (e.g.
                ``EnharmonicPitchField``).  If ``None``, returns the most
                expressive available pitch field from the priority list.
            format: Format specifier for on-the-fly conversion.
                Reserved for future conversion support.

        Raises:
            KeyError: If no matching pitch field is found / afforded.
        """
        from timetoalign.core.events import (
            EnharmonicPitchField,
            GenericPitchClassField,
            SpecificPitchClassField,
            SpecificPitchField,
        )
        from timetoalign.core.protocols import GenericPitchLike

        if pitch_field_type is not None:
            return self.get_field(pitch_field_type)

        # Discover all pitch-like fields, then pick the most informative
        # via the priority list (SP > EP > SPC > GPC).  Other pitch
        # scalar fields not in the priority list fall back after the
        # priority sweep.
        all_pitch_fields = self.get_fields_satisfying(GenericPitchLike)
        if not all_pitch_fields:
            raise KeyError(
                "No pitch field found in table. "
                f"Available columns: {self._table.column_names}"
            )

        priority: list[type[SemanticField[Any]]] = [
            SpecificPitchField,
            EnharmonicPitchField,
            SpecificPitchClassField,
            GenericPitchClassField,
        ]
        for cls in priority:
            for field in all_pitch_fields:
                if isinstance(field, cls):
                    return field
        # No priority match — return the first discovered pitch field.
        return all_pitch_fields[0]

    # -- private dispatch methods --------------------------------------------

    def _get_field_by_name(self, name: str) -> SemanticField[Any]:
        """Resolve a semantic field by field name."""
        if name not in self._table.column_names:
            raise KeyError(
                f"Field {name!r} not found. " f"Available: {self._table.column_names}"
            )
        cache = self._field_cache

        # Check cache first
        for (cached_name, _), cached_field in cache.items():
            if cached_name == name:
                return cached_field

        arr = self._table.column(name)
        pa_field = self._table.schema.field(name)

        # Strategy 1: metadata-based
        ft_str = _parse_field_type_metadata(pa_field)
        if ft_str is not None:
            field = _reconstruct_field(pa_field, arr, ft_str)
            cache[(name, ft_str)] = field
            return field

        # Strategy 2: default field mapping (start/end -> Coordinate, duration -> Duration)
        field = _resolve_default_field(name, arr, pa_field)
        if field is not None:
            cache[(name, type(field).__name__)] = field
            return field

        raise KeyError(
            f"Field {name!r} has no semantic field metadata and is not "
            f"a default field. Use get_raw({name!r}) for raw access."
        )

    def _get_field_by_class(
        self, field_type: type[SemanticField[Any]]
    ) -> SemanticField[Any]:
        """Resolve a semantic field by class (strict — exact-class match).

        Raises :class:`MultipleFieldsError` when *field_type* matches more
        than one column.  Callers that expect ambiguity should pass
        ``name="<col>"`` to ``get_field`` (which routes through
        :meth:`_get_field_by_class_and_name`).
        """
        fields = self.get_fields(field_type, strict=True)
        if not fields:
            raise KeyError(
                f"No field matching {field_type.__name__!r} found in table. "
                f"Available columns: {self._table.column_names}"
            )
        if len(fields) > 1:
            names = [f.name for f in fields]
            scalar_name = (
                field_type.scalar_cls.__name__
                if field_type.scalar_cls is not None
                else field_type.__name__
            )
            raise MultipleFieldsError(
                f"{scalar_name} matches multiple columns {names}; "
                f"pass name=<col> to disambiguate (one of: {names})."
            )
        return fields[0]

    def _get_field_by_class_and_name(
        self,
        field_type: type[SemanticField[Any]],
        name: str,
    ) -> SemanticField[Any]:
        """Resolve a semantic field by class restricted to a specific column.

        Raises plain :class:`KeyError` when no column named *name* holds
        the requested scalar type.
        """
        candidates = self.get_fields(field_type, strict=True)
        for field in candidates:
            if field.name == name:
                return field
        scalar_name = (
            field_type.scalar_cls.__name__
            if field_type.scalar_cls is not None
            else field_type.__name__
        )
        raise KeyError(
            f"No column named {name!r} holds {scalar_name}. "
            f"Available columns for this scalar: "
            f"{[f.name for f in candidates]}"
        )

    def _get_field_by_blueprint(
        self, blueprint: SemanticField[Any]
    ) -> SemanticField[Any]:
        """Resolve a semantic field from a blueprint instance.

        A blueprint is a paired ``SemanticField`` constructed with a
        ``source_fields`` specification (see
        :meth:`SemanticField.__init__`).  Two spec shapes are supported
        here:

        * a ``str`` — interpreted as the name of a field exposed by the
          underlying ``pa.Table`` (must carry a struct congruent with
          ``type(blueprint).pa_schema``);
        * a ``dict`` — recursively maps the paired class's
          :attr:`pa_schema` sub-field names to source-field names on
          the table (only single-level dicts are handled today;
          full nested dispatch is reserved for a follow-up as more
          SemanticField shapes need it).
        """
        if not getattr(blueprint, "is_blueprint", False):
            raise TypeError(
                f"Blueprint resolution requires a blueprint-mode SemanticField, "
                f"got {type(blueprint).__name__}"
            )

        spec = blueprint._blueprint_source_fields
        if spec is None:
            raise TypeError(
                f"{type(blueprint).__name__} blueprint has no source_fields attached"
            )

        # Only the string-spec path is connected at this revision. Dict
        # specs are validated up front by ``resolve_source_fields`` but
        # the table-side resolver below still expects a single name.
        if not isinstance(spec, str):
            raise NotImplementedError(
                f"{type(blueprint).__name__} blueprint resolution from a dict "
                "source_fields= spec is not implemented yet; pass a single "
                "source-field name as a string."
            )
        field_name = spec

        cache = self._field_cache
        cache_key = (field_name, type(blueprint).__name__)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        if field_name not in self._table.column_names:
            raise KeyError(
                f"Blueprint source field {field_name!r} not found. "
                f"Available: {self._table.column_names}"
            )

        arr = self._table.column(field_name)
        pa_field = self._table.schema.field(field_name)

        # Keep blueprint resolution aligned with class-based affordance
        # discovery: a declared raw atomic source is packed by the paired
        # field's ``from_array()`` before the normal struct-backed constructor is
        # attempted.  This also preserves the requested EP/MP view class.
        if not pa.types.is_struct(pa_field.type):
            promoted = self._materialize_afforded(
                field_name,
                type(blueprint),
                cache,
                raise_on_failure=True,
            )
            if promoted is not None:
                return promoted

        field = type(blueprint).from_field((arr, pa_field))
        cache[cache_key] = field
        return field


# ---------------------------------------------------------------------------
# HarmonyAccessMixin
# ---------------------------------------------------------------------------


class HarmonyAccessMixin(SemanticFieldAccessMixin):
    """Mixin providing harmony field access with priority-based defaults.

    Priority order (most specific first):
    ``DcmlHarmonyField`` > ``RomanNumeralHarmonyField`` >
    ``WesternTertianHarmonyField`` > ``PitchBasedHarmonyField`` >
    ``HarmonyLabelField``.
    """

    def get_harmony_field(
        self,
        harmony_type: type[SemanticField[Any]] | None = None,
        *,
        format: str | None = None,
    ) -> SemanticField[Any]:
        """Return a harmony field, optionally filtered by class.

        Args:
            harmony_type: A specific paired harmony class to request.
                If ``None``, returns the most specific available.
            format: Format specifier for on-the-fly conversion.
                Reserved for future conversion support.

        Raises:
            KeyError: If no matching harmony field is found.
        """
        from timetoalign.core.events import (
            DcmlHarmonyField,
            HarmonyLabelField,
            PitchBasedHarmonyField,
            RomanNumeralHarmonyField,
            WesternTertianHarmonyField,
        )
        from timetoalign.core.protocols import HarmonyLabelLike

        if harmony_type is not None:
            return self.get_field(harmony_type)

        all_harmony_fields = self.get_fields_satisfying(HarmonyLabelLike)
        if not all_harmony_fields:
            raise KeyError(
                "No harmony field found in table. "
                f"Available columns: {self._table.column_names}"
            )

        priority: list[type[SemanticField[Any]]] = [
            DcmlHarmonyField,
            RomanNumeralHarmonyField,
            WesternTertianHarmonyField,
            PitchBasedHarmonyField,
            HarmonyLabelField,
        ]
        for ht in priority:
            for field in all_harmony_fields:
                if isinstance(field, ht):
                    return field
        return all_harmony_fields[0]
