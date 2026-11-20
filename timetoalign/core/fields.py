"""DataField hierarchy + pydantic → PyArrow translator + Parquet metadata.

This module is the foundation of TimeToAlign!'s fielded storage
layer.  It bundles four closely-related concerns:

* The **DataField hierarchy** wraps ``pa.Array`` / ``pa.ChunkedArray``
  with typed accessors.  ``SemanticField[T]`` is the strictly-typed
  bridge between a pydantic scalar ``T`` and the underlying raw field.
* The **pydantic → PyArrow translator** derives a ``pa.Schema`` (and
  individual ``pa.Field`` entries) from a pydantic ``BaseModel`` once at
  class-definition time.  Value projectors handle denormalised storage
  (``Coordinate.value`` → ``{value, numerator, denominator}``) and
  selective drops (``Note.pitch`` → removed from the pa.Schema so that
  pitch lives in its own field).
* The **store builder** assembles a ``pa.StructArray`` from many
  validated scalar instances by traversing each scalar once and
  accumulating the per-sub-field arrays (NEVER ``model_dump`` row-wise).
* The **Parquet-metadata** helpers produce the ``b"timetoalign"`` JSON
  blob that travels with every TimeToAlign!-written ``pa.Field``.

Layout convention: five labelled sections — base hierarchy → translator
→ store builder → parquet metadata → DenominateNumberField hierarchy.
"""

from __future__ import annotations

import enum
import inspect
import json
import sys
from abc import ABC, abstractmethod
from fractions import Fraction
from functools import lru_cache
from importlib.machinery import PathFinder
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    Sequence,
    TypeVar,
    get_args,
    get_origin,
)

import pyarrow as pa
import pyarrow.compute as pc
from pydantic import BaseModel
from pydantic.fields import FieldInfo

T = TypeVar("T", bound=BaseModel)
"""The pydantic scalar a ``SemanticField`` is paired with."""


_PAIRED_SEMANTIC_FIELDS: dict[type[BaseModel], type["SemanticField[Any]"]] = {}
"""Registered pydantic scalar → paired semantic-field classes."""


def _register_paired_semantic_field(cls: type["SemanticField[Any]"]) -> None:
    """Record a concrete scalar/field pair as its field class is declared."""
    scalar_cls = cls.scalar_cls
    if scalar_cls is not None:
        _PAIRED_SEMANTIC_FIELDS.setdefault(scalar_cls, cls)


def install_paired_field_registry() -> None:
    """Expose core-declared scalar/field pairs to the storage field registry.

    The storage registry is lazy to avoid its import cycle with ``core``.
    Extending it at lookup time keeps subsequently declared core pairs in
    the scalar-to-field lookup without maintaining a second static list.
    """
    mixins = sys.modules.get("timetoalign.storage.mixins")
    if mixins is None or not hasattr(mixins, "_get_field_type_map"):
        _install_paired_field_registry_import_hook()
        return

    get_field_type_map = mixins._get_field_type_map
    if getattr(get_field_type_map, "_uses_paired_field_registry", False):
        return

    def registered_field_type_map() -> dict[str, type[SemanticField[Any]]]:
        field_type_map = get_field_type_map()
        for field_cls in _PAIRED_SEMANTIC_FIELDS.values():
            field_type_map.setdefault(field_cls.field_type(), field_cls)
        return field_type_map

    registered_field_type_map._uses_paired_field_registry = True  # type: ignore[attr-defined]
    mixins._get_field_type_map = registered_field_type_map


def _install_paired_field_registry_import_hook() -> None:
    """Install the registry bridge after the storage mixin finishes loading."""
    if any(
        getattr(finder, "_paired_field_registry_hook", False)
        for finder in sys.meta_path
    ):
        return

    class _RegistryLoader:
        def __init__(self, loader: Any) -> None:
            self._loader = loader

        def create_module(self, spec: Any) -> Any:
            create_module = getattr(self._loader, "create_module", None)
            return create_module(spec) if create_module is not None else None

        def exec_module(self, module: Any) -> None:
            self._loader.exec_module(module)
            install_paired_field_registry()

    class _RegistryFinder:
        _paired_field_registry_hook = True

        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
            if fullname != "timetoalign.storage.mixins":
                return None
            spec = PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _RegistryLoader(spec.loader)
                sys.meta_path.remove(self)
            return spec

    sys.meta_path.insert(0, _RegistryFinder())


def data_shaped(fn: Any) -> Any:
    """Mark a scalar method as needing a vectorized field mirror.

    A scalar method (regular function, ``@property``, or
    ``@computed_field`` wrapping a function) tagged ``@data_shaped``
    MUST appear (by the same name) on the paired :class:`SemanticField`
    subclass — otherwise :meth:`SemanticField.__init_subclass__` raises
    :class:`TypeError` at import time.

    The marker is purely a metadata flag (``__data_shaped__ = True``)
    on the underlying function.  Behavior-shaped methods are NOT
    decorated — they only run on a single materialised scalar at a
    system edge and have no vectorized dual.

    Stacking notes:

    * For plain methods, decorate normally::

          @data_shaped
          def foo(self) -> ...:

    * For properties, the marker goes on the underlying function so
      that order with ``@property`` does not matter::

          @property
          @data_shaped
          def pitch_class(self) -> int: ...

      and the corresponding pydantic ``@computed_field``::

          @computed_field
          @property
          @data_shaped
          def fifths(self) -> int: ...
    """
    fn.__data_shaped__ = True
    return fn


def _is_data_shaped(member: Any) -> bool:
    """Return True if *member* is (or wraps) a data-shaped function.

    Detects the marker on:

    * plain functions / methods (``fn.__data_shaped__``);
    * properties (``property.fget.__data_shaped__``);
    * pydantic ``@computed_field`` decorators (``ComputedFieldInfo``
      exposes the underlying function via ``.wrapped_property.fget``
      or ``.fn``; we probe defensively).
    """
    if getattr(member, "__data_shaped__", False):
        return True
    fget = getattr(member, "fget", None)
    if fget is not None and getattr(fget, "__data_shaped__", False):
        return True
    # pydantic v2 ComputedFieldInfo / wrapped descriptor — best effort.
    wrapped = getattr(member, "wrapped_property", None)
    if wrapped is not None:
        wrapped_fget = getattr(wrapped, "fget", None)
        if wrapped_fget is not None and getattr(wrapped_fget, "__data_shaped__", False):
            return True
    inner = getattr(member, "fn", None)
    if inner is not None and getattr(inner, "__data_shaped__", False):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATAFIELD HIERARCHY
# ═══════════════════════════════════════════════════════════════════════════


class DataField(ABC):
    """Abstract base class for typed PyArrow field wrappers.

    A DataField pairs a ``pa.Array`` (or ``pa.ChunkedArray``) with its
    ``pa.Field`` schema descriptor, providing convenient access to data,
    metadata, and type information.

    Two construction modes:

    * **Live mode:** pass ``data`` (a ``pa.Array``/``pa.ChunkedArray``)
      plus an explicit ``field``.  The DataField wraps the pair.
    * **Blueprint mode:** pass only ``name=<str>`` (without ``field=``).
      The DataField is materialised as schema-only with a ``pa.Field``
      whose type comes from the subclass's class-level
      ``_blueprint_pa_type`` default.  ``data`` may also be omitted
      (defaults to ``None``).  Blueprints are intended to be
      materialised later via :meth:`from_array`.

    Args:
        data: The PyArrow array holding field values, or ``None`` for
            schema-only (empty) fields.
        field: The PyArrow field descriptor carrying name, type, and
            metadata.  When omitted, ``name=`` triggers blueprint mode.
        name: When ``field`` is ``None``, the blueprint name for a
            schema-only construction; the type comes from
            ``cls._blueprint_pa_type``.

    Attributes:
        _data: The underlying array (may be ``None``).
        _field: The PyArrow field descriptor.
    """

    # Default ``pa.DataType`` used by blueprint construction (``name=``
    # only, without ``field=``).  Concrete leaves override this; the
    # abstract base leaves it as ``None`` so :meth:`_default_blueprint_pa_type`
    # raises an explicit error.
    _blueprint_pa_type: ClassVar[pa.DataType | None] = None

    def __init__(
        self,
        data: pa.Array | pa.ChunkedArray | None = None,
        field: pa.Field | None = None,
        *,
        name: str | None = None,
    ) -> None:
        if field is not None:
            self._data = data
            self._field = field
            return
        if name is None:
            raise TypeError(f"{type(self).__name__} requires either field= or name=")
        pa_type = self._default_blueprint_pa_type()
        self._data = data  # typically ``None`` for a blueprint
        self._field = pa.field(name, pa_type)

    @classmethod
    def _default_blueprint_pa_type(cls) -> pa.DataType:
        """Return the default ``pa.DataType`` for blueprint construction.

        Override on each concrete leaf via the ``_blueprint_pa_type``
        ClassVar.  Classes without a blueprint default (e.g.
        :class:`StructField` and :class:`MapField`, whose concrete shape
        is user-defined) MUST be constructed with an explicit ``field=``
        argument.
        """
        t = getattr(cls, "_blueprint_pa_type", None)
        if t is None:
            raise TypeError(
                f"{cls.__name__} has no blueprint pa_type; pass field= explicitly"
            )
        return t

    # -- properties ----------------------------------------------------------

    @property
    def data(self) -> pa.Array | pa.ChunkedArray | None:
        """The underlying PyArrow array, or ``None`` if schema-only."""
        return self._data

    @property
    def field(self) -> pa.Field:
        """The PyArrow field descriptor (name, type, metadata)."""
        return self._field

    @property
    def name(self) -> str:
        """Column name taken from the PyArrow field."""
        return self._field.name

    @property
    def is_empty(self) -> bool:
        """``True`` when this field carries no data (schema-only)."""
        return self._data is None

    @property
    def pa_type(self) -> pa.DataType:
        """The PyArrow data type of this field."""
        return self._field.type

    @property
    def metadata(self) -> dict[str, str]:
        """Decoded metadata from the PyArrow field.

        Returns a plain ``dict[str, str]`` with byte keys/values decoded
        to UTF-8 strings.  Returns an empty dict when no metadata is
        present.
        """
        raw = self._field.metadata
        if not raw:
            return {}
        return {
            (k.decode("utf-8") if isinstance(k, bytes) else k): (
                v.decode("utf-8") if isinstance(v, bytes) else v
            )
            for k, v in raw.items()
        }

    # -- dunder methods ------------------------------------------------------

    def __len__(self) -> int:
        """Number of elements in the data array.

        Raises:
            TypeError: If the field is schema-only (no data).
        """
        if self._data is None:
            raise TypeError(f"Cannot compute length of schema-only field {self.name!r}")
        return len(self._data)

    def __getitem__(self, i: int) -> Any:
        """Return the *i*-th element as a Python scalar via ``.as_py()``.

        Handles both ``pa.Array`` and ``pa.ChunkedArray`` transparently.

        Args:
            i: Zero-based index.

        Returns:
            The Python-native scalar at position *i*.

        Raises:
            TypeError: If the field is schema-only (no data).
            IndexError: If *i* is out of range.
        """
        if self._data is None:
            raise TypeError(f"Cannot index schema-only field {self.name!r}")
        if isinstance(self._data, pa.ChunkedArray):
            return self._data.combine_chunks()[i].as_py()
        return self._data[i].as_py()

    def __repr__(self) -> str:
        length = len(self) if not self.is_empty else 0
        return f"{type(self).__name__}(name={self.name!r}, type={self.pa_type}, len={length})"

    def _repr_sample(self) -> str:
        """Return a head/tail sample of element ``repr``s.

        Shows the first up-to-3 and last up-to-2 elements via ``repr``,
        inserting an ellipsis when the field is longer than the shown
        window.  Returns ``"(schema-only)"`` for empty fields.
        """
        if self.is_empty:
            return "(schema-only)"
        length = len(self)
        if length == 0:
            return "(empty)"
        head = [repr(self[i]) for i in range(min(3, length))]
        if length <= 3:
            return ", ".join(head)
        tail_start = max(3, length - 2)
        tail = [repr(self[i]) for i in range(tail_start, length)]
        gap = "…" if tail_start > 3 else ""
        middle = [gap] if gap else []
        return ", ".join(head + middle + tail)

    def _repr_rows(self) -> list[tuple[str, str]]:
        """Return the ``(label, value)`` rows for :meth:`_repr_html_`.

        Base (scalar-less) rows: Length, Arrow type, Sample.  The
        :class:`SemanticField` override prepends the scalar-type row.
        """
        import html as _html

        length = 0 if self.is_empty else len(self)
        return [
            ("Length", str(length)),
            ("Arrow type", _html.escape(str(self.pa_type))),
            ("Sample", _html.escape(self._repr_sample())),
        ]

    def _repr_affordances(self) -> list[str]:
        """Return the invocation snippets for the ``Try`` row."""
        return ["field[i]", "field.get_raw()"]

    def _repr_html_(self) -> str:
        """Rich HTML affordance card for Jupyter notebooks."""
        from timetoalign.display.html import affordance_html

        return affordance_html(
            type(self).__name__,
            self._repr_rows(),
            affordances=self._repr_affordances(),
        )

    # -- conversion helpers --------------------------------------------------

    def to_pyarrow(self) -> pa.Array:
        """Return data as a contiguous ``pa.Array``.

        If the underlying storage is a ``ChunkedArray``, it is combined
        into a single chunk first.

        Raises:
            TypeError: If the field is schema-only (no data).
        """
        if self._data is None:
            raise TypeError(
                f"Cannot convert schema-only field {self.name!r} to PyArrow array"
            )
        if isinstance(self._data, pa.ChunkedArray):
            return self._data.combine_chunks()
        return self._data

    def to_field(self) -> pa.Field:
        """Return the ``pa.Field`` descriptor (with metadata)."""
        return self._field

    # -- blueprint materialisation ------------------------------------------

    def from_array(
        self,
        source: pa.Array | pa.ChunkedArray,
        *,
        name: str | None = None,
    ) -> "DataField":
        """Materialise a live DataField from a source array.

        The default implementation casts ``source`` to this field's
        :attr:`pa_type` and wraps the result in a new instance of
        ``type(self)`` carrying the blueprint's name (or the explicit
        ``name=`` override).  Subclasses whose semantics are not a
        simple cast (e.g. :class:`RedundantNumberField` parses ``"n/d"``
        strings; :class:`DenominateNumberField` also stamps unit
        metadata) override this method.

        Args:
            source: The raw source column.
            name: Optional override for the emitted field's name.
                Defaults to :attr:`name`.

        Returns:
            A new live DataField of the same class.

        Raises:
            TypeError: If called on a non-blueprint (live) field.
        """
        if not self.is_empty:
            raise TypeError(
                f"{type(self).__name__}.from_array() can only be called on a blueprint "
                f"(empty DataField); this instance has live data"
            )
        out_name = name if name is not None else self.name
        casted = pc.cast(source, self.pa_type)
        return type(self)(casted, pa.field(out_name, self.pa_type))

    # -- abstract factory ----------------------------------------------------

    @classmethod
    @abstractmethod
    def from_field(cls, source: Any, **kw: Any) -> "DataField":
        """Construct a DataField from a source.

        Subclasses define what *source* means (e.g., a ``pa.Table``
        field, a ``pa.Field`` + ``pa.Array`` pair, etc.).

        Args:
            source: The data source.
            **kw: Additional keyword arguments for subclass-specific
                construction logic.

        Returns:
            A new DataField instance.
        """
        ...


class NumericField(DataField):
    """A DataField backed by a numeric (integer or floating-point) array.

    Validates at construction time that the PyArrow type is numeric.
    Subclasses :class:`IntField` and :class:`FloatField` pin a concrete
    blueprint ``pa.DataType`` so they can be constructed with ``name=``
    alone; the abstract ``NumericField`` has no blueprint default.

    Args:
        data: Numeric PyArrow array, or ``None``.
        field: PyArrow field descriptor (live mode).
        name: Blueprint name (when ``field`` is ``None``); the
            ``pa.DataType`` comes from the subclass's
            :attr:`_blueprint_pa_type`.

    Raises:
        TypeError: If the field type is not numeric.
    """

    def __init__(
        self,
        data: pa.Array | pa.ChunkedArray | None = None,
        field: pa.Field | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(data, field, name=name)
        if not (
            pa.types.is_integer(self._field.type)
            or pa.types.is_floating(self._field.type)
        ):
            raise TypeError(
                f"NumericField requires a numeric type, got {self._field.type}"
            )

    def __getitem__(self, i: int) -> int | float:
        """Return the *i*-th element as ``int`` or ``float``.

        Args:
            i: Zero-based index.

        Returns:
            An ``int`` for integer types, ``float`` for floating-point types.
        """
        value = super().__getitem__(i)
        return value  # type: ignore[return-value]

    @classmethod
    def from_field(
        cls, source: tuple[pa.Array | pa.ChunkedArray | None, pa.Field], **kw: Any
    ) -> "NumericField":
        """Create a NumericField from a ``(data, field)`` tuple.

        Args:
            source: A tuple of ``(pa.Array | None, pa.Field)``.

        Returns:
            A new NumericField.
        """
        data, field = source
        return cls(data, field)


class IntField(NumericField):
    """A :class:`NumericField` pinned to ``pa.int64`` for blueprint use.

    Constructing ``IntField(name="x")`` yields a schema-only field of
    type ``pa.int64()`` suitable for :meth:`from_array`-time materialisation.
    Live-mode construction (``IntField(data, field)``) still validates
    the field type via the :class:`NumericField` base.
    """

    _blueprint_pa_type: ClassVar[pa.DataType] = pa.int64()


class FloatField(NumericField):
    """A :class:`NumericField` pinned to ``pa.float64`` for blueprint use.

    Constructing ``FloatField(name="x")`` yields a schema-only field of
    type ``pa.float64()`` suitable for :meth:`from_array`-time materialisation.
    Live-mode construction still validates the field type via the
    :class:`NumericField` base.
    """

    _blueprint_pa_type: ClassVar[pa.DataType] = pa.float64()


class StringField(DataField):
    """A DataField backed by a string (utf8 / large_utf8) array.

    Usable as a blueprint directly: ``StringField(name="x")`` constructs
    a schema-only field with :func:`pa.string` as its type.

    Args:
        data: String PyArrow array, or ``None``.
        field: PyArrow field descriptor (live mode).
        name: Blueprint name (when ``field`` is ``None``).

    Raises:
        TypeError: If the field type is not string-like.
    """

    _blueprint_pa_type: ClassVar[pa.DataType] = pa.string()

    def __init__(
        self,
        data: pa.Array | pa.ChunkedArray | None = None,
        field: pa.Field | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(data, field, name=name)
        if not (
            pa.types.is_string(self._field.type)
            or pa.types.is_large_string(self._field.type)
        ):
            raise TypeError(
                f"StringField requires a string type, got {self._field.type}"
            )

    def from_array(
        self,
        source: pa.Array | pa.ChunkedArray,
        *,
        name: str | None = None,
    ) -> "StringField":
        """Materialise a live StringField from a source array.

        Non-string source dtypes are cast to ``pa.string()`` before
        wrapping.
        """
        if not self.is_empty:
            raise TypeError(
                f"{type(self).__name__}.from_array() can only be called on a blueprint "
                f"(empty DataField); this instance has live data"
            )
        out_name = name if name is not None else self.name
        if pa.types.is_string(source.type) or pa.types.is_large_string(source.type):
            casted = source
        else:
            casted = pc.cast(source, pa.string())
        return StringField(casted, pa.field(out_name, pa.string()))

    def __getitem__(self, i: int) -> str | None:
        return super().__getitem__(i)

    @classmethod
    def from_field(
        cls, source: tuple[pa.Array | pa.ChunkedArray | None, pa.Field], **kw: Any
    ) -> "StringField":
        data, field = source
        return cls(data, field)


class StructField(DataField):
    """A DataField backed by a struct array.

    Provides sub-field extraction and field-name introspection on top of
    the base ``DataField`` interface.  Has no blueprint default — the
    concrete struct shape is user-defined, so callers MUST pass an
    explicit ``field=`` argument.

    Args:
        data: Struct PyArrow array, or ``None``.
        field: PyArrow field descriptor.

    Raises:
        TypeError: If the field type is not a struct, or if constructed
            in blueprint mode without an explicit field.
    """

    def __init__(
        self,
        data: pa.Array | pa.ChunkedArray | None = None,
        field: pa.Field | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(data, field, name=name)
        if not pa.types.is_struct(self._field.type):
            raise TypeError(
                f"StructField requires a struct type, got {self._field.type}"
            )

    @property
    def field_names(self) -> list[str]:
        """Names of the sub-fields in this struct."""
        struct_type: pa.StructType = self._field.type
        return [struct_type.field(i).name for i in range(struct_type.num_fields)]

    def get_sub_field(self, name: str) -> DataField:
        """Extract a sub-field as a new ``DataField``.

        The returned field uses ``NumericField`` or ``StringField`` when
        the sub-field type matches; otherwise a minimal concrete
        ``DataField`` subclass is returned.

        Args:
            name: Name of the sub-field to extract.

        Returns:
            A DataField wrapping the sub-field's array and schema.

        Raises:
            KeyError: If *name* is not a sub-field of this struct.
            TypeError: If this field has no data.
        """
        struct_type: pa.StructType = self._field.type
        sub_idx = struct_type.get_field_index(name)
        if sub_idx < 0:
            raise KeyError(
                f"Struct field {self.name!r} has no sub-field {name!r}. Available: {self.field_names}"
            )

        sub_pa_field = struct_type.field(sub_idx)

        if self._data is None:
            sub_data = None
        else:
            arr = (
                self.to_pyarrow()
                if isinstance(self._data, pa.ChunkedArray)
                else self._data
            )
            sub_data = arr.field(name)

        if pa.types.is_integer(sub_pa_field.type) or pa.types.is_floating(
            sub_pa_field.type
        ):
            return NumericField(sub_data, sub_pa_field)
        if pa.types.is_string(sub_pa_field.type) or pa.types.is_large_string(
            sub_pa_field.type
        ):
            return StringField(sub_data, sub_pa_field)
        if pa.types.is_struct(sub_pa_field.type):
            return StructField(sub_data, sub_pa_field)
        if pa.types.is_map(sub_pa_field.type):
            return MapField(sub_data, sub_pa_field)
        return _GenericField(sub_data, sub_pa_field)

    def __getitem__(self, i: int) -> dict[str, Any] | None:
        return super().__getitem__(i)

    @classmethod
    def from_field(
        cls, source: tuple[pa.Array | pa.ChunkedArray | None, pa.Field], **kw: Any
    ) -> "StructField":
        data, field = source
        return cls(data, field)


class MapField(DataField):
    """A DataField backed by a map array.

    Has no blueprint default — the concrete key/value types are
    user-defined, so callers MUST pass an explicit ``field=``.

    Args:
        data: Map PyArrow array, or ``None``.
        field: PyArrow field descriptor.

    Raises:
        TypeError: If the field type is not a map.
    """

    def __init__(
        self,
        data: pa.Array | pa.ChunkedArray | None = None,
        field: pa.Field | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(data, field, name=name)
        if not pa.types.is_map(self._field.type):
            raise TypeError(f"MapField requires a map type, got {self._field.type}")

    def __getitem__(self, i: int) -> dict[Any, Any] | None:
        raw = super().__getitem__(i)
        if raw is None:
            return None
        if isinstance(raw, list):
            return dict(raw)
        return raw  # type: ignore[return-value]

    @classmethod
    def from_field(
        cls, source: tuple[pa.Array | pa.ChunkedArray | None, pa.Field], **kw: Any
    ) -> "MapField":
        data, field = source
        return cls(data, field)


class _GenericField(DataField):
    """Concrete fallback DataField for types not covered by specialised classes."""

    @classmethod
    def from_field(
        cls, source: tuple[pa.Array | pa.ChunkedArray | None, pa.Field], **kw: Any
    ) -> "_GenericField":
        data, field = source
        return cls(data, field)


class _VocabularyOwner:
    """Shared derivation of the ``field_type`` metadata vocabulary entry.

    ``field_type`` is the discriminator that lets a reader map a stored
    column back onto the :class:`SemanticField` subclass that owns it, so
    the strings are part of the on-disk contract.  They are derived from
    the class name — never hand-written — with
    :attr:`field_type_name` as the escape hatch for the handful of
    classes whose stored name differs from their own.

    Attributes:
        field_type_name: Explicit override of the derived name.  Set it
            on a subclass whose stored vocabulary entry is not
            ``<ClassName><suffix>``; leave it ``None`` otherwise.
    """

    field_type_name: ClassVar[str | None] = None

    _FIELD_TYPE_SUFFIX: ClassVar[str] = ""
    """Appended to the class name when deriving the vocabulary entry."""

    @classmethod
    def field_type(cls) -> str:
        """Return the ``field_type`` vocabulary entry for this class."""
        if cls.field_type_name is not None:
            return cls.field_type_name
        return f"{cls.__name__}{cls._FIELD_TYPE_SUFFIX}"

    def metadata_dict(self) -> dict[str, str]:
        """Return the metadata payload describing this object.

        Subclasses that carry more than the discriminator extend the
        base payload rather than rebuilding it::

            def metadata_dict(self) -> dict[str, str]:
                return {**super().metadata_dict(), "unit": self.unit.value}
        """
        return {"field_type": type(self).field_type()}


class FieldVocabulary(_VocabularyOwner):
    """Vocabulary owner for :class:`SemanticField` subclasses.

    A field's ``field_type`` is its own class name (``CoordinateField``,
    ``NoteField``, …).
    """


class ScalarVocabulary(_VocabularyOwner):
    """Vocabulary owner for the pydantic scalars paired with fields.

    A scalar records the name of the *field* that stores it, so the
    derived entry is ``<ScalarName>Field`` (``Coordinate`` →
    ``"CoordinateField"``).  Scalars that are stored by an inherited
    field class pin :attr:`~_VocabularyOwner.field_type_name` explicitly.
    """

    _FIELD_TYPE_SUFFIX: ClassVar[str] = "Field"


class SemanticField(DataField, FieldVocabulary, Generic[T]):
    """A DataField that wraps a raw field and adds semantic identity.

    ``SemanticField[T]`` is the strictly-typed bridge between a pydantic
    scalar ``T`` and a struct-shaped ``DataField``.  The class is
    parametrised on the scalar (``T``, a ``BaseModel`` subclass), not on
    the inner field — the inner field is pinned via the
    :attr:`_raw_cls` class variable, which defaults to :class:`StructField`
    and may be refined by subclasses (e.g. ``TimeScalarField`` sets it to
    ``DenominateNumberField``).  Storing the inner field via a ClassVar
    keeps the call-site signature ``XField(SemanticField[X])`` tidy
    while letting subclasses lock down the underlying storage shape when
    it matters.

    Composition contract.  Every concrete subclass is paired with a
    pydantic scalar ``T`` via ``XField(SemanticField[T])``.  At subclass
    declaration time, ``__init_subclass__`` derives ``cls.pa_schema``
    (the ``pa.StructType`` for ``T``, with value-projector expansions
    applied) and caches it on the subclass.  Callers use
    ``XField.pa_schema`` whenever they need the struct layout for an
    on-disk field without instantiating a live one.

    Subclasses are expected to:

    - Override ``__getitem__`` to return semantic scalar objects.
    - Implement ``from_field`` for construction from external sources.
    - Implement ``SemanticTypeLike`` properties (``semantic_type``,
      ``metadata_dict``).

    Args:
        raw: The wrapped raw DataField (an instance of ``_raw_cls`` or
            a compatible subclass).  ``None`` triggers blueprint mode.
        source_fields: Deferred-resolution specification when ``raw`` is
            ``None`` — see :meth:`__init__`.

    Attributes:
        _raw: The inner raw field instance.
        _is_blueprint: ``True`` when constructed in blueprint mode.
        _blueprint_source_fields: The unresolved ``source_fields`` spec.
    """

    # The pydantic scalar class this SemanticField is paired with.
    # Populated by ``__init_subclass__`` from ``Generic[T]`` on the bases
    # of the concrete subclass.  ``None`` on the abstract base.
    scalar_cls: ClassVar[type[BaseModel] | None] = None

    # Cached pa.StructType derived from ``scalar_cls``'s pydantic model.
    # ``None`` on the abstract base; populated for every concrete subclass.
    pa_schema: ClassVar[pa.StructType | None] = None

    # The DataField subclass used as the inner raw field.  Defaults to
    # the plain ``StructField``; subclasses refine it when the inner
    # storage is more specific (e.g. ``DenominateNumberField``).
    _raw_cls: ClassVar[type["StructField"]] = StructField

    def __init_subclass__(cls, **kw: Any) -> None:
        """Resolve ``cls.scalar_cls`` and cache ``cls.pa_schema``.

        Resolution order:

        1. If the subclass declares ``scalar_cls`` in its own ``__dict__``
           (concrete override), use that directly.  This is the escape
           hatch for paired fields that inherit through a non-parametrised
           intermediate (e.g. ``CoordinateField`` inherits through
           ``TimeScalarField`` and supplies ``scalar_cls = Coordinate``
           directly).
        2. Otherwise walk ``__orig_bases__`` for a parametrised
           ``SemanticField[T]`` whose ``T`` is a pydantic ``BaseModel``.
           This is the common path used by every paired ``XField``.

        Intermediate abstract classes that do not pin ``T`` (e.g.
        ``TimeScalarField`` extends ``SemanticField`` unparametrised)
        leave ``scalar_cls`` / ``pa_schema`` as ``None``; their concrete
        children resolve normally.
        """
        super().__init_subclass__(**kw)
        own_scalar = cls.__dict__.get("scalar_cls")
        if isinstance(own_scalar, type) and issubclass(own_scalar, BaseModel):
            scalar = own_scalar
        else:
            scalar = _resolve_scalar_cls(cls)
            if scalar is not None:
                cls.scalar_cls = scalar

        if scalar is not None:
            cls.pa_schema = derive_arrow_struct(scalar)

        # Parity check: every @data_shaped method on the paired scalar
        # MUST be mirrored (by name) on this Field subclass or any of
        # its ancestors in the SemanticField hierarchy.  Behavior-shaped
        # methods are NOT mirrored.
        scalar_cls = cls.scalar_cls
        if scalar_cls is None:
            return
        _register_paired_semantic_field(cls)
        required: set[str] = set()
        for klass in scalar_cls.__mro__:
            for name, member in list(vars(klass).items()):
                if _is_data_shaped(member):
                    required.add(name)
        if not required:
            return
        # Names defined on cls itself.
        provided: set[str] = set(vars(cls))
        # Names inherited from a parent SemanticField subclass — a mirror
        # implemented at a higher field-class layer (e.g. TimeScalarField
        # for the Coord/Dur hierarchy) satisfies the parity check for
        # every descendant.
        for base in cls.__mro__[1:]:
            if base is object:
                continue
            provided.update(vars(base))
        missing = required - provided
        if missing:
            raise TypeError(
                f"{cls.__name__} is missing vectorized mirrors for "
                f"@data_shaped methods on {scalar_cls.__name__}: "
                f"{sorted(missing)}"
            )

    def __init__(
        self,
        raw: "StructField | None" = None,
        *,
        source_fields: "str | dict[str, Any] | None" = None,
        name: str | None = None,
    ) -> None:
        """Construct a SemanticField.

        Three construction modes:

        * **Live mode:** pass a non-``None`` raw ``StructField`` (or any
          subclass of :attr:`_raw_cls`).  The SemanticField wraps it as
          usual.
        * **Source-fields blueprint:** pass ``source_fields=<spec>``.
          The SemanticField acts as a deferred specification of
          "find me this raw field (or this combination of raw fields)
          on an EventData table".  :attr:`is_blueprint` becomes ``True``;
          :attr:`_blueprint_source_fields` holds the spec.  Materialise
          via ``EventData.get_field(blueprint)``.
        * **Plain blueprint:** pass only ``name=<str>``.  The
          SemanticField is materialised with the class's
          :attr:`pa_schema` and the supplied name, and is intended to
          be promoted to a live field by :meth:`from_array` against a source
          array.  :attr:`is_blueprint` is also ``True`` in this mode;
          :attr:`_blueprint_source_fields` remains ``None``.

        The ``source_fields`` spec accepts:

        * a ``str`` naming a single source field exposed by the
          EventData (matched against existing DataField instances first,
          then against raw ``pa.Field`` entries in the underlying
          ``pa.Table``);
        * a ``dict`` mapping this class's :attr:`pa_schema` sub-field
          names to source field specifications.  Each value is itself
          either a ``str`` (resolved like a top-level string) or a
          ``dict`` continuing the recursion down a nested struct
          sub-schema.  The dict must be congruent with :attr:`pa_schema`
          — see :func:`resolve_source_fields`.

        Args:
            raw: The wrapped raw DataField (live mode), or ``None`` for
                blueprint modes.
            source_fields: Either a name (string) or a nested mapping of
                schema-sub-field → source spec (source-fields blueprint).
            name: Plain-blueprint name (used when neither ``raw`` nor
                ``source_fields`` is supplied).
        """
        if raw is None:
            schema = type(self).pa_schema
            if schema is None:
                raise TypeError(
                    f"{type(self).__name__} cannot be a blueprint without a derived "
                    "pa_schema; check Generic parametrisation"
                )
            raw_cls = type(self)._raw_cls
            if source_fields is not None:
                # Validate congruence (and normalise) up front so misshapen
                # dicts are rejected before reaching the table resolver.
                normalised = resolve_source_fields(source_fields, schema)
                dummy_name = (
                    source_fields
                    if isinstance(source_fields, str)
                    else type(self).__name__
                )
                dummy_field = pa.field(dummy_name, schema)
                self._blueprint_source_fields: "str | dict[str, Any] | None" = (
                    normalised
                )
            elif name is not None:
                dummy_field = pa.field(name, schema)
                self._blueprint_source_fields = None
            else:
                raise TypeError(
                    f"{type(self).__name__} requires either a live raw DataField, "
                    "source_fields= (deferred lookup), or name= (plain blueprint)"
                )
            # A raw class that must be bound to something (a unit, say)
            # cannot be built yet: a blueprint has not resolved what it
            # would be bound to.  Stand a plain StructField in its place;
            # the binding happens when the blueprint is materialised
            # against an EventData table.
            if getattr(raw_cls, "_requires_binding", False):
                raw = StructField(None, dummy_field)
            else:
                raw = raw_cls(None, dummy_field)
            self._is_blueprint = True
        else:
            self._is_blueprint = False
            self._blueprint_source_fields = None

        super().__init__(raw.data, raw.field)
        self._raw: "StructField" = raw

    @property
    def is_blueprint(self) -> bool:
        """``True`` if this instance is a deferred-resolution blueprint."""
        return self._is_blueprint

    @property
    def value(self) -> "StructField":
        """The inner raw field, providing access to its schema and methods.

        Statically typed as :class:`StructField` (the upper bound of the
        :attr:`_raw_cls` ClassVar).  Subclasses that pin a more specific
        ``_raw_cls`` (e.g. ``TimeScalarField`` uses
        :class:`DenominateNumberField`) MAY override this property's
        annotation in their own body to expose the refined type to
        callers.
        """
        return self._raw

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped raw field."""
        return getattr(self._raw, name)

    def __repr__(self) -> str:
        length = len(self) if not self.is_empty else 0
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"raw={type(self._raw).__name__}, len={length})"
        )

    def to_field(self) -> pa.Field:
        """Return the ``pa.Field`` descriptor carrying the versioned TTA blob.

        Every SemanticField stamps exactly once, here: the payload from
        :meth:`metadata_dict` is encoded by
        :func:`metadata_blob_from_dict` and written under
        :data:`TIMETOALIGN_METADATA_KEY`.  Metadata entries owned by
        anyone else are carried over untouched.
        """
        metadata = dict(self._field.metadata or {})
        metadata[TIMETOALIGN_METADATA_KEY] = metadata_blob_from_dict(
            self.metadata_dict()
        )
        return self._field.with_metadata(metadata)

    def _repr_rows(self) -> list[tuple[str, str]]:
        """Prepend the scalar-type row to the base Length/Arrow/Sample rows."""
        scalar_name = self.scalar_cls.__name__ if self.scalar_cls is not None else "—"
        return [("Scalar type", scalar_name)] + super()._repr_rows()

    def _repr_affordances(self) -> list[str]:
        """Return the invocation snippets for the ``Try`` row."""
        return [
            "field[i] -> <Scalar>",
            "field.convert_to(<TargetScalar>)",
            "field.get_raw()",
        ]

    def get_raw(self) -> "StructField":
        """Return the underlying raw field (strips the semantic layer)."""
        return self._raw

    def __getitem__(self, i: int) -> Any:
        """Return the *i*-th element as a typed scalar instance.

        Default implementation:

        * If ``cls.scalar_cls`` exposes a classmethod ``from_row``, the
          inner raw dict (or value) is passed through it — yielding a
          fully-validated pydantic scalar.
        * Otherwise falls back to the underlying ``DataField`` behaviour
          (``.as_py()`` on the element, typically a dict).

        Subclasses that need a richer materialisation override this
        method (e.g. ``CoordinateField`` carries a ``unit`` that is not
        in the field's data, so its ``__getitem__`` injects the unit).
        """
        raw = self._raw[i]
        cls = type(self)
        if cls.scalar_cls is not None and raw is not None:
            from_row = getattr(cls.scalar_cls, "from_row", None)
            if callable(from_row):
                return from_row(raw)
        return raw

    @classmethod
    def from_field(cls, source: Any, **kw: Any) -> "SemanticField[T]":
        """Construct a SemanticField from a source.

        Default implementation accepts the four common source shapes
        (``pa.Array``, ``pa.ChunkedArray``, ``StructField``, ``pa.Field``,
        or a ``(data, pa.Field)`` tuple) and wraps each in an instance
        of :attr:`_raw_cls` (defaults to :class:`StructField`).
        Subclasses that need additional semantic state (e.g.
        ``TimeScalarField`` carrying a ``unit``) override this method.

        Args:
            source: One of the accepted source shapes.
            **kw: Subclass-specific keyword arguments (e.g. ``name``).

        Returns:
            A new SemanticField instance.

        Raises:
            TypeError: If *source* is not a recognised shape.
        """
        name = kw.pop("name", cls.__name__.removesuffix("Field").lower() or "value")
        raw_cls = cls._raw_cls

        if isinstance(source, tuple):
            data, pa_field = source
            raw = raw_cls(data, pa_field)
            return cls(raw)

        if isinstance(source, pa.Field):
            raw = raw_cls(None, source)
            return cls(raw)

        if isinstance(source, StructField):
            return cls(source)

        if isinstance(source, (pa.Array, pa.ChunkedArray)):
            pa_field = pa.field(name, source.type)
            raw = raw_cls(source, pa_field)
            return cls(raw)

        raise TypeError(
            f"Unsupported source type for {cls.__name__}.from_field: {type(source).__name__}"
        )

    @classmethod
    def supports_plain_blueprint(cls) -> bool:
        """Whether ``cls(name=...)`` builds a blueprint on its own.

        The plain-blueprint mode of :meth:`__init__` is what lets a
        caller hand a bare source column to a field class and get that
        class's own packing rule applied.  Subclasses that cannot exist
        without further state — a bound time unit, a pair of endpoints —
        override ``__init__`` without that mode, so there is no way to
        ask them to read an atomic column.  Callers that pack a source
        into a target shape check this first, and say so plainly when
        the answer is no, rather than letting a constructor fail.

        Returns:
            True when the class accepts ``name=`` on its own.
        """
        return "name" in inspect.signature(cls.__init__).parameters

    def from_array(
        self,
        source: pa.Array | pa.ChunkedArray,
        *,
        name: str | None = None,
    ) -> "SemanticField[T]":
        """Materialise a live SemanticField from a source column.

        Resolution by ``cls.pa_schema``:

        * If the schema is the canonical rational struct
          (``{value, numerator, denominator}``), parse via
          :func:`~timetoalign.core.time.build_number_struct_array`.
        * If the schema is a single-sub-field struct (e.g.
          ``{value: string}`` for :class:`IdField`, ``{midi_number: int}``
          for :class:`EnharmonicPitchField`), pack the (cast) atomic
          source into the struct shape directly.
        * Otherwise fall back to a row-wise ``model_validate`` of each
          source row followed by the column-builder.  This path is
          reserved for richer scalars whose storage shape exceeds the
          single-atomic-value case; a class whose source column fills
          only some of its sub-fields overrides this method instead
          (see :meth:`MeasureNumberField.from_array`, where a label
          column fills ``mn`` and leaves ``mc`` and ``volta`` null).
        """
        if not self.is_empty:
            raise TypeError(
                f"{type(self).__name__}.from_array() can only be called on a blueprint "
                f"(empty DataField); this instance has live data"
            )
        cls = type(self)
        schema = cls.pa_schema
        if schema is None:
            raise TypeError(
                f"{cls.__name__} has no derived pa_schema; "
                "cannot materialise from an array"
            )
        out_name = name if name is not None else self.name
        field_names = [schema.field(i).name for i in range(schema.num_fields)]

        # Canonical number shape — build through the one number builder.
        if field_names == ["value", "numerator", "denominator"]:
            from .time import build_number_struct_array, infer_number_type

            struct_arr = build_number_struct_array(
                source, number_type=infer_number_type(source), on_error="null"
            )
            return cls.from_field((struct_arr, pa.field(out_name, struct_arr.type)))

        # Single-sub-field struct — pack the atomic source.
        if schema.num_fields == 1:
            sub_field = schema.field(0)
            sub_type = sub_field.type
            if isinstance(source, pa.ChunkedArray):
                source = source.combine_chunks()
            if not (
                pa.types.is_string(source.type)
                or pa.types.is_large_string(source.type)
                or pa.types.is_integer(source.type)
                or pa.types.is_floating(source.type)
            ):
                source = pc.cast(source, pa.string())
            if pa.types.is_integer(sub_type) or pa.types.is_floating(sub_type):
                casted = pc.cast(source, sub_type)
            elif pa.types.is_string(sub_type):
                casted = (
                    source if source.type == sub_type else pc.cast(source, sub_type)
                )
            else:
                raise TypeError(
                    f"{cls.__name__}.from_array: unsupported sub-field type {sub_type!r}"
                )
            struct_arr = pa.StructArray.from_arrays([casted], fields=list(schema))
            pa_field = pa.field(out_name, schema)
            return cls.from_field((struct_arr, pa_field))

        # Generic fallback: row-wise validate + column-builder.
        scalar_cls = cls.scalar_cls
        if scalar_cls is None:
            raise TypeError(
                f"{cls.__name__}.from_array fallback requires a paired scalar_cls"
            )
        if isinstance(source, pa.ChunkedArray):
            source = source.combine_chunks()
        objects: list[Any] = []
        for row in source.to_pylist():
            objects.append(None if row is None else scalar_cls.model_validate(row))
        struct_arr = build_struct_array(
            scalar_cls, [o for o in objects if o is not None]
        )
        pa_field = pa.field(out_name, schema)
        return cls.from_field((struct_arr, pa_field))

    @classmethod
    def matches_pa_field(cls, pa_field: pa.Field) -> bool:
        """Return True iff *pa_field* is a field this class can wrap.

        Two checks, applied in order:

        1. **Metadata-based identification (round-trip contract).** If
           ``pa_field`` carries the ``b"timetoalign"`` JSON blob with a
           ``field_type`` matching ``cls.__name__``, the field is
           definitively claimed.
        2. **Structural identification.** If ``cls.pa_schema`` is
           populated (every concrete subclass) and ``pa_field.type``
           matches it exactly (struct with identical sub-field names
           and types), the field is claimed.

        Subclasses MAY override to relax or tighten these checks.

        Args:
            pa_field: A ``pa.Field`` from the underlying table schema.

        Returns:
            ``True`` if this class can wrap the field described by
            *pa_field*; ``False`` otherwise.
        """
        # Metadata-based identification
        if (
            pa_field.metadata is not None
            and TIMETOALIGN_METADATA_KEY in pa_field.metadata
        ):
            blob = pa_field.metadata[TIMETOALIGN_METADATA_KEY]
            try:
                if isinstance(blob, bytes):
                    blob = blob.decode("utf-8")
                meta = json.loads(blob)
            except (json.JSONDecodeError, UnicodeDecodeError):
                meta = {}
            if isinstance(meta, dict) and meta.get("field_type") == cls.__name__:
                return True

        # Structural identification — exact struct-shape match.
        expected = cls.pa_schema
        if expected is None:
            return False
        if not pa.types.is_struct(pa_field.type):
            return False
        return _struct_types_match(pa_field.type, expected)


def _resolve_scalar_cls(cls: type) -> type[BaseModel] | None:
    """Walk ``__orig_bases__`` to find the ``SemanticField[T]`` parametrisation.

    Returns the pydantic ``BaseModel`` subclass ``T`` when found, or
    ``None`` when the class is an intermediate abstract subclass that
    does not pin ``T`` (e.g. ``TimeScalarField`` extends ``SemanticField``
    without a concrete parametrisation; its children ``CoordinateField``
    and ``DurationField`` set ``scalar_cls`` in their bodies).
    """
    for base in getattr(cls, "__orig_bases__", ()) or ():
        origin = get_origin(base)
        if origin is None:
            continue
        if origin is SemanticField or (
            isinstance(origin, type) and issubclass(origin, SemanticField)
        ):
            args = get_args(base)
            if not args:
                continue
            arg = args[0]
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
    return None


def _struct_types_match(actual: pa.DataType, expected: pa.StructType) -> bool:
    """Strict structural match of two struct types — names and types both."""
    if not pa.types.is_struct(actual):
        return False
    if actual.num_fields != expected.num_fields:
        return False
    for i in range(expected.num_fields):
        a, e = actual.field(i), expected.field(i)
        if a.name != e.name:
            return False
        if not a.type.equals(e.type):
            return False
    return True


def resolve_source_fields(
    source: "str | dict[str, Any]",
    target_schema: pa.StructType,
) -> "str | dict[str, Any]":
    """Validate and normalise a ``source_fields`` blueprint specification.

    The blueprint can be either a single name or a mapping of
    *target-schema-sub-field-name* → spec.  This helper enforces
    structural congruence between the dict shape and ``target_schema``;
    actual lookup against an EventData / table happens later, in the
    resolver that owns the table context.

    Accepted specs:

    * ``str``: a name (of a DataField on the EventData, or a top-level
      field on its ``pa.Table``).  Returned unchanged.  Top-level
      strings imply "this name carries a struct already congruent with
      ``target_schema``" — the resolver will validate that downstream.
    * ``dict[str, str | dict]``: keys MUST match the names of the
      sub-fields in ``target_schema`` exactly (same set, no extras, no
      gaps).  Each value is recursively resolved:

        - if the corresponding target sub-field is itself a struct, the
          value may again be a dict (recursion) or a string (resolver
          will look the name up at the EventData / table level);
        - if the target sub-field is atomic, the value MUST be a
          string.

    Args:
        source: The user-supplied spec.
        target_schema: The ``pa.StructType`` the SemanticField pins.

    Returns:
        The same spec (after recursive validation), suitable for
        downstream resolvers.

    Raises:
        TypeError: If *source* is neither a string nor a mapping, or if
            an inner value has the wrong shape.
        ValueError: If a dict's key set is not congruent with the target
            schema's sub-field names.
    """
    if isinstance(source, str):
        if not source:
            raise ValueError("source_fields string must be non-empty")
        return source

    if not isinstance(source, dict):
        raise TypeError(
            "source_fields must be a string or a dict mapping target "
            f"sub-field names to sub-specs, got {type(source).__name__}"
        )

    expected_names = {
        target_schema.field(i).name for i in range(target_schema.num_fields)
    }
    got_names = set(source.keys())
    if got_names != expected_names:
        missing = sorted(expected_names - got_names)
        extra = sorted(got_names - expected_names)
        raise ValueError(
            "source_fields dict is not congruent with the target schema. "
            f"Expected keys: {sorted(expected_names)}; "
            f"missing: {missing}; unexpected: {extra}."
        )

    normalised: dict[str, Any] = {}
    for i in range(target_schema.num_fields):
        sub_field = target_schema.field(i)
        sub_spec = source[sub_field.name]
        if pa.types.is_struct(sub_field.type):
            normalised[sub_field.name] = resolve_source_fields(sub_spec, sub_field.type)
        else:
            if not isinstance(sub_spec, str) or not sub_spec:
                raise TypeError(
                    f"source_fields[{sub_field.name!r}] expects a non-empty "
                    f"string (atomic sub-field), got {sub_spec!r}"
                )
            normalised[sub_field.name] = sub_spec
    return normalised


# ═══════════════════════════════════════════════════════════════════════════
# 2. PYDANTIC → PYARROW TRANSLATOR
# ═══════════════════════════════════════════════════════════════════════════

# A value projector maps a single pydantic field to a *sequence* of
# ``pa.Field`` entries that will replace the field in the derived struct.
# Used by ``Coordinate``: the scalar's single ``value: int|float|Fraction``
# field expands to three storage fields ``{value, numerator, denominator}``
# inside the Arrow struct.  An empty list drops the field entirely
# (used to keep e.g. ``Note.pitch`` out of the pa.Schema).
_ValueProjector = Callable[[type[BaseModel], str, FieldInfo], list[pa.Field]]

_VALUE_PROJECTORS: dict[tuple[type[BaseModel], str], _ValueProjector] = {}


def _lookup_value_projector(
    model_cls: type[BaseModel], field_name: str
) -> _ValueProjector | None:
    """Find the most-specific projector registered for ``(C, field_name)``
    where ``C`` is ``model_cls`` or any of its bases. Subclasses inherit
    projectors from their ancestors; a subclass registration overrides."""
    for c in model_cls.__mro__:
        proj = _VALUE_PROJECTORS.get((c, field_name))
        if proj is not None:
            return proj
    return None


def register_value_projector(
    model_cls: type[BaseModel],
    field_name: str,
    projector: _ValueProjector,
) -> None:
    """Register a custom field-expansion rule for a scalar field.

    The projector receives ``(model_cls, field_name, field_info)`` and
    returns a list of ``pa.Field`` entries that replace the single
    pydantic field in the derived pa.Schema.  Used for denormalised
    storage projections that the type alone cannot express (e.g.
    ``Coordinate.value`` → ``{value, numerator, denominator}``) and for
    columnar-separation drops (e.g. ``Note.pitch`` → ``[]``).

    Registration is idempotent on the (cls, field_name) pair — later
    registrations override earlier ones.  Calling this also invalidates
    the per-class derivation cache.

    Projectors are inherited by subclasses via MRO walk — register on the
    class that declares the pydantic field; subclasses automatically pick
    it up. Register on a subclass to override.

    Args:
        model_cls: The pydantic ``BaseModel`` subclass.
        field_name: The pydantic field name to project.
        projector: Callable returning a list of ``pa.Field`` replacements.
    """
    _VALUE_PROJECTORS[(model_cls, field_name)] = projector
    _derive_arrow_fields.cache_clear()


def _is_tuple_type(py_type: Any) -> bool:
    origin = get_origin(py_type)
    return origin is tuple


def _atomic_arrow_type(py_type: Any) -> pa.DataType:
    """Map a pydantic atomic / structural type annotation to a ``pa.DataType``."""
    if py_type is str:
        return pa.string()
    if py_type is int:
        return pa.int64()
    if py_type is float:
        return pa.float64()
    if py_type is bool:
        return pa.bool_()
    if py_type is Fraction:
        # An exact ratio stores the way every other number in the library
        # stores: the canonical redundant struct, not a second {num, den}
        # shape invented per scalar.
        from .time import RATIONAL_STRUCT_TYPE

        return RATIONAL_STRUCT_TYPE

    origin = get_origin(py_type)

    if origin is Literal:
        args = get_args(py_type)
        if all(isinstance(a, str) for a in args):
            return pa.string()
        if all(isinstance(a, int) for a in args):
            return pa.int64()
        raise TypeError(
            f"Mixed-type Literal not supported in pa.Schema derivation: {py_type!r}"
        )

    if isinstance(py_type, type) and issubclass(py_type, BaseModel):
        return pa.struct(_derive_arrow_fields(py_type))

    # A StrEnum stores as its string value; the scalar's validator rebuilds
    # the member from that string on read (pydantic coerces str -> StrEnum).
    if isinstance(py_type, type) and issubclass(py_type, enum.StrEnum):
        return pa.string()

    if _is_tuple_type(py_type):
        args = get_args(py_type)
        if len(args) == 2 and args[1] is Ellipsis:
            inner_pa = _atomic_arrow_type(args[0])
            return pa.list_(inner_pa)
        if len(args) >= 1 and Ellipsis not in args:
            children = [
                pa.field(f"_{i}", _atomic_arrow_type(a), nullable=True)
                for i, a in enumerate(args)
            ]
            return pa.struct(children)
        raise TypeError(
            f"Unsupported tuple annotation {py_type!r}; use tuple[T, ...] "
            "for variadic or tuple[T1, T2, ...] for fixed-length."
        )

    raise TypeError(
        f"Cannot derive PyArrow type for {py_type!r}: not a supported type. "
        "Supported: str, int, float, bool, Literal[str|int, ...], nested "
        "BaseModel, tuple[T, ...] (variadic), tuple[T1, T2, ...] (fixed). "
        "Nested unsupported shapes need to be added when a future scalar "
        "migration encounters them — extend timetoalign.core.fields."
    )


def _unwrap_optional(py_type: Any) -> tuple[Any, bool]:
    """Strip an ``Optional`` wrapper, returning ``(inner_type, was_optional)``.

    Accepts any of the spellings ``T | None``, ``Optional[T]``, and
    ``Union[T, None]`` (the latter coming via ``typing.Union``).  Returns
    the inner type ``T`` together with a flag indicating whether the
    annotation was wrapped (``True``) or already bare (``False``).
    Non-union types pass through unchanged with the flag ``False``.

    Rejected shapes — these are intentionally not normalised because the
    pydantic → PyArrow translator has no principled way to emit them:

    * A union of two or more pydantic ``BaseModel`` subclasses.  Storing
      such a union as a single field would defeat the rule that distinct
      scalar types live in distinct fields; the calling site should
      drop the offending field from the derived ``pa.Schema`` via
      :func:`register_value_projector` returning ``[]``.
    * Any other multi-arm union (e.g. ``str | int``).  Such cases must
      be split into separate fields or modelled with a tagged-union
      schema by the caller.

    Args:
        py_type: A type annotation harvested from ``BaseModel.model_fields``.

    Returns:
        ``(inner_type, was_optional)``.  When the annotation is already
        bare, ``inner_type is py_type`` and ``was_optional is False``.

    Raises:
        TypeError: When the annotation is a union shape this helper
            refuses to normalise (see "Rejected shapes" above).
    """
    origin = get_origin(py_type)
    union_type = getattr(sys.modules.get("types"), "UnionType", None)
    is_union = origin is union_type or (
        origin is not None and getattr(origin, "__name__", None) == "Union"
    )
    if not is_union:
        return py_type, False
    args = [a for a in get_args(py_type) if a is not type(None)]
    if len(args) != 1:
        basemodel_arms = [
            a for a in args if isinstance(a, type) and issubclass(a, BaseModel)
        ]
        if len(basemodel_arms) >= 2:
            raise TypeError(
                "Union of BaseModel subclasses is not supported: distinct "
                f"scalar types must live in distinct fields. Got {py_type!r}. "
                "Drop the offending field from the pa.Schema via "
                "register_value_projector(cls, name, lambda *_: [])."
            )
        raise TypeError(
            f"Only Optional[T] / T | None unions are supported, got {py_type!r}"
        )
    return args[0], True


@lru_cache(maxsize=None)
def _derive_arrow_fields(model_cls: type[BaseModel]) -> tuple[pa.Field, ...]:
    """Translate a pydantic model class into a tuple of ``pa.Field``.

    Cached per class — derivation runs exactly once per scalar.  Computed
    fields (``@computed_field``) are NOT included: they derive on access
    and are intentionally absent from both ``pa.Schema`` and the
    underlying Arrow data.

    All derived fields are marked ``nullable=True``.  Pydantic
    ``required`` describes what a single scalar instance must contain;
    the ``pa.Schema`` describes fielded storage where nulls are
    first-class.
    """
    out: list[pa.Field] = []
    for name, info in model_cls.model_fields.items():
        proj = _lookup_value_projector(model_cls, name)
        if proj is not None:
            out.extend(proj(model_cls, name, info))
            continue

        ann = info.annotation
        if ann is None:
            raise TypeError(
                f"Field {model_cls.__name__}.{name} has no annotation; cannot "
                "derive PyArrow type"
            )
        inner, _optional = _unwrap_optional(ann)
        pa_type = _atomic_arrow_type(inner)
        out.append(pa.field(name, pa_type, nullable=True))
    return tuple(out)


def derive_arrow_struct(model_cls: type[BaseModel]) -> pa.StructType:
    """Return the PyArrow struct type for a pydantic scalar.

    The struct's children are derived once per class and cached.

    Args:
        model_cls: A pydantic v2 ``BaseModel`` subclass.

    Returns:
        ``pa.StructType`` whose children mirror the model's fields
        (with value-projector expansions applied).
    """
    return pa.struct(_derive_arrow_fields(model_cls))


def derive_arrow_schema(model_cls: type[BaseModel]) -> pa.Schema:
    """Return a ``pa.Schema`` whose fields mirror a pydantic scalar's fields.

    Convenience for callers that want a top-level schema (rather than a
    nested struct).  The schema is **not** decorated with the
    ``b"timetoalign"`` metadata blob — use
    :func:`parquet_metadata_for_model` when constructing the ``pa.Field``
    that holds the struct.
    """
    return pa.schema(_derive_arrow_fields(model_cls))


# ═══════════════════════════════════════════════════════════════════════════
# 3. STORE BUILDER — pa.StructArray from a sequence of validated scalars
# ═══════════════════════════════════════════════════════════════════════════


def build_struct_array(
    model_cls: type[BaseModel],
    objects: Sequence[BaseModel | None],
) -> pa.StructArray:
    """Construct a ``pa.StructArray`` from a sequence of pydantic instances.

    Implements the bulk-build pattern: for each pydantic field name,
    materialise one ``pa.Array`` by pulling that attribute from every
    scalar, then assemble the per-field arrays into a single struct
    array.  This is the canonical bulk-construction path.

    **Nested ``BaseModel`` fields** (e.g. ``Note.start: Coordinate``,
    ``Measure.duration: Duration``) are recursed into — the inner field
    is built by recursive ``build_struct_array`` (or
    ``build_coordinate_struct_array`` for ``Coordinate`` / ``Duration``)
    so that the result is a nested ``pa.StructArray`` matching the
    derived storage shape.

    **Projector-replaced fields** (Coordinate's ``value``/``unit``;
    ``Note.pitch``) are honoured.

    ``None`` entries in *objects* are represented as null struct entries.

    This function is a thin facade over :class:`StructArrayBuilder` —
    the class is the unified store-builder; the function preserves the
    historical name.

    Args:
        model_cls: The pydantic ``BaseModel`` subclass whose fields drive
            the build.
        objects: A sequence of *model_cls* instances (or ``None`` for null
            rows).

    Returns:
        A ``pa.StructArray`` matching ``derive_arrow_struct(model_cls)``.

    Raises:
        TypeError: If a non-``None`` entry is not an instance of *model_cls*.
    """
    return StructArrayBuilder.from_model(model_cls).build(objects)


class StructArrayBuilder:
    """Traverse a sequence of scalars once; emit a ``pa.StructArray``.

    The builder generalises the bulk-build pattern: given a target
    ``pa.StructType`` and a strategy for extracting per-sub-field values
    from each scalar, it walks every object exactly once, accumulating
    Python lists of per-sub-field values, then materialises each list
    into a ``pa.Array`` and composes the result into a ``pa.StructArray``.

    Two construction paths cover the cases the library needs today:

    * :meth:`from_model` — for pydantic ``BaseModel`` subclasses.  The
      target schema is derived (with value-projector expansions
      applied), and per-sub-field extraction uses pydantic attribute
      access plus recursion into nested ``BaseModel`` fields.  The
      ``Coordinate`` / ``Duration`` denormalisation is dispatched to
      :func:`build_coordinate_struct_array` for the relevant nested
      attributes; once a registered rational-denormalisation hook is in
      place, that branch will collapse into the generic path.
    * :meth:`from_schema_and_extractors` — for raw shapes that are not
      driven by a pydantic model.  Used by
      :func:`build_coordinate_struct_array` to extract
      ``{value, numerator, denominator}`` directly from objects that
      merely expose a ``.value`` attribute.

    Why the builder is a class rather than a procedure: it lets callers
    cache the derived schema and sub-block map, and gives a single
    obvious extension point (a new ``from_*`` classmethod) when a new
    extraction strategy is required.
    """

    def __init__(
        self,
        *,
        target_schema: pa.StructType,
        pa_fields: list[pa.Field],
        validate_obj: "Callable[[Any], None] | None",
        emit_sub_arrays: "Callable[[Sequence[Any | None]], list[pa.Array]]",
    ) -> None:
        self._target_schema = target_schema
        self._pa_fields = pa_fields
        self._validate_obj = validate_obj
        self._emit_sub_arrays = emit_sub_arrays

    # -- public construction paths -----------------------------------------

    @classmethod
    def from_model(cls, model_cls: type[BaseModel]) -> "StructArrayBuilder":
        """Build a builder driven by a pydantic ``BaseModel`` subclass."""
        target_schema = derive_arrow_struct(model_cls)
        pa_fields: list[pa.Field] = list(target_schema)
        sub_blocks = _group_pa_fields_by_pydantic_field(model_cls, pa_fields)

        def _validate(obj: Any) -> None:
            if obj is None:
                return
            if not isinstance(obj, model_cls):
                raise TypeError(
                    f"Expected {model_cls.__name__} (or None), got "
                    f"{type(obj).__name__}"
                )

        def _emit(objects: Sequence[Any | None]) -> list[pa.Array]:
            arrays: list[pa.Array] = []
            for py_field_name, pa_block in sub_blocks.items():
                if py_field_name is None:
                    raise TypeError(
                        "pa.Schema field cannot be mapped back to a pydantic "
                        "field; extend the store builder for this projector shape."
                    )
                py_field_info = model_cls.model_fields[py_field_name]
                values: list[Any] = [
                    None if obj is None else getattr(obj, py_field_name)
                    for obj in objects
                ]
                arrays.extend(
                    _build_field_arrays(
                        model_cls=model_cls,
                        field_name=py_field_name,
                        field_info=py_field_info,
                        pa_subfields=pa_block,
                        values=values,
                    )
                )
            return arrays

        return cls(
            target_schema=target_schema,
            pa_fields=pa_fields,
            validate_obj=_validate,
            emit_sub_arrays=_emit,
        )

    @classmethod
    def from_schema_and_extractors(
        cls,
        target_schema: pa.StructType,
        extractors: "dict[str, Callable[[Sequence[Any | None]], pa.Array]]",
    ) -> "StructArrayBuilder":
        """Build a builder driven by an explicit schema and per-sub-field extractors.

        Each key in ``extractors`` MUST match a sub-field name in
        ``target_schema``.  The extractor receives the full object
        sequence (with ``None`` for null rows) and returns a single
        ``pa.Array`` whose type matches the sub-field.
        """
        pa_fields: list[pa.Field] = list(target_schema)
        expected = {f.name for f in pa_fields}
        if set(extractors.keys()) != expected:
            missing = sorted(expected - set(extractors.keys()))
            extra = sorted(set(extractors.keys()) - expected)
            raise ValueError(
                "Extractors are not congruent with target_schema. "
                f"Expected: {sorted(expected)}; missing: {missing}; "
                f"unexpected: {extra}."
            )

        def _emit(objects: Sequence[Any | None]) -> list[pa.Array]:
            return [extractors[f.name](objects) for f in pa_fields]

        return cls(
            target_schema=target_schema,
            pa_fields=pa_fields,
            validate_obj=None,
            emit_sub_arrays=_emit,
        )

    # -- build --------------------------------------------------------------

    def build(self, objects: Sequence[Any | None]) -> pa.StructArray:
        """Walk *objects* once and assemble the ``pa.StructArray``."""
        objects = list(objects)
        null_mask: list[bool] = []
        validate = self._validate_obj
        for obj in objects:
            if obj is None:
                null_mask.append(True)
                continue
            if validate is not None:
                validate(obj)
            null_mask.append(False)

        sub_arrays = self._emit_sub_arrays(objects)
        return pa.StructArray.from_arrays(
            sub_arrays,
            fields=self._pa_fields,
            mask=pa.array(null_mask) if any(null_mask) else None,
        )


def _group_pa_fields_by_pydantic_field(
    model_cls: type[BaseModel],
    pa_fields: list[pa.Field],
) -> "dict[str | None, list[pa.Field]]":
    """Map each ``pa.Field`` back to the pydantic field that produced it."""
    pyd_names = list(model_cls.model_fields.keys())
    pyd_name_set = set(pyd_names)
    out: dict[str | None, list[pa.Field]] = {n: [] for n in pyd_names}

    matched_pa: list[pa.Field] = []
    for pf in pa_fields:
        if pf.name in pyd_name_set:
            out[pf.name].append(pf)
            matched_pa.append(pf)

    leftover = [pf for pf in pa_fields if pf not in matched_pa]
    if not leftover:
        return {k: v for k, v in out.items() if v}

    for pyd_name in pyd_names:
        proj = _lookup_value_projector(model_cls, pyd_name)
        if proj is None:
            continue
        emitted = proj(model_cls, pyd_name, model_cls.model_fields[pyd_name])
        emitted_names = [f.name for f in emitted]
        if not emitted_names:
            continue
        captured = [pf for pf in leftover if pf.name in emitted_names]
        for pf in captured:
            out[pyd_name].append(pf)
        leftover = [pf for pf in leftover if pf not in captured]

    if leftover:
        return {**out, None: leftover}

    return {k: v for k, v in out.items() if v}


def _build_field_arrays(
    *,
    model_cls: type[BaseModel],
    field_name: str,
    field_info: Any,
    pa_subfields: list[pa.Field],
    values: list[Any],
) -> list[pa.Array]:
    """Produce one or more ``pa.Array`` for a single pydantic field's slot."""
    if not pa_subfields:
        return []

    annotation = field_info.annotation
    inner_type, _nullable = _unwrap_optional_annotation(annotation)

    if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
        if _is_coordinate_like(inner_type):
            # Imported here rather than at module scope: core/time.py owns the
            # number storage struct and builds on this module, so the
            # dependency only runs in this direction at call time.
            from .time import build_coordinate_struct_array

            return [build_coordinate_struct_array(values)]
        sub_array = build_struct_array(inner_type, values)
        return [sub_array]

    if len(pa_subfields) != 1:
        raise TypeError(
            f"Cannot build {len(pa_subfields)} arrays for atomic pydantic field "
            f"{model_cls.__name__}.{field_name}"
        )
    pa_field = pa_subfields[0]
    pa_type = pa_field.type
    if inner_type is Fraction:
        from .enums import NumberType
        from .time import build_number_struct_array

        return [build_number_struct_array(values, number_type=NumberType.fraction)]
    if pa.types.is_struct(pa_type) and all(
        pa_type.field(i).name == f"_{i}" for i in range(pa_type.num_fields)
    ):
        n = pa_type.num_fields
        rows: list[Any] = []
        for v in values:
            if v is None:
                rows.append(None)
            else:
                rows.append({f"_{i}": v[i] for i in range(n)})
        return [pa.array(rows, type=pa_type)]
    if pa.types.is_list(pa_type):
        rows = []
        for v in values:
            rows.append(None if v is None else list(v))
        return [pa.array(rows, type=pa_type)]
    return [pa.array(values, type=pa_type)]


def _unwrap_optional_annotation(annotation: Any) -> tuple[Any, bool]:
    origin = get_origin(annotation)
    union_type = getattr(sys.modules.get("types"), "UnionType", None)
    is_union = origin is union_type or (
        origin is not None and getattr(origin, "__name__", None) == "Union"
    )
    if not is_union:
        return annotation, False
    args = [a for a in get_args(annotation) if a is not type(None)]
    if len(args) == 1:
        return args[0], True
    return annotation, False


def _is_coordinate_like(tp: type) -> bool:
    """Detect ``Coordinate`` / ``Duration`` (and Id-variants) without a hard import.

    The TimeScalar value-projector denormalises ``value`` into three
    fields and drops ``unit`` (and ``timeline_id`` for the Id-variants);
    this helper is the signal that the nested-field builder must route
    through :func:`build_coordinate_struct_array` rather than the generic
    recursion.
    """
    if tp.__name__ in (
        "TimeScalar",
        "Coordinate",
        "IdCoordinate",
        "Duration",
        "IdDuration",
    ) and tp.__module__.endswith(".core.time"):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# 4. PARQUET METADATA
# ═══════════════════════════════════════════════════════════════════════════

TIMETOALIGN_METADATA_KEY: bytes = b"timetoalign"
"""The bytes key used inside ``pa.Field.metadata`` for the TTA blob.

This module is the **sole** owner of the key: nothing else in the
library spells the byte string out.  Import this constant instead.
"""

TIMETOALIGN_BLOB_VERSION: int = 1
"""Schema version of the payload stored under :data:`TIMETOALIGN_METADATA_KEY`.

Every blob written by :func:`metadata_blob_from_dict` or
:func:`metadata_blob_for_model` carries this number under the
``"version"`` key, and :func:`parse_metadata_blob` refuses any blob that
lacks it or declares a newer version than this build understands.  Bump
the constant whenever the payload's meaning changes.
"""

_BLOB_VERSION_KEY: str = "version"
"""Payload key holding :data:`TIMETOALIGN_BLOB_VERSION`."""


@lru_cache(maxsize=None)
def _cached_json_schema_bytes(model_cls: type[BaseModel]) -> bytes:
    """Return the versioned ``model_json_schema()`` serialised to UTF-8 bytes."""
    payload = dict(model_cls.model_json_schema())
    payload[_BLOB_VERSION_KEY] = TIMETOALIGN_BLOB_VERSION
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def metadata_blob_for_model(model_cls: type[BaseModel]) -> bytes:
    """Return the JSON-encoded, versioned ``model_json_schema()`` for a model.

    This is the **payload** that lives under
    ``pa.Field.metadata[TIMETOALIGN_METADATA_KEY]``.  It is identical for
    every ``pa.Field`` carrying the same scalar type and is cached so
    repeated calls return the same bytes object.  The model's JSONSchema
    is emitted verbatim except for the added
    ``"version": TIMETOALIGN_BLOB_VERSION`` entry.
    """
    return _cached_json_schema_bytes(model_cls)


def metadata_blob_from_dict(payload: dict[str, Any]) -> bytes:
    """Return versioned, JSON-encoded UTF-8 bytes from a payload dict.

    Used by SemanticField subclasses (Coordinate, Duration) to keep
    their per-instance payload shape (which records the *runtime*
    ``unit``) while still routing through the unified metadata helper.
    Migrated scalars whose schema is entirely pydantic-derived can use
    :func:`metadata_blob_for_model` directly.

    ``"version"`` is injected (overwriting any caller-supplied entry) so
    that no unversioned blob can leave this module.
    """
    return json.dumps(
        {**payload, _BLOB_VERSION_KEY: TIMETOALIGN_BLOB_VERSION}, sort_keys=True
    ).encode("utf-8")


def parquet_metadata_for_model(
    model_cls: type[BaseModel],
    *,
    extra: dict[bytes, bytes] | None = None,
) -> dict[bytes, bytes]:
    """Return the ``pa.Field.metadata`` dict for a scalar's pydantic model.

    The returned dict is suitable for passing directly into
    ``pa.field(..., metadata=...)`` or ``pa.Field.with_metadata(...)``.
    Always contains the :data:`TIMETOALIGN_METADATA_KEY` entry with the
    versioned JSONSchema payload; *extra* entries are merged in if
    provided.
    """
    metadata: dict[bytes, bytes] = {
        TIMETOALIGN_METADATA_KEY: metadata_blob_for_model(model_cls)
    }
    if extra:
        metadata.update(extra)
    return metadata


def field_metadata(pa_field: pa.Field) -> dict[str, Any]:
    """Return the TimeToAlign! payload attached to *pa_field*, or ``{}``.

    The one way to read a field's unit, representation or timeline id.
    Everything the library writes goes into the versioned blob under
    :data:`TIMETOALIGN_METADATA_KEY`, so nothing needs to know a second
    place to look, and a field written by another tool simply has no
    payload rather than a half-understood one.
    """
    raw_meta = pa_field.metadata
    if not raw_meta or TIMETOALIGN_METADATA_KEY not in raw_meta:
        return {}
    try:
        return parse_metadata_blob(raw_meta[TIMETOALIGN_METADATA_KEY])
    except (UnicodeDecodeError, ValueError):
        return {}


def blob_metadata(**payload: Any) -> dict[bytes, bytes]:
    """Build a field-metadata mapping carrying *payload* in the versioned blob.

    The counterpart to :func:`field_metadata`: every producer in the library
    writes through here, so there is exactly one on-disk convention and a
    reader never has to try two.  Enum-valued entries are recorded by their
    value, so callers can pass a ``TimeUnit`` straight in.
    """
    encoded = {
        key: (value.value if hasattr(value, "value") else str(value))
        for key, value in payload.items()
        if value is not None
    }
    return {TIMETOALIGN_METADATA_KEY: metadata_blob_from_dict(encoded)}


def parse_metadata_blob(blob: bytes | str | None) -> dict[str, Any]:
    """Parse a :data:`TIMETOALIGN_METADATA_KEY` payload back into a dict.

    The ``"version"`` entry is validated and kept in the result, so a
    caller can round-trip the dict straight back through
    :func:`metadata_blob_from_dict`.

    Args:
        blob: The bytes (or already-decoded string) stored under
            :data:`TIMETOALIGN_METADATA_KEY` in ``pa.Field.metadata``.
            ``None`` (or empty) returns an empty dict.

    Returns:
        The decoded payload dictionary, or ``{}`` if *blob* is empty.

    Raises:
        ValueError: If the payload is not a JSON object, carries no
            integer ``"version"``, or declares a version newer than
            :data:`TIMETOALIGN_BLOB_VERSION`.
    """
    if blob is None:
        return {}
    if isinstance(blob, bytes):
        blob = blob.decode("utf-8")
    if not blob:
        return {}
    payload = json.loads(blob)
    if not isinstance(payload, dict):
        raise ValueError(
            f"TimeToAlign! metadata blob must be a JSON object, got "
            f"{type(payload).__name__}"
        )
    version = payload.get(_BLOB_VERSION_KEY)
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(
            "TimeToAlign! metadata blob carries no integer "
            f"{_BLOB_VERSION_KEY!r} entry; every blob must be written by "
            "metadata_blob_from_dict() or metadata_blob_for_model()"
        )
    if version > TIMETOALIGN_BLOB_VERSION:
        raise ValueError(
            f"TimeToAlign! metadata blob declares version {version}, but this "
            f"build understands at most {TIMETOALIGN_BLOB_VERSION}"
        )
    return payload
