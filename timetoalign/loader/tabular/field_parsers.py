"""FieldParser — column processors for the tabular loader pipeline.

A :class:`FieldParser` is a *recipe* for producing one or more
:class:`DataField` instances from a source ``pa.Array`` (a column in a
source table).  Parsers handle the genuinely-procedural cases that a
plain :class:`DataField` blueprint cannot express on its own:
splitting a single column into many fields (separator / regex), or
wrapping an arbitrary user-supplied callable.

The hierarchy:

* :class:`FieldParser` — abstract base; every concrete subclass
  implements :meth:`FieldParser.from_array`.
* :class:`CompositeFieldParser` — one source column splits into many
  fields by separator or named/positional regex.
* :class:`CallableFieldParser` — escape hatch wrapping a user-provided
  ``(pa.Array) -> DataField`` callable.

Atomic / typed column emission is handled by the :class:`DataField`
hierarchy directly — :class:`IntField`, :class:`FloatField`,
:class:`StringField`, :class:`RedundantNumberField`, and every paired
:class:`SemanticField` subclass support blueprint construction
(``IntField(name="x")``) and :meth:`DataField.from_array`-time
materialisation.  The Step-1 dispatcher
:func:`resolve_field_parser` returns either a DataField blueprint or a
FieldParser — both expose a uniform ``from_array(source, name=...)`` API.

The resolution table maps user-friendly inputs (``int`` / ``float`` /
``str`` / ``Fraction`` types, ``pa.DataType`` instances, raw / paired
``DataField`` subclasses, ``FieldParser`` instances, or callables) to
the appropriate producer.

Bulk-emission contract: every parser / blueprint ``from_array()`` is
vectorized over the source array — never a per-row Python loop in
client code.  ``RedundantNumberField.from_array`` hands the column to
:func:`~timetoalign.core.time.build_number_struct_array`, which takes a
numpy route for plain numeric input and reads values one at a time only
for mixed, textual or rational columns, because no array kernel parses
``"3/8"``.  That is the single accepted exception, and it is contained
inside the one number builder.
"""

from __future__ import annotations

import inspect
import re
from fractions import Fraction
from typing import Any, Callable, Sequence, TypeGuard

import pyarrow as pa

from timetoalign.core import (
    FloatField,
    IntField,
    RedundantNumberField,
    SemanticField,
    StringField,
    StructField,
)
from timetoalign.core.fields import DataField

__all__ = [
    "FieldParser",
    "CompositeFieldParser",
    "CallableFieldParser",
    "resolve_field_parser",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. ABSTRACT BASE
# ═══════════════════════════════════════════════════════════════════════════


class FieldParser:
    """Abstract base for column-to-field processors.

    Subclasses implement :meth:`from_array` to turn a source ``pa.Array``
    into a single :class:`DataField`.  The output's name comes from the
    parser's own ``name`` (when set) or from the source column name
    passed to :meth:`from_array`.

    Args:
        name: Optional override.  When ``None``, :meth:`from_array` falls back
            to the source column name supplied by the caller.
    """

    def __init__(self, name: str | None = None) -> None:
        self._name = name

    @property
    def name(self) -> str | None:
        """Override name, or ``None`` if the source column name is used."""
        return self._name

    def from_array(self, source: pa.Array, *, name: str | None = None) -> DataField:
        """Resolve *source* into a typed :class:`DataField`.

        Args:
            source: The raw column from the source ``pa.Table``.
            name: The source column name (used as a fallback when this
                parser carries no ``name=`` override).  Optional —
                parsers with their own ``name=`` ignore it.

        Returns:
            A single :class:`DataField` whose name is ``self.name`` if
            set, otherwise *name*.
        """
        raise NotImplementedError  # pragma: no cover

    def _resolve_name(self, fallback: str | None) -> str:
        """Return the parser's name override, or the *fallback*.

        Raises:
            ValueError: If neither is supplied.
        """
        if self._name is not None:
            return self._name
        if fallback is None:
            raise ValueError(
                f"{type(self).__name__}.from_array requires a name= override "
                "or a fallback name from the caller"
            )
        return fallback


# ═══════════════════════════════════════════════════════════════════════════
# 2. COMPOSITE PARSER — one source column → many fields
# ═══════════════════════════════════════════════════════════════════════════


def _default_part_name(part_spec: Any, index: int) -> str:
    """Pick a default name for an unnamed part in an iterable ``parts=``.

    Order of preference:

    * If *part_spec* is a paired :class:`SemanticField` subclass, use
      its ``__name__`` minus the ``Field`` suffix in snake_case
      (``MeasureNumberField`` → ``measure_number``).
    * If *part_spec* is a :class:`DataField` instance with a ``name``
      attribute, use that name.
    * If *part_spec* is a :class:`FieldParser` instance with a
      non-``None`` :attr:`FieldParser.name`, use that name.
    * Otherwise fall back to ``f"part_{index}"``.
    """
    if isinstance(part_spec, type) and issubclass(part_spec, SemanticField):
        stem = part_spec.__name__.removesuffix("Field")
        return re.sub(r"(?<!^)(?=[A-Z])", "_", stem).lower()
    if isinstance(part_spec, DataField):
        return part_spec.name
    if isinstance(part_spec, FieldParser) and part_spec.name is not None:
        return part_spec.name
    return f"part_{index}"


class CompositeFieldParser(FieldParser):
    """Split one source column into many fields by separator or regex.

    The split strategy is determined at construction time:

    * ``separator="<sep>"`` — split each row's string on a literal
      separator.  The number of resulting parts MUST match
      ``len(parts)`` after splitting (otherwise the row is treated as
      null).
    * ``pattern=<regex>`` — apply a regex with named or positional
      groups.  Named groups must match the keys of a ``dict``-shaped
      ``parts``; positional groups feed the sequence form in order.

    ``parts`` follows the same dict-or-iterable rule as
    ``column_specs``, and each value undergoes the universal
    :func:`resolve_field_parser` resolution — including nested
    :class:`CompositeFieldParser` subclasses for recursive composition.

    Args:
        separator: Literal separator string.  Mutually exclusive with
            ``pattern``.
        pattern: Regex (a ``str`` or compiled :class:`re.Pattern`) with
            named groups (dict-form) or positional groups
            (iterable-form).
        parts: Either a ``dict[str, producer-like]`` or a sequence of
            producer-like entries.  Resolution goes through
            :func:`resolve_field_parser`.
        name: Optional name for the resulting :class:`StructField`.
            Defaults to the source column name passed to :meth:`from_array`.
    """

    def __init__(
        self,
        *,
        separator: str | None = None,
        pattern: str | re.Pattern[str] | None = None,
        parts: dict[str, Any] | Sequence[Any],
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if (separator is None) == (pattern is None):
            raise ValueError(
                "CompositeFieldParser requires exactly one of " "separator= or pattern="
            )
        self._separator = separator
        self._pattern: re.Pattern[str] | None = (
            re.compile(pattern) if isinstance(pattern, str) else pattern
        )

        # Resolve parts to producers (DataField blueprints OR FieldParsers).
        if isinstance(parts, dict):
            self._part_keys: list[str] = list(parts.keys())
            self._part_producers: list[DataField | FieldParser] = [
                resolve_field_parser(value, default_name=key)
                for key, value in parts.items()
            ]
        else:
            seq = list(parts)
            self._part_keys = [_default_part_name(seq[i], i) for i in range(len(seq))]
            self._part_producers = [
                resolve_field_parser(v, default_name=self._part_keys[i])
                for i, v in enumerate(seq)
            ]

    @property
    def part_keys(self) -> list[str]:
        """Names of the sub-fields produced by this composite parser."""
        return list(self._part_keys)

    @property
    def part_producers(self) -> list[DataField | FieldParser]:
        """Resolved producer for each part, in part-key order."""
        return list(self._part_producers)

    def _split_to_parts(self, source: pa.Array) -> dict[str, pa.Array]:
        """Run the split strategy and bin the results by part key.

        Returns a dict ``{part_key: pa.Array}`` where each array is a
        Pythonic string array.  Rows that fail to split produce
        ``None`` in every part.
        """
        as_strings = source.to_pylist()
        n = len(as_strings)
        bins: dict[str, list[Any]] = {key: [None] * n for key in self._part_keys}

        if self._separator is not None:
            sep = self._separator
            expected = len(self._part_keys)
            for i, value in enumerate(as_strings):
                if value is None:
                    continue
                if not isinstance(value, str):
                    value = str(value)
                pieces = value.split(sep)
                if len(pieces) != expected:
                    continue
                for key, piece in zip(self._part_keys, pieces):
                    bins[key][i] = piece.strip()
        else:
            assert self._pattern is not None
            pattern = self._pattern
            named = pattern.groupindex
            for i, value in enumerate(as_strings):
                if value is None:
                    continue
                if not isinstance(value, str):
                    value = str(value)
                m = pattern.match(value)
                if m is None:
                    continue
                if named:
                    for key in self._part_keys:
                        if key in named:
                            bins[key][i] = m.group(key)
                else:
                    groups = m.groups()
                    if len(groups) != len(self._part_keys):
                        continue
                    for key, piece in zip(self._part_keys, groups):
                        bins[key][i] = piece

        return {key: pa.array(values, type=pa.string()) for key, values in bins.items()}

    def from_array(self, source: pa.Array, *, name: str | None = None) -> StructField:
        """Build a :class:`StructField` whose sub-fields are the parts.

        Each sub-producer's ``from_array()`` defines the typed semantics of
        that part.  Nested composites and rational sub-fields nest
        cleanly inside the resulting struct.
        """
        out_name = self._resolve_name(name)
        parts_in = self._split_to_parts(source)

        pa_fields: list[pa.Field] = []
        pa_arrays: list[pa.Array] = []
        for key, producer in zip(self._part_keys, self._part_producers):
            sub_field = producer.from_array(parts_in[key], name=key)
            pa_fields.append(sub_field.field)
            data = sub_field.data
            if data is None:
                raise RuntimeError(
                    f"CompositeFieldParser part {key!r} produced no data array"
                )
            pa_arrays.append(data)

        struct_type = pa.struct(pa_fields)
        struct_arr = pa.StructArray.from_arrays(pa_arrays, fields=list(struct_type))
        pa_field = pa.field(out_name, struct_type)
        return StructField(struct_arr, pa_field)


# ═══════════════════════════════════════════════════════════════════════════
# 3. CALLABLE WRAPPER
# ═══════════════════════════════════════════════════════════════════════════


class CallableFieldParser(FieldParser):
    """Wrap a user-provided ``(pa.Array) -> DataField`` callable.

    The escape hatch in the resolution table — when the standard
    blueprints and the composite parser are insufficient, callers may
    supply any function that takes a ``pa.Array`` and returns a
    :class:`DataField`.  The parser defers entirely to the callable;
    the *name* parameter is forwarded as a keyword argument when the
    callable's signature accepts it (a ``name`` parameter or ``**kwargs``).
    The callable is called once, and any error it raises propagates.

    Args:
        fn: Callable ``(pa.Array) -> DataField`` (optionally
            accepting ``name=`` as a keyword).
        name: Optional override.
    """

    def __init__(
        self,
        fn: Callable[..., DataField],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._fn = fn
        self._takes_name = _accepts_name(fn)

    def from_array(self, source: pa.Array, *, name: str | None = None) -> DataField:
        """Call the wrapped function on *source*, once.

        The output name (this parser's ``name=``, else *name*, else ``"value"``)
        is passed as ``name=`` when the function's signature accepts it;
        otherwise the function is called with *source* alone.

        Args:
            source: The raw column from the source ``pa.Table``.
            name: The source column name, the fallback output name.

        Returns:
            The :class:`DataField` the function returns.
        """
        out_name = (
            self._name
            if self._name is not None
            else (name if name is not None else "value")
        )
        if self._takes_name:
            return self._fn(source, name=out_name)
        return self._fn(source)


def _is_field_function(value: object) -> TypeGuard[Callable[..., DataField]]:
    """Whether a ``column_specs`` entry is taken as a function making a field.

    Any callable is: what it returns is the function's contract with its
    caller (see :class:`CallableFieldParser`), not checked here.
    """
    return callable(value)


def _accepts_name(fn: Callable[..., Any]) -> bool:
    """Whether *fn* can be called with a ``name=`` keyword argument.

    True when its signature has a ``name`` parameter that can be passed by
    keyword, or a ``**kwargs`` parameter. A callable whose signature cannot be
    read (some built-ins) is called without ``name``.
    """
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == "name"
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )
        for parameter in parameters
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. UNIVERSAL RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════


def resolve_field_parser(
    value: Any, *, default_name: str | None = None
) -> DataField | FieldParser:
    """Resolve an entry in ``column_specs`` to a producer.

    A producer is either a :class:`DataField` blueprint (whose
    :meth:`DataField.from_array` materialises the live field) or a
    :class:`FieldParser` instance (whose :meth:`FieldParser.from_array`
    consumes the source array).  Both expose a uniform
    ``from_array(source, name=...)`` API so the loader does not need to
    branch on the producer's category.

    Resolution chain:

    * :class:`FieldParser` instance — returned as-is.
    * :class:`DataField` instance (must be a blueprint, i.e.
      ``is_empty``) — returned as-is.
    * :class:`DataField` subclass (raw or paired
      :class:`SemanticField`) — instantiated as a blueprint with
      ``name=default_name``.
    * Python type (``int`` / ``float`` / ``str`` / ``Fraction``) —
      matched to the corresponding :class:`DataField` blueprint.
    * :class:`pa.DataType` — matched to the corresponding blueprint
      (integer → :class:`IntField`, floating → :class:`FloatField`,
      string → :class:`StringField`).
    * Callable — wrapped in :class:`CallableFieldParser`.

    Args:
        value: The user-supplied entry.
        default_name: Name to use when constructing a blueprint from a
            type / class entry.  Comes from the dict-key in dict-form
            ``column_specs`` and is ``None`` for iterable-form entries
            (in which case the entry MUST carry its own ``name=``).

    Returns:
        A producer with a uniform ``from_array(source, name=...)`` API.

    Raises:
        TypeError: When *value* matches no recognised shape.
    """
    if isinstance(value, FieldParser):
        return value

    if isinstance(value, DataField):
        if not value.is_empty:
            raise TypeError(
                "DataField in column_specs must be a blueprint (empty); "
                f"got live {type(value).__name__}"
            )
        return value

    if isinstance(value, type):
        if issubclass(value, DataField):
            # Paired SemanticField subclasses and raw blueprint-friendly
            # DataField leaves both expose ``name=`` blueprint mode.
            return value(name=default_name)
        if value is int:
            return IntField(name=default_name)
        if value is float:
            return FloatField(name=default_name)
        if value is str:
            return StringField(name=default_name)
        if value is Fraction:
            return RedundantNumberField(name=default_name)

    if isinstance(value, pa.DataType):
        if pa.types.is_integer(value):
            return IntField(name=default_name)
        if pa.types.is_floating(value):
            return FloatField(name=default_name)
        if pa.types.is_string(value) or pa.types.is_large_string(value):
            return StringField(name=default_name)
        raise TypeError(
            f"Cannot resolve pa.DataType {value!r} to a blueprint; supply "
            "a DataField blueprint or FieldParser instance directly"
        )

    if _is_field_function(value):
        return CallableFieldParser(value)

    raise TypeError(
        f"Cannot resolve {value!r} (type {type(value).__name__}) to a "
        "DataField blueprint or FieldParser"
    )
