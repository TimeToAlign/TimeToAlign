"""Shared validation and projection for coordinate-retrieval APIs."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Set
from fractions import Fraction
from typing import Any, Literal, TypeAlias, TypeVar

import numpy as np
import pandas as pd

from .enums import NumberType, TimeUnit
from .events import Address
from .time import Coordinate, IdCoordinate, number_from_json, number_to_json

CoordinateInput: TypeAlias = (
    int | float | Fraction | Coordinate | IdCoordinate | Address
)
CoordinateCollection: TypeAlias = (
    Sequence[CoordinateInput] | np.ndarray | pd.Index | pd.Series
)
KeyCollection: TypeAlias = Sequence[str] | pd.Index | pd.Series
CoordinateResult: TypeAlias = IdCoordinate | Coordinate | float | int | Fraction
CoordinateFormat: TypeAlias = Literal[
    "id_coordinate", "coordinate", "float", "int", "fraction", "series"
]
Rounding: TypeAlias = Literal["round", "floor", "ceil", "truncate"]
TableFormat: TypeAlias = Literal["table", "dataframe"]

COORDINATE_FORMATS: tuple[str, ...] = (
    "id_coordinate",
    "coordinate",
    "float",
    "int",
    "fraction",
    "series",
)
ROUNDING_MODES: tuple[str, ...] = ("round", "floor", "ceil", "truncate")
TABLE_FORMATS: tuple[str, ...] = ("table", "dataframe")

ResultT = TypeVar("ResultT")
EntryT = TypeVar("EntryT")


def number_type_for_converted_unit(preferred: NumberType, unit: TimeUnit) -> NumberType:
    """Return a valid canonical type for a converted unit coordinate.

    Args:
        preferred: Number type declared by the selected timeline axis.
        unit: Unit requested for the converted representation.

    Returns:
        The preferred type when the target unit admits it, otherwise the
        target unit's default type.
    """
    if preferred in unit.allowed_number_types:
        return preferred
    return unit.default_number_type


def validate_retrieval_options(format: str, rounding: str) -> None:
    """Validate the closed output-format and rounding vocabularies.

    Args:
        format: Requested coordinate output format.
        rounding: Integral projection mode.

    Raises:
        ValueError: If either option is outside its closed vocabulary.
    """
    if format not in COORDINATE_FORMATS:
        raise ValueError(
            f"Unknown coordinate format {format!r}. Use one of "
            f"{', '.join(COORDINATE_FORMATS)}."
        )
    if rounding not in ROUNDING_MODES:
        raise ValueError(
            f"Unknown rounding mode {rounding!r}. Use one of "
            f"{', '.join(ROUNDING_MODES)}."
        )


def validate_table_format(format: str) -> None:
    """Validate the closed table-output-format vocabulary.

    Args:
        format: Requested table output format.

    Raises:
        ValueError: If the value is outside the closed vocabulary.
    """
    if format not in TABLE_FORMATS:
        raise ValueError(
            f"Unknown table format {format!r}. Use one of {', '.join(TABLE_FORMATS)}."
        )


def reject_dataframe_options(format: str, **options: object) -> None:
    """Raise when a dataframe-shaping option is supplied for an Arrow result.

    The shaping options (field naming, unit suffixes, an event-ID index) only
    describe a pandas rendering; an Arrow table carries its names and units in
    field metadata instead. Passing one alongside ``format="table"`` therefore
    asks for something the result cannot express, and silently ignoring it
    would hide that.

    Args:
        format: Requested table output format.
        **options: Shaping options, ``None`` where the caller left them unset.

    Raises:
        ValueError: If any option is set while *format* is not
            ``"dataframe"``.
    """
    if format == "dataframe":
        return
    offending = [name for name, value in options.items() if value is not None]
    if offending:
        raise ValueError(
            f"{', '.join(sorted(offending))} only shape a DataFrame result; "
            f'pass format="dataframe" to use them.'
        )


def is_coordinate_input(value: object) -> bool:
    """Return whether *value* is one accepted scalar coordinate input."""
    return not isinstance(value, bool) and isinstance(
        value, (int, float, Fraction, Coordinate, Address)
    )


def collection_values(value: object) -> tuple[list[object], pd.Index | None]:
    """Validate and eagerly materialize one accepted one-dimensional collection.

    Args:
        value: Candidate coordinate or key collection.

    Returns:
        Materialized values and a pandas index to preserve, when supplied.

    Raises:
        TypeError: If the value is not an accepted finite, one-dimensional
            collection.
    """
    if isinstance(value, (str, bytes, bytearray, Mapping, Set, Iterator)):
        raise TypeError(
            f"Unsupported collection type: {type(value).__name__}; expected a "
            "finite one-dimensional sequence, NumPy array, pandas Index, or Series"
        )
    if isinstance(value, pd.Series):
        return [_python_numeric(item) for item in value.array], value.index
    if isinstance(value, pd.Index):
        return [_python_numeric(item) for item in value], value
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise TypeError("NumPy coordinate collections must be one-dimensional")
        if value.dtype.kind not in "biufO":
            raise TypeError(
                f"Unsupported NumPy dtype {value.dtype}; expected numeric or object"
            )
        return [_python_numeric(item) for item in value], None
    if isinstance(value, Sequence):
        return list(value), None
    raise TypeError(
        f"Unsupported collection type: {type(value).__name__}; expected a "
        "finite one-dimensional sequence, NumPy array, pandas Index, or Series"
    )


def _python_numeric(value: object) -> object:
    """Normalize a NumPy scalar to the corresponding Python scalar."""
    return value.item() if isinstance(value, np.generic) else value


def validate_coordinate_collection(
    value: object,
) -> tuple[list[CoordinateInput], pd.Index | None]:
    """Validate a plural coordinate input atomically and preserve pandas index.

    Args:
        value: Candidate collection.

    Returns:
        Coordinate inputs and optional preserved pandas index.

    Raises:
        TypeError: If any element is not a scalar coordinate input.
    """
    values, index = collection_values(value)
    for position, item in enumerate(values):
        if not is_coordinate_input(item):
            raise TypeError(
                f"Coordinate collection element {position} has unsupported type "
                f"{type(item).__name__}"
            )
    return values, index  # type: ignore[return-value]


def validate_key_collection(value: object) -> tuple[list[str], pd.Index | None]:
    """Validate a plural key input atomically and preserve pandas index.

    Args:
        value: Candidate key collection.

    Returns:
        String keys and optional preserved pandas index.

    Raises:
        TypeError: If any element is not a string.
    """
    values, index = collection_values(value)
    for position, item in enumerate(values):
        if not isinstance(item, str):
            raise TypeError(
                f"Key collection element {position} has unsupported type "
                f"{type(item).__name__}"
            )
    return values, index  # type: ignore[return-value]


def classify_dispatch_input(value: object, *, empty_is_keys: bool = False) -> str:
    """Classify a dispatcher input as scalar/plural coordinate or key input.

    Args:
        value: Dispatcher input.
        empty_is_keys: Route an empty collection to the key branch.

    Returns:
        One of ``"coordinate"``, ``"coordinates"``, ``"key"``, or
        ``"keys"``.

    Raises:
        TypeError: If the runtime form is unsupported or mixes key and
            coordinate elements.
    """
    if isinstance(value, str):
        return "key"
    if is_coordinate_input(value):
        return "coordinate"
    values, _ = collection_values(value)
    if not values:
        return "keys" if empty_is_keys else "coordinates"
    are_keys = [isinstance(item, str) for item in values]
    are_coordinates = [is_coordinate_input(item) for item in values]
    if all(are_keys):
        return "keys"
    if all(are_coordinates):
        return "coordinates"
    raise TypeError(
        "Dispatcher collections must contain only string keys or only coordinate inputs"
    )


def dispatch_retrieval(
    receiver: object,
    singular: str,
    plural: str,
    at: object,
    timeline_id: str | None = None,
    *,
    empty_is_keys: bool = False,
    position_options: Mapping[str, Any] | None = None,
    positions_only: str | None = None,
    **options: Any,
) -> Any:
    """Route one query to the precise getter its input form names.

    Every ``get_<thing>(at=...)`` convenience method on every receiver selects
    among the same four precise getters by the same rule, so the rule lives
    here once: a position goes to ``_at``, an identity to ``_for``, and a
    collection to the plural spelling of whichever it is.

    Args:
        receiver: Object exposing the four precise getters.
        singular: Scalar method stem, e.g. ``"get_timestamp"``.
        plural: Plural method stem, e.g. ``"get_timestamps"``.
        at: Scalar or plural position or key input.
        timeline_id: Second positional argument of every precise getter.
        empty_is_keys: Route an empty collection to the key branch.
        position_options: Keyword options the positional getters accept and
            the key getters do not — a query unit, for instance.
        positions_only: Message for a receiver whose dispatcher has no key
            branch; a key input then raises instead of falling through.
        **options: Keyword options every precise getter accepts.

    Returns:
        Whatever the selected precise getter returns.

    Raises:
        TypeError: If the runtime form is unsupported, mixes key and
            coordinate elements, or names a branch the receiver omits.
    """
    branch = classify_dispatch_input(at, empty_is_keys=empty_is_keys)
    stem = singular if branch in ("coordinate", "key") else plural
    positional = branch in ("coordinate", "coordinates")
    if not positional and positions_only is not None:
        raise TypeError(positions_only)
    suffix = "_at" if positional else "_for"
    if positional and position_options:
        options = {**options, **position_options}
    return getattr(receiver, f"{stem}{suffix}")(at, timeline_id, **options)


def resolve_coordinate_collection(
    at: object, resolve: Callable[[CoordinateInput], ResultT]
) -> tuple[list[ResultT], pd.Index | None]:
    """Validate a coordinate collection and resolve every element in order.

    Validation runs over the whole collection before the first lookup, and a
    failing lookup propagates immediately, so a plural query either answers
    completely or raises — it never returns a partial result.

    Args:
        at: Candidate coordinate collection.
        resolve: Scalar resolver applied to each element.

    Returns:
        Resolved results in input order and any preserved pandas index.
    """
    values, index = validate_coordinate_collection(at)
    return [resolve(value) for value in values], index


def resolve_key_collection(
    keys: object, resolve: Callable[[str], ResultT]
) -> tuple[list[ResultT], pd.Index | None]:
    """Validate a key collection and resolve every element in order.

    Args:
        keys: Candidate key collection.
        resolve: Scalar resolver applied to each key.

    Returns:
        Resolved results in input order and any preserved pandas index.
    """
    values, index = validate_key_collection(keys)
    return [resolve(key) for key in values], index


def is_key_input(value: object) -> bool:
    """Return whether a query input names identities rather than positions.

    Args:
        value: Scalar or plural query input.

    Returns:
        ``True`` for a key or key collection, ``False`` for coordinates.

    Raises:
        TypeError: If the runtime form is unsupported or mixes key and
            coordinate elements.
    """
    return classify_dispatch_input(value) in ("key", "keys")


def _series_from_coordinates(
    coordinates: Sequence[IdCoordinate],
    *,
    index: pd.Index | None,
    name: str,
    empty_number_type: NumberType | None,
) -> pd.Series:
    """Build the canonical numeric pandas view for coordinate results."""
    values = [coordinate.value for coordinate in coordinates]
    result_index = index if index is not None else pd.RangeIndex(len(values))
    if not values:
        if empty_number_type is NumberType.float:
            dtype: str | type[object] = "float64"
        elif empty_number_type is NumberType.int:
            dtype = "int64"
        else:
            dtype = object
        return pd.Series(values, index=result_index, name=name, dtype=dtype)
    axes = {coordinate.timeline_id for coordinate in coordinates}
    if len(axes) != 1:
        return pd.Series(values, index=result_index, name=name, dtype=object)
    kinds = {coordinate.number_type for coordinate in coordinates}
    if len(kinds) != 1:
        return pd.Series(values, index=result_index, name=name, dtype=object)
    kind = coordinates[0].number_type.name
    if kind == "fraction":
        return pd.Series(values, index=result_index, name=name, dtype=object)
    if kind == "float":
        return pd.Series(values, index=result_index, name=name, dtype="float64")
    try:
        return pd.Series(values, index=result_index, name=name, dtype="int64")
    except OverflowError:
        return pd.Series(values, index=result_index, name=name, dtype=object)


def format_coordinates(
    coordinates: Sequence[IdCoordinate],
    *,
    format: CoordinateFormat = "id_coordinate",
    rounding: Rounding = "round",
    scalar: bool,
    index: pd.Index | None = None,
    series_name: str | None = None,
    empty_number_type: NumberType | None = None,
) -> CoordinateResult | list[CoordinateResult] | pd.Series:
    """Project canonical ID coordinates into one requested public format.

    Args:
        coordinates: Canonical result coordinates.
        format: Requested output format.
        rounding: Integral projection mode.
        scalar: Whether the caller supplied a scalar query.
        index: Optional pandas index to preserve.
        series_name: Required Series name; defaults to the sole result axis.
        empty_number_type: Declared axis representation for an empty Series.

    Returns:
        One scalar, a list of scalar leaves, or a pandas Series.

    Raises:
        ValueError: If an option is invalid or a scalar result does not have
            exactly one coordinate.
    """
    validate_retrieval_options(format, rounding)
    if scalar and len(coordinates) != 1:
        raise ValueError("A scalar coordinate query must resolve exactly one result")
    if format == "series":
        name = series_name
        if name is None:
            name = coordinates[0].timeline_id if coordinates else "coordinate"
        return _series_from_coordinates(
            coordinates,
            index=index,
            name=name,
            empty_number_type=empty_number_type,
        )

    def project(coordinate: IdCoordinate) -> CoordinateResult:
        if format == "id_coordinate":
            return coordinate
        if format == "coordinate":
            return Coordinate(
                coordinate.value,
                coordinate.unit,
                number_type=coordinate.number_type,
            )
        if format == "float":
            return coordinate.to_float()
        if format == "int":
            return coordinate.to_int(rounding)
        return coordinate.to_fraction()

    projected = [project(coordinate) for coordinate in coordinates]
    return projected[0] if scalar else projected


def coordinate_to_dict(coordinate: Coordinate) -> dict[str, object]:
    """Serialize one coordinate using the TimeSkeleton representation.

    Args:
        coordinate: Coordinate to serialize.

    Returns:
        A typed coordinate dictionary.
    """
    return {
        "unit": coordinate.unit.value,
        "number_type": coordinate.number_type.name,
        "value": number_to_json(coordinate.value),
    }


def coordinate_from_dict(entry: Mapping[str, Any]) -> Coordinate:
    """Restore one coordinate from the TimeSkeleton representation.

    Args:
        entry: Typed coordinate dictionary.

    Returns:
        The exact coordinate declared by the entry.
    """
    if not isinstance(entry, Mapping):
        raise ValueError(f"coordinate must be an object, got {entry!r}")
    for key in ("unit", "number_type", "value"):
        if key not in entry:
            raise ValueError(f"coordinate is missing {key!r}: {entry!r}")
    try:
        unit = TimeUnit(entry["unit"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"unknown coordinate unit {entry['unit']!r} for key 'unit'"
        ) from error
    try:
        number_type = NumberType[entry["number_type"]]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"unknown coordinate number type {entry['number_type']!r} for key 'number_type'"
        ) from error
    try:
        number_from_json(entry["value"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"invalid coordinate value for key 'value': {entry['value']!r}"
        ) from error
    try:
        value = number_from_json(entry["value"], number_type)
    except ValueError as error:
        raise ValueError(
            f"coordinate value for key 'value' does not match declared number_type "
            f"{number_type.name!r}: {entry['value']!r}"
        ) from error
    return Coordinate(value, unit, number_type=number_type)


def canonical_json(data: Any) -> str:
    """Return the canonical JSON text of a JSON-safe value.

    Object keys are sorted by Unicode code point and no whitespace is written,
    so equal values always give equal text.

    Args:
        data: A JSON-safe value.

    Returns:
        The compact JSON text with sorted keys.

    Raises:
        ValueError: If the value holds a NaN or an infinity, which JSON cannot
            represent.
    """
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_order(
    entries: Iterable[EntryT],
    key: Callable[[EntryT], Sequence[str | Mapping[str, Any]]],
    *,
    json_tie_break: bool = False,
) -> list[EntryT]:
    """Return serialized entries in the canonical order of a timeline dictionary.

    *key* names the members an entry is ordered by, most significant first.
    A coordinate dictionary compares by its exact numeric value, never by its
    JSON spelling (``9`` before ``10``, ``[5, 8]`` before ``[2, 3]``), and a
    string by Unicode code point. With *json_tie_break*, entries equal on every
    key are further ordered by their :func:`canonical_json` text. The sort is
    stable, so entries that remain equal keep their given order.

    Args:
        entries: Serialized entries, as written by a ``to_dict`` method.
        key: Returns the ordering members of one entry.
        json_tie_break: Whether the canonical JSON text of the entry breaks
            remaining ties.

    Returns:
        The entries in canonical order.
    """

    def sort_key(entry: EntryT) -> tuple[Any, ...]:
        parts = tuple(
            number_from_json(part["value"]) if isinstance(part, Mapping) else part
            for part in key(entry)
        )
        return (*parts, canonical_json(entry)) if json_tie_break else parts

    return sorted(entries, key=sort_key)
