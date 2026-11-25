"""Provide incoming external-reference storage for timeline instances.

An *external reference* records that an annotation living outside TimeToAlign
points **at** an event of this timeline. The direction is deliberately
incoming: the timeline never stores what it points to, only what points to
it, so a resource can be annotated by many independent tools without the
timeline knowing about any of them.

References are kept in a single PyArrow table per timeline whose schema is
:data:`EXTERNAL_REFERENCE_SCHEMA`:

============== =========================================== ========
column         type                                        nullable
============== =========================================== ========
event_id       ``string``                                  no
external_id    ``string``                                  no
access_points  ``list<struct<uri: string, kind: string>>``  no
comment        ``string``                                  yes
============== =========================================== ========

``event_id`` names an event of this timeline, ``external_id`` is the
identifier the annotation carries inside the external resource, and
``access_points`` lists the locators that resolve the resource. An access
point's ``kind`` is an open vocabulary -- ``"relative_path"`` and ``"url"``
are the common values, but any string is accepted. The list may be empty
when no locator is known; it is never null.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from typing_extensions import Self

# region Schema

ACCESS_POINT_TYPE = pa.struct(
    [
        pa.field("uri", pa.string(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
    ]
)

EXTERNAL_REFERENCE_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("external_id", pa.string(), nullable=False),
        pa.field("access_points", pa.list_(ACCESS_POINT_TYPE), nullable=False),
        pa.field("comment", pa.string(), nullable=True),
    ]
)

if TYPE_CHECKING:
    from .host import TimelineHost as _Host
else:
    _Host = object

# Column names in canonical order; also the exact set a row dict may carry.
EXTERNAL_REFERENCE_COLUMNS: tuple[str, ...] = (
    "event_id",
    "external_id",
    "access_points",
    "comment",
)

# Keys an access-point mapping may carry.
ACCESS_POINT_KEYS: tuple[str, ...] = ("uri", "kind")


def empty_external_reference_table() -> pa.Table:
    """Create an empty table with the canonical external-reference schema.

    Returns:
        A zero-row PyArrow table carrying :data:`EXTERNAL_REFERENCE_SCHEMA`.
    """
    return EXTERNAL_REFERENCE_SCHEMA.empty_table()


# endregion


# region Row Normalization


def _normalize_access_points(value: Any, row_index: int) -> list[dict[str, str]]:
    """Coerce the ``access_points`` cell of one row into canonical dicts.

    Args:
        value: The raw cell -- ``None``, or an iterable of mappings with
            ``uri`` and ``kind`` keys.
        row_index: Position of the row, used in error messages.

    Returns:
        A list of ``{"uri": ..., "kind": ...}`` dicts, possibly empty.

    Raises:
        TypeError: If the cell or one of its entries has the wrong type.
        ValueError: If an entry carries unknown keys or non-string values.
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(
            f"External reference row {row_index}: 'access_points' must be a "
            f"list of mappings, got {value!r}"
        )

    points: list[dict[str, str]] = []
    for point in value:
        if not isinstance(point, Mapping):
            raise TypeError(
                f"External reference row {row_index}: each access point must "
                f"be a mapping, got {point!r}"
            )
        unknown = sorted(set(point) - set(ACCESS_POINT_KEYS))
        if unknown:
            raise ValueError(
                f"External reference row {row_index}: unknown access-point "
                f"key(s) {', '.join(repr(key) for key in unknown)}"
            )
        uri = point.get("uri")
        kind = point.get("kind")
        if not isinstance(uri, str) or not isinstance(kind, str):
            raise ValueError(
                f"External reference row {row_index}: each access point needs "
                f"string 'uri' and 'kind', got {dict(point)!r}"
            )
        points.append({"uri": uri, "kind": kind})
    return points


def _normalize_rows(
    rows: Iterable[Mapping[str, Any]] | pa.Table,
) -> list[dict[str, Any]]:
    """Coerce external-reference input into canonical row dicts.

    Accepts either an iterable of mappings or a PyArrow table; a table is
    read through :meth:`pyarrow.Table.to_pylist` so both inputs follow the
    same validation path. Missing ``access_points`` become an empty list
    and a missing ``comment`` becomes ``None``.

    Args:
        rows: The rows to normalize.

    Returns:
        A list of dicts with exactly the canonical columns.

    Raises:
        TypeError: If a row is not a mapping.
        ValueError: If a row carries unknown columns, or if ``event_id`` /
            ``external_id`` / ``comment`` have the wrong type.
    """
    raw: list[Any] = rows.to_pylist() if isinstance(rows, pa.Table) else list(rows)

    normalized: list[dict[str, Any]] = []
    for row_index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise TypeError(
                f"External reference row {row_index} must be a mapping, got {row!r}"
            )
        unknown = sorted(set(row) - set(EXTERNAL_REFERENCE_COLUMNS))
        if unknown:
            raise ValueError(
                f"External reference row {row_index}: unknown column(s) "
                f"{', '.join(repr(key) for key in unknown)}"
            )

        for name in ("event_id", "external_id"):
            if not isinstance(row.get(name), str):
                raise ValueError(
                    f"External reference row {row_index}: {name!r} must be a "
                    f"string, got {row.get(name)!r}"
                )

        comment = row.get("comment")
        if comment is not None and not isinstance(comment, str):
            raise ValueError(
                f"External reference row {row_index}: 'comment' must be a "
                f"string or None, got {comment!r}"
            )

        normalized.append(
            {
                "event_id": row["event_id"],
                "external_id": row["external_id"],
                "access_points": _normalize_access_points(
                    row.get("access_points"), row_index
                ),
                "comment": comment,
            }
        )
    return normalized


# endregion


class ExternalReferencesMixin(_Host):
    """Provide incoming external-reference storage for timeline instances."""

    @property
    def external_references(self) -> pa.Table:
        """Incoming external references pointing at this timeline's events.

        The table always carries :data:`EXTERNAL_REFERENCE_SCHEMA`; when
        nothing has been added it is empty rather than ``None``. There are
        no query helpers -- filter the returned table with PyArrow directly.

        Returns:
            The PyArrow table of external references.

        Examples:
            >>> tl.external_references.num_rows
            0
            >>> tl.add_external_references(
            ...     [{"event_id": "e1", "external_id": "p2"}]
            ... ).external_references.column("external_id").to_pylist()
            ['p2']
        """
        return self._external_references

    def add_external_references(
        self,
        rows: list[dict[str, Any]] | pa.Table,
        *,
        validate: bool = True,
    ) -> Self:
        """Append incoming external references to this timeline.

        Rows are appended to the existing table; nothing is replaced or
        deduplicated. ``access_points`` defaults to an empty list and
        ``comment`` to ``None`` when a row omits them.

        Args:
            rows: Row dicts (or a PyArrow table) with the canonical columns
                ``event_id``, ``external_id``, ``access_points``, and
                ``comment``. Unknown columns are rejected.
            validate: If True (default), every ``event_id`` must name an
                event of this timeline's own event table.

        Returns:
            ``self``, so calls can be chained.

        Raises:
            KeyError: If ``validate`` is True and one or more ``event_id``
                values are absent from this timeline's events. The message
                names all missing ids, sorted.
            TypeError: If a row or an access point has the wrong type.
            ValueError: If a row carries unknown columns or ill-typed values.

        Examples:
            >>> tl.add_external_references([
            ...     {
            ...         "event_id": "e1",
            ...         "external_id": "p2",
            ...         "access_points": [
            ...             {"uri": "Analisi_1/L1.pnml", "kind": "relative_path"}
            ...         ],
            ...         "comment": "Analisi_1_L1_A",
            ...     }
            ... ])
        """
        normalized = _normalize_rows(rows)

        if validate:
            self._check_external_reference_event_ids(normalized)

        if not normalized:
            return self

        addition = pa.Table.from_pylist(normalized, schema=EXTERNAL_REFERENCE_SCHEMA)
        self._external_references = pa.concat_tables(
            [self._external_references, addition]
        ).combine_chunks()
        return self

    def _check_external_reference_event_ids(self, rows: list[dict[str, Any]]) -> None:
        """Verify that every referenced event id exists on this timeline.

        Args:
            rows: Normalized external-reference rows.

        Raises:
            KeyError: Naming every missing id, sorted, in one message.
        """
        known = set(self._events.table.column("id").to_pylist())
        missing = sorted(
            {row["event_id"] for row in rows if row["event_id"] not in known}
        )
        if missing:
            raise KeyError(
                f"Timeline {self._id!r} has no event(s) with id: "
                f"{', '.join(repr(event_id) for event_id in missing)}"
            )
