"""Helpers shared across the whole test suite."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import timetoshare

from timetoalign.core import (
    NumberType,
    build_number_struct_array,
    struct_to_coordinate,
    timestamp_table_to_dataframe,
)


def table_column(table: pa.Table, name: str) -> list[Any]:
    """Return one timestamp-table column as plain scalars.

    Timestamp-table cells are coordinate structs, so ``to_pylist()`` on a
    column yields one dict per row. Tests that care about positions read the
    column through the library's own decoding boundary instead, which is what
    a user reading the table gets.
    """
    frame = timestamp_table_to_dataframe(table, units=False)
    return frame[name].tolist()


def coordinate_values(
    column: pa.Array | pa.ChunkedArray,
    number_type: NumberType = NumberType.float,
) -> list[Any]:
    """Return a coordinate column's positions in one declared representation.

    The axis-collecting helpers carry storage cells, not bare numbers, so a
    test asserting positions has to say which side of the cell it means. That
    is the same thing every reader of a column does, and it is declared, not
    guessed from the digits.
    """
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    return [
        None if cell is None else struct_to_coordinate(cell, number_type)
        for cell in column.to_pylist()
    ]


def coordinate_column(
    values: Any, number_type: NumberType = NumberType.float
) -> pa.StructArray:
    """Encode positions as a coordinate column under one declared type."""
    return build_number_struct_array(values, number_type=number_type)


def filed(value: Any) -> Any:
    """Return a manifest written with bare IDs in the file form, ``#`` + the ID.

    A test writes a manifest as the library holds it, with bare IDs; a
    manifest dict, which ``from_dict`` reads, is in the file form. Only a
    whole manifest (a dict carrying the ``timetoshare`` marker) is
    converted; any other value is returned as it is.
    """
    if isinstance(value, dict) and "timetoshare" in value:
        return timetoshare.with_file_ids(value)
    return value


def bare(value: Any) -> Any:
    """Return a manifest in the file form with bare IDs, as the tests write it.

    The inverse of :func:`filed`, for a whole manifest only: the dict of one
    object inside a manifest is bare already, so handing one here is a
    mistake of the test and raises.
    """
    if not (isinstance(value, dict) and "timetoshare" in value):
        raise TypeError("bare() takes a whole manifest, a dict with 'timetoshare'")
    return timetoshare.with_bare_ids(value)


def reading(reader: Any) -> Any:
    """Return *reader* taking a manifest written with bare IDs (:func:`filed`)."""

    def read(data: Any, **selection: Any) -> Any:
        return reader(filed(data), **selection)

    read.__name__ = reader.__name__
    read.__qualname__ = reader.__qualname__
    return read


def refusal(error: BaseException) -> tuple[str, str, str]:
    """Return the ``(code, path, message)`` of the format validator's refusal.

    The library validates every manifest it reads and writes with
    TimeToShare's validator and passes its refusal on unchanged, a
    ``timetoshare.SchemaValidationError``; a test asserts its code, the
    JSON pointer of the fault (``#/timelines/1/regions/0/id``) and its
    message exactly. The path is where the fault lies, which the message
    alone does not always say: an ID held twice is refused at its second
    holder, the message naming the first.
    """
    assert isinstance(error, timetoshare.SchemaValidationError), repr(error)
    return error.code, timetoshare.json_pointer(error.path), error.message
