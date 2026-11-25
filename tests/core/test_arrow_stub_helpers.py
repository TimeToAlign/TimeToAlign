"""The runtime types the PyArrow stub helpers in ``core/fields.py`` state.

Each helper states a type for a PyArrow call that the type stubs declare
differently; these tests check every stated type against PyArrow itself.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from timetoalign.core.fields import (
    _cast_column,
    _dictionary_chunked_array,
    _filled_mask,
    _joined_strings,
    _struct_array_from,
    _struct_array_of,
    _struct_child,
    _struct_if_else,
    _struct_member,
    _valid_struct_cell,
)

PAIR = pa.struct([pa.field("a", pa.int64()), pa.field("b", pa.string())])


def test_struct_array_of_is_a_struct_array() -> None:
    result = _struct_array_of([{"a": 1, "b": "x"}, None], PAIR)
    assert type(result) is pa.StructArray
    assert result.type == PAIR
    assert result.to_pylist() == [{"a": 1, "b": "x"}, None]
    assert result.null_count == 1


def test_struct_array_of_no_rows_is_a_struct_array() -> None:
    result = _struct_array_of([], PAIR)
    assert type(result) is pa.StructArray
    assert result.type == PAIR
    assert len(result) == 0


def test_valid_struct_cell_is_a_dict() -> None:
    array = pa.array([{"a": 1, "b": "x"}, None], type=PAIR)
    result = _valid_struct_cell(array[0])
    assert type(result) is dict
    assert result == {"a": 1, "b": "x"}


def test_filled_mask_is_a_boolean_array() -> None:
    result = _filled_mask(pa.array([True, None, False]), pa.scalar(False))
    assert type(result) is pa.BooleanArray
    assert result.to_pylist() == [True, False, False]
    assert result.null_count == 0


def test_joined_strings_is_a_string_array() -> None:
    result = _joined_strings(pa.array(["C", None]), pa.array(["4", "5"]), "")
    assert type(result) is pa.StringArray
    assert result.to_pylist() == ["C4", None]


def test_struct_if_else_is_a_struct_array() -> None:
    left = pa.array([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], type=PAIR)
    right = pa.array([{"a": 3, "b": "z"}, None], type=PAIR)
    result = _struct_if_else(pa.array([True, False]), left, right)
    assert type(result) is pa.StructArray
    assert result.type == PAIR
    assert result.to_pylist() == [{"a": 1, "b": "x"}, None]


def test_row_wise_helpers_on_no_rows_keep_their_class() -> None:
    mask = _filled_mask(pa.array([], pa.bool_()), pa.scalar(False))
    assert type(mask) is pa.BooleanArray
    assert len(mask) == 0
    joined = _joined_strings(pa.array([], pa.string()), pa.array([], pa.string()), "")
    assert type(joined) is pa.StringArray
    assert len(joined) == 0
    empty = pa.array([], type=PAIR)
    chosen = _struct_if_else(pa.array([], pa.bool_()), empty, empty)
    assert type(chosen) is pa.StructArray
    assert chosen.type == PAIR
    assert len(chosen) == 0


def test_dictionary_chunked_array_is_a_chunked_array_of_the_type() -> None:
    dictionary_type = pa.dictionary(pa.int8(), pa.string())
    chunks = [
        pa.DictionaryArray.from_arrays(
            pa.array([0, 1], pa.int8()), pa.array(["p", "q"])
        ),
        pa.DictionaryArray.from_arrays(pa.array([1], pa.int8()), pa.array(["p", "q"])),
    ]
    result = _dictionary_chunked_array(chunks, dictionary_type)
    assert type(result) is pa.ChunkedArray
    assert result.type == dictionary_type
    assert result.num_chunks == 2
    assert result.to_pylist() == ["p", "q", "q"]


def test_dictionary_chunked_array_of_no_chunks_keeps_the_type() -> None:
    dictionary_type = pa.dictionary(pa.int8(), pa.string())
    result = _dictionary_chunked_array([], dictionary_type)
    assert type(result) is pa.ChunkedArray
    assert result.type == dictionary_type
    assert result.num_chunks == 0
    assert len(result) == 0


def test_cast_column_keeps_the_array_kind() -> None:
    array = _cast_column(pa.array([1, None, 3]), pa.float64())
    assert type(array) is pa.DoubleArray
    assert array.to_pylist() == [1.0, None, 3.0]
    chunked = _cast_column(pa.chunked_array([[1], [2, 3]]), pa.float64())
    assert type(chunked) is pa.ChunkedArray
    assert chunked.type == pa.float64()
    assert chunked.to_pylist() == [1.0, 2.0, 3.0]
    for sequence in ([1, None, 3], (1, None, 3)):
        packed = _cast_column(sequence, pa.float64())
        assert type(packed) is pa.DoubleArray
        assert packed.to_pylist() == [1.0, None, 3.0]
    numpy_packed = _cast_column(np.array([1, 3], dtype=np.int64), pa.float64())
    assert type(numpy_packed) is pa.DoubleArray
    assert numpy_packed.to_pylist() == [1.0, 3.0]


def test_struct_child_is_the_stored_child() -> None:
    fields = [pa.field("a", pa.int64())]
    array = pa.StructArray.from_arrays(
        [pa.array([1, 2])], fields=fields, mask=pa.array([False, True])
    )
    result = _struct_child(array, "a")
    assert type(result) is pa.Int64Array
    assert result.to_pylist() == [1, 2]


def test_struct_member_applies_the_struct_validity() -> None:
    fields = [pa.field("a", pa.int64())]
    array = pa.StructArray.from_arrays(
        [pa.array([1, 2, None])], fields=fields, mask=pa.array([False, True, False])
    )
    result = _struct_member(array, "a")
    assert type(result) is pa.Int64Array
    assert result.to_pylist() == [1, None, None]


def test_struct_member_of_a_chunked_column_is_chunked() -> None:
    array = pa.array([{"a": 1, "b": "x"}, None, {"a": None, "b": "z"}], type=PAIR)
    chunked = pa.chunked_array([array[:1], array[1:]])
    result = _struct_member(chunked, "b")
    assert type(result) is pa.ChunkedArray
    assert result.type == pa.string()
    assert result.num_chunks == 2
    assert result.to_pylist() == ["x", None, "z"]


def test_struct_member_of_no_rows_keeps_the_kind() -> None:
    plain = _struct_member(pa.array([], type=PAIR), "a")
    assert type(plain) is pa.Int64Array
    assert len(plain) == 0
    no_chunks = _struct_member(pa.chunked_array([], type=PAIR), "a")
    assert type(no_chunks) is pa.ChunkedArray
    assert no_chunks.type == pa.int64()
    assert no_chunks.num_chunks == 0


def test_struct_array_from_is_a_struct_array() -> None:
    members = [pa.array([1, 2]), pa.array(["x", "y"])]
    fields = list(PAIR)
    unmasked = _struct_array_from(members, fields=fields)
    assert type(unmasked) is pa.StructArray
    assert unmasked.type == PAIR
    assert unmasked.to_pylist() == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    masked = _struct_array_from(members, fields=fields, mask=pa.array([False, True]))
    assert type(masked) is pa.StructArray
    assert masked.to_pylist() == [{"a": 1, "b": "x"}, None]
