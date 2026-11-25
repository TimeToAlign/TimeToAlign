"""The runtime types the PyArrow stub helpers in ``core/fields.py`` state.

Each helper states a type for a PyArrow call that the type stubs declare
differently; these tests check every stated type against PyArrow itself.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from timetoalign.core.fields import (
    _cast_column,
    _compared,
    _ComparisonKernel,
    _dictionary_chunked_array,
    _filled,
    _if_else,
    _joined_strings,
    _null_column,
    _struct_array_from,
    _struct_array_of,
    _struct_child,
    _struct_member,
    _struct_members,
    _valid_struct_cell,
)
from timetoalign.core.time import RATIONAL_STRUCT_TYPE

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


def test_filled_boolean_mask_is_a_boolean_array() -> None:
    result = _filled(pa.array([True, None, False]), pa.scalar(False))
    assert type(result) is pa.BooleanArray
    assert result.to_pylist() == [True, False, False]
    assert result.null_count == 0


def test_filled_chunked_mask_is_chunked() -> None:
    chunked = pa.chunked_array([pa.array([True, None]), pa.array([False])])
    result = _filled(chunked, pa.scalar(False))
    assert type(result) is pa.ChunkedArray
    assert result.type == pa.bool_()
    assert result.to_pylist() == [True, False, False]
    assert result.null_count == 0


def test_filled_column_of_no_chunks_is_chunked() -> None:
    result = _filled(pa.chunked_array([], type=pa.bool_()), pa.scalar(False))
    assert type(result) is pa.ChunkedArray
    assert result.type == pa.bool_()
    assert result.num_chunks == 0


def test_filled_strings_are_a_string_array() -> None:
    result = _filled(pa.array(["C", None]), pa.scalar(""))
    assert type(result) is pa.StringArray
    assert result.to_pylist() == ["C", ""]
    only_nulls = _filled(pa.nulls(2, pa.string()), pa.scalar(""))
    assert type(only_nulls) is pa.StringArray
    assert only_nulls.to_pylist() == ["", ""]


def test_joined_strings_is_a_string_array() -> None:
    result = _joined_strings(pa.array(["C", None]), pa.array(["4", "5"]), "")
    assert type(result) is pa.StringArray
    assert result.to_pylist() == ["C4", None]


def test_if_else_of_struct_arrays_is_a_struct_array() -> None:
    left = pa.array([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], type=PAIR)
    right = pa.array([{"a": 3, "b": "z"}, None], type=PAIR)
    result = _if_else(pa.array([True, False]), left, right)
    assert type(result) is pa.StructArray
    assert result.type == PAIR
    assert result.to_pylist() == [{"a": 1, "b": "x"}, None]


def test_if_else_of_boolean_arrays_is_a_boolean_array() -> None:
    result = _if_else(
        pa.array([True, None, False]),
        pa.array([True, True, True]),
        pa.array([False, False, None]),
    )
    assert type(result) is pa.BooleanArray
    assert result.to_pylist() == [True, None, None]


def test_row_wise_helpers_on_no_rows_keep_their_class() -> None:
    mask = _filled(pa.array([], pa.bool_()), pa.scalar(False))
    assert type(mask) is pa.BooleanArray
    assert len(mask) == 0
    joined = _joined_strings(pa.array([], pa.string()), pa.array([], pa.string()), "")
    assert type(joined) is pa.StringArray
    assert len(joined) == 0
    empty = pa.array([], type=PAIR)
    chosen = _if_else(pa.array([], pa.bool_()), empty, empty)
    assert type(chosen) is pa.StructArray
    assert chosen.type == PAIR
    assert len(chosen) == 0


def test_null_column_is_an_array_of_the_type() -> None:
    struct_nulls = _null_column(2, PAIR)
    assert type(struct_nulls) is pa.StructArray
    assert struct_nulls.type == PAIR
    assert struct_nulls.to_pylist() == [None, None]
    assert struct_nulls.null_count == 2
    no_strings = _null_column(0, pa.string())
    assert type(no_strings) is pa.StringArray
    assert len(no_strings) == 0
    untyped = _null_column(2, pa.null())
    assert type(untyped) is pa.NullArray
    assert len(untyped) == 2
    dictionary_type = pa.dictionary(pa.int8(), pa.string())
    labels = _null_column(2, dictionary_type)
    assert type(labels) is pa.DictionaryArray
    assert labels.type == dictionary_type
    assert labels.to_pylist() == [None, None]


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


def _rational_column() -> pa.StructArray:
    """Three rational cells, the second masked over the placeholders 9.0, 9, 1."""
    return _struct_array_from(
        [
            pa.array([0.5, 9.0, 2.0]),
            pa.array([1, 9, 2]),
            pa.array([2, 1, None]),
        ],
        fields=list(RATIONAL_STRUCT_TYPE),
        mask=pa.array([False, True, False]),
    )


def test_struct_members_are_typed_arrays_with_the_struct_nulls() -> None:
    result = _struct_members(_rational_column())
    assert type(result) is list
    assert [type(member) for member in result] == [
        pa.DoubleArray,
        pa.Int64Array,
        pa.Int64Array,
    ]
    assert [member.to_pylist() for member in result] == [
        [0.5, None, 2.0],
        [1, None, 2],
        [2, None, None],
    ]


def test_struct_members_of_a_slice_apply_the_offset() -> None:
    result = _struct_members(_rational_column()[1:])
    assert [type(member) for member in result] == [
        pa.DoubleArray,
        pa.Int64Array,
        pa.Int64Array,
    ]
    assert [member.to_pylist() for member in result] == [
        [None, 2.0],
        [None, 2],
        [None, None],
    ]


def test_struct_members_of_no_rows_keep_their_class() -> None:
    combined = pa.chunked_array([], type=RATIONAL_STRUCT_TYPE).combine_chunks()
    assert type(combined) is pa.StructArray
    result = _struct_members(combined)
    assert type(result) is list
    assert [type(member) for member in result] == [
        pa.DoubleArray,
        pa.Int64Array,
        pa.Int64Array,
    ]
    assert [len(member) for member in result] == [0, 0, 0]


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        ("equal", [False, True, False, None]),
        ("less", [True, False, False, None]),
        ("less_equal", [True, True, False, None]),
        ("greater", [False, False, True, None]),
        ("greater_equal", [False, True, True, None]),
    ],
)
def test_compared_is_the_kernel_as_a_boolean_array(
    kernel: _ComparisonKernel, expected: list[bool | None]
) -> None:
    left = pa.array([-1, 0, 1, None])
    result = _compared(kernel, left, pa.scalar(0))
    assert type(result) is pa.BooleanArray
    assert result.to_pylist() == expected
    assert result.equals(pc.call_function(kernel, [left, pa.scalar(0)]))
    assert result.equals(getattr(pc, kernel)(left, pa.scalar(0)))


def test_compared_with_a_python_number_is_a_boolean_array() -> None:
    left = pa.array([0.5, 2.0, None])
    as_int = _compared("less", left, 1)
    assert type(as_int) is pa.BooleanArray
    assert as_int.to_pylist() == [True, False, None]
    as_float = _compared("greater_equal", left, 2.0)
    assert type(as_float) is pa.BooleanArray
    assert as_float.to_pylist() == [False, True, None]


def test_compared_with_a_string_scalar_is_a_boolean_array() -> None:
    result = _compared("equal", pa.array(["int", "float", None]), pa.scalar("int"))
    assert type(result) is pa.BooleanArray
    assert result.to_pylist() == [True, False, None]


def test_compared_with_a_python_string_is_the_equal_kernel() -> None:
    left = pa.array(["score", "clock", None])
    result = _compared("equal", left, "score")
    assert type(result) is pa.BooleanArray
    assert result.to_pylist() == [True, False, None]
    assert result.equals(pc.equal(left, "score"))


def test_compared_with_a_second_array_is_a_boolean_array() -> None:
    wide = pa.decimal256(37, 0)
    left = pa.array([1, 2, None], pa.int64()).cast(wide)
    right = pa.array([2, 2, 3], pa.int64()).cast(wide)
    result = _compared("less", left, right)
    assert type(result) is pa.BooleanArray
    assert result.to_pylist() == [True, False, None]


def test_compared_on_no_rows_is_a_boolean_array() -> None:
    result = _compared("equal", pa.array([], pa.int64()), pa.scalar(0))
    assert type(result) is pa.BooleanArray
    assert len(result) == 0
