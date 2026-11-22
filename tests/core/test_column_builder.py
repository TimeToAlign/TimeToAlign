"""Tests for the store-builder pattern (canonical bulk construction).

See README.md "test_column_builder.py" for the gold-standard plan.
"""

from __future__ import annotations

from fractions import Fraction

import pyarrow as pa
import pytest

from timetoalign.core.enums import NumberType, TimeUnit
from timetoalign.core.events import SpecificPitch
from timetoalign.core.fields import (
    build_struct_array,
    derive_arrow_struct,
)
from timetoalign.core.time import (
    Coordinate,
    build_coordinate_struct_array,
    struct_to_coordinate,
)


class TestBuildStructArraySpecificPitch:
    """§1, §2, §3: generic store-builder on SpecificPitch."""

    def test_byte_equivalent_to_model_dump_row_wise(self) -> None:
        """§1: store-builder matches row-wise pa.array(model_dump) field-by-field.

        ``model_dump`` row-wise is the LEGACY/forbidden path; this test
        proves the store-builder produces the same on-disk bytes for
        valid input.
        """
        scalars = [
            SpecificPitch(step="C", alter=0, octave=4),
            SpecificPitch(step="G", alter=1, octave=3, cents=5.0),
            SpecificPitch(step="B", alter=-1, octave=5),
        ]
        struct = derive_arrow_struct(SpecificPitch)
        store_builder_arr = build_struct_array(SpecificPitch, scalars)

        # Legacy / forbidden path, for parity comparison only.
        rows = [
            {name: getattr(sp, name) for name in SpecificPitch.model_fields}
            for sp in scalars
        ]
        legacy_arr = pa.array(rows, type=struct)

        # Field-by-field equality (struct-level .equals may diff on
        # internal layout but pylist equality is the durable contract).
        assert store_builder_arr.to_pylist() == legacy_arr.to_pylist()

    def test_handles_none_entries(self) -> None:
        """§2: a None entry produces a null struct row."""
        scalars: list[SpecificPitch | None] = [
            SpecificPitch(step="C", alter=0, octave=4),
            None,
            SpecificPitch(step="D", alter=0, octave=4),
        ]
        arr = build_struct_array(SpecificPitch, scalars)
        assert len(arr) == 3
        assert arr.is_valid().to_pylist() == [True, False, True]
        assert arr[1].as_py() is None

    def test_computed_fields_omitted(self) -> None:
        """§3: result has no 'fifths' field (it's a computed_field)."""
        scalars = [SpecificPitch(step="C", alter=0, octave=4)]
        arr = build_struct_array(SpecificPitch, scalars)
        field_names = {arr.type.field(i).name for i in range(arr.type.num_fields)}
        assert "fifths" not in field_names
        assert "midi_number" not in field_names

    def test_rejects_wrong_type(self) -> None:
        """Type mismatch raises TypeError (catches misuse early)."""
        scalars = ["not a scalar"]  # type: ignore[list-item]
        with pytest.raises(TypeError):
            build_struct_array(SpecificPitch, scalars)  # type: ignore[arg-type]


class TestBuildCoordinateStructArray:
    """§4, §5: Coordinate-specific store-builder."""

    def test_fraction_preserved(self) -> None:
        """§4: Fraction round-trips via numerator/denominator."""
        coords = [Coordinate(Fraction(3, 4), TimeUnit.quarters)]
        arr = build_coordinate_struct_array(coords)
        row = arr[0].as_py()
        assert row == {"value": 0.75, "numerator": 3, "denominator": 4}

    def test_float_mirrors_as_its_exact_dyadic(self) -> None:
        """A float is canonical; the ratio mirrors the double it actually is."""
        coords = [Coordinate(1.5, TimeUnit.seconds)]
        arr = build_coordinate_struct_array(coords)
        row = arr[0].as_py()
        assert row == {"value": 1.5, "numerator": 3, "denominator": 2}

    def test_int_preserved_via_num_den(self) -> None:
        """§5: int stores as float64 value AND numerator=int, denominator=1."""
        coords = [Coordinate(120, TimeUnit.ticks)]
        arr = build_coordinate_struct_array(coords)
        row = arr[0].as_py()
        assert row == {"value": 120.0, "numerator": 120, "denominator": 1}

    def test_int_round_trips_to_int_via_struct_to_coordinate(self) -> None:
        """§5: struct_to_coordinate(row, NumberType.int) returns int(120)."""
        coords = [Coordinate(120, TimeUnit.ticks)]
        arr = build_coordinate_struct_array(coords)
        row = arr[0].as_py()
        result = struct_to_coordinate(row, NumberType.int)
        assert result == 120
        assert isinstance(result, int) and not isinstance(result, bool)

    def test_handles_none_entries(self) -> None:
        """A None entry produces a null parent struct."""
        coords: list[Coordinate | None] = [
            Coordinate(1.5, TimeUnit.seconds),
            None,
            Coordinate(Fraction(1, 2), TimeUnit.seconds),
        ]
        arr = build_coordinate_struct_array(coords)
        assert len(arr) == 3
        assert arr.is_valid().to_pylist() == [True, False, True]


class TestColumnBuilderWork:
    """§6: the store-builder reads attributes; it never dumps a row.

    The row-wise ``model_dump`` path is what the store-builder replaces, so
    the work it saves is counted directly: building 10k scalars calls
    ``model_dump`` zero times and gives the array the row-wise path gives.
    """

    def test_column_builder_never_dumps_a_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scalars = [SpecificPitch(step="C", alter=0, octave=4) for _ in range(10_000)]
        struct = derive_arrow_struct(SpecificPitch)
        row_wise = pa.array([sp.model_dump() for sp in scalars], type=struct)

        dumps: list[SpecificPitch] = []
        original = SpecificPitch.model_dump

        def counting(self, *args, **kwargs):
            dumps.append(self)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(SpecificPitch, "model_dump", counting)
        built = build_struct_array(SpecificPitch, scalars)

        assert dumps == []
        assert built.equals(row_wise)
