"""End-to-end Parquet round-trip tests for the pilot pydantic scalars.

See README.md "test_pilot_round_trip.py" for the gold-standard plan.

These tests exercise the three validation regimes:

* Trust boundary — ``Model.model_validate(...)`` per row.
* Internal round-trip — ``Model.model_construct(...)`` per dict.
* Bulk construction — column-builder over ``T.model_fields``.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from timetoalign.core.enums import NumberType, TimeUnit
from timetoalign.core.events import SpecificPitch
from timetoalign.core.fields import (
    TIMETOALIGN_METADATA_KEY,
    build_struct_array,
    derive_arrow_struct,
    parquet_metadata_for_model,
)
from timetoalign.core.time import (
    Coordinate,
    build_coordinate_struct_array,
    struct_to_coordinate,
)


class TestSpecificPitchRoundTrip:
    """§1, §3: SpecificPitch -> Parquet -> SpecificPitch."""

    def test_full_round_trip_via_column_builder(self, tmp_path: Path) -> None:
        """§1: write column-builder array, read back, model_construct each row."""
        originals = [
            SpecificPitch(step="C", alter=0, octave=4),
            SpecificPitch(step="G", alter=1, octave=3, cents=5.0),
            SpecificPitch(step="B", alter=-1, octave=5),
        ]
        struct = derive_arrow_struct(SpecificPitch)
        # regime: bulk construction (column-builder)
        arr = build_struct_array(SpecificPitch, originals)
        pa_field = pa.field(
            "pitch",
            struct,
            nullable=True,
            metadata=parquet_metadata_for_model(SpecificPitch),
        )
        table = pa.Table.from_arrays([arr], schema=pa.schema([pa_field]))

        path = tmp_path / "specific_pitch.parquet"
        pq.write_table(table, path)
        read = pq.read_table(path)

        rows = read.column(0).to_pylist()
        # regime: internal round-trip (model_construct bypasses validators)
        reconstructed = [SpecificPitch.model_construct(**r) for r in rows]

        assert reconstructed == originals

    def test_metadata_blob_survives_round_trip(self, tmp_path: Path) -> None:
        """§3: b"timetoalign" payload bytes are preserved through Parquet."""
        scalars = [SpecificPitch(step="C", alter=0, octave=4)]
        struct = derive_arrow_struct(SpecificPitch)
        arr = build_struct_array(SpecificPitch, scalars)
        meta = parquet_metadata_for_model(SpecificPitch)
        pa_field = pa.field("pitch", struct, nullable=True, metadata=meta)
        table = pa.Table.from_arrays([arr], schema=pa.schema([pa_field]))

        path = tmp_path / "with_meta.parquet"
        pq.write_table(table, path)
        read = pq.read_table(path)

        read_field = read.schema.field("pitch")
        assert read_field.metadata is not None
        assert TIMETOALIGN_METADATA_KEY in read_field.metadata
        # Same payload bytes survive the round-trip.
        assert (
            read_field.metadata[TIMETOALIGN_METADATA_KEY]
            == meta[TIMETOALIGN_METADATA_KEY]
        )


class TestCoordinateRoundTrip:
    """§2: Coordinate -> Parquet -> Coordinate (3 numeric types)."""

    def test_fraction_float_int_all_round_trip(self, tmp_path: Path) -> None:
        """§2: a Fraction, a float, and an int all survive the round trip."""
        coords = [
            Coordinate(Fraction(3, 4), TimeUnit.quarters),
            Coordinate(1.5, TimeUnit.quarters),
            Coordinate(120, TimeUnit.quarters),
        ]
        struct = derive_arrow_struct(Coordinate)
        # regime: bulk construction
        arr = build_coordinate_struct_array(coords)
        pa_field = pa.field("coord", struct, nullable=True)
        table = pa.Table.from_arrays([arr], schema=pa.schema([pa_field]))

        path = tmp_path / "coord.parquet"
        pq.write_table(table, path)
        read = pq.read_table(path)

        rows = read.column(0).to_pylist()
        # regime: internal round-trip
        frac_v = struct_to_coordinate(rows[0], NumberType.fraction)
        flo_v = struct_to_coordinate(rows[1], NumberType.float)
        int_v = struct_to_coordinate(rows[2], NumberType.int)

        assert frac_v == Fraction(3, 4)
        assert isinstance(frac_v, Fraction)
        assert flo_v == 1.5
        assert isinstance(flo_v, float)
        assert int_v == 120
        assert isinstance(int_v, int) and not isinstance(int_v, bool)


class TestValidationRegimes:
    """§4, §5, §6: the three regimes connected."""

    def test_model_construct_and_validate_parity_on_valid_input(self) -> None:
        """§4: for valid input, model_construct and model_validate match."""
        valid = {"step": "C", "alter": 0, "octave": 4, "cents": None}
        sp_v = SpecificPitch.model_validate(valid)
        sp_c = SpecificPitch.model_construct(**valid)
        assert sp_v == sp_c

    def test_model_validate_rejects_invalid_step(self) -> None:
        """§5a: model_validate raises on bad Literal value."""
        with pytest.raises(ValidationError):
            SpecificPitch.model_validate({"step": "X", "alter": 0, "octave": 4})

    def test_model_construct_accepts_invalid_silently(self) -> None:
        """§5b: model_construct bypasses validators (the round-trip regime)."""
        # ``model_construct`` does NOT call validators; this is the
        # internal-round-trip contract (pa.Schema is the trusted artifact).
        sp = SpecificPitch.model_construct(step="X", alter=0, octave=4, cents=None)
        assert sp.step == "X"

    def test_from_row_trust_boundary_rejects_invalid(self) -> None:
        """§6: from_row uses validators -> trust-boundary rejection.

        The canonical row shape is ``{step, alter, octave, cents}``;
        invalid ``step`` triggers the Literal validator.
        """
        bad_row = {
            "step": "X",  # invalid step
            "alter": 0,
            "octave": 4,
            "cents": 0.0,
        }
        with pytest.raises(ValidationError):
            SpecificPitch.from_row(bad_row)


class TestCoordinateValidationRegimes:
    """The same three regimes on Coordinate."""

    def test_coordinate_model_validate_rejects_bool(self) -> None:
        """Bool is rejected at the validator (trust-boundary)."""
        with pytest.raises(Exception, match="Boolean"):
            Coordinate.model_validate({"value": True, "unit": "ticks"})

    def test_coordinate_model_construct_bypasses_bool_check(self) -> None:
        """model_construct sidesteps validators — by design."""
        # We do NOT recommend this in user code; the test pins that
        # ``model_construct`` does not re-run validators.
        c = Coordinate.model_construct(value=True, unit=TimeUnit.ticks)
        # The internal value will be bool here; reading number_type
        # will still raise because the property itself rejects bool.
        # The regime contract is: model_construct trusts the input;
        # post-construction property access remains free to validate.
        with pytest.raises(TypeError, match="Boolean"):
            _ = c.number_type
