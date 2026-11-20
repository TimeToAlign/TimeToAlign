"""One struct, one canonical side, one builder.

Validation logic is documented in ``tests/core/README.md`` under
"test_number_storage.py"; every expectation here is an exact value.
"""

from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import timetoalign
from timetoalign.core.enums import NumberType, TimeUnit
from timetoalign.core.time import (
    RATIONAL_STRUCT_TYPE,
    Coordinate,
    CoordinateField,
    Duration,
    DurationField,
    Interval,
    _pair_from_float,
    build_number_struct_array,
    number_cell,
    quantize_to_unit,
    struct_to_coordinate,
    to_canonical,
)

# The exact dyadic ratio of each probe double, spelled out rather than
# computed, so a change in the builder cannot quietly change the expectation
# alongside it.
DYADIC = {
    0.1: Fraction(3602879701896397, 36028797018963968),
    1 / 3: Fraction(6004799503160661, 18014398509481984),
    2.0: Fraction(2, 1),
    1.5: Fraction(3, 2),
}
PROBES = [0.1, 1 / 3, 2.0, 1.5]


class TestBothSidesAlwaysPopulated:
    """A non-null row carries the number twice, and the two sides agree."""

    @pytest.mark.parametrize("number_type", [NumberType.float, NumberType.fraction])
    def test_no_null_members_on_non_null_rows(self, number_type: NumberType) -> None:
        rows = build_number_struct_array(
            np.array(PROBES), number_type=number_type
        ).to_pylist()
        assert [r["numerator"] for r in rows].count(None) == 0
        assert [r["denominator"] for r in rows].count(None) == 0

    def test_null_row_nulls_every_member(self) -> None:
        rows = build_number_struct_array(
            np.array([1.5, np.nan]), number_type=NumberType.float
        ).to_pylist()
        assert rows[1] is None

    @pytest.mark.parametrize(
        ("values", "number_type"),
        [
            ([0, 1, -3], NumberType.int),
            ([1.4, 2.5, -0.6], NumberType.int),
            ([0.1, 1 / 3, 2.0], NumberType.float),
            ([0.1, 1 / 3, 2.0], NumberType.fraction),
            (["1/3", "5/24", "2"], NumberType.fraction),
            ([1, 2, 3], NumberType.fraction),
        ],
    )
    def test_the_librarys_own_writers_never_emit_a_partial_cell(
        self, values: list[object], number_type: NumberType
    ) -> None:
        """Read tolerance for value-only cells must not excuse writing one.

        Decoding accepts a cell carrying only a ``value``, so that older
        artifacts and hand-written fixtures stay readable. That tolerance is
        only safe while nothing in the library *produces* one — otherwise a
        writer bug emits partial cells forever and the tolerant reader
        quietly absorbs them. This is the guard on the writing side.
        """
        for row in build_number_struct_array(
            values, number_type=number_type
        ).to_pylist():
            assert row is not None
            assert row["numerator"] is not None
            assert row["denominator"] is not None
            assert row["value"] is not None

    def test_scalar_to_dict_also_completes_both_sides(self) -> None:
        for scalar in (
            Coordinate(1.5, TimeUnit.seconds),
            Coordinate(Fraction(5, 24), TimeUnit.quarters),
            Coordinate(120, TimeUnit.ticks),
            Duration(0.25, TimeUnit.seconds),
        ):
            cell = scalar.to_dict()
            assert cell["numerator"] is not None
            assert cell["denominator"] is not None
            assert cell["value"] is not None


class TestFloatCanonical:
    """``float``: the double is authoritative, the ratio mirrors it exactly."""

    def test_value_is_the_double_and_pair_is_its_exact_dyadic(self) -> None:
        rows = build_number_struct_array(
            np.array(PROBES), number_type=NumberType.float
        ).to_pylist()
        for probe, row in zip(PROBES, rows, strict=True):
            assert row["value"] == probe
            assert Fraction(row["numerator"], row["denominator"]) == DYADIC[probe]


class TestFractionCanonical:
    """``fraction``: the ratio is authoritative, the double mirrors it."""

    def test_pair_is_the_exact_dyadic_and_value_mirrors_it(self) -> None:
        rows = build_number_struct_array(
            np.array(PROBES), number_type=NumberType.fraction
        ).to_pylist()
        for probe, row in zip(PROBES, rows, strict=True):
            exact = Fraction(row["numerator"], row["denominator"])
            assert exact == DYADIC[probe]
            assert row["value"] == float(exact)

    def test_exact_ratio_strings_stay_exact(self) -> None:
        rows = build_number_struct_array(
            ["1/3", "5/24", "7/8"], number_type=NumberType.fraction
        ).to_pylist()
        assert [Fraction(r["numerator"], r["denominator"]) for r in rows] == [
            Fraction(1, 3),
            Fraction(5, 24),
            Fraction(7, 8),
        ]


class TestIntCanonical:
    """``int``: the integer is authoritative, ties settle on the even one."""

    def test_round_uses_half_to_even(self) -> None:
        rows = build_number_struct_array(
            np.array([1.4, 2.5, -0.6]), number_type=NumberType.int, rounding="round"
        ).to_pylist()
        assert [r["numerator"] for r in rows] == [1, 2, -1]
        assert [r["numerator"] for r in rows] == [round(x) for x in (1.4, 2.5, -0.6)]
        assert [r["denominator"] for r in rows] == [1, 1, 1]
        assert [r["value"] for r in rows] == [1.0, 2.0, -1.0]

    @pytest.mark.parametrize(
        ("rounding", "expected"),
        [
            ("floor", [1, 2, -1]),
            ("ceil", [2, 3, 0]),
            ("truncate", [1, 2, 0]),
        ],
    )
    def test_other_rounding_modes(self, rounding: str, expected: list[int]) -> None:
        rows = build_number_struct_array(
            np.array([1.4, 2.5, -0.6]), number_type=NumberType.int, rounding=rounding
        ).to_pylist()
        assert [r["numerator"] for r in rows] == expected

    def test_negative_ties_also_settle_on_the_even_integer(self) -> None:
        # The case where half-to-even and half-away-from-zero visibly differ
        # in sign, and the one a positive-only probe would miss.
        rows = build_number_struct_array(
            np.array([-2.5, -1.5, -0.5]), number_type=NumberType.int
        ).to_pylist()
        assert [r["numerator"] for r in rows] == [-2, -2, 0]
        assert [r["numerator"] for r in rows] == [round(x) for x in (-2.5, -1.5, -0.5)]

    def test_source_floats_can_be_kept_as_provenance(self) -> None:
        """The opt-in flag records what was measured beside what was stored."""
        rows = build_number_struct_array(
            np.array([1.4, 2.5]),
            number_type=NumberType.int,
            preserve_source_floats=True,
        ).to_pylist()
        assert [r["numerator"] for r in rows] == [1, 2]
        assert [r["denominator"] for r in rows] == [1, 1]
        assert [r["value"] for r in rows] == [1.4, 2.5]
        # Without the flag the float side mirrors the stored integer.
        assert [
            r["value"]
            for r in build_number_struct_array(
                np.array([1.4, 2.5]), number_type=NumberType.int
            ).to_pylist()
        ] == [1.0, 2.0]

    def test_provenance_floats_do_not_survive_arithmetic(self) -> None:
        """A result is always an exact mirror; provenance is construction-only."""
        field = CoordinateField.from_field(
            build_number_struct_array(
                np.array([1.4, 2.6]),
                number_type=NumberType.int,
                preserve_source_floats=True,
            ),
            unit=TimeUnit.ticks,
            number_type=NumberType.int,
        )
        assert [r["value"] for r in field.to_pyarrow().to_pylist()] == [1.4, 2.6]
        after = (field + 0).to_pyarrow().to_pylist()
        assert [r["value"] for r in after] == [1.0, 3.0]
        assert [r["numerator"] for r in after] == [1, 3]


class TestVectorisedEqualsScalar:
    """Two routes into one struct must not be two answers.

    The vectorised route is the one that can differ *by platform*, since it
    alone reaches numpy — so the tests that bind it to the scalar definition
    take ``platform_contract`` and run under both widths numpy binds its
    C-typed ufunc arguments to. A rule the two routes share everywhere except
    on the platform nobody here runs is not a rule the storage contract holds.
    """

    def test_routes_agree_and_canonical_side_survives(self, platform_contract) -> None:
        rng = np.random.default_rng(20260809)
        samples = np.concatenate(
            [
                rng.uniform(-1000.0, 1000.0, 3000),
                rng.uniform(-1.0, 1.0, 3000),
                # The band where the exact dyadic denominator overflows int64.
                rng.uniform(-1e-3, 1e-3, 2000),
                np.array([0.0, -0.0, 1.0, -1.0, 0.5, 1e-8, -1e-8, 1e18]),
            ]
        )
        rows = build_number_struct_array(
            samples, number_type=NumberType.float
        ).to_pylist()
        for sample, row in zip(samples, rows, strict=True):
            assert (row["numerator"], row["denominator"]) == _pair_from_float(
                float(sample)
            )
            assert row["value"] == float(sample)

    def test_mirror_is_exact_wherever_int64_allows(self, platform_contract) -> None:
        # Everything at or above 2**-10 fits: its dyadic denominator is at
        # most 2**62. 1e-8 does not, and is covered by the test below.
        samples = np.array([0.1, 1 / 3, 1.5, 2.0, 0.5, 2**-10])
        rows = build_number_struct_array(
            samples, number_type=NumberType.float
        ).to_pylist()
        for sample, row in zip(samples, rows, strict=True):
            assert Fraction(row["numerator"], row["denominator"]) == Fraction(
                float(sample)
            )

    def test_below_the_int64_limit_only_the_mirror_gives(
        self, platform_contract
    ) -> None:
        # 0.0001 needs denominator 2**66, past int64. The fallback is the
        # nearest ratio over 2**62 — round(value * 2**62) / 2**62, reduced —
        # and the canonical side is untouched.
        row = number_cell(0.0001, NumberType.float)
        assert row["value"] == 0.0001
        expected = Fraction(round(0.0001 * 2**62), 2**62)
        assert (row["numerator"], row["denominator"]) == (
            expected.numerator,
            expected.denominator,
        )
        assert (row["numerator"], row["denominator"]) == (
            461168601842739,
            4611686018427387904,
        )
        assert Fraction(row["numerator"], row["denominator"]) != Fraction(0.0001)

    def test_as_fraction_recomputes_rather_than_reading_the_mirror(self) -> None:
        """The storage ceiling must not leak into the accessor.

        ``Fraction`` is arbitrary-precision, so the exact dyadic of an
        overflow-band double is perfectly representable in Python — the
        int64 ceiling constrains storage, not retrieval. Reading the stored
        mirror here would silently hand back the approximation.
        """
        coordinate = Coordinate(0.0001, TimeUnit.seconds)
        stored = number_cell(0.0001, NumberType.float)
        assert Fraction(stored["numerator"], stored["denominator"]) != Fraction(0.0001)
        assert coordinate.to_fraction() == Fraction(0.0001)
        assert coordinate.to_fraction().denominator == 2**66

    def test_an_exact_field_refuses_a_value_it_cannot_store_exactly(self) -> None:
        """A canonical value is exact or it is an error; only mirrors approximate."""
        with pytest.raises(ValueError, match="number_type float"):
            number_cell(0.0001, NumberType.fraction)
        # quarters is fraction-canonical, so the scalar refuses it too.
        with pytest.raises(ValueError, match="number_type float"):
            build_number_struct_array([0.0001], number_type=NumberType.fraction)


class TestTheBuilderRunsOnEveryPlatform:
    """A column of doubles must become a column of ratios anywhere.

    The mirror is a *binary* decomposition, so every quantity in it is a bit
    count or a power of two — nothing whose meaning depends on how wide a C
    ``int`` happens to be. These tests hold the implementation to that:
    ``platform_contract`` runs each of them once as this machine offers numpy
    and once as an LLP64 build does.
    """

    def test_the_scale_needs_no_c_typed_exponent(self, platform_contract) -> None:
        """Scaling a double by a power of two must not go through a C int.

        The exponent is a bit count capped at 62 and the scale factor is the
        denominator itself, exact as a double — so the mirror can be built
        with a multiply, and nothing has to be handed to a ufunc argument
        whose width the platform, rather than the library, decides.
        """
        samples = np.array([0.1, 1 / 3, 1.5, 2.0, 0.5, 2**-10, 0.0, -1.0])
        rows = build_number_struct_array(
            samples, number_type=NumberType.float
        ).to_pylist()

        for sample, row in zip(samples, rows, strict=True):
            assert (row["numerator"], row["denominator"]) == _pair_from_float(
                float(sample)
            )
            assert row["value"] == float(sample)

    def test_the_denominator_keeps_its_full_64_bit_width(
        self, platform_contract
    ) -> None:
        """What the exponent counts still needs all 64 bits.

        ``1 << 62`` overflows anything narrower, so a denominator accumulated
        in a 32-bit width would wrap into nonsense — silently, and only for
        the smallest values, which is the worst way for it to be wrong.
        """
        row = build_number_struct_array(
            np.array([2**-10]), number_type=NumberType.float
        ).to_pylist()[0]

        assert (row["numerator"], row["denominator"]) == (1, 1024)
        assert Fraction(row["numerator"], row["denominator"]) == Fraction(2**-10)

        # 0.0001 is past the int64 ceiling, so it lands on the capped mirror —
        # the one place the full 2**62 denominator is actually stored.
        capped = build_number_struct_array(
            np.array([0.0001]), number_type=NumberType.float
        ).to_pylist()[0]
        assert capped["denominator"] == 2**62
        assert capped["numerator"] == 461168601842739

    def test_the_reduction_finds_every_common_power_of_two(
        self, platform_contract
    ) -> None:
        """Reducing by the lowest set bit must equal reducing step by step.

        The dyadic ratio's only common factor is a power of two, so the whole
        reduction is the numerator's lowest set bit — one pass, not one per
        bit. Values are chosen so that the bit to strip is at the bottom
        (``2**-10``), in the middle (``0.75``), absent (``0.1``), and the
        whole numerator (``0.0``, which reduces to 0/1 and has no lowest bit
        to find at all).
        """
        samples = np.array([2**-10, 0.75, 0.1, 0.0, -0.0, 3.0, 2.0**52])
        rows = build_number_struct_array(
            samples, number_type=NumberType.float
        ).to_pylist()

        pairs = [(row["numerator"], row["denominator"]) for row in rows]
        assert pairs == [
            (1, 1024),
            (3, 4),
            (3602879701896397, 36028797018963968),
            (0, 1),
            (0, 1),
            (3, 1),
            (2**52, 1),
        ]
        # A pair equal to the value's Fraction *component for component* is a
        # pair in lowest terms, since Fraction normalises on construction.
        for sample, (numerator, denominator) in zip(samples, pairs, strict=True):
            exact = Fraction(float(sample))
            assert (numerator, denominator) == (exact.numerator, exact.denominator)


class TestExactValuesAreNeverSilentlyRounded:
    """An exact non-integral value is a mistake upstream, not a rounding job."""

    @pytest.mark.parametrize("scalar_cls", [Coordinate, Duration])
    @pytest.mark.parametrize("unit", [TimeUnit.ticks, TimeUnit.samples])
    def test_exact_fraction_into_a_discrete_unit_raises(
        self, scalar_cls: type, unit: TimeUnit
    ) -> None:
        with pytest.raises(ValueError):
            scalar_cls(Fraction(5, 24), unit)

    @pytest.mark.parametrize("scalar_cls", [Coordinate, Duration])
    def test_inexact_float_into_a_discrete_unit_rounds(self, scalar_cls: type) -> None:
        assert scalar_cls(120.7, TimeUnit.ticks).value == 121
        assert scalar_cls(120.4, TimeUnit.ticks).value == 120

    def test_builder_refuses_an_exact_ratio_for_an_int_field(self) -> None:
        with pytest.raises(ValueError):
            number_cell(Fraction(5, 24), NumberType.int)

    def test_conversion_may_quantize_where_construction_may_not(self) -> None:
        assert quantize_to_unit(Fraction(1, 2), TimeUnit.ticks) == 0
        assert quantize_to_unit(Fraction(3, 2), TimeUnit.ticks) == 2
        # A unit that can express the ratio keeps it untouched.
        assert quantize_to_unit(Fraction(1, 2), TimeUnit.seconds) == Fraction(1, 2)


class TestUnitOwnsTheRepresentation:
    """The unit is the single source of number policy."""

    @pytest.mark.parametrize(
        ("unit", "default"),
        [
            (TimeUnit.ticks, NumberType.int),
            (TimeUnit.samples, NumberType.int),
            (TimeUnit.frames, NumberType.int),
            (TimeUnit.pixels, NumberType.int),
            (TimeUnit.quarters, NumberType.fraction),
            (TimeUnit.floating_measures, NumberType.float),
            (TimeUnit.seconds, NumberType.float),
            (TimeUnit.milliseconds, NumberType.float),
            (TimeUnit.minutes, NumberType.float),
            (TimeUnit.meters, NumberType.float),
            (TimeUnit.centimeters, NumberType.float),
            (TimeUnit.millimeters, NumberType.float),
            (TimeUnit.inches, NumberType.float),
            (TimeUnit.points, NumberType.float),
            (TimeUnit.number, NumberType.float),
        ],
    )
    def test_default_number_type(self, unit: TimeUnit, default: NumberType) -> None:
        assert unit.default_number_type is default

    @pytest.mark.parametrize(
        "unit", [TimeUnit.ticks, TimeUnit.samples, TimeUnit.frames, TimeUnit.pixels]
    )
    def test_discrete_units_admit_int_only(self, unit: TimeUnit) -> None:
        assert unit.allowed_number_types == frozenset({NumberType.int})
        with pytest.raises(ValueError, match="does not admit"):
            unit.resolve_number_type(NumberType.float)

    def test_continuous_units_admit_both_exact_and_inexact(self) -> None:
        assert TimeUnit.quarters.allowed_number_types == frozenset(
            {NumberType.float, NumberType.fraction}
        )
        assert TimeUnit.quarters.resolve_number_type(NumberType.float) is (
            NumberType.float
        )
        with pytest.raises(ValueError, match="does not admit"):
            TimeUnit.quarters.resolve_number_type(NumberType.int)

    def test_generic_unit_admits_everything(self) -> None:
        assert TimeUnit.number.allowed_number_types == frozenset(NumberType)


class TestArithmeticRunsInTheDeclaredRepresentation:
    """One engine, and the left operand decides."""

    @staticmethod
    def _coord(values: list[object], number_type: NumberType) -> CoordinateField:
        return CoordinateField.from_field(
            build_number_struct_array(values, number_type=number_type),
            unit=TimeUnit.quarters,
            number_type=number_type,
        )

    @staticmethod
    def _dur(values: list[object], number_type: NumberType) -> DurationField:
        return DurationField.from_field(
            build_number_struct_array(values, number_type=number_type),
            unit=TimeUnit.quarters,
            number_type=number_type,
        )

    def test_left_operand_decides_and_both_sides_mirror(self) -> None:
        result = self._coord(["1/3", "1/2", "3/4"], NumberType.fraction) + self._dur(
            [0.5, 0.25, 0.125], NumberType.float
        )
        assert isinstance(result, CoordinateField)
        assert result.number_type is NumberType.fraction
        expected = [Fraction(5, 6), Fraction(3, 4), Fraction(7, 8)]
        for i, row in enumerate(result.to_pyarrow().to_pylist()):
            assert Fraction(row["numerator"], row["denominator"]) == expected[i]
            assert row["value"] == float(expected[i])
            assert result[i] == Coordinate(expected[i], TimeUnit.quarters)

    def test_scalar_addition_keeps_the_left_operand_exact(self) -> None:
        total = Coordinate(Fraction(1, 3), TimeUnit.quarters) + 0.5
        assert total.value == Fraction(5, 6)

    def test_scaling_quantizes_the_result_never_the_operand(self) -> None:
        """The two cases that separate every candidate implementation.

        Coercing the operand first would make the tick case ``0``
        (``round(0.5)`` is zero); doing the arithmetic in float would make
        the quarters case an ugly dyadic instead of a sixth. Running the
        operation exactly and rounding once, at the end, gives both.
        """
        # 101 * 1/2 is 50.5 exactly; rounding once, at the end, ties to even.
        assert (Coordinate(101, TimeUnit.ticks) * 0.5).value == 50
        assert (Coordinate(100, TimeUnit.ticks) * 0.5).value == 50
        assert (Coordinate(100, TimeUnit.ticks) * Fraction(1, 2)).value == 50
        # The exact lane stays exact: 0.5 is exactly 1/2, so this is a sixth.
        assert (Coordinate(Fraction(1, 3), TimeUnit.quarters) * 0.5).value == Fraction(
            1, 6
        )

    def test_the_four_rounding_modes_share_one_vocabulary(self) -> None:
        """``to_int`` uses the same modes and the same default as the builder."""
        coordinate = Coordinate(Fraction(7, 4), TimeUnit.quarters)
        assert coordinate.to_int() == 2
        assert coordinate.to_int("round") == 2
        assert coordinate.to_int("floor") == 1
        assert coordinate.to_int("ceil") == 2
        assert coordinate.to_int("truncate") == 1

    def test_mismatched_unit_raises(self) -> None:
        field = self._coord(["1/3"], NumberType.fraction)
        with pytest.raises(TypeError, match="different units"):
            field + Duration(1, TimeUnit.seconds)
        with pytest.raises(TypeError, match="different units"):
            Coordinate(1, TimeUnit.quarters) + Duration(1, TimeUnit.seconds)


class TestErrorPolicy:
    """Unreadable cells: raise for a file's coordinates, null for a loose column."""

    def test_raise_is_the_default(self) -> None:
        with pytest.raises((ValueError, ZeroDivisionError)):
            build_number_struct_array(["1/0"], number_type=NumberType.fraction)

    def test_null_policy_keeps_the_good_rows(self) -> None:
        rows = build_number_struct_array(
            ["1/2", "1/0", "3/4"], number_type=NumberType.fraction, on_error="null"
        ).to_pylist()
        assert rows[1] is None
        assert Fraction(rows[0]["numerator"], rows[0]["denominator"]) == Fraction(1, 2)
        assert Fraction(rows[2]["numerator"], rows[2]["denominator"]) == Fraction(3, 4)


class TestDecoding:
    """Reading a cell back in a chosen representation."""

    def test_requested_representation_is_honoured(self) -> None:
        cell = number_cell(Fraction(5, 24), NumberType.fraction)
        assert struct_to_coordinate(cell, NumberType.fraction) == Fraction(5, 24)
        assert struct_to_coordinate(cell, NumberType.float) == float(Fraction(5, 24))

    def test_value_only_cell_decodes_through_the_float_side(self) -> None:
        # Hand-built and legacy cells carry the number in one place only.
        assert struct_to_coordinate({"value": 1.5}, NumberType.fraction) == Fraction(
            3, 2
        )
        assert struct_to_coordinate({"value": 1.5}, NumberType.float) == 1.5

    def test_to_canonical_is_total_over_the_supported_inputs(self) -> None:
        assert to_canonical("3/8", NumberType.fraction) == Fraction(3, 8)
        assert to_canonical("7", NumberType.int) == 7
        assert to_canonical(7, NumberType.float) == 7.0


class TestDisplayShowsWhatIsCarried:
    """One formatter behind every pretty rendering, admitting the whole value."""

    @pytest.mark.parametrize(
        ("value", "unit", "number_type", "expected"),
        [
            # An exact ratio reads as a ratio, never as a truncated decimal.
            (Fraction(1, 3), TimeUnit.quarters, "fraction", "1/3 quarters"),
            (Fraction(19, 2), TimeUnit.quarters, "fraction", "19/2 quarters"),
            # Integral values drop the denominator and the decimal point.
            (Fraction(10, 1), TimeUnit.quarters, "fraction", "10 quarters"),
            (Fraction(0, 1), TimeUnit.quarters, "fraction", "0 quarters"),
            (2.0, TimeUnit.seconds, "float", "2 seconds"),
            # Floats keep every digit they round-trip with.
            (0.1, TimeUnit.seconds, "float", "0.1 seconds"),
            (45.5, TimeUnit.seconds, "float", "45.5 seconds"),
            (1 / 3, TimeUnit.seconds, "float", "0.3333333333333333 seconds"),
            # Large and small values never round away or go scientific.
            (1234567.5, TimeUnit.seconds, "float", "1234567.5 seconds"),
            (1e-7, TimeUnit.seconds, "float", "0.0000001 seconds"),
            # Discrete units render as integers.
            (160, TimeUnit.ticks, None, "160 ticks"),
        ],
    )
    def test_scalar_str_renders_the_whole_value(
        self,
        value: object,
        unit: TimeUnit,
        number_type: str | None,
        expected: str,
    ) -> None:
        assert str(Coordinate(value, unit, number_type=number_type)) == expected

    def test_duration_renders_the_same_way(self) -> None:
        assert str(Duration(Fraction(1, 3), TimeUnit.quarters)) == "1/3 quarters"

    def test_scalar_and_stamp_agree(self) -> None:
        """The two display paths are one formatter, so they cannot drift."""
        from timetoalign.core.timestamp import _format_coordinate_value

        for value, unit in (
            (Fraction(1, 3), TimeUnit.quarters),
            (Fraction(19, 2), TimeUnit.quarters),
            (0.1, TimeUnit.seconds),
            (1e-7, TimeUnit.seconds),
        ):
            number_type = "fraction" if isinstance(value, Fraction) else "float"
            scalar = str(Coordinate(value, unit, number_type=number_type))
            assert _format_coordinate_value(value, unit.value) == scalar

    @pytest.mark.parametrize(
        ("start", "end", "unit", "number_type", "expected"),
        [
            # Exact endpoints stay exact; the unit is named once.
            (
                Fraction(1, 2),
                Fraction(3, 2),
                TimeUnit.quarters,
                "fraction",
                "[1/2, 3/2) quarters",
            ),
            (
                Fraction(1, 3),
                Fraction(2, 3),
                TimeUnit.quarters,
                "fraction",
                "[1/3, 2/3) quarters",
            ),
            # Integral endpoints drop the denominator.
            (
                Fraction(10, 1),
                Fraction(12, 1),
                TimeUnit.quarters,
                "fraction",
                "[10, 12) quarters",
            ),
            # Float endpoints render as floats, integral ones without a point.
            (0.1, 2.0, TimeUnit.seconds, "float", "[0.1, 2) seconds"),
            (
                1e-7,
                1234567.5,
                TimeUnit.seconds,
                "float",
                "[0.0000001, 1234567.5) seconds",
            ),
            # Discrete units render as integers.
            (160, 480, TimeUnit.ticks, None, "[160, 480) ticks"),
            # A zero-length span is valid and renders both endpoints.
            (
                Fraction(5, 4),
                Fraction(5, 4),
                TimeUnit.quarters,
                "fraction",
                "[5/4, 5/4) quarters",
            ),
        ],
    )
    def test_interval_str_renders_both_endpoints_whole(
        self,
        start: object,
        end: object,
        unit: TimeUnit,
        number_type: str | None,
        expected: str,
    ) -> None:
        interval = Interval(
            start=Coordinate(start, unit, number_type=number_type),
            end=Coordinate(end, unit, number_type=number_type),
        )
        assert str(interval) == expected

    def test_interval_endpoints_render_like_bare_coordinates(self) -> None:
        """The interval reuses the scalar formatter, so it cannot show less."""
        for value, unit, number_type in (
            (Fraction(1, 3), TimeUnit.quarters, "fraction"),
            (Fraction(19, 2), TimeUnit.quarters, "fraction"),
            (0.1, TimeUnit.seconds, "float"),
            (1e-7, TimeUnit.seconds, "float"),
            (160, TimeUnit.ticks, None),
        ):
            coordinate = Coordinate(value, unit, number_type=number_type)
            interval = Interval(start=coordinate, end=coordinate)
            endpoint = str(coordinate).removesuffix(f" {unit}")
            assert str(interval) == f"[{endpoint}, {endpoint}) {unit}"

    def test_repr_stays_exact_and_typed(self) -> None:
        """repr is the other lane and already showed the whole value."""
        assert repr(Coordinate(Fraction(1, 3), TimeUnit.quarters)) == (
            "Coordinate(Fraction(1, 3), quarters)"
        )
        assert repr(
            Interval(
                start=Coordinate(Fraction(1, 2), TimeUnit.quarters),
                end=Coordinate(Fraction(3, 2), TimeUnit.quarters),
            )
        ) == (
            "Interval(start=Coordinate(Fraction(1, 2), quarters), "
            "end=Coordinate(Fraction(3, 2), quarters))"
        )


class TestConversionRowsAgreeWithGetUnit:
    """A C-Map row and ``get_unit`` are two readings of one number."""

    @staticmethod
    def _stamp() -> tuple[object, object]:
        """A fraction axis and an int axis, each mapped to all three classes.

        Every map is chosen to give a non-integral exact reading at the probe
        coordinate, because that is the value the construction path refused
        (into pixels) and the value it kept undegraded (into seconds).
        """
        from timetoalign.maps import LinearMap
        from timetoalign.timelines import (
            ContinuousLogicalTimeline,
            DiscreteGraphicalTimeline,
        )

        quarters_axis = ContinuousLogicalTimeline(length=Fraction(100), uid="clt_1")
        for scalar, target in (
            (Fraction(7104000, 3278347), TimeUnit.pixels),
            (Fraction(3278347, 7350), TimeUnit.seconds),
            (Fraction(1, 3), TimeUnit.number),
        ):
            quarters_axis.add_conversion_map(
                LinearMap(
                    scalar=scalar,
                    offset=0,
                    source_unit=TimeUnit.quarters,
                    target_unit=target,
                )
            )

        pixels_axis = DiscreteGraphicalTimeline(length=20000, uid="dgt_holes")
        for scalar, target in (
            (Fraction(3278347, 7350), TimeUnit.seconds),
            (Fraction(1, 3), TimeUnit.quarters),
        ):
            pixels_axis.add_conversion_map(
                LinearMap(
                    scalar=scalar,
                    offset=0,
                    source_unit=TimeUnit.pixels,
                    target_unit=target,
                )
            )

        return quarters_axis.get_timestamp(Fraction(1)), pixels_axis.get_timestamp(1)

    def test_display_never_raises_on_an_int_locked_target(self) -> None:
        """An exact ratio converted into pixels used to break ``repr``."""
        from_fraction_axis, _ = self._stamp()

        assert "pixels=2" in repr(from_fraction_axis)
        assert "2 pixels" in str(from_fraction_axis)
        assert "pixels" in from_fraction_axis._repr_html_()

    def test_rows_carry_the_target_representation(self) -> None:
        """The target decides alone, so the source axis leaves no trace.

        The two ``seconds`` rows come off different axes -- one exact, one
        integer-locked -- through the same conversion, and report the same
        number. Letting the source axis's representation ride along produced
        a forty-digit ratio on a float-canonical reading and made one map
        answer two ways depending on where it was read from.
        """
        from_fraction_axis, from_int_axis = self._stamp()

        assert from_fraction_axis._conversion_rows() == [
            ("pixels", 2, "pixels"),
            ("seconds", 446.03360544217685, "seconds"),
            ("number", 0.3333333333333333, "number"),
        ]
        assert from_int_axis._conversion_rows() == [
            ("seconds", 446.03360544217685, "seconds"),
            ("quarters", Fraction(1, 3), "quarters"),
        ]

    def test_every_row_equals_the_typed_getter(self) -> None:
        """The property, not a table of expectations: the two lanes agree."""
        for stamp in self._stamp():
            rows = stamp._conversion_rows()
            assert rows
            for label, value, _suffix in rows:
                converted = stamp.get_unit(TimeUnit(label), format="coordinate")
                assert converted.value == value
                assert type(converted.value) is type(value)

    def test_the_selector_getter_reads_the_same_boundary(self) -> None:
        """``get_conversion_for`` names a map; the number is still the axis's."""
        from_fraction_axis, from_int_axis = self._stamp()

        assert from_fraction_axis.get_conversion_for("pixels") == 2
        assert from_fraction_axis.get_conversion_for("seconds") == 446.03360544217685
        assert from_int_axis.get_conversion_for("seconds") == 446.03360544217685
        assert from_fraction_axis.get_conversion_for("number") == 0.3333333333333333

    def test_rendered_text_shows_the_row_values(self) -> None:
        """Both display lanes print the same numbers the rows carry."""
        from_fraction_axis, from_int_axis = self._stamp()

        text = str(from_fraction_axis)
        assert "  pixels   2 pixels" in text
        assert "  seconds  446.03360544217685 seconds" in text
        assert "  number   0.3333333333333333 number" in text

        html = from_int_axis._repr_html_()
        assert "446.03360544217685 seconds" in html
        assert "1/3 quarters" in html


class TestMapCallsWriteTheirTargetAxis:
    """A public map evaluation lands on the target unit's own representation."""

    @pytest.mark.parametrize(
        ("target", "value", "expected"),
        [
            # float-canonical: an exact input degrades on the way out.
            (TimeUnit.seconds, 1, 1.0),
            (TimeUnit.seconds, Fraction(1), 1.0),
            # fraction-canonical: a float input becomes its exact dyadic.
            (TimeUnit.quarters, 1.5, Fraction(3, 2)),
            (TimeUnit.quarters, Fraction(3, 2), Fraction(3, 2)),
            # int-locked: quantized, because there is no fractional pixel.
            (TimeUnit.pixels, Fraction(4, 3), 1),
            (TimeUnit.pixels, 2.5, 2),
        ],
    )
    def test_target_unit_decides(
        self, target: TimeUnit, value: object, expected: object
    ) -> None:
        from timetoalign.maps import LinearMap

        cmap = LinearMap(scalar=1, offset=0, target_unit=target)
        result = cmap(value)

        assert result == expected
        assert type(result) is type(expected)

    def test_a_map_without_a_target_unit_declares_nothing(self) -> None:
        from timetoalign.maps import LinearMap

        cmap = LinearMap(scalar=1, offset=0)

        assert cmap.output_number_type is None
        assert cmap(Fraction(3, 2)) == Fraction(3, 2)

    def test_ordinal_maps_keep_their_own_whole_numbers(self) -> None:
        """A measure count and a bucket index are not axis positions."""
        from timetoalign.maps.meter import MetricMap
        from timetoalign.maps.periodic import FloorMap

        meter = MetricMap.from_uniform(
            n_measures=4,
            quarters_per_measure=Fraction(4, 1),
            start_mc=1,
            start_mn="1",
        )
        floors = FloorMap(4.0, base=1, target_unit=TimeUnit.number)

        # A measure count names no axis at all, so the map declares no
        # target unit and still answers a whole number.
        assert meter.target_unit is None
        assert meter(4.0) == 2
        assert type(meter(4.0)) is int
        assert floors.target_unit is TimeUnit.number
        assert floors(5.0) == 2
        assert type(floors(5.0)) is int


class TestNoFabricatedExactness:
    """Denominator limiting invents ratios; the package must not use it."""

    def test_zero_occurrences_in_the_package(self) -> None:
        package_root = Path(timetoalign.__file__).parent
        forbidden = "limit" + "_denominator"
        hits = subprocess.run(
            ["grep", "-rn", forbidden, "--include=*.py", str(package_root)],
            capture_output=True,
            text=True,
        )
        assert hits.stdout == ""

    def test_to_fraction_of_a_float_is_its_exact_dyadic(self) -> None:
        assert Coordinate(0.1, TimeUnit.seconds).to_fraction() == DYADIC[0.1]


class TestStructShape:
    """The one struct, and nothing shaped almost like it."""

    def test_builder_output_matches_the_canonical_type(self) -> None:
        built = build_number_struct_array(
            np.array([1.0, 2.0]), number_type=NumberType.float
        )
        assert built.type.equals(RATIONAL_STRUCT_TYPE)

    def test_empty_column_keeps_the_type(self) -> None:
        built = build_number_struct_array(
            np.array([], dtype=np.float64), number_type=NumberType.float
        )
        assert built.type.equals(RATIONAL_STRUCT_TYPE)
        assert len(built) == 0

    def test_pyarrow_accepts_the_rows(self) -> None:
        built = build_number_struct_array(
            ["1/3", None, "2"], number_type=NumberType.fraction, on_error="null"
        )
        assert pa.types.is_struct(built.type)
        assert built.to_pylist()[1] is None
