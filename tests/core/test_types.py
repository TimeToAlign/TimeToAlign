"""Tests for core/types.py."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from timetoalign.core.enums import Domain, NumberType, TimeUnit
from timetoalign.core.time import Coordinate, Duration, IdCoordinate, IdDuration


class TestCoordinateCreation:
    """Tests for Coordinate instantiation."""

    def test_coordinate_creation_int(self) -> None:
        """Can create coordinate with integer value."""
        c = Coordinate(120, TimeUnit.ticks)
        assert c.value == 120
        assert c.unit == TimeUnit.ticks

    def test_coordinate_creation_float(self) -> None:
        """Can create coordinate with float value."""
        c = Coordinate(1.5, TimeUnit.seconds)
        assert c.value == 1.5
        assert c.unit == TimeUnit.seconds

    def test_coordinate_creation_fraction(self) -> None:
        """Can create coordinate with Fraction value."""
        c = Coordinate(Fraction(3, 4), TimeUnit.quarters)
        assert c.value == Fraction(3, 4)
        assert c.unit == TimeUnit.quarters

    def test_coordinate_creation_string_unit(self) -> None:
        """Can create coordinate with string unit (coerces to enum)."""
        c = Coordinate(1, "seconds")  # type: ignore[arg-type]
        assert c.unit == TimeUnit.seconds

    def test_coordinate_creation_enum_unit(self) -> None:
        """Can create coordinate with TimeUnit enum (no coercion needed)."""
        c = Coordinate(1, TimeUnit.seconds)
        assert c.unit == TimeUnit.seconds
        assert isinstance(c.unit, TimeUnit)

    def test_coordinate_invalid_value_type(self) -> None:
        """Invalid value types raise TypeError."""
        with pytest.raises(TypeError, match="must be int, float, or Fraction"):
            Coordinate("bad", TimeUnit.seconds)  # type: ignore[arg-type]

    def test_coordinate_invalid_value_none(self) -> None:
        """None value raises TypeError."""
        with pytest.raises(TypeError, match="must be int, float, or Fraction"):
            Coordinate(None, TimeUnit.seconds)  # type: ignore[arg-type]

    def test_coordinate_is_frozen(self) -> None:
        """Coordinate is immutable."""
        c = Coordinate(120, TimeUnit.ticks)
        # Pydantic v2 BaseModel with ``frozen=True`` raises ValidationError
        # on attribute assignment (not the dataclass AttributeError).
        with pytest.raises(ValidationError) as exc_info:
            c.value = 240  # type: ignore[misc]
        assert "frozen" in str(exc_info.value).lower() or isinstance(
            exc_info.value, (AttributeError, ValueError)
        )

    def test_coordinate_is_hashable(self) -> None:
        """Coordinate can be used in sets and as dict keys."""
        c1 = Coordinate(120, TimeUnit.ticks)
        c2 = Coordinate(120, TimeUnit.ticks)
        c3 = Coordinate(240, TimeUnit.ticks)

        assert hash(c1) == hash(c2)
        assert c1 in {c2}
        assert len({c1, c2, c3}) == 2


class TestCoordinateConversions:
    """Tests for Coordinate type conversion methods."""

    def test_to_float_from_int(self) -> None:
        """to_float() works with int value."""
        c = Coordinate(120, TimeUnit.ticks)
        assert c.to_float() == 120.0
        assert isinstance(c.to_float(), float)

    def test_to_float_from_float(self) -> None:
        """to_float() works with float value."""
        c = Coordinate(1.5, TimeUnit.seconds)
        assert c.to_float() == 1.5

    def test_to_float_from_fraction(self) -> None:
        """to_float() works with Fraction value."""
        c = Coordinate(Fraction(3, 4), TimeUnit.quarters)
        assert c.to_float() == 0.75

    def test_to_int_from_int(self) -> None:
        """to_int() works with int value."""
        c = Coordinate(120, TimeUnit.ticks)
        assert c.to_int() == 120

    def test_to_int_from_float(self) -> None:
        """to_int() rounds to nearest by default, like everywhere else."""
        c = Coordinate(1.9, TimeUnit.seconds)
        assert c.to_int() == 2
        assert c.to_int("truncate") == 1

    def test_to_int_from_fraction(self) -> None:
        """to_int() rounds to nearest by default, like everywhere else."""
        c = Coordinate(Fraction(7, 4), TimeUnit.quarters)
        assert c.to_int() == 2
        assert c.to_int("truncate") == 1

    def test_to_int_ties_settle_on_the_even_integer(self) -> None:
        """The default is half-to-even, the same rule the builder applies."""
        assert Coordinate(Fraction(5, 2), TimeUnit.quarters).to_int() == 2
        assert Coordinate(Fraction(7, 2), TimeUnit.quarters).to_int() == 4
        assert Coordinate(-2.5, TimeUnit.seconds).to_int() == -2

    def test_to_fraction_from_int(self) -> None:
        """to_fraction() works with int value."""
        c = Coordinate(120, TimeUnit.ticks)
        assert c.to_fraction() == Fraction(120, 1)

    def test_to_fraction_from_fraction(self) -> None:
        """to_fraction() returns existing Fraction."""
        frac = Fraction(3, 4)
        c = Coordinate(frac, TimeUnit.quarters)
        assert c.to_fraction() == frac
        assert c.to_fraction() is frac

    def test_to_fraction_from_float(self) -> None:
        """to_fraction() approximates float value."""
        c = Coordinate(0.75, TimeUnit.seconds)
        assert c.to_fraction() == Fraction(3, 4)

    def test_to_dict_matches_exact_storage_cell(self) -> None:
        coordinate = Coordinate(Fraction(10, 3), TimeUnit.quarters)
        assert coordinate.to_dict() == {
            "value": 10 / 3,
            "numerator": 10,
            "denominator": 3,
        }

    def test_float_to_dict_mirrors_without_fabricating(self) -> None:
        # 2.5 IS 5/2 exactly, so the mirror states a fact rather than
        # guessing a tidy ratio near an untidy number.
        coordinate = Coordinate(2.5, TimeUnit.seconds)
        assert coordinate.to_dict() == {
            "value": 2.5,
            "numerator": 5,
            "denominator": 2,
        }


class TestCoordinateProperties:
    """Tests for Coordinate properties."""

    def test_number_type_int(self) -> None:
        """number_type is INT for integer values."""
        c = Coordinate(120, TimeUnit.ticks)
        assert c.number_type == NumberType.int

    def test_number_type_float(self) -> None:
        """number_type is FLOAT for float values."""
        c = Coordinate(1.5, TimeUnit.seconds)
        assert c.number_type == NumberType.float

    def test_number_type_fraction(self) -> None:
        """number_type is FRACTION for Fraction values."""
        c = Coordinate(Fraction(3, 4), TimeUnit.quarters)
        assert c.number_type == NumberType.fraction

    def test_number_type_bool_raises(self) -> None:
        """Boolean values are rejected at Coordinate construction.

        With the pydantic v2 BaseModel implementation, the ``value``
        field validator rejects bool eagerly at construction, before
        ``number_type`` handling. A validator raising ``TypeError``
        propagates unwrapped (pydantic only re-wraps ``ValueError`` /
        ``AssertionError`` into ``ValidationError``).
        """
        with pytest.raises(TypeError, match="Boolean values"):
            Coordinate(True, TimeUnit.ticks)  # type: ignore[arg-type]

    def test_domain_physical(self) -> None:
        """domain property returns PHYSICAL for physical units."""
        c = Coordinate(1.5, TimeUnit.seconds)
        assert c.domain == Domain.physical

    def test_domain_logical(self) -> None:
        """domain property returns logical for logical units."""
        c = Coordinate(120, TimeUnit.ticks)
        assert c.domain == Domain.logical

    def test_domain_graphical(self) -> None:
        """domain property returns GRAPHICAL for graphical units."""
        c = Coordinate(100, TimeUnit.pixels)
        assert c.domain == Domain.graphical


class TestCoordinateArithmetic:
    """Tests for Coordinate arithmetic operations (post-TimeScalar unification).

    Under the unified semantics:

    * ``Coordinate + Coordinate`` → ``TypeError`` (subtract for a Duration)
    * ``Coordinate + Duration`` / ``Coordinate - Duration`` → ``Coordinate``
    * ``Coordinate - Coordinate`` → ``Duration``
    * ``Coordinate ± number`` → ``Coordinate``
    """

    def test_add_two_coordinates_raises(self) -> None:
        """Coordinate + Coordinate is no longer legal — subtract for a Duration."""
        c1 = Coordinate(100, TimeUnit.ticks)
        c2 = Coordinate(50, TimeUnit.ticks)
        with pytest.raises(TypeError, match="add two Coordinates"):
            c1 + c2

    def test_add_coordinate_and_duration_different_units_raises(self) -> None:
        """Adding scalars with different units raises TypeError."""
        c = Coordinate(100, TimeUnit.ticks)
        d = Duration(1.5, TimeUnit.seconds)
        with pytest.raises(TypeError, match="different units"):
            c + d

    def test_add_number(self) -> None:
        """Coordinate + number returns a Coordinate with the offset value."""
        c = Coordinate(100, TimeUnit.ticks)
        result = c + 50
        assert isinstance(result, Coordinate)
        assert result.value == 150
        assert result.unit == TimeUnit.ticks

    def test_sub_two_coordinates_returns_duration(self) -> None:
        """Coordinate - Coordinate returns a Duration."""
        c1 = Coordinate(100, TimeUnit.ticks)
        c2 = Coordinate(30, TimeUnit.ticks)
        result = c1 - c2
        assert isinstance(result, Duration)
        assert result.value == 70
        assert result.unit == TimeUnit.ticks

    def test_sub_different_units_raises(self) -> None:
        """Subtraction with different units raises TypeError."""
        c1 = Coordinate(100, TimeUnit.ticks)
        c2 = Coordinate(1.5, TimeUnit.seconds)
        with pytest.raises(TypeError, match="different units"):
            c1 - c2

    def test_sub_number(self) -> None:
        """Coordinate - number returns a Coordinate with the offset value."""
        c = Coordinate(100, TimeUnit.ticks)
        result = c - 50
        assert isinstance(result, Coordinate)
        assert result.value == 50
        assert result.unit == TimeUnit.ticks

    def test_mul_scalar_int(self) -> None:
        """Multiplication with int scalar."""
        c = Coordinate(100, TimeUnit.ticks)
        result = c * 2
        assert result.value == 200
        assert result.unit == TimeUnit.ticks

    def test_mul_scalar_float(self) -> None:
        """Multiplication with float scalar."""
        c = Coordinate(100, TimeUnit.ticks)
        result = c * 0.5
        assert result.value == 50.0

    def test_mul_scalar_fraction(self) -> None:
        """Multiplication with Fraction scalar."""
        c = Coordinate(100, TimeUnit.ticks)
        result = c * Fraction(1, 2)
        assert result.value == 50

    def test_rmul_scalar(self) -> None:
        """Right multiplication works (scalar * coordinate)."""
        c = Coordinate(100, TimeUnit.ticks)
        result = 2 * c
        assert result.value == 200

    def test_mul_invalid_type_raises(self) -> None:
        """Multiplication with invalid type raises TypeError."""
        c = Coordinate(100, TimeUnit.ticks)
        with pytest.raises(TypeError, match="Cannot multiply"):
            c * "bad"  # type: ignore[operator]

    def test_truediv_scalar(self) -> None:
        """True division with scalar."""
        c = Coordinate(100, TimeUnit.ticks)
        result = c / 4
        assert result.value == 25.0

    def test_truediv_zero_raises(self) -> None:
        """Division by zero raises ZeroDivisionError."""
        c = Coordinate(100, TimeUnit.ticks)
        with pytest.raises(ZeroDivisionError):
            c / 0

    def test_truediv_invalid_type_raises(self) -> None:
        """Division with invalid type raises TypeError."""
        c = Coordinate(100, TimeUnit.ticks)
        with pytest.raises(TypeError, match="Cannot divide"):
            c / "bad"  # type: ignore[operator]

    def test_floordiv_scalar(self) -> None:
        """Floor division with scalar."""
        c = Coordinate(100, TimeUnit.ticks)
        result = c // 3
        assert result.value == 33

    def test_floordiv_zero_raises(self) -> None:
        """Floor division by zero raises ZeroDivisionError."""
        c = Coordinate(100, TimeUnit.ticks)
        with pytest.raises(ZeroDivisionError):
            c // 0

    def test_floordiv_invalid_type_raises(self) -> None:
        """Floor division with invalid type raises TypeError."""
        c = Coordinate(100, TimeUnit.ticks)
        with pytest.raises(TypeError, match="Cannot floor-divide"):
            c // "bad"  # type: ignore[operator]

    def test_mul_two_timescalars_raises(self) -> None:
        """Multiplying two TimeScalars together raises TypeError."""
        c1 = Coordinate(100, TimeUnit.ticks)
        c2 = Coordinate(2, TimeUnit.ticks)
        with pytest.raises(TypeError, match="multiply two TimeScalars"):
            c1 * c2  # type: ignore[operator]

    def test_truediv_two_timescalars_raises(self) -> None:
        """Dividing two TimeScalars raises TypeError."""
        c = Coordinate(100, TimeUnit.ticks)
        d = Duration(2, TimeUnit.ticks)
        with pytest.raises(TypeError, match="divide two TimeScalars"):
            c / d  # type: ignore[operator]


class TestCoordinateDurationArithmetic:
    """Arithmetic that crosses the Coordinate / Duration boundary."""

    def test_coord_plus_duration(self) -> None:
        """Coordinate + Duration returns a Coordinate."""
        c = Coordinate(100, TimeUnit.ticks)
        d = Duration(25, TimeUnit.ticks)
        result = c + d
        assert isinstance(result, Coordinate)
        assert result.value == 125
        assert result.unit == TimeUnit.ticks

    def test_coord_minus_duration(self) -> None:
        """Coordinate - Duration returns a Coordinate."""
        c = Coordinate(100, TimeUnit.ticks)
        d = Duration(30, TimeUnit.ticks)
        result = c - d
        assert isinstance(result, Coordinate)
        assert result.value == 70

    def test_duration_plus_duration(self) -> None:
        """Duration + Duration returns a Duration."""
        d1 = Duration(10, TimeUnit.ticks)
        d2 = Duration(5, TimeUnit.ticks)
        result = d1 + d2
        assert isinstance(result, Duration)
        assert result.value == 15

    def test_duration_minus_duration_negative(self) -> None:
        """Duration - Duration may be negative."""
        d1 = Duration(5, TimeUnit.ticks)
        d2 = Duration(10, TimeUnit.ticks)
        result = d1 - d2
        assert isinstance(result, Duration)
        assert result.value == -5
        assert result.is_negative() is True

    def test_duration_plus_coordinate_raises(self) -> None:
        """Duration + Coordinate raises TypeError (operand order matters)."""
        d = Duration(10, TimeUnit.ticks)
        c = Coordinate(100, TimeUnit.ticks)
        with pytest.raises(TypeError, match="Cannot add a Coordinate"):
            d + c

    def test_duration_minus_coordinate_raises(self) -> None:
        """Duration - Coordinate raises TypeError."""
        d = Duration(10, TimeUnit.ticks)
        c = Coordinate(100, TimeUnit.ticks)
        with pytest.raises(TypeError, match="subtract a Coordinate"):
            d - c

    def test_duration_mul_scalar(self) -> None:
        """Duration * number returns a Duration."""
        d = Duration(5, TimeUnit.ticks)
        result = d * 2
        assert isinstance(result, Duration)
        assert result.value == 10

    def test_duration_truediv_scalar(self) -> None:
        """Duration / number returns a Duration."""
        d = Duration(10, TimeUnit.ticks)
        result = d / 2
        assert isinstance(result, Duration)
        assert result.value == 5.0

    def test_duration_floordiv_scalar(self) -> None:
        """Duration // number returns a Duration."""
        d = Duration(10, TimeUnit.ticks)
        result = d // 3
        assert isinstance(result, Duration)
        assert result.value == 3

    def test_duration_mul_timescalar_raises(self) -> None:
        """Duration * TimeScalar raises TypeError."""
        d = Duration(5, TimeUnit.ticks)
        c = Coordinate(2, TimeUnit.ticks)
        with pytest.raises(TypeError, match="multiply two TimeScalars"):
            d * c  # type: ignore[operator]


class TestIdCoordinateArithmetic:
    """Arithmetic involving IdCoordinate / IdDuration (timeline-id propagation)."""

    def test_idcoord_minus_idcoord_same_id(self) -> None:
        """IdCoordinate - IdCoordinate returns IdDuration (id preserved)."""
        a = IdCoordinate(120, TimeUnit.ticks, "tl1")
        b = IdCoordinate(80, TimeUnit.ticks, "tl1")
        result = a - b
        assert isinstance(result, IdDuration)
        assert result.value == 40
        assert result.timeline_id == "tl1"

    def test_idcoord_minus_idcoord_mismatched_id_raises(self) -> None:
        """IdCoordinate - IdCoordinate with mismatched ids raises TypeError."""
        a = IdCoordinate(120, TimeUnit.ticks, "tl1")
        b = IdCoordinate(80, TimeUnit.ticks, "tl2")
        with pytest.raises(TypeError, match="mismatched timeline_id"):
            a - b

    def test_idcoord_plus_duration(self) -> None:
        """IdCoordinate + Duration returns IdCoordinate (id preserved)."""
        a = IdCoordinate(120, TimeUnit.ticks, "tl1")
        d = Duration(40, TimeUnit.ticks)
        result = a + d
        assert isinstance(result, IdCoordinate)
        assert result.value == 160
        assert result.timeline_id == "tl1"

    def test_coord_minus_idcoord(self) -> None:
        """Coordinate - IdCoordinate returns IdDuration (picks up the id)."""
        a = Coordinate(120, TimeUnit.ticks)
        b = IdCoordinate(80, TimeUnit.ticks, "tl1")
        result = a - b
        assert isinstance(result, IdDuration)
        assert result.value == 40
        assert result.timeline_id == "tl1"

    def test_coord_plus_idduration(self) -> None:
        """Coordinate + IdDuration returns IdCoordinate (picks up the id)."""
        c = Coordinate(50, TimeUnit.ticks)
        d = IdDuration(10, TimeUnit.ticks, "tl1")
        result = c + d
        assert isinstance(result, IdCoordinate)
        assert result.value == 60
        assert result.timeline_id == "tl1"

    def test_idcoord_plus_idcoord_raises(self) -> None:
        """IdCoordinate + IdCoordinate is still rejected (two Coordinates)."""
        a = IdCoordinate(120, TimeUnit.ticks, "tl1")
        b = IdCoordinate(80, TimeUnit.ticks, "tl1")
        with pytest.raises(TypeError, match="add two Coordinates"):
            a + b


class TestCoordinateComparison:
    """Tests for Coordinate comparison operations."""

    def test_lt(self) -> None:
        """Less than comparison."""
        c1 = Coordinate(100, TimeUnit.ticks)
        c2 = Coordinate(200, TimeUnit.ticks)
        assert c1 < c2
        assert not c2 < c1
        assert not c1 < c1

    def test_le(self) -> None:
        """Less than or equal comparison."""
        c1 = Coordinate(100, TimeUnit.ticks)
        c2 = Coordinate(200, TimeUnit.ticks)
        c3 = Coordinate(100, TimeUnit.ticks)
        assert c1 <= c2
        assert c1 <= c3
        assert not c2 <= c1

    def test_gt(self) -> None:
        """Greater than comparison."""
        c1 = Coordinate(100, TimeUnit.ticks)
        c2 = Coordinate(200, TimeUnit.ticks)
        assert c2 > c1
        assert not c1 > c2
        assert not c1 > c1

    def test_ge(self) -> None:
        """Greater than or equal comparison."""
        c1 = Coordinate(100, TimeUnit.ticks)
        c2 = Coordinate(200, TimeUnit.ticks)
        c3 = Coordinate(100, TimeUnit.ticks)
        assert c2 >= c1
        assert c1 >= c3
        assert not c1 >= c2

    def test_comparison_different_units_raises(self) -> None:
        """Comparison with different units raises TypeError."""
        c1 = Coordinate(100, TimeUnit.ticks)
        c2 = Coordinate(1.5, TimeUnit.seconds)
        with pytest.raises(TypeError, match="different units"):
            c1 < c2

    def test_comparison_with_number(self) -> None:
        """Comparison with a raw number compares against ``self.value``."""
        c = Coordinate(100, TimeUnit.ticks)
        assert c < 200
        assert c <= 100
        assert c > 50
        assert c >= 100
        assert not c < 100
        assert not c > 100

    def test_comparison_with_bad_type_raises(self) -> None:
        """Comparison against a non-numeric, non-TimeScalar value raises TypeError."""
        c = Coordinate(100, TimeUnit.ticks)
        with pytest.raises(TypeError, match="Cannot compare"):
            c < "bad"  # type: ignore[operator]


class TestCoordinateUtilities:
    """Tests for Coordinate utility methods."""

    def test_repr(self) -> None:
        """__repr__ returns valid representation."""
        c = Coordinate(120, TimeUnit.ticks)
        assert "Coordinate" in repr(c)
        assert "120" in repr(c)
        assert "ticks" in repr(c)

    def test_str(self) -> None:
        """__str__ returns human-readable string."""
        c = Coordinate(120, TimeUnit.ticks)
        assert str(c) == "120 ticks"

    def test_str_float(self) -> None:
        """__str__ works with float."""
        c = Coordinate(1.5, TimeUnit.seconds)
        assert str(c) == "1.5 seconds"

    def test_is_zero_true(self) -> None:
        """is_zero() returns True for zero coordinate."""
        c = Coordinate(0, TimeUnit.ticks)
        assert c.is_zero() is True

    def test_is_zero_false(self) -> None:
        """is_zero() returns False for non-zero coordinate."""
        c = Coordinate(1, TimeUnit.ticks)
        assert c.is_zero() is False

    def test_is_zero_float(self) -> None:
        """is_zero() works with float zero."""
        c = Coordinate(0.0, TimeUnit.seconds)
        assert c.is_zero() is True

    def test_is_positive_true(self) -> None:
        """is_positive() returns True for positive coordinate."""
        c = Coordinate(1, TimeUnit.ticks)
        assert c.is_positive() is True

    def test_is_positive_false(self) -> None:
        """is_positive() returns False for zero or negative."""
        assert Coordinate(0, TimeUnit.ticks).is_positive() is False
        assert Coordinate(-1, TimeUnit.ticks).is_positive() is False

    def test_is_negative_true(self) -> None:
        """is_negative() returns True for negative coordinate."""
        c = Coordinate(-1, TimeUnit.ticks)
        assert c.is_negative() is True

    def test_is_negative_false(self) -> None:
        """is_negative() returns False for zero or positive."""
        assert Coordinate(0, TimeUnit.ticks).is_negative() is False
        assert Coordinate(1, TimeUnit.ticks).is_negative() is False

    def test_with_value(self) -> None:
        """with_value() returns new Coordinate with different value."""
        c1 = Coordinate(100, TimeUnit.ticks)
        c2 = c1.with_value(200)
        assert c2.value == 200
        assert c2.unit == TimeUnit.ticks
        assert c1.value == 100  # Original unchanged

    def test_with_unit(self) -> None:
        """with_unit() returns new Coordinate with different unit."""
        c1 = Coordinate(100, TimeUnit.ticks)
        c2 = c1.with_unit(TimeUnit.pulses)
        assert c2.value == 100
        assert c2.unit == TimeUnit.pulses
        assert c1.unit == TimeUnit.ticks  # Original unchanged


class TestCoordinateNumericProtocols:
    """Tests for Coordinate numeric protocol methods (__float__, __int__, __index__)."""

    # --- __float__ tests ---

    def test_float_from_int(self) -> None:
        """__float__ converts int value to float."""
        c = Coordinate(15343, TimeUnit.pixels)
        result = float(c)
        assert result == 15343.0
        assert isinstance(result, float)

    def test_float_from_float(self) -> None:
        """__float__ returns float value unchanged."""
        c = Coordinate(3.14159, TimeUnit.seconds)
        result = float(c)
        assert result == 3.14159
        assert isinstance(result, float)

    def test_float_from_fraction(self) -> None:
        """__float__ converts Fraction value to float."""
        c = Coordinate(Fraction(3, 4), TimeUnit.quarters)
        result = float(c)
        assert result == 0.75
        assert isinstance(result, float)

    def test_float_enables_math_functions(self) -> None:
        """__float__ allows Coordinate to be used with math module."""
        import math

        c = Coordinate(16, TimeUnit.pixels)
        # math.sqrt uses __float__ implicitly
        assert math.sqrt(c) == 4.0

    # --- __int__ tests ---

    def test_int_from_int(self) -> None:
        """__int__ returns int value unchanged."""
        c = Coordinate(15343, TimeUnit.pixels)
        result = int(c)
        assert result == 15343
        assert isinstance(result, int)

    def test_int_from_float_truncates(self) -> None:
        """__int__ truncates float value towards zero."""
        c = Coordinate(3.9, TimeUnit.seconds)
        assert int(c) == 3

        c_neg = Coordinate(-3.9, TimeUnit.seconds)
        assert int(c_neg) == -3

    def test_int_from_fraction(self) -> None:
        """__int__ truncates Fraction value."""
        c = Coordinate(Fraction(7, 4), TimeUnit.quarters)
        assert int(c) == 1

    # --- __index__ tests ---

    def test_index_from_int(self) -> None:
        """__index__ returns int value for use as index."""
        c = Coordinate(5, TimeUnit.pixels)
        # Test as string index
        assert "hello world"[c] == " "  # 5th character

    def test_index_enables_slice_notation(self) -> None:
        """__index__ allows Coordinate in slice notation."""
        c = Coordinate(3, TimeUnit.pixels)
        assert [0, 1, 2, 3, 4, 5][c] == 3

    def test_index_from_float_raises(self) -> None:
        """__index__ raises TypeError for float values."""
        c = Coordinate(3.5, TimeUnit.seconds)
        with pytest.raises(TypeError, match="Only integer Coordinates"):
            "hello"[c]

    def test_index_from_fraction_raises(self) -> None:
        """__index__ raises TypeError for Fraction values."""
        c = Coordinate(Fraction(3, 4), TimeUnit.quarters)
        with pytest.raises(TypeError, match="Only integer Coordinates"):
            [1, 2, 3][c]

    # --- Integration tests ---

    def test_float_in_tuple_construction(self) -> None:
        """float(coord) can be used in tuple construction."""
        c = Coordinate(15343, TimeUnit.pixels)
        result = (float(c), "dgt_1")
        assert result == (15343.0, "dgt_1")

    def test_float_in_arithmetic_with_numbers(self) -> None:
        """float(coord) enables arithmetic with plain numbers."""
        c = Coordinate(100, TimeUnit.pixels)
        # This would fail without float() conversion
        result = 10.0 + float(c)
        assert result == 110.0
