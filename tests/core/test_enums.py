"""Tests for core/enums.py."""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign.core.enums import Domain, EventType, NumberType, TimeUnit


class TestFancyStrEnum:
    """Tests for the FancyStrEnum base class."""

    def test_get_abbreviations_dict(self) -> None:
        """get_abbreviations() returns a dict when string=False."""
        abbrevs = Domain.get_abbreviations(string=False)
        assert isinstance(abbrevs, dict)
        assert "logical" in abbrevs
        assert "lo" in abbrevs["logical"]

    def test_get_abbreviations_string(self) -> None:
        """get_abbreviations() returns a formatted string when string=True."""
        abbrevs = Domain.get_abbreviations(string=True)
        assert isinstance(abbrevs, str)
        assert "logical" in abbrevs
        assert "lo" in abbrevs

    def test_repr(self) -> None:
        """FancyStrEnum __repr__ returns quoted name."""
        assert repr(Domain.logical) == '"logical"'

    def test_str(self) -> None:
        """FancyStrEnum __str__ returns the name."""
        assert str(Domain.logical) == "logical"


class TestDomain:
    """Tests for the Domain enum."""

    def test_domain_values(self) -> None:
        """All domains have correct string values."""
        assert Domain.logical.value == "logical"
        assert Domain.physical.value == "physical"
        assert Domain.graphical.value == "graphical"

    def test_domain_str(self) -> None:
        """str() returns the name."""
        assert str(Domain.logical) == "logical"
        assert str(Domain.physical) == "physical"
        assert str(Domain.graphical) == "graphical"

    def test_domain_is_str_subclass(self) -> None:
        """Domain is a string enum, usable as a string."""
        assert isinstance(Domain.logical, str)
        assert Domain.logical == "logical"

    def test_domain_from_string(self) -> None:
        """Can create Domain from string."""
        assert Domain("logical") == Domain.logical
        assert Domain("physical") == Domain.physical
        assert Domain("graphical") == Domain.graphical

    def test_domain_from_alias(self) -> None:
        """Can create Domain from alias."""
        assert Domain("lo") == Domain.logical
        assert Domain("ph") == Domain.physical
        assert Domain("gr") == Domain.graphical

    def test_domain_alias_identity(self) -> None:
        """Aliases are the same object as the canonical member."""
        assert Domain.lo is Domain.logical
        assert Domain.ph is Domain.physical
        assert Domain.gr is Domain.graphical

    def test_domain_invalid_value(self) -> None:
        """Invalid values raise ValueError."""
        with pytest.raises(ValueError, match="not a valid Domain"):
            Domain("invalid")

    def test_domain_invalid_non_string_value(self) -> None:
        """Non-string invalid values raise ValueError."""
        with pytest.raises(ValueError, match="not a valid Domain"):
            Domain(123)  # type: ignore[arg-type]


class TestTimeUnit:
    """Tests for the TimeUnit enum."""

    def test_physical_units(self) -> None:
        """Physical domain units have correct values."""
        assert TimeUnit.seconds.value == "seconds"
        assert TimeUnit.milliseconds.value == "milliseconds"
        assert TimeUnit.samples.value == "samples"
        assert TimeUnit.frames.value == "frames"

    def test_logical_units(self) -> None:
        """Musical domain units have correct values."""
        assert TimeUnit.quarters.value == "quarters"
        assert TimeUnit.number.value == "number"
        assert TimeUnit.floating_measures.value == "floating_measures"
        assert TimeUnit.ticks.value == "ticks"

    def test_graphical_units(self) -> None:
        """Graphical domain units have correct values."""
        assert TimeUnit.pixels.value == "pixels"
        assert TimeUnit.points.value == "points"
        assert TimeUnit.inches.value == "inches"
        assert TimeUnit.millimeters.value == "millimeters"

    def test_timeunit_str(self) -> None:
        """str() returns the name."""
        assert str(TimeUnit.seconds) == "seconds"
        assert str(TimeUnit.ticks) == "ticks"
        assert str(TimeUnit.pixels) == "pixels"

    def test_timeunit_aliases(self) -> None:
        """Aliases work correctly."""
        # Musical
        assert TimeUnit.q is TimeUnit.quarters
        assert TimeUnit.fm is TimeUnit.floating_measures
        assert TimeUnit.pulses is TimeUnit.ticks

        # Physical
        assert TimeUnit.s is TimeUnit.seconds
        assert TimeUnit.ms is TimeUnit.milliseconds

        # Graphical
        assert TimeUnit.px is TimeUnit.pixels
        assert TimeUnit.pt is TimeUnit.points
        assert TimeUnit.mm is TimeUnit.millimeters
        assert TimeUnit.cm is TimeUnit.centimeters

    @pytest.mark.parametrize("unit", [TimeUnit.css_pixels, TimeUnit.user_units])
    def test_vector_graphics_units_are_continuous_graphical(
        self, unit: TimeUnit
    ) -> None:
        """CSS pixels and user units are graphical, continuous and float by default."""
        assert unit.domain is Domain.graphical
        assert unit.is_discrete is False
        assert unit.default_number_type is NumberType.float
        assert unit.allowed_number_types == {NumberType.float, NumberType.fraction}
        assert unit.allowed_number_types == TimeUnit.points.allowed_number_types

    def test_vector_graphics_unit_abbreviations(self) -> None:
        """``uu`` spells user units; ``px`` stays the discrete pixel; CSS pixels have none."""
        abbreviations = TimeUnit.get_abbreviations()
        assert abbreviations["css_pixels"] == []
        assert abbreviations["user_units"] == ["uu"]
        assert TimeUnit("css_pixels") is TimeUnit.css_pixels
        assert TimeUnit("user_units") is TimeUnit.user_units
        assert TimeUnit("uu") is TimeUnit.user_units
        assert TimeUnit.uu is TimeUnit.user_units
        assert TimeUnit("px") is TimeUnit.pixels

    def test_a_generic_beat_axis_is_the_number_unit(self) -> None:
        """There is no ``beats`` unit, and no alias resurrects one.

        A bare "beat 217 of this timeline" carries no metrical meaning and
        may cumulate beats of different sizes, so beats are typed keys
        rather than a coordinate unit. A source-given cumulative beat rod
        rides the generic ``number`` axis, which admits an exact ratio.
        """
        for spelling in ("beats", "b"):
            with pytest.raises(ValueError, match="is not a valid TimeUnit"):
                TimeUnit(spelling)

        assert not hasattr(TimeUnit, "beats")
        assert TimeUnit.number.domain == Domain.logical
        assert TimeUnit.number.is_discrete is False
        assert NumberType.fraction in TimeUnit.number.allowed_number_types

    def test_timeunit_from_alias(self) -> None:
        """Can create TimeUnit from alias string."""
        assert TimeUnit("q") == TimeUnit.quarters
        assert TimeUnit("ms") == TimeUnit.milliseconds
        assert TimeUnit("px") == TimeUnit.pixels

    def test_timeunit_domain_physical(self) -> None:
        """Physical units map to physical domain."""
        assert TimeUnit.seconds.domain == Domain.physical
        assert TimeUnit.milliseconds.domain == Domain.physical
        assert TimeUnit.samples.domain == Domain.physical
        assert TimeUnit.frames.domain == Domain.physical

    def test_timeunit_domain_logical(self) -> None:
        """Musical units map to logical domain."""
        assert TimeUnit.quarters.domain == Domain.logical
        assert TimeUnit.floating_measures.domain == Domain.logical
        assert TimeUnit.ticks.domain == Domain.logical
        assert TimeUnit.number.domain == Domain.logical

    def test_timeunit_domain_graphical(self) -> None:
        """Graphical units map to graphical domain."""
        assert TimeUnit.pixels.domain == Domain.graphical
        assert TimeUnit.points.domain == Domain.graphical
        assert TimeUnit.inches.domain == Domain.graphical
        assert TimeUnit.millimeters.domain == Domain.graphical

    def test_timeunit_is_discrete(self) -> None:
        """Discrete units are correctly identified."""
        # Discrete
        assert TimeUnit.samples.is_discrete is True
        assert TimeUnit.frames.is_discrete is True
        assert TimeUnit.ticks.is_discrete is True
        assert TimeUnit.pixels.is_discrete is True

        # Continuous
        assert TimeUnit.seconds.is_discrete is False
        assert TimeUnit.milliseconds.is_discrete is False
        assert TimeUnit.quarters.is_discrete is False
        assert TimeUnit.floating_measures.is_discrete is False
        assert TimeUnit.points.is_discrete is False
        assert TimeUnit.inches.is_discrete is False
        assert TimeUnit.millimeters.is_discrete is False

    def test_timeunit_from_string(self) -> None:
        """Can create TimeUnit from string."""
        assert TimeUnit("seconds") == TimeUnit.seconds
        assert TimeUnit("ticks") == TimeUnit.ticks
        assert TimeUnit("pixels") == TimeUnit.pixels

    def test_timeunit_invalid_value(self) -> None:
        """Invalid values raise ValueError."""
        with pytest.raises(ValueError, match="not a valid TimeUnit"):
            TimeUnit("invalid")


class TestNumberType:
    """Tests for the NumberType enum."""

    def test_numbertype_values(self) -> None:
        """NumberType has correct values (Python types)."""
        assert NumberType.int.value is int
        assert NumberType.float.value is float
        assert NumberType.fraction.value is Fraction

    def test_numbertype_str(self) -> None:
        """str() returns the name."""
        assert str(NumberType.int) == "int"
        assert str(NumberType.float) == "float"
        assert str(NumberType.fraction) == "fraction"

    def test_numbertype_python_type(self) -> None:
        """python_type returns the correct Python type."""
        assert NumberType.int.python_type is int
        assert NumberType.float.python_type is float
        assert NumberType.fraction.python_type is Fraction

    def test_numbertype_from_string(self) -> None:
        """Can create NumberType from string."""
        assert NumberType("int") == NumberType.int
        assert NumberType("float") == NumberType.float
        assert NumberType("fraction") == NumberType.fraction

    def test_numbertype_from_type(self) -> None:
        """Can create NumberType from Python type."""
        assert NumberType(int) == NumberType.int
        assert NumberType(float) == NumberType.float
        assert NumberType(Fraction) == NumberType.fraction

    def test_numbertype_from_number(self) -> None:
        """from_number() creates NumberType from number instance."""
        assert NumberType.from_number(42) == NumberType.int
        assert NumberType.from_number(3.14) == NumberType.float
        assert NumberType.from_number(Fraction(1, 2)) == NumberType.fraction

    def test_numbertype_invalid_value(self) -> None:
        """Invalid values raise ValueError."""
        with pytest.raises(ValueError):
            NumberType("invalid")

    def test_numbertype_invalid_non_string_value(self) -> None:
        """Non-string invalid values (not a recognized type) raise ValueError."""
        with pytest.raises(ValueError):
            NumberType(str)  # str type is not a valid NumberType value


class TestEventType:
    """Tests for the EventType enum."""

    def test_eventtype_values(self) -> None:
        """EventType has correct values."""
        assert EventType.instant.value == "instant"
        assert EventType.interval.value == "interval"

    def test_eventtype_str(self) -> None:
        """str() returns the name."""
        assert str(EventType.instant) == "instant"
        assert str(EventType.interval) == "interval"

    def test_eventtype_aliases(self) -> None:
        """Aliases work correctly."""
        assert EventType.inst is EventType.instant
        assert EventType.intv is EventType.interval

    def test_eventtype_from_alias(self) -> None:
        """Can create EventType from alias string."""
        assert EventType("inst") == EventType.instant
        assert EventType("intv") == EventType.interval

    def test_eventtype_from_string(self) -> None:
        """Can create EventType from string."""
        assert EventType("instant") == EventType.instant
        assert EventType("interval") == EventType.interval
