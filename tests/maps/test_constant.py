"""Tests for ConstantMap.

ConstantMap returns a fixed value for any input coordinate. It is used
inside CombinationMap to attach metadata labels (e.g., filenames) to
every coordinate on a timeline.
"""

from fractions import Fraction

import numpy as np
import pytest

from timetoalign.core.enums import NumberType
from timetoalign.maps import CombinationMap, ConstantMap, LinearMap, ScalarMap
from timetoalign.maps.base import ConversionMap


class TestConstantMap:
    """Tests for ConstantMap basic functionality."""

    def test_initialization(self):
        """Test basic initialisation with a string value."""
        cmap = ConstantMap(value="page1.jpeg", name="filename")
        assert cmap.value == "page1.jpeg"
        assert cmap.name == "filename"
        assert cmap.source_unit is None
        assert cmap.target_unit is None
        assert cmap.is_invertible is False

    def test_scalar_conversion_string(self):
        """Test scalar conversion returns the constant string."""
        cmap = ConstantMap(value="page1.jpeg")
        assert cmap(42.0) == "page1.jpeg"
        assert cmap(0) == "page1.jpeg"
        assert cmap(-100.5) == "page1.jpeg"

    def test_scalar_conversion_numeric(self):
        """Test scalar conversion with a numeric constant."""
        cmap = ConstantMap(value=3.14)
        assert cmap(0.0) == 3.14
        assert cmap(999) == 3.14

    def test_scalar_conversion_none(self):
        """Test scalar conversion with None as the constant."""
        cmap = ConstantMap(value=None)
        assert cmap(42.0) is None

    def test_array_conversion(self):
        """Test array conversion returns an array filled with the constant."""
        cmap = ConstantMap(value="page1.jpeg")
        arr = np.array([1.0, 2.0, 3.0])
        result = cmap(arr)

        assert isinstance(result, np.ndarray)
        assert result.dtype == object
        assert len(result) == 3
        assert result[0] == "page1.jpeg"
        assert result[1] == "page1.jpeg"
        assert result[2] == "page1.jpeg"

    def test_array_conversion_preserves_shape(self):
        """Test that array output shape matches input shape."""
        cmap = ConstantMap(value="x")
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = cmap(arr)
        assert result.shape == arr.shape

    def test_not_invertible(self):
        """ConstantMap cannot be inverted."""
        cmap = ConstantMap(value="test")
        with pytest.raises(NotImplementedError, match="cannot be inverted"):
            cmap.inverse()

    def test_repr(self):
        """Test repr includes the value."""
        cmap = ConstantMap(value="page1.jpeg", name="filename")
        r = repr(cmap)
        assert "ConstantMap" in r
        assert "page1.jpeg" in r
        assert "filename" in r

    def test_repr_without_name(self):
        """Test repr when no explicit name is set."""
        cmap = ConstantMap(value="test")
        r = repr(cmap)
        assert "ConstantMap" in r
        assert "test" in r


class TestConstantMapSerialization:
    """Tests for ConstantMap serialization round-trips."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        cmap = ConstantMap(value="page1.jpeg", name="filename", uid="cmap_1")
        d = cmap.to_dict()

        assert d["type"] == "ConstantMap"
        assert d["value"] == "page1.jpeg"
        assert d["id"] == "cmap_1"
        assert d["source_unit"] is None
        assert d["target_unit"] is None

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "type": "ConstantMap",
            "value": "page2.jpeg",
            "id": "cmap_2",
            "source_unit": None,
            "target_unit": None,
            "name": "source_file",
        }
        cmap = ConstantMap.from_dict(d)
        assert cmap.value == "page2.jpeg"
        assert cmap.id == "cmap_2"
        assert cmap.name == "source_file"

    def test_round_trip(self):
        """Test full serialization round-trip."""
        original = ConstantMap(value="my_image.png", name="img", uid="rt1")
        d = original.to_dict()
        restored = ConstantMap.from_dict(d)

        assert restored.value == original.value
        assert restored.id == original.id
        assert restored(100.0) == original(100.0)

    def test_base_class_dispatch(self):
        """Test that ConversionMap.from_dict dispatches to ConstantMap."""
        d = {
            "type": "ConstantMap",
            "value": "dispatched.jpeg",
            "id": "dispatch_1",
            "source_unit": None,
            "target_unit": None,
        }
        cmap = ConversionMap.from_dict(d)
        assert isinstance(cmap, ConstantMap)
        assert cmap.value == "dispatched.jpeg"


class TestConstantMapWithCombinationMap:
    """Tests for ConstantMap inside CombinationMap (the primary use case)."""

    def test_combination_with_scalar_and_constant(self):
        """Test CombinationMap with ScalarMap + ConstantMap."""
        px_to_sec = ScalarMap(
            scalar=30.0 / 866, source_unit="pixels", target_unit="seconds"
        )
        filename = ConstantMap(value="page1_1.jpeg", name="filename")

        combo = CombinationMap(
            maps={"seconds": px_to_sec, "filename": filename},
            source_unit="pixels",
        )

        result = combo(433.0)
        assert isinstance(result, dict)
        assert "seconds" in result
        assert "filename" in result
        assert result["filename"] == "page1_1.jpeg"
        # A ScalarMap writes scalar × input on its float target axis.
        assert result["seconds"] == 433.0 * (30.0 / 866)

    def test_combination_array(self):
        """Test array conversion in CombinationMap with ConstantMap."""
        px_to_sec = ScalarMap(scalar=0.1, source_unit="pixels", target_unit="seconds")
        filename = ConstantMap(value="img.jpeg", name="filename")

        combo = CombinationMap(
            maps={"seconds": px_to_sec, "filename": filename},
            source_unit="pixels",
        )

        arr = np.array([100.0, 200.0, 300.0])
        result = combo(arr)

        np.testing.assert_array_almost_equal(
            result["seconds"], np.array([10.0, 20.0, 30.0])
        )
        assert all(v == "img.jpeg" for v in result["filename"])

    def test_combination_source_unit_none_skipped(self):
        """ConstantMap with source_unit=None does not conflict in CombinationMap."""
        m1 = LinearMap(scalar=2.0, source_unit="pixels", target_unit="seconds")
        m2 = ConstantMap(value="no_unit")  # source_unit=None

        # This should NOT raise, because CombinationMap skips None source_units
        combo = CombinationMap(
            maps={"converted": m1, "label": m2},
            source_unit="pixels",
        )
        result = combo(5.0)
        assert result["converted"] == 10.0
        assert result["label"] == "no_unit"

    def test_combination_serialization_with_constant(self):
        """Test CombinationMap serialization when it contains a ConstantMap."""
        px_to_sec = ScalarMap(
            scalar=0.5, source_unit="pixels", target_unit="seconds", uid="s1"
        )
        filename = ConstantMap(value="test.jpeg", name="filename", uid="c1")

        combo = CombinationMap(
            maps={"seconds": px_to_sec, "filename": filename},
            source_unit="pixels",
            uid="combo1",
        )

        d = combo.to_dict()
        assert d["maps"]["filename"]["type"] == "ConstantMap"
        assert d["maps"]["filename"]["value"] == "test.jpeg"

        # Round-trip via base class dispatch
        restored = CombinationMap.from_dict(d)
        assert isinstance(restored["filename"], ConstantMap)
        assert restored["filename"].value == "test.jpeg"
        assert restored(10.0) == {"seconds": 5.0, "filename": "test.jpeg"}


class TestConstantMapImport:
    """Test that ConstantMap is properly exported at all levels."""

    def test_import_from_maps(self):
        """Test import from timetoalign.maps."""
        from timetoalign.maps import ConstantMap as CM

        assert CM is ConstantMap

    def test_import_from_package(self):
        """Test import from top-level timetoalign package."""
        from timetoalign import ConstantMap as CM

        assert CM is ConstantMap


class TestFloatParameters:
    """The constant is the map's parameter: a float one is answered as a float."""

    def test_float_constant_answers_the_float(self):
        cmap = ConstantMap(value=0.1, source_unit="quarters", target_unit="quarters")
        assert cmap.output_number_type is NumberType.float
        result = cmap(0)
        assert result == 0.1
        assert type(result) is float

    def test_exact_constant_stays_exact(self):
        cmap = ConstantMap(
            value=Fraction(1, 10), source_unit="quarters", target_unit="quarters"
        )
        assert cmap.output_number_type is NumberType.fraction
        result = cmap(0)
        assert result == Fraction(1, 10)
        assert type(result) is Fraction

    def test_label_is_no_parameter(self):
        cmap = ConstantMap(value="page1.jpeg")
        assert cmap(0.1) == "page1.jpeg"
