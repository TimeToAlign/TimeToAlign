"""Typed measure and beat keys.

Reasoning, specimens and gold values: ``tests/core/README.md``,
section "test_addresses.py".
"""

from __future__ import annotations

from fractions import Fraction

import pyarrow as pa
import pytest
from pydantic import ValidationError

from timetoalign.core import (
    Address,
    Beat,
    BeatPolicy,
    Coordinate,
    MeasureId,
    MeasureIdAddress,
    MeasureNumber,
    MeasureNumberAddress,
    MeasureNumberField,
    TimeUnit,
)
from timetoalign.core.retrieval import classify_dispatch_input


class TestParse:
    """``Address.parse`` reads the two printed forms."""

    def test_measure_plus_offset(self) -> None:
        """The offset counts whole notes, so 3/8 is 3/2 quarters."""
        address = Address.parse("12+3/8")

        assert address == MeasureNumberAddress(
            mn="12", at=Coordinate(Fraction(3, 2), TimeUnit.quarters)
        )
        assert address.at.value == Fraction(3, 2)
        assert address.at.unit is TimeUnit.quarters
        assert address.selects_span is False

    def test_bare_label_keeps_its_suffix(self) -> None:
        """A split-bar label is one label, not a number with decoration."""
        address = Address.parse("12a")

        assert address == MeasureNumber(mn="12a")
        assert address.mn == "12a"
        assert address.selects_span is True

    def test_offset_denomination_is_configurable(self) -> None:
        """A source counting quarters says so; the conversion stays exact."""
        address = Address.parse("12+3/8", offset_denomination=Fraction(1, 4))

        assert address.at.value == Fraction(3, 8)

    def test_unreadable_offset_raises(self) -> None:
        with pytest.raises(ValueError, match="within-measure offset"):
            Address.parse("12+not-a-fraction")


class TestMeasureId:
    """An int is a position; a str is an identifier. Never both."""

    def test_int_is_positional(self) -> None:
        assert MeasureId(value=13).is_positional is True

    def test_str_is_an_identifier(self) -> None:
        assert MeasureId(value="m-0023").is_positional is False

    def test_count_refuses_a_string(self) -> None:
        """``"13"`` is an identifier; coercing it would invent a position."""
        with pytest.raises(ValueError, match="int position"):
            MeasureId.count("13")

    def test_identifier_refuses_an_int(self) -> None:
        with pytest.raises(ValueError, match="str id"):
            MeasureId.identifier(13)

    def test_strict_construction_does_not_coerce(self) -> None:
        assert MeasureId.count(13).value == 13
        assert MeasureId.identifier("13").value == "13"

    def test_selects_span(self) -> None:
        assert MeasureId(value=13).selects_span is True
        assert MeasureIdAddress(value=13, at=Beat(index=2)).selects_span is False


class TestBeat:
    """A beat knows its index; a policy tells it how long it is."""

    def test_downbeat(self) -> None:
        assert Beat(index=1).is_downbeat is True
        assert Beat(index=2).is_downbeat is False
        assert Beat(index=1, level=1).is_downbeat is False

    def test_size_is_none_without_a_policy(self) -> None:
        """A key does not know the bar's default counting; resolution does."""
        assert Beat(index=2).size is None

    def test_size_under_a_policy(self) -> None:
        policy = BeatPolicy.from_time_signature("6/8")

        assert Beat(index=2, policy=policy).size == Fraction(3, 2)

    def test_offset_takes_the_bar_policy_when_the_beat_has_none(self) -> None:
        policy = BeatPolicy.from_time_signature("3/8+2/8+3/8")

        assert Beat(index=3).offset(policy) == Fraction(5, 2)

    def test_offset_without_any_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="BeatPolicy"):
            Beat(index=3).offset()

    def test_index_is_one_based(self) -> None:
        with pytest.raises(ValidationError):
            Beat(index=0)


class TestBareStringsAreRejectedInAt:
    """Strings are parsed explicitly, never accepted as a position."""

    def test_measure_number_address_refuses_a_string(self) -> None:
        with pytest.raises(ValidationError):
            MeasureNumberAddress(mn="12", at="3/8")


class TestDispatchClassification:
    """An address is a structural position, so it uses positional getters."""

    def test_scalar_address_is_a_position(self) -> None:
        assert classify_dispatch_input(MeasureNumber(mn="12")) == "coordinate"
        assert classify_dispatch_input(MeasureId(value=13)) == "coordinate"
        assert classify_dispatch_input(Address.parse("12+3/8")) == "coordinate"

    def test_collection_of_addresses_is_positions(self) -> None:
        assert (
            classify_dispatch_input([MeasureNumber(mn="12"), MeasureId(value=13)])
            == "coordinates"
        )

    def test_addresses_do_not_mix_with_string_keys(self) -> None:
        with pytest.raises(TypeError):
            classify_dispatch_input([MeasureNumber(mn="12"), "n1b8"])


class TestMeasureNumberField:
    """The storage struct holds every fact a source can state."""

    def test_struct_shape(self) -> None:
        schema = MeasureNumberField.pa_schema

        assert [field.name for field in schema] == [
            "rendition",
            "mc",
            "mn",
            "volta",
            "section",
        ]
        assert schema.field(0).type == pa.int64()
        assert schema.field(1).type == pa.int64()
        assert schema.field(2).type == pa.string()

    def test_round_trip_of_a_split_bar_label(self) -> None:
        """WoO 71 bar 237b is measure count 261 in its second ending."""
        from timetoalign.core.fields import build_struct_array

        scalar = MeasureNumber(mn="237b", mc=261, volta=2)
        array = build_struct_array(MeasureNumber, [scalar])
        field = MeasureNumberField.from_field(
            (array, pa.field("measure_number", array.type))
        )

        assert array.to_pylist() == [
            {
                "rendition": None,
                "mc": 261,
                "mn": "237b",
                "volta": 2,
                "section": None,
            }
        ]
        assert field[0] == scalar

    def test_a_label_column_fills_only_the_label(self) -> None:
        """A source that prints only labels states neither count nor volta."""
        field = MeasureNumberField(name="measure_number").from_array(
            pa.array(["0", "1", "237b"])
        )

        assert field[0] == MeasureNumber(mn="0")
        assert field[2] == MeasureNumber(mn="237b")
        assert field.data.to_pylist()[2] == {
            "rendition": None,
            "mc": None,
            "mn": "237b",
            "volta": None,
            "section": None,
        }

    def test_int_labels_coerce_to_the_printed_string(self) -> None:
        assert MeasureNumber(mn=12).mn == "12"
