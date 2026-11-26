"""Tests for the AlignmentAnchor and MatchClaim scalars.

Validation strategy and exact values: ``tests/alignment/README.md`` (the
"AlignmentAnchor Tests" and "MatchClaim Tests" sections). The provenance entry
a claim holds is tested for itself in ``tests/core/test_provenance.py``.
"""

from __future__ import annotations

import json
import re
from fractions import Fraction

import pytest
from pydantic import ValidationError

from timetoalign.alignment import (
    AlignmentAnchor,
    AlignmentBundle,
    Belief,
    MatchClaim,
    MatchClaimField,
    Provenance,
)
from timetoalign.core import (
    CHUNK,
    NOMATCH,
    Coordinate,
    IdCoordinate,
    NumberType,
    TimeUnit,
)
from timetoalign.maps import SamplesToSeconds
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteLogicalTimeline,
    DiscretePhysicalTimeline,
)

# region Fixtures

SCORE = ContinuousLogicalTimeline(length=200, uid="score:1")
RECORDING = ContinuousPhysicalTimeline(length=100.0, uid="recording:1")
MIDI = DiscreteLogicalTimeline(length=10_000, uid="midi")
FLOAT_SCORE = ContinuousLogicalTimeline(
    length=16, uid="float_score", number_type="float"
)


def _wav() -> DiscretePhysicalTimeline:
    """A 44,100 Hz samples timeline with its C-Map to seconds."""
    wav = DiscretePhysicalTimeline(length=44_100 * 60, uid="wav")
    wav.add_conversion_map(SamplesToSeconds(sample_rate=44_100))
    return wav


def _anchor(
    a: str, coordinate_a: Coordinate, b: str, coordinate_b: Coordinate
) -> AlignmentAnchor:
    return AlignmentAnchor(
        timeline_a_id=a,
        coordinate_a=coordinate_a,
        timeline_b_id=b,
        coordinate_b=coordinate_b,
    )


def _number(value: float) -> Coordinate:
    return Coordinate(value, TimeUnit.number)


@pytest.fixture
def basic_anchor() -> AlignmentAnchor:
    """A score quarter paired with a recording second."""
    return _anchor(
        "score:1",
        Coordinate(100, TimeUnit.quarters),
        "recording:1",
        Coordinate(45.5, TimeUnit.seconds),
    )


def _claim(anchor: AlignmentAnchor, **fields: object) -> MatchClaim:
    """An explicit synchronous claim on *anchor* between two named events."""
    return MatchClaim(
        timeline_a_id=anchor.timeline_a_id,
        timeline_b_id=anchor.timeline_b_id,
        start_anchor=anchor,
        event_a_id="a1",
        event_b_id="b1",
        **fields,
    )


#: The check's refusal, for a coordinate of the given description.
def _off_axis(where: str, unit: str, number_type: str, timeline: str, axis: str) -> str:
    return (
        f"{where} is in {unit} ({number_type}), but timeline '{timeline}' counts in "
        f"{axis}: a MatchClaim's coordinates are on its timelines' own axes. "
        f"Convert a position through the timeline's C-Map before the MatchClaim "
        f"is made."
    )


# endregion


# region AlignmentAnchor Tests


class TestAlignmentAnchor:
    """Tests for AlignmentAnchor (pure coordinate pair)."""

    def test_basic_creation(self) -> None:
        anchor = _anchor("tl1", _number(10.0), "tl2", _number(20.0))
        assert anchor.timeline_a_id == "tl1"
        assert anchor.coordinate_a.value == 10.0
        assert anchor.timeline_b_id == "tl2"
        assert anchor.coordinate_b.value == 20.0

    def test_raw_float_coordinates_are_rejected(self) -> None:
        """Anchors require explicit unit-bearing Coordinate values."""
        with pytest.raises(ValidationError):
            AlignmentAnchor(
                timeline_a_id="tl1",
                coordinate_a=10.0,
                timeline_b_id="tl2",
                coordinate_b=20.0,
            )

    def test_no_id_or_claim_fields(self) -> None:
        anchor = _anchor("tl1", _number(0.0), "tl2", _number(0.0))
        assert not hasattr(anchor, "id")
        assert not hasattr(anchor, "is_explicit")
        assert not hasattr(anchor, "is_synchronous")

    def test_timelines_and_coordinates(self, basic_anchor: AlignmentAnchor) -> None:
        assert basic_anchor.timelines == ("score:1", "recording:1")
        assert basic_anchor.coordinates == (
            Coordinate(100, TimeUnit.quarters),
            Coordinate(45.5, TimeUnit.seconds),
        )

    def test_get_coordinate_for(self, basic_anchor: AlignmentAnchor) -> None:
        assert basic_anchor.get_coordinate_for(
            "score:1", format="coordinate"
        ) == Coordinate(100, TimeUnit.quarters)
        assert basic_anchor.get_coordinate_for("recording:1") == IdCoordinate(
            45.5, TimeUnit.seconds, "recording:1"
        )
        with pytest.raises(KeyError):
            basic_anchor.get_coordinate_for("other")

    def test_connects(self, basic_anchor: AlignmentAnchor) -> None:
        assert basic_anchor.connects("score:1") is True
        assert basic_anchor.connects("recording:1") is True
        assert basic_anchor.connects("other") is False
        assert basic_anchor.connects_both("recording:1", "score:1") is True
        assert basic_anchor.connects_both("score:1", "other") is False

    def test_to_dict(self, basic_anchor: AlignmentAnchor) -> None:
        assert basic_anchor.to_dict() == {
            "timeline_a_id": "score:1",
            "coordinate_a": {
                "unit": "quarters",
                "number_type": "fraction",
                "value": [100, 1],
            },
            "timeline_b_id": "recording:1",
            "coordinate_b": {"unit": "seconds", "number_type": "float", "value": 45.5},
        }

    def test_from_dict_roundtrip(self, basic_anchor: AlignmentAnchor) -> None:
        assert AlignmentAnchor.from_dict(basic_anchor.to_dict()) == basic_anchor

    def test_coordinates_kept_as_given(self) -> None:
        """The anchor rewrites no coordinate: the timeline, not the unit, decides."""
        anchor = _anchor(
            "audio",
            Coordinate(Fraction(25, 2), TimeUnit.seconds),
            "float_score",
            Coordinate(1.5, TimeUnit.quarters, number_type=NumberType.float),
        )
        assert anchor.coordinate_a.value == Fraction(25, 2)
        assert anchor.coordinate_a.number_type is NumberType.fraction
        assert anchor.coordinate_b.value == 1.5
        assert anchor.coordinate_b.number_type is NumberType.float
        assert repr(anchor) == (
            "AlignmentAnchor(audio@25/2 seconds <-> float_score@1.5 quarters)"
        )
        assert anchor.to_dict()["coordinate_a"] == {
            "unit": "seconds",
            "number_type": "fraction",
            "value": [25, 2],
        }
        assert anchor.to_dict()["coordinate_b"] == {
            "unit": "quarters",
            "number_type": "float",
            "value": 1.5,
        }
        assert AlignmentAnchor.from_dict(anchor.to_dict()) == anchor

    def test_frozen_model(self, basic_anchor: AlignmentAnchor) -> None:
        with pytest.raises(ValidationError):
            basic_anchor.coordinate_a = 200.0  # type: ignore

    def test_repr(self, basic_anchor: AlignmentAnchor) -> None:
        assert repr(basic_anchor) == (
            "AlignmentAnchor(score:1@100 quarters <-> recording:1@45.5 seconds)"
        )
        assert str(basic_anchor) == repr(basic_anchor)
        assert basic_anchor._repr_html_() == (
            "<div style='font-family: monospace;'><strong>AlignmentAnchor</strong>("
            "score:1@100 quarters &lt;-&gt; recording:1@45.5 seconds)</div>"
        )

    def test_value_equality(self) -> None:
        assert _anchor("tl1", _number(10.0), "tl2", _number(20.0)) == _anchor(
            "tl1", _number(10.0), "tl2", _number(20.0)
        )

    @pytest.mark.parametrize(
        ("coordinate", "expected"),
        [
            (Coordinate(480, TimeUnit.ticks), "480 ticks"),
            (Coordinate(12.3456789, TimeUnit.seconds), "12.3456789 seconds"),
        ],
    )
    def test_integer_and_float_coordinate_rendering(
        self, coordinate: Coordinate, expected: str
    ) -> None:
        anchor = _anchor("a", coordinate, "b", coordinate)
        assert repr(anchor) == f"AlignmentAnchor(a@{expected} <-> b@{expected})"


# endregion


# region MatchClaim: structure


class TestMatchClaim:
    """Direct construction: anchors, events, ids."""

    def test_instant_creation(self, basic_anchor: AlignmentAnchor) -> None:
        claim = _claim(basic_anchor)
        assert claim.is_interval is False
        assert claim.start_anchor is basic_anchor
        assert claim.end_anchor is None
        assert claim.timelines == ("score:1", "recording:1")
        assert claim.anchors == [basic_anchor]

    def test_interval_creation(self) -> None:
        start = _anchor("dgt_1", _number(0.0), "dgt_2", _number(0.0))
        end = _anchor("dgt_1", _number(975.0), "dgt_2", _number(866.0))
        claim = _claim(start, end_anchor=end)
        assert claim.is_interval is True
        assert claim.anchors == [start, end]
        assert claim.get_interval_for("dgt_1").end.value == 975.0
        assert claim.get_interval_for("dgt_2").end.value == 866.0

    def test_auto_generated_and_explicit_id(
        self, basic_anchor: AlignmentAnchor
    ) -> None:
        first, second = _claim(basic_anchor), _claim(basic_anchor)
        assert re.fullmatch(r"claim_[0-9a-z]{13}", first.id)
        assert re.fullmatch(r"claim_[0-9a-z]{13}", second.id)
        assert first.id != second.id
        assert _claim(basic_anchor, id="my_claim").id == "my_claim"

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("id_minted", True),
            ("is_explicit", False),
            ("source_claim_id", "claim_1"),
            ("source_coordinate", Coordinate(100, TimeUnit.quarters)),
        ],
    )
    def test_removed_properties_are_refused(
        self, basic_anchor: AlignmentAnchor, name: str, value: object
    ) -> None:
        assert name not in MatchClaim.model_fields
        with pytest.raises(
            ValidationError,
            match=re.escape(f"{name}\n  Extra inputs are not permitted"),
        ):
            _claim(basic_anchor, **{name: value})

    def test_mismatched_anchors_raises(self) -> None:
        start = _anchor("tl1", _number(0.0), "tl2", _number(0.0))
        end = _anchor("tl1", _number(100.0), "tl3", _number(100.0))
        with pytest.raises(
            ValueError,
            match=re.escape(
                "An anchor of a MatchClaim joins its timeline A and its timeline B "
                "in that order: the claim joins 'tl1' and 'tl2', the anchor 'tl1' "
                "and 'tl3'."
            ),
        ):
            _claim(start, end_anchor=end)

    def test_anchor_timeline_mismatch_raises(self) -> None:
        anchor = _anchor("tl1", _number(0.0), "tl2", _number(0.0))
        for timeline_a, timeline_b in (("tl1", "tl3"), ("tl2", "tl1")):
            with pytest.raises(
                ValueError,
                match=re.escape(
                    f"the claim joins '{timeline_a}' and '{timeline_b}', the "
                    f"anchor 'tl1' and 'tl2'."
                ),
            ):
                MatchClaim(
                    timeline_a_id=timeline_a,
                    timeline_b_id=timeline_b,
                    start_anchor=anchor,
                    event_a_id="a",
                    event_b_id="b",
                )

    def test_one_timeline_twice_refused(self) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "A MatchClaim connects two timelines; both sides are 'tl1'."
            ),
        ):
            MatchClaim(
                timeline_a_id="tl1",
                timeline_b_id="tl1",
                is_synchronous=False,
                event_a_id="a",
                event_b_id="b",
            )

    def test_synchronous_requires_anchor(self) -> None:
        with pytest.raises(ValueError, match="require a start_anchor"):
            MatchClaim(
                timeline_a_id="tl1", timeline_b_id="tl2", event_a_id="a", event_b_id="b"
            )

    def test_non_synchronous_rejects_anchor(self) -> None:
        anchor = _anchor("tl1", _number(0.0), "tl2", _number(0.0))
        with pytest.raises(ValueError, match="must not have anchors"):
            _claim(anchor, is_synchronous=False)

    def test_claim_names_its_events(self, basic_anchor: AlignmentAnchor) -> None:
        synchronous = {
            "timeline_a_id": "score:1",
            "timeline_b_id": "recording:1",
            "start_anchor": basic_anchor,
        }
        not_synchronous = {
            "timeline_a_id": "score:1",
            "timeline_b_id": "recording:1",
            "is_synchronous": False,
        }
        refused = (
            (
                {**synchronous, "event_a_id": "a1", "event_b_id": NOMATCH},
                "NOMATCH_SENTINEL stands only on a MatchClaim that is not "
                "synchronous: a synchronous MatchClaim names an event or "
                "CHUNK_SENTINEL on each side.",
            ),
            (
                {**not_synchronous, "event_a_id": CHUNK, "event_b_id": "b1"},
                "CHUNK_SENTINEL stands only on a synchronous MatchClaim, whose "
                "anchor holds its coordinate.",
            ),
            (
                {**not_synchronous, "event_a_id": NOMATCH, "event_b_id": NOMATCH},
                "A NOMATCH MatchClaim names an event on its other side.",
            ),
            (
                {
                    **not_synchronous,
                    "event_a_id": "a1",
                    "event_b_id": NOMATCH,
                    "event_b_name": "bar 1",
                },
                "event_b_name is 'bar 1', but side B is NOMATCH_SENTINEL: a "
                "NOMATCH side has no name.",
            ),
            (
                {**synchronous, "event_a_id": "a1"},
                "event_b_id\n  Field required",
            ),
        )
        for properties, message in refused:
            with pytest.raises(ValidationError, match=re.escape(message)):
                MatchClaim(**properties)

    def test_get_coordinate_for(self, basic_anchor: AlignmentAnchor) -> None:
        claim = _claim(basic_anchor)
        assert claim.get_coordinate_for("score:1") == IdCoordinate(
            100, TimeUnit.quarters, "score:1"
        )
        with pytest.raises(KeyError, match="other"):
            claim.get_coordinate_for("other")
        nomatch = MatchClaim.nomatch({"id": "x"}, SCORE, RECORDING)
        with pytest.raises(ValueError, match="no anchors"):
            nomatch.get_coordinate_for("score:1")

    def test_connects(self, basic_anchor: AlignmentAnchor) -> None:
        claim = _claim(basic_anchor)
        assert claim.connects("score:1") is True
        assert claim.connects("other") is False
        assert claim.connects_both("recording:1", "score:1") is True
        assert claim.connects_both("score:1", "other") is False

    def test_frozen_model(self, basic_anchor: AlignmentAnchor) -> None:
        with pytest.raises(ValidationError):
            _claim(basic_anchor).is_synchronous = False  # type: ignore


# endregion


# region MatchClaim: certainty and provenance


class TestCertaintyAndProvenance:
    def test_certainty_validation(self, basic_anchor: AlignmentAnchor) -> None:
        assert _claim(basic_anchor).certainty == 1.0
        for certainty in (-0.1, 1.1):
            with pytest.raises(
                ValidationError,
                match=re.escape(f"Certainty must be in [0, 1], got {certainty}"),
            ):
                _claim(basic_anchor, certainty=certainty)

    def test_certainty_boundaries(self, basic_anchor: AlignmentAnchor) -> None:
        assert _claim(basic_anchor, certainty=0.0).certainty == 0.0
        assert _claim(basic_anchor, certainty=1.0).certainty == 1.0
        assert _claim(basic_anchor, certainty=None).certainty is None

    def test_to_dict_literal_with_provenance(self) -> None:
        provenance = Provenance(
            stated_by=("https://orcid.org/0000-0001",),
            method="manual",
            beliefs=(Belief(actor="jo", belief="unlikely", date="2026-11-21"),),
        )
        claim = MatchClaim.from_events(
            {"id": "note_7", "name": "C♯4", "start": Fraction(7, 3)},
            SCORE,
            {"id": "onset_7", "start": 1.25},
            RECORDING,
            certainty=0.75,
            provenance=provenance,
            id="c7",
        )
        data = claim.to_dict()
        assert data == {
            "timeline_a_id": "score:1",
            "timeline_b_id": "recording:1",
            "start_anchor": {
                "coordinate_a": {
                    "unit": "quarters",
                    "number_type": "fraction",
                    "value": [7, 3],
                },
                "coordinate_b": {
                    "unit": "seconds",
                    "number_type": "float",
                    "value": 1.25,
                },
            },
            "end_anchor": None,
            "is_synchronous": True,
            "provenance": {
                "stated_by": ["https://orcid.org/0000-0001"],
                "method": "manual",
                "beliefs": [
                    {"actor": "jo", "belief": "unlikely", "date": "2026-11-21"}
                ],
            },
            "certainty": 0.75,
            "event_a_id": "note_7",
            "event_a_name": "C♯4",
            "event_b_id": "onset_7",
            "event_b_name": None,
            "id": "c7",
        }
        # The keys are the MatchClaim table's columns, in its order.
        assert list(data) == MatchClaimField.pa_schema.names
        json.dumps(data)
        restored = MatchClaim.from_dict(data)
        assert restored == claim
        assert restored.start_anchor == AlignmentAnchor(
            timeline_a_id="score:1",
            coordinate_a=Coordinate(Fraction(7, 3), TimeUnit.quarters),
            timeline_b_id="recording:1",
            coordinate_b=Coordinate(1.25, TimeUnit.seconds),
        )
        stated_twice = {
            **data,
            "start_anchor": {**data["start_anchor"], "timeline_a_id": "score:1"},
        }
        with pytest.raises(
            ValueError,
            match=re.escape(
                "A MatchClaim's start_anchor holds the keys ['coordinate_a', "
                "'coordinate_b'], not ['coordinate_a', 'coordinate_b', "
                "'timeline_a_id']: its timelines are the claim's timeline_a_id "
                "and timeline_b_id."
            ),
        ):
            MatchClaim.from_dict(stated_twice)


# endregion


# region MatchClaim: coordinates on the timelines


class TestFromEvents:
    """``from_events`` places every coordinate on its timeline's axis."""

    def test_from_events_places_numbers_on_the_axes(self) -> None:
        claim = MatchClaim.from_events(
            {"id": "n8", "name": "Note C4", "start": 8},
            SCORE,
            {"id": "t96", "start": 96},
            MIDI,
        )
        assert claim.start_anchor.coordinate_a == Coordinate(
            Fraction(8, 1), TimeUnit.quarters
        )
        assert type(claim.start_anchor.coordinate_b.value) is int
        assert claim.start_anchor.coordinate_b == Coordinate(96, TimeUnit.ticks)
        assert (claim.event_a_id, claim.event_a_name, claim.event_b_id) == (
            "n8",
            "Note C4",
            "t96",
        )
        cell = MatchClaim.from_events(
            {"id": "n1", "start": 1},
            SCORE,
            {"id": "o1", "start": {"value": 2.5, "numerator": 5, "denominator": 2}},
            RECORDING,
        )
        assert cell.start_anchor.coordinate_b.value == 2.5
        assert cell.start_anchor.coordinate_b.number_type is NumberType.float

    def test_from_events_interval_and_conceptual(self) -> None:
        interval = MatchClaim.from_events(
            {"id": "a", "start": 0, "end": Fraction(1, 2)},
            SCORE,
            {"id": "b", "start": 0, "end": 261},
            MIDI,
            end_coord_key="end",
        )
        assert interval.get_interval_for("score:1").end == Coordinate(
            Fraction(1, 2), TimeUnit.quarters
        )
        assert interval.get_interval_for("midi").end == Coordinate(261, TimeUnit.ticks)
        conceptual = MatchClaim.from_events(
            {"id": "a", "start": 0},
            SCORE,
            {"id": "b", "start": 0},
            MIDI,
            is_synchronous=False,
        )
        assert (conceptual.is_synchronous, conceptual.anchors) == (False, [])

    def test_from_events_converts_another_unit(self) -> None:
        wav = _wav()
        claim = MatchClaim.from_events(
            {"id": "n1", "start": Fraction(1, 2)},
            SCORE,
            {"id": "o1", "start": Coordinate(3.16952381, TimeUnit.seconds)},
            wav,
        )
        assert type(claim.start_anchor.coordinate_b.value) is int
        assert claim.start_anchor.coordinate_b == Coordinate(139776, TimeUnit.samples)
        as_dict = MatchClaim.from_events(
            {"id": "n1", "start": 1},
            SCORE,
            {
                "id": "o1",
                "start": {
                    "unit": "seconds",
                    "number_type": "float",
                    "value": 19.278367347,
                },
            },
            wav,
        )
        assert as_dict.start_anchor.coordinate_b == Coordinate(850176, TimeUnit.samples)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "start_anchor.coordinate_b names timeline 'other', not 'wav'"
            ),
        ):
            MatchClaim.from_events(
                {"id": "n1", "start": 1},
                SCORE,
                {"id": "o1", "start": IdCoordinate(3.0, TimeUnit.seconds, "other")},
                wav,
            )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "No C-Map available to convert coordinate from unit 'seconds' to "
                "'quarters' on timeline 'score:1'"
            ),
        ):
            MatchClaim.from_events(
                {"id": "n1", "start": Coordinate(3.0, TimeUnit.seconds)},
                SCORE,
                {"id": "o1", "start": 3},
                wav,
            )

    def test_from_events_refuses_another_number_type(self) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                _off_axis(
                    "start_anchor.coordinate_a",
                    "quarters",
                    "float",
                    "score:1",
                    "quarters (fraction)",
                )
            ),
        ):
            MatchClaim.from_events(
                {
                    "id": "n1",
                    "start": Coordinate(0.5, TimeUnit.quarters, number_type="float"),
                },
                SCORE,
                {"id": "o1", "start": 3},
                MIDI,
            )
        kept = MatchClaim.from_events(
            {
                "id": "n1",
                "start": Coordinate(0.5, TimeUnit.quarters, number_type="float"),
            },
            FLOAT_SCORE,
            {"id": "o1", "start": 3},
            MIDI,
        )
        assert kept.start_anchor.coordinate_a.number_type is NumberType.float

    def test_from_events_requires_event_ids(self) -> None:
        for event_a, event_b, side in (
            ({"start": 0}, {"id": "b", "start": 0}, "event_a"),
            ({"id": "a", "start": 0}, {"start": 0}, "event_b"),
        ):
            with pytest.raises(
                ValueError,
                match=re.escape(
                    f"{side} names no event id: a conceptual MatchClaim names two "
                    f"events. Give a position without an event of its own a "
                    f"Projection event first."
                ),
            ):
                MatchClaim.from_events(
                    event_a, SCORE, event_b, MIDI, is_synchronous=False
                )


class TestCheckOnTimelines:
    """The one check: every coordinate in its timeline's unit and number type."""

    def test_check_on_timelines(self) -> None:
        wav = _wav()
        seconds_on_samples = _claim(
            _anchor(
                "score:1",
                Coordinate(Fraction(1, 2), TimeUnit.quarters),
                "wav",
                Coordinate(3.16952381, TimeUnit.seconds),
            )
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                _off_axis(
                    "start_anchor.coordinate_b",
                    "seconds",
                    "float",
                    "wav",
                    "samples (int)",
                )
            ),
        ):
            seconds_on_samples.check_on_timelines(SCORE, wav)
        assert (
            seconds_on_samples.check_on_timelines(SCORE, ("seconds", "float"))
            is seconds_on_samples
        )

        fraction_on_float = _claim(
            _anchor(
                "float_score",
                Coordinate(Fraction(1, 3), TimeUnit.quarters),
                "midi",
                Coordinate(96, TimeUnit.ticks),
            )
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                _off_axis(
                    "start_anchor.coordinate_a",
                    "quarters",
                    "fraction",
                    "float_score",
                    "quarters (float)",
                )
            ),
        ):
            fraction_on_float.check_on_timelines(FLOAT_SCORE, MIDI)
        assert (
            fraction_on_float.check_on_timelines(("quarters", "fraction"), MIDI)
            is fraction_on_float
        )

        with pytest.raises(
            ValueError,
            match=re.escape("timeline_b is 'wav', but the MatchClaim names 'midi'"),
        ):
            fraction_on_float.check_on_timelines(FLOAT_SCORE, wav)

    def test_a_claim_without_anchors_holds_no_coordinate(self) -> None:
        nomatch = MatchClaim(
            timeline_a_id="recording:1",
            timeline_b_id="score:1",
            is_synchronous=False,
            event_a_id=NOMATCH,
            event_b_id="n9",
        )
        assert nomatch.check_on_timelines(RECORDING, SCORE) is nomatch
        assert nomatch.check_on_timelines(None, None) is nomatch


class TestNomatch:
    def test_nomatch_factory(self) -> None:
        claim = MatchClaim.nomatch(
            {"id": "orphan", "name": "E5", "start": Fraction(27, 2)},
            SCORE,
            "perf:Chopin_Ashkenazy",
            certainty=0.5,
        )
        assert (claim.is_synchronous, claim.anchors, claim.certainty) == (
            False,
            [],
            0.5,
        )
        assert (claim.timeline_a_id, claim.timeline_b_id) == (
            "score:1",
            "perf:Chopin_Ashkenazy",
        )
        assert (claim.event_a_id, claim.event_a_name, claim.event_b_id) == (
            "orphan",
            "E5",
            "NOMATCH_SENTINEL",
        )
        # A NOMATCH claim holds no coordinate: the event's own start is the
        # event's, read from its timeline, never repeated in the claim.
        assert claim.to_dict()["start_anchor"] is None
        assert "source_coordinate" not in claim.to_dict()
        with pytest.raises(ValueError, match=re.escape("event names no event id")):
            MatchClaim.nomatch({"start": 1}, SCORE, RECORDING)


# endregion


# region Serialisation and display


class TestSerialisationAndDisplay:
    def test_from_dict_roundtrips(self, basic_anchor: AlignmentAnchor) -> None:
        instant = _claim(basic_anchor)
        assert MatchClaim.from_dict(instant.to_dict()) == instant
        interval = _claim(
            _anchor("tl1", _number(0.0), "tl2", _number(0.0)),
            end_anchor=_anchor("tl1", _number(100.0), "tl2", _number(200.0)),
            provenance=Provenance(service="https://example.org/aligner"),
        )
        assert MatchClaim.from_dict(interval.to_dict()) == interval
        conceptual = MatchClaim(
            timeline_a_id="tl1",
            timeline_b_id="tl2",
            is_synchronous=False,
            event_a_id="a",
            event_b_id="b",
        )
        assert MatchClaim.from_dict(conceptual.to_dict()) == conceptual

    def test_repr_interval(self) -> None:
        claim = _claim(
            _anchor("dgt_1", _number(0.0), "dgt_2", _number(0.0)),
            end_anchor=_anchor("dgt_1", _number(975.0), "dgt_2", _number(866.0)),
        )
        assert repr(claim) == (
            "MatchClaim(interval: dgt_1[0 number-975 number] <-> "
            "dgt_2[0 number-866 number])"
        )

    def test_repr_non_synchronous(self) -> None:
        claim = MatchClaim(
            timeline_a_id="tl1",
            timeline_b_id="tl2",
            is_synchronous=False,
            event_a_id="e_orphan",
            event_b_id=NOMATCH,
        )
        assert repr(claim) == "MatchClaim(tl1 <-> tl2 [NOMATCH])"

    def test_repr_nomatch_names_the_timelines(self) -> None:
        claim = MatchClaim.nomatch(
            {"id": "orphan", "start": 188.8},
            ContinuousPhysicalTimeline(length=300.0, uid="score:clt_1"),
            "perf:Chopin_Ashkenazy",
        )
        assert repr(claim) == (
            "MatchClaim(score:clt_1 <-> perf:Chopin_Ashkenazy [NOMATCH])"
        )
        assert str(claim) == (
            "MatchClaim (NOMATCH)\n"
            "  Timeline A:  score:clt_1\n"
            "  Event A:     orphan\n"
            "  Timeline B:  perf:Chopin_Ashkenazy\n"
            "  Event B:     NOMATCH"
        )

    def test_nomatch_roundtrip(self) -> None:
        claim = MatchClaim.nomatch(
            {"id": "x", "start": Fraction(27, 2)}, SCORE, RECORDING
        )
        restored = MatchClaim.from_dict(claim.to_dict())
        assert restored == claim

    def test_repr_synchronous_instant_with_units(self) -> None:
        claim = _claim(_anchor("score:1", _number(100.0), "recording:1", _number(45.5)))
        assert repr(claim) == (
            "MatchClaim(instant: score:1@100 number <-> recording:1@45.5 number)"
        )

    def test_str_shows_certainty_provenance_and_beliefs(self) -> None:
        claim = MatchClaim.from_events(
            {"id": "e001", "name": "Intro", "start": 0, "end": Fraction(1, 2)},
            SCORE,
            {"id": "e042", "name": "Intro", "start": 0, "end": 261},
            MIDI,
            end_coord_key="end",
            certainty=0.9,
            provenance=Provenance(
                stated_by=("ana", "jo"),
                method="imported",
                beliefs=(
                    Belief(actor="ana", belief="true", date="2026-11-21"),
                    Belief(actor="jo", belief="false", date="2026-11-22"),
                ),
            ),
        )
        assert str(claim) == (
            "MatchClaim (synchronous, interval)\n"
            "  Timeline A:  score:1  [0 quarters -- 1/2 quarters]\n"
            '  Event A:     e001 "Intro"\n'
            "  Timeline B:  midi  [0 ticks -- 261 ticks]\n"
            '  Event B:     e042 "Intro"\n'
            "  Certainty:   0.9\n"
            "  Provenance:  stated_by=ana jo, method=imported\n"
            "  Beliefs:     ana true (2026-11-21); jo false (2026-11-22)"
        )
        assert claim._repr_html_() == (
            "<div style='font-family: monospace;'><strong>MatchClaim</strong> "
            "<span style='background: #e3f2fd; padding: 0 4px; border-radius: 3px; "
            "font-size: 0.8em;'>synchronous, interval</span><table "
            "style='border-collapse: collapse; margin-top: 4px;'><tbody><tr><td>"
            "Timeline A</td><td><strong>score:1</strong></td><td>[0 quarters &ndash; "
            "1/2 quarters]</td></tr><tr style='color: #555; font-size: 0.9em;'><td>"
            'Event A</td><td>e001</td><td>"Intro"</td></tr><tr><td>Timeline B</td>'
            "<td><strong>midi</strong></td><td>[0 ticks &ndash; 261 ticks]</td></tr>"
            "<tr style='color: #555; font-size: 0.9em;'><td>Event B</td><td>e042</td>"
            "<td>\"Intro\"</td></tr><tr style='color: #666;'><td>Certainty</td>"
            "<td colspan='2'>0.9</td></tr><tr style='color: #666;'><td>Provenance</td>"
            "<td colspan='2'>stated_by=ana jo, method=imported</td></tr><tr "
            "style='color: #666;'><td>Beliefs</td><td colspan='2'>ana true "
            "(2026-11-21); jo false (2026-11-22)</td></tr></tbody></table><div "
            "style='margin-top: 4px; color: #666; font-size: 0.85em;'>Try: <code>"
            "claim.get_matchstamp()</code></div></div>"
        )


class TestMatchstampQuery:
    def test_matchstamp_query_reads_the_claim_s_timeline(self) -> None:
        bundle = AlignmentBundle()
        score = ContinuousLogicalTimeline(length=16, uid="score")
        perf = ContinuousPhysicalTimeline(length=40.0, uid="perf")
        bundle.add_timeline(score, as_group="score")
        bundle.add_timeline(perf, as_group="perf")
        with pytest.raises(ValueError, match="An anchor of a MatchClaim joins"):
            MatchClaim(
                timeline_a_id="score",
                timeline_b_id="perf",
                start_anchor=_anchor(
                    "perf",
                    Coordinate(20.0, TimeUnit.seconds),
                    "score",
                    Coordinate(8, TimeUnit.quarters),
                ),
                event_a_id="note_8",
                event_b_id="onset_8",
            )
        claim = MatchClaim(
            timeline_a_id="score",
            timeline_b_id="perf",
            start_anchor=_anchor(
                "score",
                Coordinate(8, TimeUnit.quarters),
                "perf",
                Coordinate(20.0, TimeUnit.seconds),
            ),
            event_a_id="note_8",
            event_b_id="onset_8",
        ).check_on_timelines(score, perf)
        bundle.add_match_claims([claim])
        stamp = claim.get_matchstamp()
        assert stamp.get_coordinate_for("score", format="coordinate") == Coordinate(
            8, TimeUnit.quarters
        )
        assert stamp.get_coordinate_for("perf", format="coordinate") == Coordinate(
            20.0, TimeUnit.seconds
        )
        reduced = claim.get_matchstamp(from_graph=False)
        assert reduced.get_coordinate_for("score", format="coordinate") == Coordinate(
            8, TimeUnit.quarters
        )


# endregion


# region Integration


class TestClaimIntegration:
    def test_thoresen_segment_claims(self) -> None:
        """Five interval claims between the segments of two page images.

        DGT1 segments are five of 975 px; DGT2 segments 866, 867, 867, 864, 864.
        """
        dgt_1 = DiscreteLogicalTimeline(length=4875, uid="dgt_1")
        dgt_2 = DiscreteLogicalTimeline(length=4328, uid="dgt_2")
        lengths_1 = [975, 975, 975, 975, 975]
        lengths_2 = [866, 867, 867, 864, 864]
        provenance = Provenance(
            stated_by=("https://example.org/analyst",), method="manual"
        )
        claims = []
        start_1 = start_2 = 0
        for i in range(5):
            claims.append(
                MatchClaim.from_events(
                    {
                        "id": f"seg_1_{i}",
                        "start": start_1,
                        "end": start_1 + lengths_1[i],
                    },
                    dgt_1,
                    {
                        "id": f"seg_2_{i}",
                        "start": start_2,
                        "end": start_2 + lengths_2[i],
                    },
                    dgt_2,
                    end_coord_key="end",
                    provenance=provenance,
                    id=f"segment_{i + 1}",
                )
            )
            start_1 += lengths_1[i]
            start_2 += lengths_2[i]

        assert all(claim.is_interval for claim in claims)
        assert claims[0].get_interval_for("dgt_2").end.value == 866
        assert claims[4].get_interval_for("dgt_1").end.value == 4875
        assert claims[4].get_interval_for("dgt_2").end.value == 4328
        assert {claim.provenance for claim in claims} == {provenance}


# endregion
