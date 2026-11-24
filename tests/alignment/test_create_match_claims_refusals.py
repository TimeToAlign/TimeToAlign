"""What ``AlignmentBundle.create_match_claims`` refuses, and what it accepts beside.

Validation logic: ``tests/alignment/README.md``, section
"``create_match_claims`` refusals".
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign.alignment import AlignmentBundle
from timetoalign.core import Coordinate, NumberType, TimeUnit
from timetoalign.maps import SamplesToSeconds
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteLogicalTimeline,
    DiscretePhysicalTimeline,
)


def _bundle(
    *, audio_locked: bool = False, clock_locked: bool = False
) -> AlignmentBundle:
    score = ContinuousLogicalTimeline(length=16, uid="score")
    score.add_events([{"id": "n1", "event_type": "Note", "instant": 4}])
    audio = DiscretePhysicalTimeline(
        length=44100, uid="audio", unit="samples", locked=audio_locked
    )
    audio.add_conversion_map(SamplesToSeconds(sample_rate=44100))
    clock = ContinuousPhysicalTimeline(length=230.0, uid="clock", locked=clock_locked)
    ticks = DiscreteLogicalTimeline(length=1000, uid="ticks", unit="ticks")
    bundle = AlignmentBundle()
    for timeline in (score, audio, clock, ticks):
        bundle.add_timeline(timeline, as_group=timeline.id)
    return bundle


def _timeline(bundle: AlignmentBundle, timeline_id: str):
    return bundle.timelines[timeline_id]


def _state(bundle: AlignmentBundle) -> tuple[int, ...]:
    """Claims held, then each timeline's events and length, to prove a refusal changed nothing."""
    return (
        len(bundle.match_claim_table),
        *(bundle.timelines[t].n_events for t in ("score", "audio", "clock")),
    )


def _refused(bundle: AlignmentBundle, pairs: list, *parts: str) -> str:
    """Assert that *pairs* is refused with a message holding every part, nothing changed."""
    before = _state(bundle)
    with pytest.raises(ValueError) as caught:
        bundle.create_match_claims(pairs)
    message = str(caught.value)
    for part in parts:
        assert part in message, (part, message)
    assert _state(bundle) == before
    return message


class TestFloatOnSamples:
    def test_a_float_is_refused(self) -> None:
        bundle = _bundle()
        _refused(
            bundle,
            [({"start": 1}, "score", {"start": 10.5}, "audio")],
            "Pair 0: the side on timeline 'audio', its start is the float 10.5",
            "counts samples",
            "a sample count given in the wrong number type (give it as an int)",
            "a position in seconds that needs converting",
            "give Coordinate(10.5, 'seconds')",
            "Give positions as Coordinates",
        )

    def test_a_whole_float_is_refused_too(self) -> None:
        bundle = _bundle()
        _refused(
            bundle,
            [({"start": 1}, "score", {"start": 0, "end": 22050.0}, "audio")],
            "its end is the float 22050.0",
        )

    def test_an_int_is_accepted(self) -> None:
        bundle = _bundle()
        (claim,) = bundle.create_match_claims(
            [({"start": 1}, "score", {"start": 22050}, "audio")]
        )
        coordinate = claim.start_anchor.coordinate_b
        assert (coordinate.value, coordinate.unit, coordinate.number_type) == (
            22050,
            TimeUnit.samples,
            NumberType.int,
        )
        assert type(coordinate.value) is int

    def test_a_coordinate_in_seconds_is_converted(self) -> None:
        bundle = _bundle()
        (claim,) = bundle.create_match_claims(
            [({"start": 1}, "score", {"start": Coordinate(0.5, "seconds")}, "audio")]
        )
        coordinate = claim.start_anchor.coordinate_b
        assert (coordinate.value, coordinate.unit, coordinate.number_type) == (
            22050,
            TimeUnit.samples,
            NumberType.int,
        )
        assert type(coordinate.value) is int

    def test_a_float_on_ticks_is_not_refused(self) -> None:
        bundle = _bundle()
        (claim,) = bundle.create_match_claims(
            [({"start": 1}, "score", {"start": 10.5}, "ticks")]
        )
        assert claim.start_anchor.coordinate_b.value == 10


class TestBackwardsSpan:
    def test_refused_on_seconds(self) -> None:
        bundle = _bundle()
        _refused(
            bundle,
            [
                (
                    {"start": 4, "end": 8},
                    "score",
                    {"start": 226.08, "end": 201.18},
                    "clock",
                )
            ],
            "Pair 0: the side on timeline 'clock' runs backwards",
            "its end 201.18 seconds lies before its start 226.08 seconds",
            "swap the two if they were given in the wrong order, or correct the one",
        )

    def test_refused_on_quarters(self) -> None:
        bundle = _bundle()
        _refused(
            bundle,
            [({"start": 8, "end": 4}, "score", {"start": 1.0, "end": 2.0}, "clock")],
            "the side on timeline 'score' runs backwards",
            "its end 4 quarters lies before its start 8 quarters",
        )

    def test_a_forward_span_is_accepted(self) -> None:
        bundle = _bundle()
        (claim,) = bundle.create_match_claims(
            [
                (
                    {"start": 4, "end": 8},
                    "score",
                    {"start": 201.18, "end": 226.08},
                    "clock",
                )
            ]
        )
        assert claim.start_anchor.coordinate_b.value == 201.18
        assert claim.end_anchor.coordinate_b.value == 226.08

    def test_an_empty_span_is_accepted(self) -> None:
        bundle = _bundle()
        (claim,) = bundle.create_match_claims(
            [({"start": 2, "end": 2}, "score", {"start": 30, "end": 30}, "audio")]
        )
        event = _timeline(bundle, "audio").get_event(claim.event_b_id)
        assert event["temporal_type"] == "interval"
        assert event["duration"] == {"value": 0.0, "numerator": 0, "denominator": 1}
        assert claim.start_anchor.coordinate_b.value == 30
        assert claim.end_anchor.coordinate_b.value == 30


class TestPastTheEndOfALockedTimeline:
    def test_a_span_is_refused(self) -> None:
        bundle = _bundle(audio_locked=True)
        _refused(
            bundle,
            [
                (
                    {"start": 4, "end": 8},
                    "score",
                    {"start": 44000, "end": 50000},
                    "audio",
                )
            ],
            "Pair 0: the side on timeline 'audio' reaches 50000 samples, past the end "
            "of timeline 'audio' at 44100 samples",
            "The timeline is locked",
            "Correct the span",
            "make the timeline with the source's length",
        )
        assert _timeline(bundle, "audio").length == Coordinate(44100, "samples")

    def test_an_instant_is_refused(self) -> None:
        bundle = _bundle(audio_locked=True)
        _refused(
            bundle,
            [({"start": 4}, "score", {"start": 50000}, "audio")],
            "reaches 50000 samples, past the end",
        )

    def test_a_coordinate_is_named_as_given(self) -> None:
        bundle = _bundle(audio_locked=True)
        _refused(
            bundle,
            [({"start": 4}, "score", {"start": Coordinate(2.0, "seconds")}, "audio")],
            "reaches 2 seconds (88200 samples on the timeline), past the end",
        )

    def test_a_span_ending_at_the_end_is_accepted(self) -> None:
        bundle = _bundle(audio_locked=True)
        (claim,) = bundle.create_match_claims(
            [({"start": 4, "end": 8}, "score", {"start": 44000, "end": 44100}, "audio")]
        )
        assert claim.end_anchor.coordinate_b.value == 44100
        assert _timeline(bundle, "audio").length == Coordinate(44100, "samples")

    def test_an_unlocked_timeline_grows(self) -> None:
        bundle = _bundle()
        bundle.create_match_claims(
            [({"start": 4, "end": 8}, "score", {"start": 44000, "end": 50000}, "audio")]
        )
        assert _timeline(bundle, "audio").length == Coordinate(50000, "samples")

    def test_no_timeline_is_changed_before_the_refusal(self) -> None:
        bundle = _bundle(audio_locked=True)
        _refused(
            bundle,
            [
                ({"start": 1}, "score", {"start": 100}, "audio"),
                ({"start": 2}, "score", {"start": 50000}, "audio"),
            ],
            "Pair 1: the side on timeline 'audio' reaches 50000 samples",
        )


class TestHeldIdWithAnotherSpan:
    @staticmethod
    def _with_m1() -> AlignmentBundle:
        bundle = _bundle()
        bundle.create_match_claims(
            [
                (
                    {"id": "m1", "start": 0, "end": 4},
                    "score",
                    {"start": 0.0, "end": 2.0},
                    "clock",
                )
            ]
        )
        return bundle

    def test_another_span_is_refused(self) -> None:
        bundle = self._with_m1()
        _refused(
            bundle,
            [
                (
                    {"id": "m1", "start": 2, "end": 6},
                    "score",
                    {"start": 1.0, "end": 3.0},
                    "clock",
                )
            ],
            "Pair 0: the side on timeline 'score' names the event 'm1' it holds, which "
            "lies from 0 quarters to 4 quarters, and states from 2 quarters to 6 quarters",
            "Give its own span or leave out 'start' and 'end'",
            "leave out the ID (or give a new one)",
        )

    def test_a_note_named_with_another_position_is_refused(self) -> None:
        bundle = _bundle()
        _refused(
            bundle,
            [({"id": "n1", "start": 5}, "score", {"start": 1.0}, "clock")],
            "names the event 'n1' it holds, which lies at 4 quarters, and states at 5 quarters",
        )

    @pytest.mark.parametrize(
        "side", [{"id": "m1", "start": 0, "end": 4}, {"id": "m1"}], ids=["same", "none"]
    )
    def test_the_same_span_or_none_is_accepted(self, side: dict) -> None:
        bundle = self._with_m1()
        (claim,) = bundle.create_match_claims(
            [(side, "score", {"start": 0.0, "end": 2.0}, "clock")]
        )
        assert claim.event_a_id == "m1"
        assert claim.start_anchor.coordinate_a.value == Fraction(0)
        assert claim.end_anchor.coordinate_a.value == Fraction(4)
        ids = _timeline(bundle, "score").get_events().table.column("id").to_pylist()
        assert ids.count("m1") == 1


class TestNameKept:
    def test_later_claims_show_the_first_name(self) -> None:
        bundle = _bundle()
        (first,) = bundle.create_match_claims(
            [({"start": 1, "name": "bar 1"}, "score", {"start": 1.0}, "clock")]
        )
        assert (first.event_a_id, first.event_a_name) == ("projection_1", "bar 1")
        (unnamed,) = bundle.create_match_claims(
            [({"start": 1}, "score", {"start": 2.0}, "clock")]
        )
        (by_id,) = bundle.create_match_claims(
            [("projection_1", "score", {"start": 3.0}, "clock")]
        )
        (renamed,) = bundle.create_match_claims(
            [({"start": 1, "name": "first bar"}, "score", {"start": 4.0}, "clock")]
        )
        assert [(c.event_a_id, c.event_a_name) for c in (unnamed, by_id, renamed)] == [
            ("projection_1", "bar 1"),
            ("projection_1", "bar 1"),
            ("projection_1", "bar 1"),
        ]

    def test_within_one_call(self) -> None:
        bundle = _bundle()
        claims = bundle.create_match_claims(
            [
                ({"start": 1, "name": "bar 1"}, "score", {"start": 1.0}, "clock"),
                ({"start": 1}, "score", {"start": 2.0}, "clock"),
            ]
        )
        assert [(c.event_a_id, c.event_a_name) for c in claims] == [
            ("projection_1", "bar 1"),
            ("projection_1", "bar 1"),
        ]

    def test_a_note_keeps_its_own_name(self) -> None:
        bundle = _bundle()
        (claim,) = bundle.create_match_claims(
            [("n1", "score", {"start": 1.0}, "clock")]
        )
        assert claim.event_a_name is None
