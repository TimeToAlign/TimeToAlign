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
    SegmentLine,
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
        assert claim.chunk_sides == ("score", "audio")
        assert _timeline(bundle, "audio").n_events == 0
        assert claim.start_anchor.coordinate_b.value == 30
        assert claim.end_anchor.coordinate_b.value == 30
        (projected,) = bundle.create_match_claims(
            [("n1", "score", {"start": 30, "end": 30}, "audio")]
        )
        event = _timeline(bundle, "audio").get_event(projected.event_b_id)
        assert event["temporal_type"] == "interval"
        assert event["duration"] == {"value": 0.0, "numerator": 0, "denominator": 1}


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

    @pytest.mark.parametrize(
        "score_side",
        [{"start": 4, "end": 8}, "n1"],
        ids=["chunk", "projection"],
    )
    def test_an_unlocked_timeline_grows(self, score_side) -> None:
        bundle = _bundle()
        bundle.create_match_claims(
            [(score_side, "score", {"start": 44000, "end": 50000}, "audio")]
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

    def test_a_held_span_named_with_its_start_only_is_refused(self) -> None:
        bundle = self._with_m1()
        _refused(
            bundle,
            [({"id": "m1", "start": 0}, "score", {"start": 1.0}, "clock")],
            "Pair 0: the side on timeline 'score' names the event 'm1' it holds, which "
            "lies from 0 quarters to 4 quarters, and states at 0 quarters",
            "Give its own span or leave out 'start' and 'end'",
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
            [("n1", "score", {"start": 1.0, "name": "beat 1"}, "clock")]
        )
        projection = first.event_b_id
        assert _own_projections(bundle, "clock") == [(projection, 1)]
        assert first.event_b_name == "beat 1"
        (unnamed,) = bundle.create_match_claims(
            [("n1", "score", {"start": 1.0}, "clock")]
        )
        (by_id,) = bundle.create_match_claims([("n1", "score", projection, "clock")])
        (renamed,) = bundle.create_match_claims(
            [("n1", "score", {"start": 1.0, "name": "first beat"}, "clock")]
        )
        assert [(c.event_b_id, c.event_b_name) for c in (unnamed, by_id, renamed)] == [
            (projection, "beat 1"),
            (projection, "beat 1"),
            (projection, "beat 1"),
        ]
        assert _own_projections(bundle, "clock") == [(projection, 1)]

    def test_within_one_call(self) -> None:
        bundle = _bundle()
        claims = bundle.create_match_claims(
            [
                ("n1", "score", {"start": 1.0, "name": "beat 1"}, "clock"),
                ("n1", "score", {"start": 1.0}, "clock"),
            ]
        )
        ((projection, _),) = _own_projections(bundle, "clock")
        assert [(c.event_b_id, c.event_b_name) for c in claims] == [
            (projection, "beat 1"),
            (projection, "beat 1"),
        ]

    def test_a_note_keeps_its_own_name(self) -> None:
        bundle = _bundle()
        (claim,) = bundle.create_match_claims(
            [("n1", "score", {"start": 1.0}, "clock")]
        )
        assert claim.event_a_name is None


def _own_projections(
    bundle: AlignmentBundle, timeline_id: str
) -> list[tuple[str, Fraction]]:
    """The Projections a timeline holds itself (not its children), as (id, start)."""
    table = bundle.get_timeline(timeline_id).events.table
    return [
        (row["id"], Fraction(row["start"]["numerator"], row["start"]["denominator"]))
        for row in table.select(["id", "event_type", "start"]).to_pylist()
        if row["event_type"] == "Projection"
    ]


class TestPastWhereANestedTimelineReaches:
    """An unlocked child grows only to an end its parent contains."""

    @staticmethod
    def _nested() -> AlignmentBundle:
        root = ContinuousLogicalTimeline(length=10, uid="root")
        root.add_child(
            ContinuousLogicalTimeline(length=5, uid="kid"), offset=0, locked=False
        )
        other = ContinuousLogicalTimeline(length=20, uid="other")
        bundle = AlignmentBundle()
        bundle.add_timeline(root, as_group="root")
        bundle.add_timeline(other, as_group="other")
        return bundle

    @staticmethod
    def _snapshot(bundle: AlignmentBundle) -> tuple:
        timelines = [bundle.get_timeline(t) for t in ("root", "kid", "other")]
        return (
            bundle.match_claim_table.table,
            *((t.events.table, t.length) for t in timelines),
        )

    @staticmethod
    def _pairs(kid_start: int) -> list:
        return [
            ({"start": 1}, "other", {"start": 2}, "root"),
            ({"start": 3}, "other", {"start": kid_start}, "kid"),
        ]

    def test_a_child_growing_past_its_parent_is_refused_first(self) -> None:
        bundle = self._nested()
        before = self._snapshot(bundle)
        assert [t.num_rows for t, _ in before[1:]] == [1, 0, 0]
        with pytest.raises(ValueError) as caught:
            bundle.create_match_claims(self._pairs(12))
        message = str(caught.value)
        for part in (
            "Pair 1: the side on timeline 'kid' reaches 12 quarters, past the end of "
            "timeline 'kid' at 5 quarters, and the timeline cannot grow to hold it: "
            "Timeline 'kid' cannot end at 12 on its parent 'root', which ends at 10.",
            "a parent never grows with its child",
            "Correct the span; or, if the position lies beyond the timeline, state it "
            "on its parent 'root'.",
        ):
            assert part in message, (part, message)
        after = self._snapshot(bundle)
        assert after[0].equals(before[0])
        assert len(after[0]) == 0
        for (table_after, length_after), (table_before, length_before) in zip(
            after[1:], before[1:], strict=True
        ):
            assert table_after.equals(table_before)
            assert length_after == length_before

    def test_a_child_within_its_length_is_accepted(self) -> None:
        bundle = self._nested()
        claims = bundle.create_match_claims(self._pairs(4))
        assert [
            (c.timeline_b_id, c.start_anchor.coordinate_b.value) for c in claims
        ] == [
            ("root", Fraction(2)),
            ("kid", Fraction(4)),
        ]
        assert bundle.get_timeline("kid").length == Coordinate(5, "quarters")
        assert [c.chunk_sides for c in claims] == [("other", "root"), ("other", "kid")]
        assert {t: _own_projections(bundle, t) for t in ("root", "kid", "other")} == {
            "root": [],
            "kid": [],
            "other": [],
        }


class TestPastWhereASegmentReaches:
    """A segment's end is its successor's offset; a MatchClaim never moves it."""

    @staticmethod
    def _line() -> AlignmentBundle:
        line = SegmentLine[ContinuousLogicalTimeline](length=0, uid="line")
        line.append_segment(ContinuousLogicalTimeline(length=4, uid="a"))
        line.append_segment(ContinuousLogicalTimeline(length=None, uid="b"))
        line.length = 9
        other = ContinuousLogicalTimeline(length=20, uid="other")
        bundle = AlignmentBundle()
        bundle.add_timeline(line, as_group="line")
        bundle.add_timeline(other, as_group="other")
        return bundle

    @staticmethod
    def _pairs(segment: str, start: int) -> list:
        return [
            ({"start": 1}, "other", {"start": 2}, "line"),
            ({"start": 3}, "other", {"start": start}, segment),
        ]

    @staticmethod
    def _snapshot(bundle: AlignmentBundle) -> tuple:
        timelines = [bundle.get_timeline(t) for t in ("line", "a", "b", "other")]
        return (
            bundle.match_claim_table.table,
            *((t.events.table, t.length) for t in timelines),
        )

    @pytest.mark.parametrize(
        ("segment", "start", "parts"),
        [
            (
                "b",
                6,
                (
                    "Pair 1: the side on timeline 'b' reaches 6 quarters, past the end of "
                    "timeline 'b' at 5 quarters, where its parent 'line' ends it while its "
                    "length is unknown; a MatchClaim never moves that end.",
                    "state it on its parent 'line'.",
                ),
            ),
            (
                "a",
                5,
                (
                    "Pair 1: the side on timeline 'a' reaches 5 quarters, past the end of "
                    "timeline 'a' at 4 quarters, and the timeline cannot grow to hold it: "
                    "SegmentLine requires contiguous segments. Segment 'a' cannot end at 5: "
                    "its successor 'b' starts at 4. A MatchClaim",
                ),
            ),
        ],
        ids=["unknown-length", "known-length"],
    )
    def test_past_the_segment_is_refused_first(
        self, segment: str, start: int, parts: tuple[str, ...]
    ) -> None:
        bundle = self._line()
        before = self._snapshot(bundle)
        with pytest.raises(ValueError) as caught:
            bundle.create_match_claims(self._pairs(segment, start))
        for part in parts:
            assert part in str(caught.value), (part, str(caught.value))
        after = self._snapshot(bundle)
        assert after[0].equals(before[0])
        for (table_after, length_after), (table_before, length_before) in zip(
            after[1:], before[1:], strict=True
        ):
            assert table_after.equals(table_before)
            assert length_after == length_before

    @pytest.mark.parametrize(
        ("segment", "start", "length"),
        [("b", 5, None), ("a", 4, Coordinate(4, "quarters"))],
        ids=["unknown-length", "known-length"],
    )
    def test_up_to_the_segment_end_is_accepted(
        self, segment: str, start: int, length: Coordinate | None
    ) -> None:
        bundle = self._line()
        claims = bundle.create_match_claims(self._pairs(segment, start))
        assert [c.start_anchor.coordinate_b.value for c in claims] == [
            Fraction(2),
            Fraction(start),
        ]
        assert bundle.get_timeline(segment).length == length
        assert [c.chunk_sides for c in claims] == [
            ("other", "line"),
            ("other", segment),
        ]
        assert _own_projections(bundle, segment) == []


NAN = float("nan")
INF = float("inf")


class TestFloatATimelineCannotStore:
    """Rule 6: refused with the number cell's message, before anything is added."""

    @pytest.mark.parametrize(
        ("side", "message"),
        [
            ({"start": NAN}, "cannot store non-finite value nan as a number cell"),
            ({"start": INF}, "cannot store non-finite value inf as a number cell"),
            ({"start": -INF}, "cannot store non-finite value -inf as a number cell"),
            (
                {"start": 1.0, "end": NAN},
                "cannot store non-finite value nan as a number cell",
            ),
            (
                {"start": 1.0, "end": INF},
                "cannot store non-finite value inf as a number cell",
            ),
            (
                {"start": 2.0**63},
                "value 9.223372036854776e+18 is too large to mirror as an int64 ratio",
            ),
            (
                {"start": 1e18, "end": 1e19},
                "value 1e+19 is too large to mirror as an int64 ratio",
            ),
            (
                {"start": NAN, "end": 1e300},
                "cannot store non-finite value nan as a number cell",
            ),
            (
                {"start": 1e300, "end": NAN},
                "value 1e+300 is too large to mirror as an int64 ratio",
            ),
        ],
    )
    def test_refused_before_the_score_side_is_added(
        self, side: dict, message: str
    ) -> None:
        bundle = _bundle()
        refused = _refused(bundle, [("n1", "score", side, "clock")], message)
        assert refused == message

    def test_refused_in_the_second_pair(self) -> None:
        bundle = _bundle()
        _refused(
            bundle,
            [
                ("n1", "score", {"start": 1.0}, "clock"),
                ("n1", "score", {"start": NAN}, "clock"),
            ],
            "cannot store non-finite value nan as a number cell",
        )

    def test_a_negative_infinite_end_runs_backwards(self) -> None:
        bundle = _bundle()
        _refused(
            bundle,
            [("n1", "score", {"start": 1.0, "end": -INF}, "clock")],
            "runs backwards: its end -inf seconds lies before its start 1.0 seconds",
        )

    @pytest.mark.parametrize("side", [{"start": 9e18}, {"start": 1.0, "end": 2.0}])
    def test_a_storable_float_is_accepted(self, side: dict) -> None:
        bundle = _bundle()
        claims = bundle.create_match_claims([("n1", "score", side, "clock")])
        assert len(claims) == 1
        assert len(bundle.match_claim_table) == 1
        assert _timeline(bundle, "score").n_events == 1
        assert _timeline(bundle, "clock").n_events == 1
