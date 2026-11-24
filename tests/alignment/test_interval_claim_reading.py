"""Interval claims read exactly, from the queried timeline or a group sibling."""

from __future__ import annotations

from fractions import Fraction

from timetoalign.alignment import MatchClaim
from timetoalign.alignment.bundle import AlignmentBundle
from timetoalign.alignment.graph import MatchStamp
from timetoalign.core import Coordinate, NumberType, TimeUnit
from timetoalign.timelines import Timeline

from .helpers import match_claim


def _timeline(
    timeline_id: str, length: int, unit: TimeUnit, number_type: NumberType
) -> Timeline:
    """Create one synthetic timeline."""
    return Timeline(length=length, uid=timeline_id, unit=unit, number_type=number_type)


def _on(unit: TimeUnit, number_type: NumberType):
    """A coordinate factory for one axis."""

    def coordinate(value: int | float | Fraction) -> Coordinate:
        return Coordinate(value, unit, number_type=number_type)

    return coordinate


SAMPLES = _on(TimeUnit.samples, NumberType.int)
QUARTERS = _on(TimeUnit.quarters, NumberType.fraction)
PIXELS = _on(TimeUnit.pixels, NumberType.int)
FRAMES = _on(TimeUnit.frames, NumberType.int)
SECONDS = _on(TimeUnit.seconds, NumberType.float)


def _interval_claim(
    timeline_a: str,
    span_a: tuple[Coordinate, Coordinate],
    timeline_b: str,
    span_b: tuple[Coordinate, Coordinate],
) -> MatchClaim:
    """Create one synchronous interval claim between two spans."""
    return match_claim(
        timeline_a,
        span_a[0],
        timeline_b,
        span_b[0],
        end_a=span_a[1],
        end_b=span_b[1],
    )


def _bundle(*, video: bool = False) -> AlignmentBundle:
    """The recording group, the score and the scan, with claims 1 and 2.

    ``track_hi`` is ``track`` at twice the rate. With *video*, claim 3 joins
    ``track_hi`` to a video.
    """
    bundle = AlignmentBundle(id="interval-reading")
    bundle.add_timeline(_timeline("track", 2000, TimeUnit.samples, NumberType.int))
    bundle.add_timeline(
        _timeline("track_hi", 4000, TimeUnit.samples, NumberType.int),
        grouped_with="track",
    )
    bundle.add_timeline(_timeline("score", 200, TimeUnit.quarters, NumberType.fraction))
    bundle.add_timeline(_timeline("scan", 2000, TimeUnit.pixels, NumberType.int))
    claims = [
        _interval_claim(
            "track",
            (SAMPLES(1000), SAMPLES(1087)),
            "score",
            (QUARTERS(56), QUARTERS(63)),
        ),
        _interval_claim(
            "track", (SAMPLES(1037), SAMPLES(1047)), "scan", (PIXELS(0), PIXELS(45))
        ),
    ]
    if video:
        bundle.add_timeline(_timeline("video", 2000, TimeUnit.frames, NumberType.int))
        claims.append(
            _interval_claim(
                "track_hi",
                (SAMPLES(2000), SAMPLES(2174)),
                "video",
                (FRAMES(100), FRAMES(274)),
            )
        )
    bundle.add_match_claims(claims)
    return bundle


def _claim_ids(stamp: MatchStamp) -> list[str | None]:
    """The ids of the interval claims a stamp lists, in its order."""
    return [claim.id for claim in stamp.interval_claims]


EXPECTED_AT_1044 = {
    "score": QUARTERS(Fraction(5180, 87)),
    "scan": PIXELS(32),
    "track": SAMPLES(1044),
    "track_hi": SAMPLES(2088),
}


class TestExactInteriorMapping:
    """An interior position computed from exact inputs comes back exact."""

    def test_fraction_axis_takes_the_exact_ratio(self) -> None:
        stamp = _bundle().get_matchstamp_at(1044, "track")

        score = stamp.coordinates["score"]
        assert score == QUARTERS(Fraction(5180, 87))
        assert type(score.value) is Fraction
        assert score.value == Fraction(5180, 87)
        assert stamp.get_coordinate_for("score", format="fraction") == Fraction(
            5180, 87
        )

    def test_int_axis_rounds_the_exact_ratio_once(self) -> None:
        # 7/10 of 45 pixels is 63/2 exactly; as doubles it is
        # 31.499999999999996, which would round to 31.
        stamp = _bundle().get_matchstamp_at(1044, "track")

        scan = stamp.coordinates["scan"]
        assert scan == PIXELS(32)
        assert type(scan.value) is int
        assert scan.value == 32

    def test_float_axis_stays_float(self) -> None:
        bundle = AlignmentBundle(id="float-reading")
        bundle.add_timeline(_timeline("mix", 200, TimeUnit.seconds, NumberType.float))
        bundle.add_timeline(_timeline("take", 200, TimeUnit.seconds, NumberType.float))
        bundle.add_match_claims(
            [
                _interval_claim(
                    "mix",
                    (SECONDS(0.0), SECONDS(87.0)),
                    "take",
                    (SECONDS(56.0), SECONDS(63.0)),
                )
            ]
        )

        stamp = bundle.get_matchstamp_at(44.0, "mix")

        take = stamp.coordinates["take"]
        assert type(take.value) is float
        assert take.value == 59.54022988505747
        assert take == SECONDS(59.54022988505747)


class TestReadingFromAGroupSibling:
    """A question on a group sibling reads the claims of the claim-carrying member."""

    def test_claim_carrying_member(self) -> None:
        stamp = _bundle().get_matchstamp_at(1044, "track")

        assert stamp.coordinates == EXPECTED_AT_1044
        assert list(stamp.coordinates) == ["track", "track_hi", "score", "scan"]
        assert _claim_ids(stamp) == ["claim_1", "claim_2"]
        assert stamp.stated_pairs == []
        assert stamp.inferred_pairs == [
            ("track", "score"),
            ("track", "scan"),
            ("track", "track_hi"),
        ]
        assert stamp.is_interpolated is True

    def test_sibling_reads_the_same_claims(self) -> None:
        bundle = _bundle()
        carrier = bundle.get_matchstamp_at(1044, "track")
        stamp = bundle.get_matchstamp_at(2088, "track_hi")

        assert stamp.coordinates == EXPECTED_AT_1044
        assert stamp.coordinates == carrier.coordinates
        assert list(stamp.coordinates) == ["track_hi", "track", "score", "scan"]
        assert _claim_ids(stamp) == ["claim_1", "claim_2"]
        assert set(_claim_ids(stamp)) == set(_claim_ids(carrier))
        assert stamp.stated_pairs == []
        assert stamp.inferred_pairs == [
            ("track", "score"),
            ("track", "scan"),
            ("track_hi", "track"),
        ]
        assert stamp.is_interpolated is True

    def test_sibling_display_rows_agree_with_the_stamp(self) -> None:
        bundle = _bundle()
        carrier = bundle.get_matchstamp_at(1044, "track")
        stamp = bundle.get_matchstamp_at(2088, "track_hi")

        rows = [
            (claim.id, timeline_id, coordinate)
            for claim, timeline_id, coordinate in stamp._interval_candidate_rows()
        ]
        assert rows == [
            ("claim_1", "score", QUARTERS(Fraction(5180, 87))),
            ("claim_2", "scan", PIXELS(32)),
        ]
        assert rows == [
            (claim.id, timeline_id, coordinate)
            for claim, timeline_id, coordinate in carrier._interval_candidate_rows()
        ]
        assert stamp._ambiguous_interval_candidates() == {}
        assert repr(stamp) == (
            "MatchStamp(track_hi=2088 samples, track=1044 samples, "
            "score=5180/87 quarters, scan=32 pixels, "
            "claim 1 track<->score: score=5180/87 quarters, "
            "claim 2 track<->scan: scan=32 pixels)"
        )

    def test_claims_on_both_members_queried_first(self) -> None:
        bundle = _bundle(video=True)
        carrier = bundle.get_matchstamp_at(1044, "track")
        stamp = bundle.get_matchstamp_at(2088, "track_hi")

        expected = {**EXPECTED_AT_1044, "video": FRAMES(188)}
        assert carrier.coordinates == expected
        assert stamp.coordinates == expected
        assert _claim_ids(carrier) == ["claim_1", "claim_2", "claim_3"]
        assert _claim_ids(stamp) == ["claim_3", "claim_1", "claim_2"]

    def test_sibling_anchor_is_stated_but_query_stays_interpolated(self) -> None:
        bundle = _bundle()
        carrier = bundle.get_matchstamp_at(1000, "track")
        stamp = bundle.get_matchstamp_at(2000, "track_hi")

        expected = {
            "score": QUARTERS(56),
            "track": SAMPLES(1000),
            "track_hi": SAMPLES(2000),
        }
        assert carrier.coordinates == expected
        assert stamp.coordinates == expected
        assert _claim_ids(stamp) == ["claim_1"]
        assert carrier.stated_pairs == [("track", "score")]
        assert stamp.stated_pairs == [("track", "score")]
        assert stamp.inferred_pairs == [("track_hi", "track")]
        assert carrier.is_interpolated is False
        assert stamp.is_interpolated is True

    def test_plural_getter_on_the_sibling(self) -> None:
        bundle = _bundle()

        stamps = bundle.get_matchstamps_at([2088, 2000], "track_hi")

        assert [stamp.coordinates for stamp in stamps] == [
            EXPECTED_AT_1044,
            {
                "score": QUARTERS(56),
                "track": SAMPLES(1000),
                "track_hi": SAMPLES(2000),
            },
        ]
        assert [_claim_ids(stamp) for stamp in stamps] == [
            ["claim_1", "claim_2"],
            ["claim_1"],
        ]

    def test_table_on_the_sibling(self) -> None:
        table = _bundle().get_matchstamp_table([2088, 2000], "track_hi")

        assert table.column_names == ["scan", "score", "track", "track_hi"]
        assert table.to_pylist() == [
            {
                "scan": {"value": 32.0, "numerator": 32, "denominator": 1},
                "score": {
                    "value": 5180 / 87,
                    "numerator": 5180,
                    "denominator": 87,
                },
                "track": {"value": 1044.0, "numerator": 1044, "denominator": 1},
                "track_hi": {"value": 2088.0, "numerator": 2088, "denominator": 1},
            },
            {
                "scan": None,
                "score": {"value": 56.0, "numerator": 56, "denominator": 1},
                "track": {"value": 1000.0, "numerator": 1000, "denominator": 1},
                "track_hi": {"value": 2000.0, "numerator": 2000, "denominator": 1},
            },
        ]
