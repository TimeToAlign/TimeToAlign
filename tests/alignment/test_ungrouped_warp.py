"""An ungrouped timeline warps in matchstamp queries as a group of its own.

The scenario and every expected value are documented in ``README.md`` under
"Ungrouped Timelines Warp".
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign.alignment.bundle import AlignmentBundle
from timetoalign.alignment.claims import AlignmentAnchor, MatchClaim
from timetoalign.core import Coordinate, TimeUnit
from timetoalign.maps import SamplesToSeconds
from timetoalign.timelines import Timeline

# region Fixture

SCORE = "score"
RECORDING = "recording"
OTHER = "other"

# score quarters -> recording seconds; every seconds x 44100 is whole.
ANCHORS = ((1, 0.5), (3, 1.5), (5, 2.0), (9, 4.0))

# score quarters -> recording samples, interpolated by hand (README table).
BETWEEN = (
    (Fraction(2), 44100),
    (Fraction(7, 3), 51450),
    (Fraction(4), 77175),
    (Fraction(7), 132300),
)

LAYOUTS = ("ungrouped", "score_grouped", "both_grouped")


def _quarters(value: int | Fraction) -> Coordinate:
    """One coordinate on the score's exact quarter axis."""
    return Coordinate(Fraction(value), TimeUnit.quarters)


def _samples(value: int) -> Coordinate:
    """One coordinate on the recording's sample axis."""
    return Coordinate(value, TimeUnit.samples)


def _bundle(layout: str = "ungrouped", *, claims: bool = True) -> AlignmentBundle:
    """Build the score/recording/other bundle in one grouping layout.

    Args:
        layout: ``"ungrouped"`` (no groups), ``"score_grouped"`` (the score in
            a group of its own) or ``"both_grouped"`` (score and recording each
            in a group of its own; ``other`` stays ungrouped in every layout).
        claims: Whether to add the four score -> recording claims.
    """
    score = Timeline(length=16, unit=TimeUnit.quarters, uid=SCORE)
    recording = Timeline(length=441000, unit=TimeUnit.samples, uid=RECORDING)
    recording.add_conversion_map(SamplesToSeconds(sample_rate=44100))
    other = Timeline(length=10, unit=TimeUnit.seconds, uid=OTHER)

    bundle = AlignmentBundle(id="ungrouped-warp")
    if layout == "ungrouped":
        bundle.add_timeline(score)
        bundle.add_timeline(recording)
    elif layout == "score_grouped":
        bundle.add_timeline(score, as_group="score_group")
        bundle.add_timeline(recording)
    else:
        assert layout == "both_grouped"
        bundle.add_timeline(score, as_group="score_group")
        bundle.add_timeline(recording, as_group="recording_group")
    bundle.add_timeline(other)

    if claims:
        bundle.add_match_claims(
            [
                MatchClaim(
                    timeline_a_id=SCORE,
                    timeline_b_id=RECORDING,
                    start_anchor=AlignmentAnchor(
                        timeline_a_id=SCORE,
                        coordinate_a=Coordinate(quarters, TimeUnit.quarters),
                        timeline_b_id=RECORDING,
                        coordinate_b=Coordinate(seconds, TimeUnit.seconds),
                    ),
                    is_synchronous=True,
                )
                for quarters, seconds in ANCHORS
            ]
        )
    return bundle


# endregion


def test_fixture_is_ungrouped() -> None:
    """The ungrouped layout holds no group, and the others group as named."""
    assert _bundle("ungrouped").timeline_to_groups == {}
    assert set(_bundle("score_grouped").timeline_to_groups) == {SCORE}
    assert set(_bundle("both_grouped").timeline_to_groups) == {SCORE, RECORDING}


@pytest.mark.parametrize(("quarters", "samples"), BETWEEN)
def test_between_anchors_matches_the_warp_map(quarters: Fraction, samples: int) -> None:
    """A position between anchors gives the bundle WarpMap's exact sample."""
    bundle = _bundle()
    warp = bundle._get_or_build_warp_map(SCORE, RECORDING)
    assert warp is not None
    assert warp.get_coordinate_at(quarters).value == samples

    stamp = bundle.get_matchstamp_at(quarters, SCORE)

    assert stamp.coordinates == {
        SCORE: _quarters(quarters),
        RECORDING: _samples(samples),
    }
    assert stamp.is_interpolated is True


def test_plural_getter_matches_the_warp_map() -> None:
    """get_matchstamps_at resolves every element through the same warp."""
    bundle = _bundle()

    stamps = bundle.get_matchstamps_at([quarters for quarters, _ in BETWEEN], SCORE)

    assert [stamp.coordinates for stamp in stamps] == [
        {SCORE: _quarters(quarters), RECORDING: _samples(samples)}
        for quarters, samples in BETWEEN
    ]


def test_table_matches_the_warp_map() -> None:
    """get_matchstamp_table carries the warped samples, struct and scalar."""
    bundle = _bundle()
    positions = [quarters for quarters, _ in BETWEEN]

    table = bundle.get_matchstamp_table(positions, SCORE)
    frame = bundle.get_matchstamp_table(positions, SCORE, format="dataframe")

    assert table.column_names == [RECORDING, SCORE]
    assert table.to_pylist() == [
        {
            RECORDING: {
                "value": float(samples),
                "numerator": samples,
                "denominator": 1,
            },
            SCORE: {
                "value": float(quarters),
                "numerator": quarters.numerator,
                "denominator": quarters.denominator,
            },
        }
        for quarters, samples in BETWEEN
    ]
    assert list(frame.columns) == ["recording (samples)", "score (quarters)"]
    assert frame["recording (samples)"].tolist() == [s for _, s in BETWEEN]
    assert frame["score (quarters)"].tolist() == positions


def test_on_anchor_gives_the_anchor() -> None:
    """An anchor position still answers the anchor, not interpolated."""
    stamp = _bundle().get_matchstamp_at(3, SCORE)

    assert stamp.coordinates == {SCORE: _quarters(3), RECORDING: _samples(66150)}
    assert stamp.is_interpolated is False


@pytest.mark.parametrize("quarters", [Fraction(1, 2), Fraction(10)])
def test_outside_the_anchors_does_not_resolve(quarters: Fraction) -> None:
    """Under the default omit policy, the recording is absent off the hull."""
    bundle = _bundle()

    stamp = bundle.get_matchstamp_at(quarters, SCORE)
    table = bundle.get_matchstamp_table([quarters, Fraction(2)], SCORE)

    assert stamp.coordinates == {SCORE: _quarters(quarters)}
    assert table.column(RECORDING).to_pylist() == [
        None,
        {"value": 44100.0, "numerator": 44100, "denominator": 1},
    ]


@pytest.mark.parametrize(("samples", "quarters"), [(44100, 2), (77175, 4)])
def test_recording_warps_to_the_score(samples: int, quarters: int) -> None:
    """The ungrouped recording is a warp source as well as a target."""
    stamp = _bundle().get_matchstamp_at(samples, RECORDING)

    assert stamp.coordinates == {
        RECORDING: _samples(samples),
        SCORE: _quarters(quarters),
    }
    assert stamp.is_interpolated is True


@pytest.mark.parametrize("layout", LAYOUTS[:2])
def test_grouping_parity(layout: str) -> None:
    """Grouping the score, or nothing, answers as grouping both timelines."""
    positions = [Fraction(1, 2), Fraction(1), Fraction(2), Fraction(7, 3)]
    positions += [Fraction(3), Fraction(4), Fraction(7), Fraction(9), Fraction(10)]

    def answers(bundle: AlignmentBundle) -> list[tuple[dict, bool]]:
        return [
            (stamp.coordinates, stamp.is_interpolated)
            for stamp in bundle.get_matchstamps_at(positions, SCORE)
        ]

    assert answers(_bundle(layout)) == answers(_bundle("both_grouped"))


def test_unclaimed_ungrouped_timeline_stays_unresolved() -> None:
    """A timeline with no claims is never reached, from or into it."""
    bundle = _bundle()

    assert OTHER not in bundle.get_matchstamp_at(2, SCORE).coordinates
    assert bundle.get_matchstamp_at(5, OTHER).coordinates == {
        OTHER: Coordinate(5.0, TimeUnit.seconds)
    }


def test_no_claims_no_answer() -> None:
    """Two ungrouped timelines with no claims between them stay apart."""
    bundle = _bundle(claims=False)

    assert bundle.get_matchstamp_at(2, SCORE).coordinates == {SCORE: _quarters(2)}
    assert bundle.get_matchstamp_at(44100, RECORDING).coordinates == {
        RECORDING: _samples(44100)
    }
