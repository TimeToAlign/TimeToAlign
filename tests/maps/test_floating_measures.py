"""The floating-measure conversion.

Reasoning, specimens and gold values: ``tests/maps/README.md``,
section "test_floating_measures.py".
"""

from __future__ import annotations

import re
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from timetoalign.core import Measure, MeasureConstituent, NumberType, TimeUnit
from timetoalign.loader.score import Ms3Loader
from timetoalign.maps.interval import QuartersToFloatingMeasures
from timetoalign.testdata import ensure_data
from timetoalign.timelines import MeasureMap

SCORE_DIR = ensure_data("score")
WAGNER_MEASURES = (
    SCORE_DIR
    / "wagner_walkure"
    / "01_RawData"
    / "score_musescore"
    / "Wagner_WWV086B-3.measures.tsv"
)
WOO71_MEASURES = SCORE_DIR / "beethoven_woo71" / "WoO71.measures.tsv"
COUPERIN_DIR = SCORE_DIR / "couperin_concerts"
MUSETE_MEASURES = COUPERIN_DIR / "c05n05_musete.measures.tsv"
RONDEAU_MEASURES = COUPERIN_DIR / "c11n08_Rondeau.measures.tsv"
OP18_DIR = SCORE_DIR / "beethoven_op18-4iv_multimodal"
SQ8913219_MEASURES = OP18_DIR / "OpenScoreSQ" / "sq8913219.measures.tsv"
OP18_FLOW_MEASURES = OP18_DIR / "op18_no4_mov4_flow" / "op18_no4_mov4_flow.measures.tsv"


def _loaded_measure_map(path: Path) -> MeasureMap:
    """The measure map a DCML measures table describes."""
    return MeasureMap.from_measure_data(Ms3Loader.from_file(path).store["measures"])


def _eroica_pickup_map() -> MeasureMap:
    """A 1/8 pickup in 2/4 followed by complete bars.

    ``nominal_length`` 2 quarters, ``actual_length`` 1/2, so the pickup's
    content sits ``3/2`` quarters into its notated bar.
    """
    records = [
        MeasureConstituent(
            id="m1",
            number=0,
            time_signature="2/4",
            nominal_length=Fraction(2),
            actual_length=Fraction(1, 2),
            offset_within_measure=Fraction(3, 2),
        )
    ]
    records += [
        Measure(
            id=f"m{count}",
            number=count - 1,
            time_signature="2/4",
            nominal_length=Fraction(2),
            actual_length=Fraction(2),
        )
        for count in range(2, 6)
    ]
    return MeasureMap(records)


def _tuplet_bar_map() -> MeasureMap:
    """Three complete bars of 7/3 quarters.

    A septuplet-length bar: no binary fraction reaches its boundaries, so
    every interior reading separates an exact walk from an interpolated
    one. The slope inside a bar is ``3/7`` fm per quarter.
    """
    records = [
        Measure(
            id=f"m{count}",
            number=count,
            nominal_length=Fraction(7, 3),
            actual_length=Fraction(7, 3),
        )
        for count in range(1, 4)
    ]
    return MeasureMap(records)


def _uniform_measure_map(n_measures: int, quarters_per_measure: Fraction) -> MeasureMap:
    return MeasureMap(
        Measure(
            id=f"m{count}",
            number=count,
            nominal_length=quarters_per_measure,
            actual_length=quarters_per_measure,
        )
        for count in range(1, n_measures + 1)
    )


def _op106_ii_opening() -> MeasureMap:
    """Op. 106/II up to bar 47: an anacrusis, then bar 46 split in two records."""
    return MeasureMap.from_rows(
        [
            {
                "count": 1,
                "number": 0,
                "qstamp": 0,
                "time_signature": "3/4",
                "nominal_length": 3,
                "actual_length": 1,
                "offset_within_measure": 2,
                "next": ["2"],
            },
            {
                "count": 47,
                "number": 46,
                "qstamp": 136,
                "nominal_length": 3,
                "actual_length": 2,
                "next": ["48"],
            },
            {
                "count": 48,
                "number": 46,
                "qstamp": 138,
                "nominal_length": 3,
                "actual_length": 1,
                "offset_within_measure": 2,
                "next": ["49"],
            },
            {
                "count": 49,
                "number": 47,
                "qstamp": 139,
                "nominal_length": 3,
                "actual_length": 3,
            },
        ]
    )


def _op106_iv_opening() -> MeasureMap:
    """Op. 106/IV rows 1-6: bar 1 in two records, bar 2 in three, ten quarters."""
    return MeasureMap.from_rows(
        [
            {
                "count": 1,
                "number": 1,
                "qstamp": 0,
                "time_signature": "4/4",
                "nominal_length": 4,
                "actual_length": [23, 4],
                "next": ["2"],
            },
            {
                "count": 2,
                "number": 1,
                "qstamp": [23, 4],
                "actual_length": 3,
                "next": ["3"],
            },
            {
                "count": 3,
                "number": 2,
                "qstamp": [35, 4],
                "actual_length": [17, 8],
                "next": ["4"],
            },
            {
                "count": 4,
                "number": 2,
                "qstamp": [87, 8],
                "actual_length": [15, 8],
                "offset_within_measure": [17, 8],
                "next": ["5"],
            },
            {
                "count": 5,
                "number": 2,
                "qstamp": [51, 4],
                "actual_length": 6,
                "next": ["6"],
            },
            {"count": 6, "number": 3, "qstamp": [75, 4], "actual_length": 4},
        ]
    )


def _volta_pair(*, jump_back: bool) -> MeasureMap:
    """Fourteen 4/4 bars, then 15a and 15b, both numbered 15.

    With *jump_back*, ``15a`` ends a repeat and its ``next`` names bar 1, as a
    first ending does; without, ``next`` is the default, the following record.
    """
    records = [
        Measure(
            id=f"m{count}",
            number=count,
            time_signature="4/4",
            nominal_length=Fraction(4),
            actual_length=Fraction(4),
        )
        for count in range(1, 15)
    ]
    records += [
        Measure(
            id="15a",
            number=15,
            name="15a",
            volta=1,
            time_signature="4/4",
            nominal_length=Fraction(4),
            actual_length=Fraction(4),
            end_repeat=jump_back,
            next=("m1",) if jump_back else None,
        ),
        Measure(
            id="15b",
            number=15,
            name="15b",
            volta=2,
            time_signature="4/4",
            nominal_length=Fraction(4),
            actual_length=Fraction(4),
        ),
    ]
    return MeasureMap(records)


def _two_records_at_one_downbeat(
    first_number: int | None,
    second_number: int | None,
    *,
    second_time_signature: str = "2/4",
    second_offset: Fraction = Fraction(1, 2),
    second_actual: Fraction = Fraction(3, 2),
    voltas: tuple[int | None, int | None] = (None, None),
) -> MeasureMap:
    """``a`` (1/2 of a 2/4 bar) and ``b`` at offset 1/2: both downbeats are 0.

    ``b`` is a 2/4 bar (``nominal_length`` 2) unless *second_time_signature*
    says 3/4 (``nominal_length`` 3), and sounds *second_actual* (3/2, the
    rest of the bar). With *second_offset* 0, ``b``'s downbeat is its
    sounding start ``1/2``.
    """
    second_nominal = {"2/4": Fraction(2), "3/4": Fraction(3)}[second_time_signature]
    return MeasureMap(
        (
            Measure(
                id="a",
                number=first_number,
                time_signature="2/4",
                nominal_length=Fraction(2),
                actual_length=Fraction(1, 2),
                volta=voltas[0],
            ),
            MeasureConstituent(
                id="b",
                number=second_number,
                time_signature=second_time_signature,
                nominal_length=second_nominal,
                actual_length=second_actual,
                offset_within_measure=second_offset,
                volta=voltas[1],
            ),
        ),
    )


def _four_four_bars(*, short: str, actual: Fraction) -> list[Measure]:
    """Full 4/4 bars ``m1``-``m5`` numbered 1-5; the *short* one sounds *actual*."""
    return [
        Measure(
            id=f"m{count}",
            number=count,
            time_signature="4/4",
            nominal_length=Fraction(4),
            actual_length=actual if f"m{count}" == short else Fraction(4),
        )
        for count in range(1, 6)
    ]


def _first_ending(*, jump_back: bool) -> MeasureMap:
    """An anacrusis, a bar, a 3-quarter first ending, a second ending, a bar.

    ``m0`` is the pickup (offset 3, one quarter); ``m2`` (volta 1) sounds 3
    quarters and, with *jump_back*, its ``next`` names ``m0``; ``m3`` (volta 2)
    carries ``m2``'s number.
    """
    pickup = MeasureConstituent(
        id="m0",
        number=0,
        time_signature="4/4",
        nominal_length=Fraction(4),
        actual_length=Fraction(1),
        offset_within_measure=Fraction(3),
    )

    def bar(id_: str, number: int, actual: int = 4, **values: object) -> Measure:
        return Measure(
            id=id_,
            number=number,
            time_signature="4/4",
            nominal_length=Fraction(4),
            actual_length=Fraction(actual),
            **values,
        )

    return MeasureMap(
        [
            pickup,
            bar("m1", 1),
            bar(
                "m2",
                2,
                3,
                volta=1,
                end_repeat=True,
                next=("m0",) if jump_back else None,
            ),
            bar("m3", 2, volta=2),
            bar("m4", 3),
        ]
    )


def _repeat_end_leading_on() -> MeasureMap:
    """The pickup ``m0`` and ``m1``; ``m2`` (3 quarters) ends a repeat, leading back and on.

    ``m2``'s ``next`` names ``m0`` and ``m3``; ``m3`` and ``m4`` are full bars
    numbered 3 and 4, no record states a volta.
    """
    records = list(_first_ending(jump_back=True).measures)
    records[2] = Measure(
        id="m2",
        number=2,
        time_signature="4/4",
        nominal_length=Fraction(4),
        actual_length=Fraction(3),
        end_repeat=True,
        next=("m0", "m3"),
    )
    records[3:] = [
        Measure(
            id=f"m{number}",
            number=number,
            time_signature="4/4",
            nominal_length=Fraction(4),
            actual_length=Fraction(4),
        )
        for number in (3, 4)
    ]
    return MeasureMap(records)


def _warns_exactly(*lines: str) -> pytest.WarningsChecker:
    """Expect the one floating-measure warning, with exactly these *lines*."""
    text = "Floating measures that do not add up to their bars:\n" + ";\n".join(
        f"  {line}" for line in lines
    )
    return pytest.warns(UserWarning, match=f"^{re.escape(text)}$")


WOO71_DOUBTS = (
    "offset mismatch: measure '43' (number 38) states offset 3/2, not 1 where '42' ends",
    "offset mismatch: measure '262' (number 237) states offset 3/2, not 4 where '261' ends",
    "overlength: measures '371', '372' (number 346) reach 9/2 quarters into a bar of 3",
)
SQ8913219_DOUBTS = (
    "overlength: measures '77', '78' (number 76) reach 8 quarters into a bar of 4",
    "short: measure '221' (number 219) ends 3 quarters into a bar of 4, "
    "and measure '222' (number 220), which it leads to, does not complete the bar",
    "offset mismatch: measure '483' (number 480) states offset 2, not 3/2 where '482' ends",
    "overlength: measures '533', '534' (number 529) reach 5 quarters into a bar of 3",
    "overlength: measures '626', '627' (number 617) reach 7 quarters into a bar of 4",
    "overlength: measures '675', '676' (number 663) reach 15/2 quarters into a bar of 4",
    "overlength: measures '684', '685' (number 671) reach 8 quarters into a bar of 4",
)


def _doubtful_refusal(first: str, second: str) -> str:
    """The refusal of a doubtful split at offset 3/4, its records named *first*, *second*."""
    return (
        f"The floating measures starting with measures {first!r} and {second!r} have "
        "nominal downbeats 0 and -1/4, which do not increase; adjacent measures form one "
        "floating measure only when they share a number or the second completes the bar "
        "the first leaves incomplete, and endings under different volta brackets count "
        "through as separate floating measures. Floating measures that do not add up to "
        f"their bars:\n  doubtful split, not joined: measure {first!r} (number 5) and "
        f"measure {second!r} (number 6) together fill a bar of 2 quarters, but {second!r} "
        f"states offset 3/4, not 1/2 where {first!r} ends"
    )


DOUBTFUL_REFUSAL = _doubtful_refusal("a", "b")
OP106_IV_DOUBTS = (
    "overlength: measures '1', '2' (number 1) reach 35/4 quarters into a bar of 4",
    "overlength: measures '3', '4', '5' (number 2) reach 10 quarters into a bar of 4",
)


class TestAnacrusisIsAnchoredOnTheNominalBar:
    """A pickup's content sits where it is notated, not at 0.000."""

    def test_wagner_pickup_reads_the_published_value(self) -> None:
        """1/8 in 9/8 is 8/9 of the way through bar 0: WRD reads 0.888."""
        loader = Ms3Loader.from_file(WAGNER_MEASURES)
        measure_map = MeasureMap.from_measure_data(loader.store["measures"])

        pickup = measure_map.measures[0]
        assert pickup.time_signature == "9/8"
        assert pickup.nominal_length == Fraction(9, 2)
        assert pickup.actual_length == Fraction(1, 2)
        assert pickup.offset_within_measure == Fraction(4)

        cmap = QuartersToFloatingMeasures.from_measure_map(measure_map)

        assert cmap(0) == 0.888
        assert cmap(Fraction(1, 2)) == 1.0
        assert cmap(5) == 2.0

    def test_truncation_not_rounding(self) -> None:
        """8/9 is 0.8888...; rounding would say 0.889, which WRD does not."""
        loader = Ms3Loader.from_file(WAGNER_MEASURES)
        measure_map = MeasureMap.from_measure_data(loader.store["measures"])

        cmap = QuartersToFloatingMeasures.from_measure_map(measure_map)

        assert cmap(0) == 0.888
        assert cmap(0) != 0.889
        assert round(Fraction(8, 9) * 1000) / 1000 == 0.889

    def test_eroica_pickup_reads_three_quarters_of_the_bar(self) -> None:
        """1/8 in 2/4 sits 3/2 quarters in: 0.750."""
        cmap = QuartersToFloatingMeasures.from_measure_map(_eroica_pickup_map())

        assert cmap(0) == 0.750
        assert cmap(Fraction(1, 2)) == 1.0
        assert cmap(Fraction(5, 2)) == 2.0

    def test_a_complete_first_bar_starts_at_one(self) -> None:
        """No pickup, no measure 0."""
        cmap = QuartersToFloatingMeasures.from_measure_map(
            _uniform_measure_map(n_measures=4, quarters_per_measure=Fraction(4))
        )

        assert cmap(0) == 1.0
        assert cmap(4) == 2.0


class TestOrdinalsCountCompoundMeasures:
    """One ordinal per measure number run; voltas stay apart; labels never count."""

    def test_a_volta_pair_left_by_a_jump_is_two_measures(self) -> None:
        """15a's next names bar 1, not 15b: 15.0 and 16.0."""
        cmap = QuartersToFloatingMeasures.from_measure_map(_volta_pair(jump_back=True))

        assert cmap(Fraction(56)) == 15.0
        assert cmap(Fraction(60)) == 16.0
        assert cmap(Fraction(64)) == 17.0

    def test_a_volta_pair_without_a_jump_stays_two_measures(self) -> None:
        """With the default next, 15a and 15b still state volta 1 and volta 2."""
        cmap = QuartersToFloatingMeasures.from_measure_map(_volta_pair(jump_back=False))

        assert cmap(Fraction(56)) == 15.0
        assert cmap(Fraction(60)) == 16.0
        assert cmap(Fraction(64)) == 17.0


class TestCompoundMeasures:
    """A split bar reads as the one bar it is notated as."""

    def test_a_split_bar_after_an_anacrusis(self) -> None:
        """Op. 106/II: bar 46 is counts 47 and 48, ordinal 46, slope 1/3."""
        measure_map = _op106_ii_opening()
        cmap = QuartersToFloatingMeasures.from_measure_map(measure_map)

        assert len(measure_map) == 49
        assert measure_map.measures[47].qstamp == Fraction(138)
        assert [cmap(q) for q in (0, 136, 137, 138, 139)] == [
            0.666,
            46.0,
            46.333,
            46.666,
            47.0,
        ]
        assert cmap(142) == 48.0

    def test_the_knots_of_a_split_bar(self) -> None:
        """One knot per compound: count 48 places none, the end closes at 142."""
        cmap = QuartersToFloatingMeasures.from_measure_map(_op106_ii_opening())

        assert cmap.x_values.tolist()[-4:] == [133.0, 136.0, 139.0, 142.0]
        assert cmap.y_values.tolist()[-4:] == [45.0, 46.0, 47.0, 48.0]
        assert len(cmap.x_values) == 49

    def test_a_three_part_compound_longer_than_its_bar(self) -> None:
        """Op. 106/IV: bar 2 is three records, ten quarters, one ordinal."""
        with _warns_exactly(*OP106_IV_DOUBTS):
            cmap = QuartersToFloatingMeasures.from_measure_map(_op106_iv_opening())

        quarters = [
            0,
            Fraction(23, 4),
            Fraction(35, 4),
            Fraction(87, 8),
            Fraction(51, 4),
        ]
        assert [cmap(q) for q in quarters] == [1.0, 1.657, 2.0, 2.212, 2.4]
        assert cmap(Fraction(75, 4)) == 3.0
        assert cmap.x_values.tolist() == [0.0, 8.75, 18.75, 22.75]

    def test_the_inverse_follows_the_compound_knots(self) -> None:
        """fm 2.0 is bar 2's downbeat, 3.0 bar 3's."""
        with _warns_exactly(*OP106_IV_DOUBTS):
            inverse = QuartersToFloatingMeasures.from_measure_map(
                _op106_iv_opening()
            ).inverse()

        assert inverse(2.0) == 8.75
        assert inverse(2.5) == 13.75
        assert inverse(3.0) == 18.75

    def test_the_two_lanes_agree_inside_a_compound(self) -> None:
        cmap = QuartersToFloatingMeasures.from_measure_map(_op106_ii_opening())

        values = cmap.convert_array(np.array([136.0, 137.0, 138.0, 139.0]))

        assert values.tolist() == [46.0, 46.333, 46.666, 47.0]

    def test_woo71_split_bars_read_as_one_bar(self) -> None:
        """WoO 71 splits bars at repeat signs; each pair reads as its one bar."""
        loader = Ms3Loader.from_file(WOO71_MEASURES)
        measure_map = MeasureMap.from_measure_data(loader.store["measures"])
        with _warns_exactly(*WOO71_DOUBTS):
            cmap = QuartersToFloatingMeasures.from_measure_map(measure_map)

        assert cmap(0) == 0.5
        assert cmap(19) == 10.0
        assert cmap(20) == 10.5
        assert cmap(21) == 11.0
        assert cmap(75) == 38.0
        assert cmap(76) == 38.666
        assert cmap(Fraction(153, 2)) == 39.0

    def test_woo71_second_ending_joins_its_rest_across_a_metre_change(self) -> None:
        """mc 260 (volta 1) and mc 261 (volta 2) count through as two ordinals.

        mc 262, the 6/8 rest of the second ending, follows mc 261 with its
        number and states no volta: one bar, although 261 is 2/4. Ordinal
        238, spread over 4 + 3/2 = 11/2 quarters.
        """
        loader = Ms3Loader.from_file(WOO71_MEASURES)
        measure_map = MeasureMap.from_measure_data(loader.store["measures"])
        with _warns_exactly(*WOO71_DOUBTS):
            cmap = QuartersToFloatingMeasures.from_measure_map(measure_map)

        first, second, rest = measure_map.measures[259:262]
        assert (first.number, second.number, rest.number) == (237, 237, 237)
        assert (first.qstamp, second.qstamp) == (Fraction(945, 2), Fraction(949, 2))
        assert (second.nominal_length, second.actual_length, second.next) == (
            Fraction(2),
            Fraction(4),
            ("262",),
        )
        assert (rest.id, rest.nominal_length, rest.offset_within_measure) == (
            "262",
            Fraction(3),
            Fraction(3, 2),
        )
        assert rest.qstamp == Fraction(957, 2)
        quarters = [
            first.qstamp,
            second.qstamp,
            Fraction(477),
            rest.qstamp,
            Fraction(480),
        ]
        assert [cmap(q) for q in quarters] == [237.0, 238.0, 238.454, 238.727, 239.0]
        assert len(measure_map) == 397
        assert (len(cmap.x_values), cmap.y_values.tolist()[-1]) == (374, 373.0)

    def test_couperin_musete_bar_split_at_a_repeat_sign_is_one_bar(self) -> None:
        """mc 31 jumps back, mc 32 continues it at offset 3/2: bar 28, slope 1/3."""
        measure_map = _loaded_measure_map(MUSETE_MEASURES)
        cmap = QuartersToFloatingMeasures.from_measure_map(measure_map)

        first, second = measure_map.measures[30:32]
        assert (first.id, first.number, first.qstamp) == ("31", 28, Fraction(165, 2))
        assert (first.actual_length, first.next) == (Fraction(3, 2), ("17", "6"))
        assert (second.id, second.number, second.qstamp) == ("32", 28, Fraction(84))
        assert second.offset_within_measure == Fraction(3, 2)
        assert first.nominal_length == second.nominal_length == Fraction(3)
        quarters = [Fraction(165, 2), Fraction(84), Fraction(85), Fraction(171, 2)]
        assert [cmap(q) for q in quarters] == [28.0, 28.5, 28.833, 29.0]
        assert len(measure_map) == 58
        assert (len(cmap.x_values), cmap.y_values.tolist()[-1]) == (56, 55.0)

    def test_couperin_rondeau_bar_split_at_a_repeat_sign_is_one_bar(self) -> None:
        """mc 18 jumps back, mc 19 continues it at offset 1: bar 16, slope 2/3."""
        measure_map = _loaded_measure_map(RONDEAU_MEASURES)
        with _warns_exactly(
            "overlength: measures '27', '28' (number 24) reach 3 quarters into a bar of 3/2"
        ):
            cmap = QuartersToFloatingMeasures.from_measure_map(measure_map)

        first, second = measure_map.measures[17:19]
        assert (first.number, first.qstamp, first.next) == (
            16,
            Fraction(23),
            ("10", "1"),
        )
        assert (second.number, second.offset_within_measure) == (16, Fraction(1))
        assert [cmap(q) for q in (Fraction(23), Fraction(24), Fraction(49, 2))] == [
            16.0,
            16.666,
            17.0,
        ]

    def test_sq8913219_bars_across_a_metre_change_read_as_one_bar(self) -> None:
        """mc 482/483 (3/8, then 3/4) and mc 583/584 (3/4, then a 2/2 pickup)
        each carry one number: each pair is one bar, and the whole score has
        a lattice of 796 floating measures, ordinals 0-795.
        """
        measure_map = _loaded_measure_map(SQ8913219_MEASURES)
        records = {record.id: record for record in measure_map.measures}
        split, rest = records["482"], records["483"]
        assert (split.number, rest.number) == (480, 480)
        assert (split.nominal_length, split.actual_length, split.next) == (
            Fraction(3, 2),
            Fraction(3, 2),
            ("483",),
        )
        assert (rest.nominal_length, rest.offset_within_measure) == (
            Fraction(3),
            Fraction(2),
        )
        bar, pickup, after = records["583"], records["584"], records["585"]
        assert (bar.number, pickup.number, after.number) == (578, 578, 579)
        assert (bar.nominal_length, bar.actual_length, bar.next) == (
            Fraction(3),
            Fraction(3),
            ("1",),
        )
        assert (pickup.nominal_length, pickup.actual_length) == (
            Fraction(4),
            Fraction(1),
        )
        assert pickup.offset_within_measure == Fraction(3)
        assert (bar.qstamp, pickup.qstamp, after.qstamp) == (
            Fraction(3131, 2),
            Fraction(3137, 2),
            Fraction(3139, 2),
        )

        compounds = [
            [record.id for record in compound]
            for compound in QuartersToFloatingMeasures._compound_measures(measure_map)
        ]
        joined = [compound for compound in compounds if len(compound) > 1]
        assert [compound[0] for compound in joined] == [
            "77",
            "482",
            "491",
            "533",
            "583",
            "592",
            "601",
            "610",
            "626",
            "660",
            "667",
            "675",
            "684",
        ]
        assert all(len(compound) == 2 for compound in joined)

        with _warns_exactly(*SQ8913219_DOUBTS):
            cmap = QuartersToFloatingMeasures.from_measure_map(measure_map)
        quarters = [
            Fraction(1270),
            Fraction(2543, 2),
            Fraction(1272),
            Fraction(2545, 2),
            bar.qstamp,
            pickup.qstamp,
            after.qstamp,
        ]
        assert [cmap(q) for q in quarters] == [
            480.0,
            480.6,
            480.8,
            481.0,
            578.0,
            578.75,
            579.0,
        ]
        assert len(measure_map) == 809
        assert (len(cmap.x_values), cmap.x_values.tolist()[-1]) == (797, 2447.0)
        assert cmap.y_values.tolist()[-2:] == [795.0, 796.0]
        assert cmap(2446) == 795.666


class TestInterpolationInsideABar:
    """The slope inside a bar is one over its nominal length."""

    def test_halfway_through_a_four_quarter_bar(self) -> None:
        cmap = QuartersToFloatingMeasures.from_measure_map(
            _uniform_measure_map(n_measures=4, quarters_per_measure=Fraction(4))
        )

        assert cmap(2) == 1.5
        assert cmap(6) == 2.5

    def test_the_lattice_closes_at_the_end_of_the_last_bar(self) -> None:
        measure_map = _uniform_measure_map(
            n_measures=4, quarters_per_measure=Fraction(4)
        )
        cmap = QuartersToFloatingMeasures.from_measure_map(measure_map)

        assert cmap(16) == 5.0


class TestInverse:
    """fm back to quarters, over the same knots, without invention."""

    def test_exact_values_invert_onto_the_quarters_axis(self) -> None:
        """0.750 is the pickup onset; 2.0 and 3.0 are printed bars 2 and 3."""
        cmap = QuartersToFloatingMeasures.from_measure_map(_eroica_pickup_map())
        inverse = cmap.inverse()

        assert inverse(0.750) == 0.0
        assert inverse(2.0) == 2.5
        assert inverse(3.0) == 4.5

    def test_the_inverse_computes_in_doubles(self) -> None:
        """Its knots are fm floats, so it answers floats even into quarters."""
        cmap = QuartersToFloatingMeasures.from_measure_map(_eroica_pickup_map())
        inverse = cmap.inverse()

        assert inverse.target_unit is TimeUnit.quarters
        assert inverse.output_number_type is NumberType.float
        assert type(inverse(2.0)) is float

    def test_a_truncated_source_value_carries_its_documented_error(self) -> None:
        """Truncation is not reconstructed: the residue is reported, not fixed.

        The Wagner pickup onset is quarter 0 and reads ``0.888``. Feeding
        that reading back lands one thousandth of a bar early: the bar is
        9/2 quarters, and ``(0.888 - 8/9) x 9/2`` is -0.004 quarters. The
        inverse reports exactly that rather than snapping to zero.
        """
        loader = Ms3Loader.from_file(WAGNER_MEASURES)
        measure_map = MeasureMap.from_measure_data(loader.store["measures"])
        cmap = QuartersToFloatingMeasures.from_measure_map(measure_map)
        inverse = cmap.inverse()

        recovered = inverse(0.888)
        assert type(recovered) is float
        assert recovered == -0.0040000000000000036


class TestAxisAndOutputType:
    """fm is float by definition."""

    def test_target_axis(self) -> None:
        cmap = QuartersToFloatingMeasures.from_measure_map(
            _uniform_measure_map(n_measures=2, quarters_per_measure=Fraction(4))
        )

        assert cmap.source_unit is TimeUnit.quarters
        assert cmap.target_unit is TimeUnit.floating_measures
        assert cmap.output_number_type is NumberType.float
        assert isinstance(cmap(2), float)

    def test_the_array_lane_truncates_too(self) -> None:
        loader = Ms3Loader.from_file(WAGNER_MEASURES)
        measure_map = MeasureMap.from_measure_data(loader.store["measures"])
        cmap = QuartersToFloatingMeasures.from_measure_map(measure_map)

        values = cmap.convert_array(np.array([0.0, 0.5, 5.0]))

        assert values.tolist() == [0.888, 1.0, 2.0]

    def test_the_array_lane_reads_interior_positions_exactly(self) -> None:
        """Thousandth boundaries inside a bar no binary fraction reaches.

        A 7/3-quarter bar rises 3/7 fm per quarter, so the exact
        thousandth boundary ``k`` sits at quarter ``7k/3000``. A column
        interpolated in floating point lands just below such a boundary
        about as often as just above it and reads ``1.002`` for the first
        of these.
        """
        cmap = QuartersToFloatingMeasures.from_measure_map(_tuplet_bar_map())

        values = cmap.convert_array(np.array([0.007, 0.014, 0.021, 0.287, 2.331]))

        assert values.tolist() == [1.003, 1.006, 1.009, 1.122, 1.998]

    def test_the_two_lanes_agree_everywhere(self) -> None:
        """One map, one answer, whether the caller asks in bulk or singly."""
        cmap = QuartersToFloatingMeasures.from_measure_map(_tuplet_bar_map())
        positions = [count / 1000 for count in range(7001)]

        values = cmap.convert_array(np.array(positions))

        assert values.tolist() == [cmap(position) for position in positions]


class TestRefusals:
    """The lattice needs measures, and knots that increase."""

    def test_a_measure_map_without_measures_raises(self) -> None:
        measure_map = MeasureMap([])

        with pytest.raises(ValueError, match="at least one measure"):
            QuartersToFloatingMeasures.from_measure_map(measure_map)

    def test_different_numbers_at_one_downbeat_raise(self) -> None:
        """5 and 6 at one downbeat, b sounding 1 quarter: 1/2 + 1 fills no bar.

        No fit, so no join; the knots coincide, and nothing is warned first.
        """
        measure_map = _two_records_at_one_downbeat(5, 6, second_actual=Fraction(1))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(
                ValueError, match="'a' and 'b' have nominal downbeats 0 and 0"
            ):
                QuartersToFloatingMeasures.from_measure_map(measure_map)

    def test_records_without_numbers_never_join(self) -> None:
        with pytest.raises(ValueError, match="do not increase"):
            QuartersToFloatingMeasures.from_measure_map(
                _two_records_at_one_downbeat(None, None)
            )

    def test_one_number_under_two_voltas_raises(self) -> None:
        """5 and 5 with the default next, but under volta 1 and volta 2."""
        measure_map = _two_records_at_one_downbeat(5, 5, voltas=(1, 2))

        with pytest.raises(ValueError, match="do not increase"):
            QuartersToFloatingMeasures.from_measure_map(measure_map)


class TestAdjacentRecordsOfOneNumber:
    """The number alone joins adjacent records: no next, length or offset is read."""

    def test_one_number_is_one_measure(self) -> None:
        """5 and 5, b continuing a at offset 1/2: one 2/4 bar."""
        cmap = QuartersToFloatingMeasures.from_measure_map(
            _two_records_at_one_downbeat(5, 5)
        )

        assert [cmap(0), cmap(1), cmap(2)] == [1.0, 1.5, 2.0]

    def test_one_number_of_two_nominal_lengths_is_one_measure(self) -> None:
        """5 and 5, b a 3/4 bar: one bar across the metre change."""
        cmap = QuartersToFloatingMeasures.from_measure_map(
            _two_records_at_one_downbeat(5, 5, second_time_signature="3/4")
        )

        assert [cmap(0), cmap(1), cmap(2)] == [1.0, 1.5, 2.0]

    def test_one_number_joins_whatever_the_offset(self) -> None:
        """5 and 5, b at offset 0: still one bar, knots 0 -> 1 and 2 -> 2.

        Apart, b's downbeat 1/2 would be a knot of its own and read 2.0.
        """
        cmap = QuartersToFloatingMeasures.from_measure_map(
            _two_records_at_one_downbeat(5, 5, second_offset=Fraction(0))
        )

        assert cmap.x_values.tolist() == [0.0, 2.0]
        assert [cmap(0), cmap(Fraction(1, 2)), cmap(2)] == [1.0, 1.25, 2.0]


class TestSplitBarNumberedAsTwo:
    """Records numbered apart join where they complete one bar exactly."""

    def test_two_numbers_that_fill_one_bar_are_one_measure(self) -> None:
        """5 and 6, b at offset 1/2 sounding 3/2: 1/2 + 3/2 = 2, one 2/4 bar."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cmap = QuartersToFloatingMeasures.from_measure_map(
                _two_records_at_one_downbeat(5, 6)
            )

        assert cmap.x_values.tolist() == [0.0, 2.0]
        assert [cmap(0), cmap(1), cmap(2)] == [1.0, 1.5, 2.0]

    def test_op18_flow_skeleton_joins_its_split_bars(self) -> None:
        """Numbers by count: mc 9/10, 18/19, 27/28, 77/78 and 84/85 each fill a bar.

        221 floating measures, ordinals 0-220, no warning: the first endings
        mc 44 and 93 are completed by the pickups they jump back to.
        """
        measure_map = _loaded_measure_map(OP18_FLOW_MEASURES)
        records = {record.id: record for record in measure_map.measures}
        bar, pickup = records["9"], records["10"]
        assert (bar.number, bar.qstamp, bar.actual_length, bar.next) == (
            9,
            Fraction(29),
            Fraction(3),
            ("1", "10"),
        )
        assert (pickup.number, pickup.qstamp, pickup.offset_within_measure) == (
            10,
            Fraction(32),
            Fraction(3),
        )
        first, rest = records["77"], records["78"]
        assert (first.number, first.qstamp, first.actual_length) == (
            77,
            Fraction(288),
            Fraction(7, 2),
        )
        assert (rest.number, rest.qstamp, rest.offset_within_measure) == (
            78,
            Fraction(583, 2),
            Fraction(7, 2),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cmap = QuartersToFloatingMeasures.from_measure_map(measure_map)

        joined = [
            [record.id for record in compound]
            for compound in QuartersToFloatingMeasures._compound_measures(measure_map)
            if len(compound) > 1
        ]
        assert joined == [
            ["9", "10"],
            ["18", "19"],
            ["27", "28"],
            ["77", "78"],
            ["84", "85"],
        ]
        assert len(measure_map) == 226
        assert (len(cmap.x_values), cmap.y_values.tolist()[-1]) == (222, 221.0)
        assert cmap(0) == 0.75
        quarters = [29, 30, 32, 33, 288, Fraction(583, 2), 292]
        assert [cmap(q) for q in quarters] == [8.0, 8.25, 8.75, 9.0, 73.0, 73.875, 74.0]


class TestMeasuresThatDoNotAddUp:
    """One warning per measure map lists every place that does not add up."""

    @pytest.mark.parametrize(
        ("offset", "knots", "readings"),
        [
            (Fraction(0), [0.0, 0.5, 2.0], [1.5, 2.0, 3.0]),
            (Fraction(1, 4), [0.0, 0.25, 2.0], [2.0, 2.142, 3.0]),
        ],
    )
    def test_a_doubtful_split_warns_and_stays_apart(
        self, offset: Fraction, knots: list[float], readings: list[float]
    ) -> None:
        """5 and 6 would fill the bar, but b's offset is not 1/2: two measures.

        Offset 0 is the fit case that must not join because the offset is
        missing.
        """
        measure_map = _two_records_at_one_downbeat(5, 6, second_offset=offset)

        with _warns_exactly(
            "doubtful split, not joined: measure 'a' (number 5) and measure 'b' "
            f"(number 6) together fill a bar of 2 quarters, but 'b' states offset "
            f"{offset}, not 1/2 where 'a' ends"
        ):
            cmap = QuartersToFloatingMeasures.from_measure_map(measure_map)

        assert cmap.x_values.tolist() == knots
        assert cmap.y_values.tolist() == [1.0, 2.0, 3.0]
        assert [cmap(q) for q in (Fraction(1, 4), Fraction(1, 2), 2)] == readings

    def test_an_overlength_compound_warns(self) -> None:
        """Op. 106/IV: bars 1 and 2 run past their 4/4 bar, bar 3 fills its own."""
        with _warns_exactly(*OP106_IV_DOUBTS):
            cmap = QuartersToFloatingMeasures.from_measure_map(_op106_iv_opening())

        assert cmap.x_values.tolist() == [0.0, 8.75, 18.75, 22.75]

    def test_a_short_bar_in_the_middle_warns(self) -> None:
        """m3 sounds 3 of 4 quarters and m4 starts a bar: short, warned."""
        measure_map = MeasureMap(_four_four_bars(short="m3", actual=Fraction(3)))

        with _warns_exactly(
            "short: measure 'm3' (number 3) ends 3 quarters into a bar of 4, "
            "and measure 'm4' (number 4), which it leads to, does not complete the bar"
        ):
            cmap = QuartersToFloatingMeasures.from_measure_map(measure_map)

        assert cmap.x_values.tolist() == [0.0, 4.0, 8.0, 11.0, 15.0, 19.0]
        assert [cmap(10), cmap(11)] == [3.666, 4.0]

    def test_short_first_and_last_bars_do_not_warn(self) -> None:
        """An anacrusis and a closing bar shorter than their bars are no doubt."""
        records = _four_four_bars(short="m5", actual=Fraction(3))
        records[0] = MeasureConstituent(
            id="m1",
            number=1,
            time_signature="4/4",
            nominal_length=Fraction(4),
            actual_length=Fraction(1),
            offset_within_measure=Fraction(3),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cmap = QuartersToFloatingMeasures.from_measure_map(MeasureMap(records))

        assert cmap.y_values.tolist()[0] == 0.0

    def test_a_first_ending_completed_across_the_jump_does_not_warn(self) -> None:
        """m2 (volta 1) sounds 3 quarters and jumps back to the pickup m0."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cmap = QuartersToFloatingMeasures.from_measure_map(
                _first_ending(jump_back=True)
            )

        assert cmap.x_values.tolist() == [-3.0, 1.0, 5.0, 8.0, 12.0, 16.0]
        assert cmap.y_values.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        assert [cmap(5), cmap(7), cmap(8)] == [2.0, 2.666, 3.0]

    def test_a_first_ending_without_the_jump_warns(self) -> None:
        """The same m2 with the default next: m3 starts at offset 0, not 3."""
        with _warns_exactly(
            "short: measure 'm2' (number 2) ends 3 quarters into a bar of 4, "
            "and measure 'm3' (number 2), which it leads to, does not complete the bar"
        ):
            QuartersToFloatingMeasures.from_measure_map(_first_ending(jump_back=False))

    def test_a_bar_ending_a_repeat_must_add_up_on_every_pass(self) -> None:
        """m2 leads back to the pickup m0 and on to the fresh bar m3: short."""
        with _warns_exactly(
            "short: measure 'm2' (number 2) ends 3 quarters into a bar of 4, "
            "and measure 'm3' (number 3), which it leads to, does not complete the bar"
        ):
            cmap = QuartersToFloatingMeasures.from_measure_map(_repeat_end_leading_on())

        assert cmap.x_values.tolist() == [-3.0, 1.0, 5.0, 8.0, 12.0, 16.0]
        assert cmap(7) == 2.666

    def test_an_offset_mismatch_inside_a_compound_warns(self) -> None:
        """b, numbered like a, states offset 1 where a ends at 1/2."""
        measure_map = _two_records_at_one_downbeat(
            5, 5, second_offset=Fraction(1), second_actual=Fraction(1)
        )

        with _warns_exactly(
            "offset mismatch: measure 'b' (number 5) states offset 1, not 1/2 where 'a' ends"
        ):
            cmap = QuartersToFloatingMeasures.from_measure_map(measure_map)

        assert cmap.x_values.tolist() == [0.0, 1.5]
        assert cmap.y_values.tolist() == [1.0, 2.0]
        assert cmap(Fraction(3, 4)) == 1.5

    def test_a_refused_lattice_names_the_places_and_does_not_warn(self) -> None:
        """b at offset 3/4 is a doubtful split whose downbeat precedes a's."""
        measure_map = _two_records_at_one_downbeat(5, 6, second_offset=Fraction(3, 4))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match=f"^{re.escape(DOUBTFUL_REFUSAL)}$"):
                QuartersToFloatingMeasures.from_measure_map(measure_map)

    def test_the_warning_names_the_caller(self) -> None:
        """The warning is attributed to this file, not to the package."""
        measure_map = MeasureMap(_four_four_bars(short="m3", actual=Fraction(3)))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            QuartersToFloatingMeasures.from_measure_map(measure_map)

        assert [(w.category, w.filename) for w in caught] == [(UserWarning, __file__)]

    def test_writing_a_lattice_does_not_warn_again(self) -> None:
        """The manifest view re-reads the measure map silently."""
        with _warns_exactly(*OP106_IV_DOUBTS):
            cmap = QuartersToFloatingMeasures.from_measure_map(_op106_iv_opening())

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert cmap._is_lattice_of(cmap.measure_map)


class TestScoreFloatingMeasures:
    """What a loaded score reads on the fm axis, and when it reads nothing."""

    def test_a_loaded_score_reads_the_structural_value(self) -> None:
        """Derived from the measure structure, never from the printed labels.

        The pickup row is printed ``1``; reading that label as an ordinal
        would put the onset at ``1.0``. The structure says the onset sits
        eight ninths of the way through a 9/8 bar, which is the published
        ``0.888``.
        """
        timeline = Ms3Loader.from_file(WAGNER_MEASURES).create_timeline()

        reading = timeline.convert_to(Fraction(0), TimeUnit.floating_measures)

        assert reading.value == 0.888
        assert reading.unit is TimeUnit.floating_measures

    def test_a_split_bar_score_reads_its_compound_measures(self) -> None:
        """WoO 71 splits bars, and loads with its fm conversion and one warning."""
        with _warns_exactly(*WOO71_DOUBTS):
            timeline = Ms3Loader.from_file(WOO71_MEASURES).create_timeline()

        assert TimeUnit.floating_measures in timeline._get_available_units()
        reading = timeline.convert_to(Fraction(20), TimeUnit.floating_measures)
        assert (reading.value, reading.unit) == (10.5, TimeUnit.floating_measures)
        assert len(timeline.measure_map) == 397

    def test_a_score_without_a_lattice_says_so(self, tmp_path: Path) -> None:
        """mc 2 states offset 3/4 where mc 1 ends at 1/2: no lattice, one warning."""
        path = tmp_path / "doubtful.measures.tsv"
        path.write_text(
            "mc\tmn\tquarterbeats\ttimesig\tact_dur\tmc_offset\n"
            "1\t5\t0\t2/4\t1/8\t0\n"
            "2\t6\t1/2\t2/4\t3/8\t3/16\n",
            encoding="utf-8",
        )
        expected = (
            f"The score {str(path)!r} has no floating measures: "
            + _doubtful_refusal("1", "2")
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            timeline = Ms3Loader.from_file(path).create_timeline()

        assert [(w.category, w.filename, str(w.message)) for w in caught] == [
            (UserWarning, __file__, expected)
        ]
        assert TimeUnit.floating_measures not in timeline._get_available_units()
        assert len(timeline.measure_map) == 2
