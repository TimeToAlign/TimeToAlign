"""The floating-measure conversion.

Reasoning, specimens and gold values: ``tests/maps/README.md``,
section "test_floating_measures.py".
"""

from __future__ import annotations

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
SQ8913219_MEASURES = (
    SCORE_DIR
    / "beethoven_op18-4iv_multimodal"
    / "OpenScoreSQ"
    / "sq8913219.measures.tsv"
)


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
    first_next: str | None = None,
    second_time_signature: str = "2/4",
    voltas: tuple[int | None, int | None] = (None, None),
) -> MeasureMap:
    """``a`` (1/2 of a 2/4 bar) and ``b`` at offset 1/2: both downbeats are 0.

    ``b`` is a 2/4 bar (``nominal_length`` 2) unless *second_time_signature*
    says 3/4 (``nominal_length`` 3).
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
                next=None if first_next is None else (first_next,),
                volta=voltas[0],
            ),
            MeasureConstituent(
                id="b",
                number=second_number,
                time_signature=second_time_signature,
                nominal_length=second_nominal,
                actual_length=Fraction(3, 2),
                offset_within_measure=Fraction(1, 2),
                volta=voltas[1],
            ),
        ),
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
        cmap = QuartersToFloatingMeasures.from_measure_map(measure_map)

        assert cmap(0) == 0.5
        assert cmap(19) == 10.0
        assert cmap(20) == 10.5
        assert cmap(21) == 11.0
        assert cmap(75) == 38.0
        assert cmap(76) == 38.666
        assert cmap(Fraction(153, 2)) == 39.0

    def test_woo71_alternative_endings_stay_apart(self) -> None:
        """mc 260 jumps back, so mc 261 starts the next ordinal.

        mc 262, the 6/8 rest of the second ending, is named by mc 261's next
        but is not the same notated bar: 261 is 2/4, so it is ordinal 239.
        """
        loader = Ms3Loader.from_file(WOO71_MEASURES)
        measure_map = MeasureMap.from_measure_data(loader.store["measures"])
        cmap = QuartersToFloatingMeasures.from_measure_map(measure_map)

        first, second, rest = measure_map.measures[259:262]
        assert (first.number, second.number, rest.number) == (237, 237, 237)
        assert (first.qstamp, second.qstamp) == (Fraction(945, 2), Fraction(949, 2))
        assert (second.nominal_length, rest.nominal_length) == (
            Fraction(2),
            Fraction(3),
        )
        assert rest.qstamp == Fraction(957, 2)
        assert cmap(first.qstamp) == 237.0
        assert cmap(second.qstamp) == 238.0
        assert cmap(Fraction(477)) == 239.0
        assert cmap(rest.qstamp) == 239.5
        assert cmap(Fraction(480)) == 240.0
        assert len(measure_map) == 397
        assert (len(cmap.x_values), cmap.y_values.tolist()[-1]) == (375, 374.0)

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

    def test_sq8913219_different_nominal_lengths_stay_apart(self) -> None:
        """One number, the offset matching, but a 3/4 bar and a 2/2 pickup.

        Both pairs of the score that share a number across a metre change are
        two compounds each; the first of them already refuses the lattice.
        """
        measure_map = _loaded_measure_map(SQ8913219_MEASURES)
        records = {record.id: record for record in measure_map.measures}
        bar, pickup = records["583"], records["584"]
        assert (bar.number, pickup.number) == (578, 578)
        assert (bar.nominal_length, bar.actual_length, bar.next) == (
            Fraction(3),
            Fraction(3),
            ("1",),
        )
        assert (pickup.nominal_length, pickup.offset_within_measure) == (
            Fraction(4),
            Fraction(3),
        )

        compounds = [
            [record.id for record in compound]
            for compound in QuartersToFloatingMeasures._compound_measures(measure_map)
        ]
        for pair in (["482", "483"], ["583", "584"]):
            assert [
                compound for compound in compounds if set(compound) & set(pair)
            ] == [
                [pair[0]],
                [pair[1]],
            ]
        with pytest.raises(
            ValueError, match="'482' and '483' have nominal downbeats 1270 and 2539/2"
        ):
            QuartersToFloatingMeasures.from_measure_map(measure_map)


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
        """Only one measure number joins records; 5 and 6 at one downbeat do not."""
        with pytest.raises(ValueError, match="do not increase"):
            QuartersToFloatingMeasures.from_measure_map(
                _two_records_at_one_downbeat(5, 6)
            )

    def test_records_without_numbers_never_join(self) -> None:
        with pytest.raises(ValueError, match="do not increase"):
            QuartersToFloatingMeasures.from_measure_map(
                _two_records_at_one_downbeat(None, None)
            )

    def test_one_number_of_two_nominal_lengths_raises(self) -> None:
        """5 and 5, b's offset continuing a, but b is a 3/4 bar."""
        measure_map = _two_records_at_one_downbeat(5, 5, second_time_signature="3/4")

        with pytest.raises(ValueError, match="do not increase"):
            QuartersToFloatingMeasures.from_measure_map(measure_map)

    def test_one_number_under_two_voltas_raises(self) -> None:
        """5 and 5 with the default next, but under volta 1 and volta 2."""
        measure_map = _two_records_at_one_downbeat(5, 5, voltas=(1, 2))

        with pytest.raises(ValueError, match="do not increase"):
            QuartersToFloatingMeasures.from_measure_map(measure_map)

    def test_a_bar_split_at_a_repeat_sign_is_one_measure(self) -> None:
        """5 and 5, a's next names a itself (a repeat), b's offset continues a."""
        cmap = QuartersToFloatingMeasures.from_measure_map(
            _two_records_at_one_downbeat(5, 5, first_next="a")
        )

        assert [cmap(0), cmap(1), cmap(2)] == [1.0, 1.5, 2.0]

    def test_the_same_pair_numbered_alike_has_a_lattice(self) -> None:
        """The control: 5 and 5 with the default next are one 2/4 bar."""
        cmap = QuartersToFloatingMeasures.from_measure_map(
            _two_records_at_one_downbeat(5, 5)
        )

        assert [cmap(0), cmap(1), cmap(2)] == [1.0, 1.5, 2.0]


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
        """WoO 71 splits bars, and loads with its fm conversion and no warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            timeline = Ms3Loader.from_file(WOO71_MEASURES).create_timeline()

        assert TimeUnit.floating_measures in timeline._get_available_units()
        reading = timeline.convert_to(Fraction(20), TimeUnit.floating_measures)
        assert (reading.value, reading.unit) == (10.5, TimeUnit.floating_measures)
        assert len(timeline.measure_map) == 397
