"""The floating-measure conversion.

Reasoning, specimens and gold values: ``tests/maps/README.md``,
section "test_floating_measures.py".
"""

from __future__ import annotations

import warnings
from fractions import Fraction

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


class TestOrdinalsCountRecords:
    """Voltas get consecutive ordinals; labels never drive the count."""

    def test_hand_built_volta_pair_is_consecutive(self) -> None:
        """15a and 15b are two bars, so 15.0 and 16.0."""
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
        measure_map = MeasureMap(records)

        cmap = QuartersToFloatingMeasures.from_measure_map(measure_map)

        assert cmap(Fraction(56)) == 15.0
        assert cmap(Fraction(60)) == 16.0


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
    """The lattice needs measures, and one knot per bar."""

    def test_a_measure_map_without_measures_raises(self) -> None:
        measure_map = MeasureMap([])

        with pytest.raises(ValueError, match="at least one measure"):
            QuartersToFloatingMeasures.from_measure_map(measure_map)

    def test_a_real_score_with_split_bars_is_refused(self) -> None:
        """WoO 71 splits bars, so its fm lattice is not yet expressible.

        A split bar's two halves are one NOTATED bar and share a nominal
        downbeat, so a lattice that gives every record its own ordinal has
        two knots at one position. The map says so instead of collapsing
        them onto a single point or inventing an order between them.
        """
        loader = Ms3Loader.from_file(WOO71_MEASURES)
        measure_map = MeasureMap.from_measure_data(loader.store["measures"])

        with pytest.raises(ValueError, match="same nominal downbeat"):
            QuartersToFloatingMeasures.from_measure_map(measure_map)

    def test_a_split_bar_cannot_anchor_two_ordinals(self) -> None:
        """Two halves of one notated bar share a nominal downbeat."""
        measure_map = MeasureMap(
            (
                Measure(
                    id="m5a",
                    number=5,
                    name="5a",
                    time_signature="2/4",
                    nominal_length=Fraction(2),
                    actual_length=Fraction(1, 2),
                ),
                MeasureConstituent(
                    id="m5b",
                    number=5,
                    name="5b",
                    time_signature="2/4",
                    nominal_length=Fraction(2),
                    actual_length=Fraction(3, 2),
                    offset_within_measure=Fraction(1, 2),
                ),
            ),
        )

        with pytest.raises(ValueError, match="same nominal downbeat"):
            QuartersToFloatingMeasures.from_measure_map(measure_map)


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

    def test_a_split_bar_score_gets_no_conversion_and_no_warning(self) -> None:
        """An absent conversion never supplies wrong numbers or a silent fallback.

        WoO 71 splits bars, so its fm lattice is not expressible (see
        ``TestRefusals``). The score still loads and every other conversion is
        intact; the fm axis is simply not among its units. The limitation is
        documented on the loader rather than warned about at load time —
        nothing a caller could act on, and one warning per load of any such
        score — so the absence of a warning is asserted here rather than left
        unobserved.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            timeline = Ms3Loader.from_file(WOO71_MEASURES).create_timeline()

        assert TimeUnit.floating_measures not in timeline._get_available_units()
        assert timeline.get_conversion_map(TimeUnit.floating_measures) is None
        assert TimeUnit.ticks in timeline._get_available_units()
        # The structure itself is intact: only the derived lattice is absent.
        assert len(timeline.measure_map) == 397
