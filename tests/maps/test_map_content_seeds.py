"""Content seeds of the ids conversion maps mint (see README, "Content seeds")."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

import timetoalign.maps
from timetoalign.core.ids import IdMinter, reset_content_minters
from timetoalign.maps import (
    ChainMap,
    CombinationMap,
    ConstantMap,
    FloatingMeasureMap,
    FloorMap,
    InterpolationMap,
    IntervalToConstantMap,
    LinearMap,
    MeasureBeatMap,
    MeasureNumberMap,
    MeasureRecordMap,
    MetricalPositionMap,
    MetricMap,
    PiecewiseMap,
    QuartersToFloatingMeasures,
    QuartersToMeasureNumber,
    QuartersToTicks,
    RotationMap,
    SamplesToSeconds,
    ScalarMap,
    SecondsToSamples,
    ShiftMap,
    TableMap,
    TicksToQuarters,
)
from timetoalign.maps.base import ConversionMap
from timetoalign.maps.meter import BeatInMeasureMap
from timetoalign.timelines import MeasureMap

Factory = Callable[[], ConversionMap[Any]]


def _bars(time_signature: str) -> MeasureMap:
    """A measure map of four bars, its id given."""
    return MeasureMap.from_rows(
        [{"count": 1, "time_signature": time_signature}, {"count": 4}],
        id=f"bars {time_signature}",
    )


def _meter(length: int, uid: str) -> MetricMap:
    """Four uniform bars of *length* quarters, the id given."""
    return MetricMap(
        starts=[Fraction(length * n) for n in range(4)],
        mcs=[1, 2, 3, 4],
        mns=["1", "2", "3", "4"],
        lengths=[Fraction(length)] * 4,
        uid=uid,
    )


def _linear(scalar: int, uid: str) -> LinearMap:
    return LinearMap(scalar=scalar, uid=uid)


#: For each concrete class, two maps differing in one defining parameter.
PAIRS: dict[type[ConversionMap[Any]], tuple[Factory, Factory]] = {
    LinearMap: (lambda: LinearMap(scalar=2), lambda: LinearMap(scalar=3)),
    ScalarMap: (lambda: ScalarMap(2), lambda: ScalarMap(3)),
    ShiftMap: (lambda: ShiftMap(offset=2), lambda: ShiftMap(offset=3)),
    TicksToQuarters: (
        lambda: TicksToQuarters(ppq=480),
        lambda: TicksToQuarters(ppq=960),
    ),
    QuartersToTicks: (
        lambda: QuartersToTicks(ppq=480),
        lambda: QuartersToTicks(ppq=960),
    ),
    SamplesToSeconds: (
        lambda: SamplesToSeconds(sample_rate=44100),
        lambda: SamplesToSeconds(sample_rate=48000),
    ),
    SecondsToSamples: (
        lambda: SecondsToSamples(sample_rate=44100),
        lambda: SecondsToSamples(sample_rate=48000),
    ),
    ConstantMap: (lambda: ConstantMap(value=1), lambda: ConstantMap(value=2)),
    RotationMap: (lambda: RotationMap(4.0), lambda: RotationMap(3.0)),
    FloorMap: (lambda: FloorMap(4.0), lambda: FloorMap(3.0)),
    TableMap: (
        lambda: TableMap(x_values=[0, 1, 2], y_values=[0, 2, 4]),
        lambda: TableMap(x_values=[0, 1, 2], y_values=[0, 2, 4], kind="previous"),
    ),
    InterpolationMap: (
        lambda: InterpolationMap(
            source_coords=[0.0, 1.0],
            target_coords=[0.0, 2.0],
            source_id="a",
            target_id="b",
        ),
        lambda: InterpolationMap(
            source_coords=[0.0, 1.0],
            target_coords=[0.0, 3.0],
            source_id="a",
            target_id="b",
        ),
    ),
    ChainMap: (
        lambda: ChainMap([_linear(2, "two"), _linear(3, "three")]),
        lambda: ChainMap([_linear(2, "two"), _linear(5, "five")]),
    ),
    PiecewiseMap: (
        lambda: PiecewiseMap(breaks=[0, 4], maps=[_linear(2, "two")]),
        lambda: PiecewiseMap(breaks=[0, 8], maps=[_linear(2, "two")]),
    ),
    CombinationMap: (
        lambda: CombinationMap({"x": _linear(2, "two")}),
        lambda: CombinationMap({"y": _linear(2, "two")}),
    ),
    IntervalToConstantMap: (
        lambda: IntervalToConstantMap(boundaries=[0, 4], values=["A", "B"]),
        lambda: IntervalToConstantMap(boundaries=[0, 6], values=["A", "B"]),
    ),
    QuartersToMeasureNumber: (
        lambda: QuartersToMeasureNumber(boundaries=[0, 4], mns=["1", "2"]),
        lambda: QuartersToMeasureNumber(boundaries=[0, 3], mns=["1", "2"]),
    ),
    QuartersToFloatingMeasures: (
        lambda: QuartersToFloatingMeasures(
            x_values=[0, 4, 8], y_values=[1.0, 2.0, 3.0]
        ),
        lambda: QuartersToFloatingMeasures(
            x_values=[0, 3, 6], y_values=[1.0, 2.0, 3.0]
        ),
    ),
    MetricMap: (
        lambda: MetricMap.from_uniform(4, Fraction(4)),
        lambda: MetricMap.from_uniform(4, Fraction(3)),
    ),
    BeatInMeasureMap: (
        lambda: BeatInMeasureMap(_meter(4, "four")),
        lambda: BeatInMeasureMap(_meter(3, "three")),
    ),
    MetricalPositionMap: (
        lambda: MetricalPositionMap(_meter(4, "four"), beat_map_id="beats"),
        lambda: MetricalPositionMap(_meter(3, "three"), beat_map_id="beats"),
    ),
    MeasureRecordMap: (
        lambda: MeasureRecordMap(_bars("4/4")),
        lambda: MeasureRecordMap(_bars("3/4")),
    ),
    MeasureNumberMap: (
        lambda: MeasureNumberMap(_bars("4/4")),
        lambda: MeasureNumberMap(_bars("3/4")),
    ),
    MeasureBeatMap: (
        lambda: MeasureBeatMap(_bars("4/4")),
        lambda: MeasureBeatMap(_bars("3/4")),
    ),
    FloatingMeasureMap: (
        lambda: FloatingMeasureMap(_bars("4/4")),
        lambda: FloatingMeasureMap(_bars("3/4")),
    ),
}


def _concrete_map_classes() -> set[type[ConversionMap[Any]]]:
    """Every concrete ConversionMap class the library defines."""
    for module in pkgutil.iter_modules(timetoalign.maps.__path__):
        importlib.import_module(f"timetoalign.maps.{module.name}")
    found: set[type[ConversionMap[Any]]] = set()
    pending = list(ConversionMap.__subclasses__())
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        if cls.__module__.startswith("timetoalign.") and not inspect.isabstract(cls):
            found.add(cls)
    return found


def _ids_built(*factories: Factory) -> list[str]:
    """The ids of the maps *factories* build, in order, in a fresh run."""
    reset_content_minters()
    return [factory().id for factory in factories]


def test_every_concrete_class_has_a_pair() -> None:
    assert {cls.__name__ for cls in _concrete_map_classes()} == {
        cls.__name__ for cls in PAIRS
    }


@pytest.mark.parametrize("cls", list(PAIRS), ids=lambda cls: cls.__name__)
def test_one_defining_parameter_seeds_the_id(cls: type[ConversionMap[Any]]) -> None:
    first, second = PAIRS[cls]
    assert type(first()) is cls and type(second()) is cls
    first_alone, second_after = _ids_built(first, second)
    second_alone, first_after = _ids_built(second, first)
    assert first_alone == first_after
    assert second_alone == second_after
    assert first_alone != second_alone


def test_numpy_scalars_and_arrays_seed_as_python_floats() -> None:
    values = [0.0, 1.5, 3.0]
    (from_floats,) = _ids_built(lambda: TableMap(x_values=values, y_values=values))
    (from_numpy_scalars,) = _ids_built(
        lambda: TableMap(
            x_values=[np.float64(v) for v in values],
            y_values=[np.float64(v) for v in values],
        )
    )
    (from_arrays,) = _ids_built(
        lambda: TableMap(x_values=np.array(values), y_values=np.array(values))
    )
    assert from_numpy_scalars == from_floats
    assert from_arrays == from_floats


def test_seed_mints_a_metrical_position_map_and_its_beat_map_in_order() -> None:
    meter = _meter(4, "four")
    reset_content_minters()
    LinearMap(scalar=7)  # something ran before: the seeded ids do not change
    position = MetricalPositionMap(meter, seed=11)
    minter = IdMinter(11)
    assert [position.id, position.maps["beat"].id] == [
        minter.mint("cmap"),
        minter.mint("cmap"),
    ]


def test_seed_mints_an_interval_map_and_its_pieces_in_order() -> None:
    reset_content_minters()
    ConstantMap(value="A")  # something ran before: the seeded ids do not change
    sections = IntervalToConstantMap(
        boundaries=[0, 4], values=["A", "B"], end=8, seed=11
    )
    minter = IdMinter(11)
    assert [sections.id, *(piece.id for piece in sections._nested_maps())] == [
        minter.mint("cmap"),
        minter.mint("cmap"),
        minter.mint("cmap"),
    ]
