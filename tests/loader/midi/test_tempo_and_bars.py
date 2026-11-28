"""The tempo map and the bars a MIDI file states, on its ticks timeline."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import mido
import pytest

from timetoalign.core import Coordinate, TimeUnit
from timetoalign.loader.midi import PerformanceMidiLoader, ScoreMidiLoader
from timetoalign.loader.midi.base import MidiLoader
from timetoalign.loader.midi.store import MidiStore
from timetoalign.maps import TableMap
from timetoalign.timelines import Timeline, TimelineGroup

LOADERS: list[Callable[[], MidiLoader]] = [
    lambda: ScoreMidiLoader(),
    lambda: ScoreMidiLoader(parser="mido"),
    lambda: PerformanceMidiLoader(),
]
LOADER_IDS = ["partitura", "mido", "performance"]


def _write(
    path: Path,
    end: int,
    *,
    tempi: tuple[tuple[int, int], ...] = (),
    signatures: tuple[tuple[int, int, int], ...] = (),
) -> Path:
    """A type-1 file at 480 ticks per quarter: a meta track and a one-note track.

    The meta track holds the given ``set_tempo`` and ``time_signature``
    messages at their absolute ticks and ends at *end*; the note track's
    note sounds from 0 to 480 and the track ends there.
    """
    messages = [
        (tick, mido.MetaMessage("set_tempo", tempo=tempo)) for tick, tempo in tempi
    ]
    messages += [
        (tick, mido.MetaMessage("time_signature", numerator=num, denominator=den))
        for tick, num, den in signatures
    ]
    messages.sort(key=lambda pair: pair[0])
    meta = mido.MidiTrack()
    now = 0
    for tick, message in messages:
        meta.append(message.copy(time=tick - now))
        now = tick
    meta.append(mido.MetaMessage("end_of_track", time=end - now))
    notes = mido.MidiTrack(
        [
            mido.Message("note_on", note=60, velocity=64, time=0),
            mido.Message("note_off", note=60, velocity=0, time=480),
            mido.MetaMessage("end_of_track", time=0),
        ]
    )
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks += [meta, notes]
    midi.save(path)
    return path


def _timeline(make: Callable[[], MidiLoader], path: Path) -> Timeline:
    return make().load(path).create_timeline()


def _tempo_map(timeline: Timeline) -> TableMap:
    tempo_map = timeline.get_conversion_map(TimeUnit.seconds)
    assert isinstance(tempo_map, TableMap)
    return tempo_map


def _knots(tempo_map: TableMap) -> list[tuple[int, F]]:
    """The map's anchors as written: ``to_dict`` reads them back exactly.

    Every seconds value must be written as a Fraction pair of two ints; a
    float would compare equal to a dyadic Fraction and pass unnoticed.
    """
    form = tempo_map.to_dict()
    for y in form["y_values"]:
        assert isinstance(y, list) and [type(part) for part in y] == [int, int], y
    return [(x, F(*y)) for x, y in zip(form["x_values"], form["y_values"], strict=True)]


def _rows(timeline: Timeline) -> list[tuple[Any, ...]]:
    measure_map = timeline.measure_map
    assert measure_map is not None
    return [
        (
            bar.count,
            bar.qstamp,
            bar.time_signature,
            bar.nominal_length,
            bar.actual_length,
        )
        for bar in measure_map
    ]


@pytest.mark.parametrize("make", LOADERS, ids=LOADER_IDS)
class TestStatedFacts:
    """Each source's metadata records the tempo changes and signatures as written."""

    def test_same_tick_later_message_replaces(
        self, make: Callable[[], MidiLoader], tmp_path: Path
    ) -> None:
        path = tmp_path / "same_tick.mid"
        meta = mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=600_000, time=0),
                mido.MetaMessage("set_tempo", tempo=400_000, time=0),
                mido.MetaMessage(
                    "time_signature", numerator=3, denominator=4, time=960
                ),
                mido.MetaMessage("time_signature", numerator=6, denominator=8, time=0),
                mido.MetaMessage("end_of_track", time=960),
            ]
        )
        # The note sounds past the signatures: Partitura's score parse
        # refuses signatures after the last note's end.
        notes = mido.MidiTrack(
            [
                mido.Message("note_on", note=60, velocity=64, time=0),
                mido.MetaMessage("set_tempo", tempo=300_000, time=960),
                mido.Message("note_off", note=60, velocity=0, time=960),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
        midi = mido.MidiFile(type=1, ticks_per_beat=480)
        midi.tracks += [meta, notes]
        midi.save(path)
        source = make().load(path).metadata["sources"][0]
        assert source["tempo_events"] == [(0, 400_000), (960, 300_000)]
        assert source["time_signatures"] == [(960, 6, 8)]
        assert all(
            type(number) is int
            for fact in (*source["tempo_events"], *source["time_signatures"])
            for number in fact
        )


@pytest.mark.parametrize("make", LOADERS, ids=LOADER_IDS)
class TestTempoMap:
    """Ticks to seconds through the file's tempo map, exact."""

    def test_two_tempi(self, make: Callable[[], MidiLoader], tmp_path: Path) -> None:
        path = _write(tmp_path / "two.mid", 1920, tempi=((0, 500_000), (960, 250_000)))
        timeline = _timeline(make, path)
        tempo_map = _tempo_map(timeline)
        assert _knots(tempo_map) == [(0, 0), (960, 1), (1920, F(3, 2))]
        assert (tempo_map.source_unit, tempo_map.kind.value) == (
            TimeUnit.ticks,
            "linear",
        )
        assert tempo_map(480) == 0.5
        assert tempo_map(1440) == 1.25
        assert tempo_map(2400) == 1.75
        stamp = timeline.get_timestamp_at(1440)
        exact = stamp.get_unit(TimeUnit.seconds, format="fraction")
        assert exact == F(5, 4) and type(exact) is F
        # Off the binary grid: 160 * 500000 / (480 * 10**6) = 1/6, which no float
        # equals. The map computes it exactly; the seconds axis writes it as a float.
        exact = tempo_map._evaluate(160)
        assert exact == F(1, 6) and type(exact) is F
        stamp = timeline.get_timestamp_at(160)
        written = stamp.get_unit(TimeUnit.seconds, format="fraction")
        assert written == F(1 / 6) and type(written) is F

    def test_default_tempo(
        self, make: Callable[[], MidiLoader], tmp_path: Path
    ) -> None:
        tempo_map = _tempo_map(_timeline(make, _write(tmp_path / "none.mid", 960)))
        assert _knots(tempo_map) == [(0, 0), (960, 1)]

    def test_quarters_map(self, make: Callable[[], MidiLoader], tmp_path: Path) -> None:
        timeline = _timeline(make, _write(tmp_path / "none.mid", 960))
        quarters = timeline.get_conversion_map(TimeUnit.quarters)
        assert quarters is not None
        assert quarters(240) == F(1, 2) and type(quarters(240)) is F

    def test_no_maps_on_request(
        self, make: Callable[[], MidiLoader], tmp_path: Path
    ) -> None:
        loader = make().load(_write(tmp_path / "none.mid", 960))
        timeline = loader.create_timeline(attach_cmaps=False)
        assert (timeline.n_conversion_maps, timeline.measure_map) == (0, None)


@pytest.mark.parametrize("make", LOADERS, ids=LOADER_IDS)
class TestBars:
    """The measure map of the file's time signatures and end of track."""

    def test_change_inside_a_bar(
        self, make: Callable[[], MidiLoader], tmp_path: Path
    ) -> None:
        path = _write(
            tmp_path / "bars.mid",
            4800,
            signatures=((0, 4, 4), (1920, 3, 4), (2880, 2, 4)),
        )
        assert _rows(_timeline(make, path)) == [
            (1, 0, "4/4", 4, 4),
            (2, 4, "3/4", 3, 2),
            (3, 6, "2/4", 2, 2),
            (4, 8, "2/4", 2, 2),
        ]

    def test_default_signature_and_end(
        self, make: Callable[[], MidiLoader], tmp_path: Path
    ) -> None:
        plain = _write(tmp_path / "plain.mid", 2640)
        at_end = _write(tmp_path / "at_end.mid", 2640, signatures=((2640, 3, 4),))
        expected = [(1, 0, "4/4", 4, 4), (2, 4, "4/4", 4, F(3, 2))]
        assert _rows(_timeline(make, plain)) == expected
        assert _rows(_timeline(make, at_end)) == expected

    def test_signatures_as_written(
        self, make: Callable[[], MidiLoader], tmp_path: Path
    ) -> None:
        path = _write(
            tmp_path / "odd.mid", 3720, signatures=((0, 6, 8), (1440, 19, 16))
        )
        assert _rows(_timeline(make, path)) == [
            (1, 0, "6/8", 3, 3),
            (2, 3, "19/16", F(19, 4), F(19, 4)),
        ]

    def test_round_trip(self, make: Callable[[], MidiLoader], tmp_path: Path) -> None:
        path = _write(
            tmp_path / "bars.mid",
            4800,
            tempi=((0, 500_000), (960, 250_000)),
            signatures=((0, 4, 4), (1920, 3, 4), (2880, 2, 4)),
        )
        timeline = _timeline(make, path)
        written = timeline.to_dict()
        (entry,) = written["measure_maps"]
        chains = [
            cmap
            for cmap in written["timelines"][0]["conversion_maps"]
            if cmap["type"] == "ChainMap"
        ]
        assert [member["type"] for member in chains[0]["maps"]] == [
            "LinearMap",
            "MeasureRecordMap",
        ]
        assert chains[0]["maps"][0]["scalar"] == [1, 480]
        assert chains[0]["maps"][1]["measure_map"] == entry["id"]
        restored = Timeline.from_dict(json.loads(json.dumps(written)))
        assert restored.measure_map == timeline.measure_map
        assert restored.to_dict() == written
        assert _knots(_tempo_map(restored)) == [(0, 0), (960, 1), (4800, 3)]


class TestSpecimens:
    """Real files, checked against values read from their bytes."""

    @pytest.mark.parametrize("make", LOADERS, ids=LOADER_IDS)
    def test_beethoven_quartet_tempo_map(
        self, make: Callable[[], MidiLoader], beethoven_score_path: Path
    ) -> None:
        path = beethoven_score_path
        tempo_map = _tempo_map(_timeline(make, path))
        knots = _knots(tempo_map)
        # The set_tempo ticks and values of the file, read off a mido scan.
        ticks = [0, 15360, 37440, 37920, 52800, 53280, 191040, 191520, 306240]
        ticks += [306720, 340320, 340560, 430560, 532800, 534240]
        tempi = [352941, 352941, 1058823, 352941, 1058823, 352941, 1058823]
        tempi += [352941, 1058823, 352941, 1058823, 352941, 300000, 450000]
        assert [tick for tick, _ in knots] == ticks
        slopes = [
            (y1 - y0) / (x1 - x0)
            for (x0, y0), (x1, y1) in zip(knots, knots[1:], strict=False)
        ]
        assert slopes == [F(tempo, 480 * 10**6) for tempo in tempi]

    @pytest.mark.parametrize("make", LOADERS, ids=LOADER_IDS)
    def test_beethoven_quartet_bars(
        self, make: Callable[[], MidiLoader], beethoven_score_path: Path
    ) -> None:
        path = beethoven_score_path
        timeline = _timeline(make, path)
        rows = _rows(timeline)
        # Every signature segment of the file holds whole bars (hand count: 290).
        assert len(rows) == 290
        assert all(row[3] == row[4] for row in rows)
        # The last of 1/4, 2/2, 1/4 at tick 30,720 (quarter 64) holds.
        assert rows[18] == (19, 64, "1/4", 1, 1)
        assert rows[120] == (121, F(897, 2), "2/16", F(1, 2), F(1, 2))
        # The 3/4 at the end of track (quarter 1,113) adds no bar.
        assert rows[-1] == (290, 1109, "2/2", 4, 4)

    @pytest.mark.parametrize("make", LOADERS, ids=LOADER_IDS)
    def test_beethoven_theme(
        self, make: Callable[[], MidiLoader], score_midi_dir: Path
    ) -> None:
        timeline = _timeline(make, score_midi_dir / "beethoven_mtd.mid")
        assert _rows(timeline) == [
            (1, 0, "1/8", F(1, 2), F(1, 2)),
            (2, F(1, 2), "4/4", 4, 4),
            (3, F(9, 2), "4/4", 4, 4),
            (4, F(17, 2), "4/4", 4, 4),
            (5, F(25, 2), "4/4", 4, 2),
        ]
        assert _knots(_tempo_map(timeline)) == [(0, 0), (6960, F(29, 4))]
        # The end of track, one tick after the last note-off (6,959).
        assert (timeline.length, timeline.is_locked) == (
            Coordinate(6960, "ticks"),
            True,
        )


class TestSeveralSources:
    """Sources sharing one ticks axis state a fact only where they agree."""

    def test_agreeing_sources(self, tmp_path: Path) -> None:
        first = _write(tmp_path / "a.mid", 960, tempi=((0, 250_000),))
        second = _write(tmp_path / "b.mid", 1920, tempi=((0, 250_000),))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            timeline = PerformanceMidiLoader().load(first, second).create_timeline()
        assert _knots(_tempo_map(timeline)) == [(0, 0), (1920, 1)]
        assert _rows(timeline) == [(1, 0, "4/4", 4, 4)]

    @staticmethod
    def _disagreeing(tmp_path: Path) -> PerformanceMidiLoader:
        first = _write(
            tmp_path / "a.mid", 960, tempi=((0, 250_000),), signatures=((0, 4, 4),)
        )
        second = _write(
            tmp_path / "b.mid", 960, tempi=((0, 500_000),), signatures=((0, 3, 4),)
        )
        return PerformanceMidiLoader().load(first, second)

    def test_reading_a_disagreeing_loader_does_not_warn(self, tmp_path: Path) -> None:
        loader = self._disagreeing(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            repr(loader)
            len(loader)
            loader["notes"]
            store = loader.store
        assert store.metadata["disagreeing_facts"] == [
            "tempo_events",
            "time_signatures",
        ]

    def test_disagreeing_facts(self, tmp_path: Path) -> None:
        loader = self._disagreeing(tmp_path)
        with pytest.warns(UserWarning) as caught:
            timeline = loader.create_timeline()
        assert [str(warning.message) for warning in caught] == [
            "The 2 MIDI sources of this timeline state different "
            "tempo_events, time_signatures; the timeline carries none of them"
        ]
        assert caught[0].filename == __file__
        quarters = timeline.get_conversion_map(TimeUnit.quarters)
        assert quarters is not None and quarters(480) == 1
        assert timeline.get_conversion_map(TimeUnit.seconds) is None
        assert timeline.measure_map is None

    def test_no_maps_no_warning(self, tmp_path: Path) -> None:
        loader = self._disagreeing(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            timeline = loader.create_timeline(attach_cmaps=False)
        assert timeline.n_conversion_maps == 0


@pytest.mark.parametrize("make", LOADERS, ids=LOADER_IDS)
class TestLength:
    """The timeline is as long as the file's end of track, and locked."""

    @pytest.mark.parametrize("attach_cmaps", [True, False])
    def test_end_of_track_after_the_last_note(
        self, make: Callable[[], MidiLoader], tmp_path: Path, attach_cmaps: bool
    ) -> None:
        loader = make().load(_write(tmp_path / "long.mid", 1920))
        timeline = loader.create_timeline(attach_cmaps=attach_cmaps)
        assert (timeline.length, timeline.is_locked) == (
            Coordinate(1920, "ticks"),
            True,
        )

    def test_the_bars_end_with_the_timeline(
        self, make: Callable[[], MidiLoader], tmp_path: Path
    ) -> None:
        timeline = _timeline(make, _write(tmp_path / "long.mid", 1920))
        assert timeline.measure_map is not None
        assert timeline.measure_map.total_actual_length == 4

    def test_the_lock_bites(
        self, make: Callable[[], MidiLoader], tmp_path: Path
    ) -> None:
        timeline = _timeline(make, _write(tmp_path / "long.mid", 1920))
        row = {"id": "late", "event_type": "Onset", "instant": 1921}
        with pytest.raises(ValueError):
            timeline.add_events([row])
        assert timeline.length == Coordinate(1920, "ticks")

    def test_a_store_without_end_of_track(
        self, make: Callable[[], MidiLoader], tmp_path: Path
    ) -> None:
        loader = make().load(_write(tmp_path / "long.mid", 1920))
        timeline = MidiStore.from_data(loader.events).create_timeline()
        assert (timeline.length, timeline.is_locked) == (
            Coordinate(480, "ticks"),
            False,
        )


@pytest.mark.parametrize("make", LOADERS, ids=LOADER_IDS)
class TestGroupTable:
    """Two MIDI timelines in a group head every C-Map column with a name of its own."""

    @staticmethod
    def _pair(
        make: Callable[[], MidiLoader], tmp_path: Path
    ) -> tuple[Timeline, Timeline]:
        a = _timeline(make, _write(tmp_path / "a.mid", 1920, signatures=((0, 4, 4),)))
        b = _timeline(
            make,
            _write(
                tmp_path / "b.mid", 3840, tempi=((0, 250_000),), signatures=((0, 4, 4),)
            ),
        )
        return a, b

    def test_each_timeline_alone_keeps_the_plain_names(
        self, make: Callable[[], MidiLoader], tmp_path: Path
    ) -> None:
        for timeline in self._pair(make, tmp_path):
            assert timeline.measure_map is not None
            assert timeline.get_timestamp_table().column_names == [
                timeline.id,
                "ticks_to_quarters",
                "ticks_to_seconds",
            ]

    def test_the_group_table(
        self, make: Callable[[], MidiLoader], tmp_path: Path
    ) -> None:
        a, b = self._pair(make, tmp_path)
        table = TimelineGroup(id="pair", timelines=[a, b]).get_timestamp_table()
        assert table.column_names == [
            a.id,
            b.id,
            f"{a.id}:ticks_to_quarters",
            f"{a.id}:ticks_to_seconds",
            f"{b.id}:ticks_to_quarters",
            f"{b.id}:ticks_to_seconds",
        ]
        cells = {
            name: [
                (c["numerator"], c["denominator"])
                for c in table.column(name).to_pylist()
            ]
            for name in table.column_names
        }
        assert cells == {
            a.id: [(0, 1), (1920, 1)],
            b.id: [(0, 1), (3840, 1)],
            f"{a.id}:ticks_to_quarters": [(0, 1), (4, 1)],
            f"{a.id}:ticks_to_seconds": [(0, 1), (2, 1)],
            f"{b.id}:ticks_to_quarters": [(0, 1), (8, 1)],
            f"{b.id}:ticks_to_seconds": [(0, 1), (2, 1)],
        }

    def test_the_group_frame(
        self, make: Callable[[], MidiLoader], tmp_path: Path
    ) -> None:
        a, b = self._pair(make, tmp_path)
        group = TimelineGroup(id="pair", timelines=[a, b])
        frame = group.get_timestamp_table(format="dataframe", units=False)
        assert {name: list(frame[name]) for name in frame.columns} == {
            a.id: [0, 1920],
            b.id: [0, 3840],
            f"{a.id}:ticks_to_quarters": [F(0), F(4)],
            f"{a.id}:ticks_to_seconds": [0.0, 2.0],
            f"{b.id}:ticks_to_quarters": [F(0), F(8)],
            f"{b.id}:ticks_to_seconds": [0.0, 2.0],
        }
        assert list(frame.columns) == [
            a.id,
            b.id,
            f"{a.id}:ticks_to_quarters",
            f"{a.id}:ticks_to_seconds",
            f"{b.id}:ticks_to_quarters",
            f"{b.id}:ticks_to_seconds",
        ]
        assert [type(q) for q in frame[f"{b.id}:ticks_to_quarters"]] == [F, F]
        assert [type(s) for s in frame[f"{b.id}:ticks_to_seconds"]] == [float, float]
        headed = group.get_timestamp_table(format="dataframe")
        assert list(headed.columns)[2] == f"{a.id}:ticks_to_quarters (quarters)"
