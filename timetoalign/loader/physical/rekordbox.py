"""Load Rekordbox XML collections as structured physical timelines."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from timetoalign.alignment import AlignmentBundle
from timetoalign.core.ids import CMAP_PREFIX, MEASURE_MAP_PREFIX
from timetoalign.loader._sources import Source, describe_source, readable
from timetoalign.loader.base import Loader

if TYPE_CHECKING:
    from timetoalign.timelines import BeatGrid, BeatGridSegment, MeasureMap


@dataclass(frozen=True)
class RekordboxTempo:
    """One Rekordbox beat-grid declaration."""

    inizio: Fraction
    bpm: Fraction
    metro: str
    battito: int

    @property
    def numerator(self) -> int:
        """Return the meter numerator."""
        return int(self.metro.split("/", maxsplit=1)[0])

    @property
    def denominator(self) -> int:
        """Return the meter denominator."""
        return int(self.metro.split("/", maxsplit=1)[1])


@dataclass(frozen=True)
class RekordboxTrack:
    """Metadata and grids parsed from one collection track."""

    track_id: str
    name: str
    total_time: Fraction
    average_bpm: float | None
    sample_rate: int | None
    location: str | None
    tempos: tuple[RekordboxTempo, ...]
    position_marks: tuple[dict[str, str], ...]


class RekordboxLoader(Loader[list[RekordboxTrack]]):
    """Parse the collection tracks in a Rekordbox XML export."""

    def __init__(self, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._tracks: list[RekordboxTrack] = []

    @property
    def tracks(self) -> tuple[RekordboxTrack, ...]:
        """Return collection tracks in XML order."""
        return tuple(self._tracks)

    def clear(self) -> None:
        """Clear parsed tracks and shared source state."""
        super().clear()
        self._tracks.clear()

    def _load_source(self, source: Source) -> list[RekordboxTrack]:
        with readable(source) as document:
            root = ET.parse(document).getroot()
        collection = root if root.tag == "COLLECTION" else root.find("COLLECTION")
        if collection is None:
            raise ValueError(
                f"Rekordbox XML {describe_source(source)} has no COLLECTION element"
            )
        return [self._parse_track(element) for element in collection.findall("TRACK")]

    def _accept_source(
        self,
        path: Source,
        source_meta: dict[str, Any],
        payload: list[RekordboxTrack],
    ) -> None:
        super()._accept_source(path, source_meta, payload)
        self._tracks.extend(payload)

    @staticmethod
    def _parse_track(element: ET.Element) -> RekordboxTrack:
        attributes = element.attrib
        try:
            track_id = attributes["TrackID"]
            name = attributes["Name"]
            total_time = Fraction(attributes["TotalTime"])
        except KeyError as exc:
            raise ValueError(f"Collection TRACK is missing {exc.args[0]}") from exc

        tempos = tuple(
            RekordboxTempo(
                inizio=Fraction(child.attrib["Inizio"]),
                bpm=Fraction(child.attrib["Bpm"]),
                metro=child.attrib["Metro"],
                battito=int(child.attrib["Battito"]),
            )
            for child in element.findall("TEMPO")
        )
        if not tempos:
            raise ValueError(f"Rekordbox track {name!r} has no TEMPO grid")
        tempos = tuple(sorted(tempos, key=lambda tempo: tempo.inizio))
        for tempo in tempos:
            if tempo.bpm <= 0:
                raise ValueError(f"Rekordbox track {name!r} has non-positive BPM")
            if not 1 <= tempo.battito <= tempo.numerator:
                raise ValueError(
                    f"Rekordbox track {name!r} has Battito {tempo.battito} "
                    f"outside meter {tempo.metro}"
                )

        average_bpm = attributes.get("AverageBpm")
        sample_rate = attributes.get("SampleRate")
        return RekordboxTrack(
            track_id=track_id,
            name=name,
            total_time=total_time,
            average_bpm=float(average_bpm) if average_bpm is not None else None,
            sample_rate=int(sample_rate) if sample_rate is not None else None,
            location=attributes.get("Location"),
            tempos=tempos,
            position_marks=tuple(
                dict(child.attrib) for child in element.findall("POSITION_MARK")
            ),
        )

    def create_timelines(self) -> list[Any]:
        """Create and structure one seconds timeline per collection track."""
        return [self._create_track_timeline(track) for track in self._loaded_tracks()]

    def create_timeline(self) -> Any:
        """Create the only loaded track, rejecting ambiguous collections."""
        return self._create_track_timeline(self._only_track("create_timeline"))

    def create_beatgrids(self) -> list[BeatGrid]:
        """Create one beat grid per collection track, in XML order.

        Each grid holds the track's ``TEMPO`` declarations as segments,
        bounded by its ``TotalTime``; it generated the track's measures
        and its floating-measure anchors.
        """
        return [self._beat_grid(track) for track in self._loaded_tracks()]

    def create_beatgrid(self) -> BeatGrid:
        """Create the only loaded track's beat grid."""
        return self._beat_grid(self._only_track("create_beatgrid"))

    def create_measure_maps(self) -> list[MeasureMap]:
        """Create one measure map per collection track, in XML order.

        The measures are anchored in quarter notes, which a seconds
        timeline cannot carry, so the map is returned here instead of
        being attached to the track's timeline.

        Raises:
            ValueError: If a track's grids reach no downbeat.
        """
        return [
            self._track_measures(track, self._id_minter.mint(MEASURE_MAP_PREFIX))[1]
            for track in self._loaded_tracks()
        ]

    def create_measure_map(self) -> MeasureMap:
        """Create the only loaded track's measure map.

        Raises:
            ValueError: If the track's grids reach no downbeat.
        """
        return self._track_measures(
            self._only_track("create_measure_map"),
            self._id_minter.mint(MEASURE_MAP_PREFIX),
        )[1]

    def create_bundle(self) -> AlignmentBundle:
        """Create an alignment bundle containing every collection track."""

        bundle = AlignmentBundle()
        for timeline in self.create_timelines():
            bundle.add_timeline(timeline)
        return bundle

    def _loaded_tracks(self) -> list[RekordboxTrack]:
        if not self._tracks:
            raise RuntimeError("No Rekordbox collection loaded. Call load() first.")
        return self._tracks

    def _only_track(self, method: str) -> RekordboxTrack:
        if len(self._tracks) != 1:
            available = ", ".join(repr(track.name) for track in self._tracks) or "none"
            raise ValueError(
                f"{method}() requires exactly one Rekordbox collection track; "
                f"available names: {available}"
            )
        return self._tracks[0]

    @classmethod
    def _track_measures(
        cls, track: RekordboxTrack, measure_map_id: str | None = None
    ) -> tuple[BeatGrid, MeasureMap, list[tuple[Fraction, BeatGridSegment]]]:
        """Return the track's grid, the measure map it generates, and its downbeats.

        The measure map takes *measure_map_id* (the loader mints one for a
        map it hands out); without one it mints its own.
        """
        from timetoalign.timelines import MeasureMap

        grid = cls._beat_grid(track)
        downbeats = [
            (beat.seconds, grid.segments[beat.segment])
            for beat in grid.iter_beats()
            if beat.is_downbeat
        ]
        measure_map = MeasureMap(cls._measure_map(grid, downbeats), id=measure_map_id)
        return grid, measure_map, downbeats

    def _create_track_timeline(self, track: RekordboxTrack) -> Any:
        from timetoalign.core import NumberType, TimeUnit
        from timetoalign.maps import SecondsToSamples, TableMap
        from timetoalign.timelines import ContinuousPhysicalTimeline
        from timetoalign.timelines.factory import mint_timeline_id

        grid, measure_map, downbeats = self._track_measures(track)
        seconds, floating_measures = self._conversion_anchors(
            track, grid, downbeats, measure_map
        )

        timeline = ContinuousPhysicalTimeline(
            length=float(track.total_time),
            unit=TimeUnit.seconds,
            number_type=NumberType.float,
            uid=mint_timeline_id(TimeUnit.seconds, self._id_minter),
            name=track.name,
            meta={
                "TrackID": track.track_id,
                "AverageBpm": track.average_bpm,
                "Location": track.location,
                "SampleRate": track.sample_rate,
                "POSITION_MARK": [dict(mark) for mark in track.position_marks],
            },
        )
        timeline.add_conversion_map(
            TableMap(
                x_values=seconds,
                y_values=floating_measures,
                source_unit=TimeUnit.seconds,
                target_unit=TimeUnit.floating_measures,
                name="rekordbox_floating_measures",
                uid=self._id_minter.mint(CMAP_PREFIX),
            )
        )
        if track.sample_rate is not None:
            timeline.add_conversion_map(
                SecondsToSamples(
                    sample_rate=track.sample_rate,
                    uid=self._id_minter.mint(CMAP_PREFIX),
                )
            )
        return timeline

    @staticmethod
    def _beat_grid(track: RekordboxTrack) -> BeatGrid:
        """Build the track's beat grid from its ``TEMPO`` declarations.

        Each grid counts one beat per ``Metro`` numerator unit, which is
        what ``Battito`` indexes; reading ``6/8`` as two dotted beats
        would put the anchor index outside its own bar.
        """
        from timetoalign.core import BeatPolicy
        from timetoalign.timelines import BeatGrid, BeatGridSegment

        segments = [
            BeatGridSegment(
                start=tempo.inizio,
                bpm=tempo.bpm,
                policy=BeatPolicy.from_time_signature(tempo.metro).as_divisions(),
                battito=tempo.battito,
            )
            for tempo in track.tempos
        ]
        return BeatGrid(segments, extent=track.total_time)

    @staticmethod
    def _measure_map(
        grid: BeatGrid, downbeats: list[tuple[Fraction, BeatGridSegment]]
    ) -> list[Any]:
        from timetoalign.core import Measure

        measures: list[Any] = []
        first = grid.segments[0]
        if first.battito != 1:
            offset = first.policy.offset_for(first.battito)
            measures.append(
                Measure.from_values(
                    number=0,
                    time_signature=first.policy.name,
                    nominal_length=first.policy.span,
                    actual_length=first.policy.span - offset,
                    offset_within_measure=offset,
                )
            )

        extent = grid.extent
        assert extent is not None
        for index, (downbeat, segment) in enumerate(downbeats):
            next_downbeat = (
                downbeats[index + 1][0] if index + 1 < len(downbeats) else extent
            )
            actual_length = grid.quarters_between(downbeat, next_downbeat)
            measures.append(
                Measure.from_values(
                    number=index + 1,
                    time_signature=segment.policy.name,
                    nominal_length=segment.policy.span,
                    actual_length=actual_length,
                )
            )

        if not measures:
            raise ValueError("A Rekordbox track's grids state no measures")
        return measures

    @staticmethod
    def _conversion_anchors(
        track: RekordboxTrack,
        grid: BeatGrid,
        downbeats: list[tuple[Fraction, BeatGridSegment]],
        measure_map: MeasureMap,
    ) -> tuple[list[Fraction], list[Fraction]]:
        from timetoalign.maps import QuartersToFloatingMeasures

        downbeat_instants = [instant for instant, _ in downbeats]
        anchors: dict[Fraction, Fraction] = {}

        def anchor_of(segment: Any) -> Fraction:
            """Read a grid anchor as a bar ordinal plus its beat offset."""
            bar_number = bisect_right(downbeat_instants, segment.start)
            return Fraction(bar_number) + Fraction(
                segment.battito - 1, segment.policy.n_beats
            )

        first = grid.segments[0]
        anchors[Fraction(0)] = anchor_of(first) - first.start / first.bar_seconds

        for segment in grid.segments:
            anchors[segment.start] = anchor_of(segment)

        for index, instant in enumerate(downbeat_instants, start=1):
            anchors[instant] = Fraction(index)

        # The bars come from a beat grid, not from notation: a tempo change
        # inside a bar leaves it a few thousandths over or under its
        # nominal length, which is no doubt about the source, so the lattice
        # is built without the warning that lists such bars.
        x_values, y_values = QuartersToFloatingMeasures._lattice(
            measure_map, warn=False
        )
        canonical = QuartersToFloatingMeasures(x_values=x_values, y_values=y_values)
        total_quarters = measure_map.total_actual_length
        assert total_quarters is not None
        anchors[track.total_time] = Fraction(str(canonical(total_quarters)))

        result = sorted(anchors.items())
        return [item[0] for item in result], [item[1] for item in result]
