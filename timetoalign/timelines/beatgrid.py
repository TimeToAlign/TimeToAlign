"""Beat grids: tempo segments that generate a labelled lattice of beats.

A :class:`BeatGridSegment` states the facts an analysis program records
about one stretch of a recording — where its anchor beat sits, how fast
it runs, how its beats group into bars, and which beat of a bar the
anchor is.  Those facts generate beats forever; a :class:`BeatGrid`
assembles the segments in time order, bounds each one by its successor,
and labels the resulting beats with measure and beat numbers.

The grid is the arithmetic engine behind measure structures read from
DJ-software exports: it answers "which second is bar 449?" and "which
bar and beat is second 4329.7?", and it integrates exact quarter-note
lengths across tempo changes.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal

from timetoalign.core import BeatPolicy, Coordinate, IdCoordinate, TimeUnit

if TYPE_CHECKING:
    import pandas as pd

#: Accepted spellings of a position or duration in seconds.
SecondsSpec = int | float | Fraction | str | Coordinate | IdCoordinate


def _as_fraction(value: Any, *, what: str) -> Fraction:
    """Read a raw number exactly.

    Floats convert through :class:`~fractions.Fraction` directly, so a
    float contributes the exact binary value it holds and nothing
    tidier; guessing a rounder ratio would claim a precision the number
    never had.

    Args:
        value: An ``int``, ``float``, ``Fraction`` or decimal string.
        what: What is being read, for the error message.

    Returns:
        The value as an exact rational.

    Raises:
        TypeError: If *value* is of a type the grid cannot read as a
            number.
        ValueError: If *value* is a string that does not spell one
            (raised by :class:`~fractions.Fraction` itself).
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError(f"{what} must be a number, got {value!r}")
    if isinstance(value, (int, float, str)):
        return Fraction(value)
    raise TypeError(f"{what} must be a number, got {type(value).__name__}")


def _as_seconds(value: SecondsSpec, *, what: str = "A position") -> Fraction:
    """Read a position on the seconds axis exactly.

    Args:
        value: A raw number, a :class:`~timetoalign.core.Coordinate` or an
            :class:`~timetoalign.core.IdCoordinate`.
        what: What is being read, for the error message.

    Returns:
        The position in seconds, as an exact rational.

    Raises:
        ValueError: If a coordinate carries a unit other than seconds.
        TypeError: If *value* is not a number the grid can read.
    """
    if isinstance(value, (Coordinate, IdCoordinate)):
        if value.unit is not TimeUnit.seconds:
            raise ValueError(
                f"{what} on a beat grid is measured in "
                f"{TimeUnit.seconds.value!r}, not {value.unit.value!r}"
            )
        return _as_fraction(value.value, what=what)
    return _as_fraction(value, what=what)


@dataclass(frozen=True)
class BeatGridSegment:
    """One tempo segment of a beat grid: an anchor beat and how it repeats.

    The segment stores only what a source states. Beat and bar lengths,
    and the first downbeat, are derived from those facts rather than
    stored beside them.

    Attributes:
        start: Seconds at which the anchor beat sounds.
        bpm: Beats per minute; one beat is one tick of the lattice.
        policy: How the lattice beats group into bars. ``policy.name``
            carries the meter as the source spells it.
        battito: The 1-based beat-in-bar index of the anchor beat.
        end: Exclusive bound in seconds, or ``None`` for a segment that
            generates beats without end.
    """

    start: Fraction
    bpm: Fraction
    policy: BeatPolicy
    battito: int
    end: Fraction | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _as_fraction(self.start, what="A grid start"))
        object.__setattr__(self, "bpm", _as_fraction(self.bpm, what="A grid tempo"))
        if self.end is not None:
            object.__setattr__(self, "end", _as_fraction(self.end, what="A grid bound"))
        if self.bpm <= 0:
            raise ValueError(f"A grid tempo must be positive, got {self.bpm}")
        if not 1 <= self.battito <= self.policy.n_beats:
            raise ValueError(
                f"Anchor beat {self.battito} is outside 1..{self.policy.n_beats} "
                f"for policy {self.policy}"
            )
        if self.end is not None and self.end <= self.start:
            raise ValueError(
                f"A grid segment starting at {self.start} cannot end at {self.end}"
            )

    def __repr__(self) -> str:
        bound = "unbounded" if self.end is None else f"end={self.end}"
        return (
            f"BeatGridSegment(start={self.start}, {bound}, bpm={self.bpm}, "
            f"policy={self.policy}, battito={self.battito})"
        )

    @property
    def beat_seconds(self) -> Fraction:
        """Length of one lattice beat, in seconds."""
        return Fraction(60) / self.bpm

    @property
    def bar_seconds(self) -> Fraction:
        """Length of one bar of this policy, in seconds."""
        return self.beat_seconds * self.policy.n_beats

    @property
    def quarters_per_second(self) -> Fraction:
        """Quarter notes sounding per second in this segment.

        One bar lasts :attr:`bar_seconds` and is notated
        ``policy.span`` quarters, so the two divide. On the one-beat-per-
        division lattices a grid export states, that is the same as
        ``division x bpm / 60``.
        """
        return self.policy.span / self.bar_seconds

    @property
    def first_downbeat(self) -> Fraction:
        """Seconds of the first beat 1 this segment generates."""
        if self.battito == 1:
            return self.start
        return self.start + (self.policy.n_beats - self.battito + 1) * self.beat_seconds


@dataclass(frozen=True)
class GridBeat:
    """One beat of a grid, labelled in both numbering schemes.

    Attributes:
        seconds: Where the beat sounds, as the exact ratio the grid's
            arithmetic produced. This is what the grid computes with and
            what a caller integrating tempi or deriving anchors reads.
        segment: 0-based index of the segment that generated it.
        measure: Measure number counted across the whole grid. Beats
            before the grid's first downbeat carry ``0``.
        segment_measure: Measure number restarted at each segment. Beats
            before the segment's first downbeat carry ``0``.
        beat: 1-based beat-in-measure index under its segment's policy.
        instant: Derived from :attr:`seconds`: the same position as a
            float-typed seconds :class:`~timetoalign.core.Coordinate`,
            built as :meth:`BeatGrid.seconds_at` builds its answer, so
            the two directions of the grid compare equal.
        is_downbeat: Derived: whether this beat opens a measure, that is
            whether :attr:`beat` is ``1``.
    """

    seconds: Fraction
    segment: int
    measure: int
    segment_measure: int
    beat: int

    @property
    def instant(self) -> Coordinate:
        """Where the beat sounds, as a published seconds coordinate.

        The seconds axis is float-declared, and this is built exactly as
        :meth:`BeatGrid.seconds_at` builds its answer, so the two
        directions of the grid compare equal whenever they name the same
        moment. Exactness stays in :attr:`seconds`, which is where the
        grid does its arithmetic.
        """
        return Coordinate(float(self.seconds), TimeUnit.seconds)

    @property
    def is_downbeat(self) -> bool:
        """Whether this beat opens a measure."""
        return self.beat == 1


class BeatGrid:
    """A sequence of tempo segments and the beats they generate.

    Segments are given as they are stated — typically without an end,
    each one generating beats forever. The grid puts them in time order
    and bounds each by the next one's start; the last segment is bounded
    by *extent*, or generates without end when no extent is given.

    Examples:
        >>> from fractions import Fraction
        >>> grid = BeatGrid.from_tempo(120, extent=8)
        >>> [str(beat.seconds) for beat in grid.iter_beats()][:5]
        ['0', '1/2', '1', '3/2', '2']
        >>> grid.seconds_at(2)
        Coordinate(2.0, seconds)
        >>> grid.position_at(2.75).instant
        Coordinate(2.5, seconds)
        >>> grid.position_at(2.75).instant == grid.seconds_at(2, 2)
        True
    """

    @classmethod
    def from_tempo(
        cls,
        bpm: int | float | Fraction | str,
        *,
        metro: str = "4/4",
        start: SecondsSpec = 0,
        battito: int = 1,
        extent: SecondsSpec | None = None,
        policy: BeatPolicy | None = None,
    ) -> BeatGrid:
        """Build a single-segment grid from one tempo statement.

        Args:
            bpm: Beats per minute of the lattice.
            metro: Meter as ``"n/d"``, read as ``n`` beats of ``4/d``
                quarters. Ignored when *policy* is given.
            start: Seconds of the anchor beat.
            battito: 1-based beat-in-bar index of the anchor beat.
            extent: Exclusive bound in seconds. ``None`` leaves the grid
                unbounded, generating beats without end.
            policy: An explicit counting policy, overriding *metro*.

        Returns:
            The one-segment grid.
        """
        segment = BeatGridSegment(
            start=_as_seconds(start, what="A grid start"),
            bpm=_as_fraction(bpm, what="A grid tempo"),
            policy=(
                policy
                if policy is not None
                else BeatPolicy.from_time_signature(metro).as_divisions()
            ),
            battito=battito,
        )
        bound = None if extent is None else _as_seconds(extent, what="A grid extent")
        return cls([segment], extent=bound)

    def __init__(
        self,
        segments: Iterable[BeatGridSegment],
        *,
        extent: SecondsSpec | None = None,
    ) -> None:
        """Assemble tempo segments into one grid.

        **Segments abut**: one runs until the next one opens, and the
        last until the grid's extent. A segment stating no ``end`` is
        bounded accordingly; a segment stating one must state that same
        bound, because a shorter or longer end would leave a gap or an
        overlap the lattice has no way to represent.

        Args:
            segments: The segments, in any order. An ``end`` of ``None``
                is filled with the successor's ``start``, or with
                *extent* for the last segment.
            extent: Exclusive bound of the whole grid, in seconds.
                ``None`` leaves the grid unbounded.

        Raises:
            ValueError: If no segments are given, if two segments share a
                start, if a segment states an ``end`` other than the one
                the grid bounds it at, or if *extent* does not lie after
                the last segment's start.
        """
        ordered = sorted(segments, key=lambda segment: segment.start)
        if not ordered:
            raise ValueError("A beat grid requires at least one segment")
        for previous, current in zip(ordered, ordered[1:]):
            if current.start == previous.start:
                raise ValueError(
                    f"Two grid segments start at {current.start}; each segment "
                    "must open at a distinct instant"
                )
        self._extent = None if extent is None else _as_seconds(extent, what="An extent")
        bounds = [segment.start for segment in ordered[1:]] + [self._extent]
        bounded: list[BeatGridSegment] = []
        for segment, bound in zip(ordered, bounds):
            if segment.end is not None and segment.end != bound:
                stated = "the grid's extent" if bound is None else f"{bound}"
                raise ValueError(
                    f"The grid segment starting at {segment.start} states "
                    f"end={segment.end}, but the grid bounds it at {stated}; "
                    "grid segments abut, so a stated end must equal the next "
                    "segment's start"
                )
            bounded.append(
                segment if segment.end is not None else replace(segment, end=bound)
            )
        self._segments = tuple(bounded)
        self._beats: tuple[GridBeat, ...] | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BeatGrid):
            return NotImplemented
        return self._segments == other._segments and self._extent == other._extent

    def __repr__(self) -> str:
        plural = "" if self.n_segments == 1 else "s"
        if self._extent is None:
            return f"BeatGrid({self.n_segments} segment{plural}, unbounded)"
        return (
            f"BeatGrid({self.n_segments} segment{plural}, extent={self._extent}, "
            f"{self.n_measures} measures)"
        )

    @property
    def segments(self) -> tuple[BeatGridSegment, ...]:
        """The segments in time order, each bounded by its successor."""
        return self._segments

    @property
    def extent(self) -> Fraction | None:
        """Exclusive bound of the grid in seconds, or ``None``."""
        return self._extent

    @property
    def n_segments(self) -> int:
        """How many tempo segments the grid holds."""
        return len(self._segments)

    @property
    def n_measures(self) -> int:
        """How many measures the grid's beats fall into.

        A grid opening off the downbeat counts the beats before its first
        downbeat as measure ``0``, so they add one measure to the total.

        Raises:
            ValueError: If the grid is unbounded.
        """
        beats = self._all_beats()
        return beats[-1].measure + (1 if beats[0].measure == 0 else 0)

    def iter_beats(self, *, stop: SecondsSpec | None = None) -> Iterator[GridBeat]:
        """Yield the grid's beats in time order.

        Args:
            stop: Exclusive bound in seconds. An unbounded grid generates
                beats without end, so a caller iterating one must give a
                stop.

        Yields:
            Every beat the grid states, labelled in both numbering
            schemes.
        """
        bound = None if stop is None else _as_seconds(stop, what="A stop")
        if self._extent is not None:
            for beat in self._all_beats():
                if bound is not None and beat.seconds >= bound:
                    return
                yield beat
            return
        yield from self._generate(stop=bound)

    def segment_at(self, seconds: SecondsSpec) -> int:
        """Return the index of the segment sounding at *seconds*.

        Args:
            seconds: A position on the grid's seconds axis.

        Returns:
            The 0-based index of the segment whose half-open span
            contains the position.

        Raises:
            ValueError: If the position lies before the first segment or
                at/after the grid's extent.
        """
        position = _as_seconds(seconds)
        self._require_within(position)
        for index in reversed(range(len(self._segments))):
            if self._segments[index].start <= position:
                return index
        raise AssertionError  # pragma: no cover - guarded by _require_within

    def seconds_at(
        self,
        measure: int,
        beat: int | float | Fraction = 1,
        *,
        policy: BeatPolicy | None = None,
    ) -> Coordinate:
        """Return the instant of a measure and beat, in seconds.

        Args:
            measure: Measure number in the grid's own numbering.
            beat: Beat within that measure, 1-based. A fractional beat
                interpolates between the beats on either side of it.
            policy: Read the beat index under this counting instead of
                the grid's own lattice — the beat becomes a quarter-note
                offset from the downbeat, converted back through the
                segments' tempi.

        Returns:
            The instant, as a seconds coordinate.

        Raises:
            ValueError: If the measure or beat is not one the grid
                states, or if the position falls outside the measure.
        """
        return Coordinate(
            float(self._instant_of(measure, beat, policy)), TimeUnit.seconds
        )

    def segment_seconds_at(
        self,
        measure: int,
        beat: int | float | Fraction = 1,
        *,
        policy: BeatPolicy | None = None,
    ) -> Coordinate:
        """Return the instant of a measure and beat within its segment.

        Args:
            measure: Measure number in the grid's own numbering.
            beat: Beat within that measure, 1-based.
            policy: Read the beat index under this counting instead of
                the grid's own lattice.

        Returns:
            Seconds since the start of the segment containing the
            instant.

        Raises:
            ValueError: If the measure or beat is not one the grid
                states.
        """
        instant = self._instant_of(measure, beat, policy)
        segment = self._segments[self.segment_at(instant)]
        return Coordinate(float(instant - segment.start), TimeUnit.seconds)

    def position_at(
        self,
        seconds: SecondsSpec,
        *,
        policy: BeatPolicy | None = None,
    ) -> GridBeat:
        """Return the beat sounding at *seconds*.

        The beat whose span contains the position, that is the last beat
        at or before it.

        Args:
            seconds: A position on the grid's seconds axis.
            policy: Read the beat index under this counting instead of
                the grid's own lattice. The measure numbering and the
                segment are unaffected.

        Returns:
            The beat, carrying both measure numberings.

        Raises:
            ValueError: If the position lies before the grid's first beat
                or at/after its extent, or if *policy* does not reach the
                position within its measure.
        """
        position = _as_seconds(seconds)
        self._require_within(position)
        found: GridBeat | None = None
        for beat in self.iter_beats(stop=position + 1):
            if beat.seconds > position:
                break
            found = beat
        if found is None:
            first = self._segments[0].start
            raise ValueError(
                f"Position {position} lies before the grid's first beat at {first}"
            )
        if policy is None:
            return found
        downbeat = self._require_downbeat_of(found.measure)
        return replace(
            found, beat=policy.index_at(self.quarters_between(downbeat, position))
        )

    def quarters_between(
        self,
        start_seconds: SecondsSpec,
        end_seconds: SecondsSpec,
    ) -> Fraction:
        """Integrate exact quarter-note length over a seconds interval.

        Each segment contributes its own tempo. **The domain is
        ``[0, extent]``** — ``[0, inf)`` on an unbounded grid — closed at
        both ends, so the extent itself may be named even though no beat
        sounds there. A grid that opens late still reads the seconds
        before it, at its first segment's tempo: that stretch carries the
        anchor the floating-measure lattice hangs on. Outside the domain
        the grid states no tempo, and this raises rather than
        extrapolating, exactly as the seconds direction does.

        Args:
            start_seconds: Where the interval opens.
            end_seconds: Where the interval closes.

        Returns:
            The interval's length in quarter notes, exactly.

        Raises:
            ValueError: If the interval runs backwards, or if either end
                lies outside the grid's domain.
        """
        start = _as_seconds(start_seconds, what="An interval start")
        end = _as_seconds(end_seconds, what="An interval end")
        if end < start:
            raise ValueError(f"Interval [{start}, {end}] runs backwards")
        self._require_in_domain(start, what="An interval start")
        self._require_in_domain(end, what="An interval end")

        active = self._segments[0]
        for segment in self._segments:
            if segment.start > start:
                break
            active = segment

        cursor = start
        quarters = Fraction(0)
        for segment in self._segments:
            if segment.start <= start:
                continue
            if segment.start >= end:
                break
            quarters += (segment.start - cursor) * active.quarters_per_second
            cursor = segment.start
            active = segment
        return quarters + (end - cursor) * active.quarters_per_second

    def get_beat_table(
        self,
        *,
        segment: int | None = None,
        numbering: Literal["set", "segment"] = "set",
    ) -> pd.DataFrame:
        """Render the grid's beats as a table.

        Args:
            segment: Restrict the table to one segment's beats.
            numbering: Whether the ``measure`` column counts across the
                whole grid (``"set"``) or restarts per segment
                (``"segment"``).

        Returns:
            One row per beat with the columns ``seconds``, ``segment``,
            ``segment_seconds``, ``measure`` and ``beat``.

        Raises:
            ValueError: If the grid is unbounded, if *segment* names no
                segment, or if *numbering* is not one of the two
                spellings.
        """
        import pandas as pd

        if self._extent is None:
            raise ValueError("Cannot tabulate an unbounded beat grid")
        if numbering not in ("set", "segment"):
            raise ValueError(
                f"Unknown numbering {numbering!r}. Use 'set' or 'segment'."
            )
        if segment is not None and not 0 <= segment < len(self._segments):
            raise ValueError(
                f"Segment {segment} is outside 0..{len(self._segments) - 1}"
            )
        beats = [
            beat
            for beat in self._all_beats()
            if segment is None or beat.segment == segment
        ]
        starts = [self._segments[beat.segment].start for beat in beats]
        return pd.DataFrame(
            {
                "seconds": [float(beat.seconds) for beat in beats],
                "segment": [beat.segment for beat in beats],
                "segment_seconds": [
                    float(beat.seconds - start) for beat, start in zip(beats, starts)
                ],
                "measure": [
                    beat.measure if numbering == "set" else beat.segment_measure
                    for beat in beats
                ],
                "beat": [beat.beat for beat in beats],
            }
        )

    def export_to_csv(
        self,
        filepath: str | Path,
        *,
        format: Literal["sonic_visualiser", "tilia"],  # noqa: A002
        labels: Literal["beats", "measures", "both"] = "beats",
    ) -> int:
        """Write the grid's beats in an audio tool's annotation format.

        Args:
            filepath: Where to write the file.
            format: ``"sonic_visualiser"`` writes a ``TIME``/``LABEL``
                label track; ``"tilia"`` writes a beat track with
                ``time``, ``measure``, ``beat`` and
                ``is_first_in_measure``.
            labels: For the label track, whether to mark every beat
                (``"M1B2"``), only the downbeats (``"M1"``), or both.

        Returns:
            How many rows were written.

        Raises:
            ValueError: If *format* or *labels* is not recognised, or if
                the grid is unbounded.
        """
        if format not in ("sonic_visualiser", "tilia"):
            raise ValueError(
                f"Unknown format {format!r}. Use 'sonic_visualiser' or 'tilia'."
            )
        if labels not in ("beats", "measures", "both"):
            raise ValueError(
                f"Unknown labels {labels!r}. Use 'beats', 'measures', or 'both'."
            )
        table = self.get_beat_table()
        if format == "tilia":
            rows = [
                (row.seconds, row.measure, row.beat, row.beat == 1)
                for row in table.itertuples()
            ]
            header = ["time", "measure", "beat", "is_first_in_measure"]
        else:
            marked = []
            for row in table.itertuples():
                if labels in ("beats", "both"):
                    marked.append((row.seconds, f"M{row.measure}B{row.beat}"))
                if labels in ("measures", "both") and row.beat == 1:
                    marked.append((row.seconds, f"M{row.measure}"))
            rows = sorted(marked, key=lambda entry: entry[0])
            header = ["TIME", "LABEL"]

        with Path(filepath).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return len(rows)

    def _all_beats(self) -> tuple[GridBeat, ...]:
        """Return every beat of a bounded grid, computed once."""
        if self._beats is None:
            if self._extent is None:
                raise ValueError(
                    "An unbounded beat grid generates beats without end; give the "
                    "grid an extent, or iterate it with a stop"
                )
            self._beats = tuple(self._generate(stop=None))
        return self._beats

    def _generate(self, *, stop: Fraction | None) -> Iterator[GridBeat]:
        """Generate the grid's beats, applying the boundary rule.

        Rekordbox rounds a grid's anchor to three decimals and, when a
        later grid's anchor is moved to the right, freezes the beats to
        its left without inserting a fill-in. A beat generated a few
        milliseconds before the next segment's anchor is therefore that
        anchor beat displaced, not a beat of its own: the next segment
        takes precedence and the generated beat is dropped.

        A beat of segment *k* is dropped when it falls within HALF of
        segment *k*'s beat duration before the next segment's start; at
        exactly half a beat it is kept. This is a heuristic — what the
        source program does for anchor displacements beyond half a beat
        has not been studied. It applies at segment boundaries only,
        never at the end of the grid.
        """
        measure = 0
        last = len(self._segments) - 1
        for index, segment in enumerate(self._segments):
            end = segment.end
            beat_seconds = segment.beat_seconds
            n_beats = segment.policy.n_beats
            segment_measure = 0
            step = 0
            while True:
                instant = segment.start + step * beat_seconds
                if end is not None and instant >= end:
                    break
                if stop is not None and instant >= stop:
                    return
                if index < last:
                    assert end is not None
                    if (end - instant) * 2 < beat_seconds:
                        break
                beat = ((segment.battito - 1 + step) % n_beats) + 1
                if beat == 1:
                    measure += 1
                    segment_measure += 1
                yield GridBeat(instant, index, measure, segment_measure, beat)
                step += 1

    def _require_in_domain(self, position: Fraction, *, what: str) -> None:
        """Raise unless *position* lies in the grid's tempo domain.

        The domain runs from zero to the extent inclusive — the span over
        which the grid states a tempo, which is wider than the span over
        which it sounds beats.
        """
        if position < 0:
            raise ValueError(f"{what} at {position} lies before zero")
        if self._extent is not None and position > self._extent:
            raise ValueError(
                f"{what} at {position} lies beyond the grid's extent {self._extent}"
            )

    def _require_within(self, position: Fraction) -> None:
        """Raise unless *position* lies inside the grid's span."""
        first = self._segments[0].start
        if position < first:
            raise ValueError(
                f"Position {position} lies before the grid, which opens at {first}"
            )
        if self._extent is not None and position >= self._extent:
            raise ValueError(
                f"Position {position} lies at or after the grid's extent "
                f"{self._extent}"
            )

    def _require_downbeat_of(self, measure: int) -> Fraction:
        """Return the instant opening *measure*, raising when it has none.

        A caller policy measures a quarter-note offset from the
        downbeat, so a measure the grid never opens — the partial one
        before a grid's first downbeat — cannot be read under one.
        """
        downbeat = self._downbeat_of(measure)
        if downbeat is None:
            raise ValueError(
                f"Measure {measure} has no downbeat in this grid, so a beat index "
                "cannot be read under a supplied policy"
            )
        return downbeat

    def _downbeat_of(self, measure: int) -> Fraction | None:
        """Return the instant opening *measure*, or ``None`` when it has none."""
        for beat in self.iter_beats():
            if beat.measure > measure:
                return None
            if beat.measure == measure and beat.beat == 1:
                return beat.seconds
        return None

    def _beats_of_measure(self, measure: int) -> dict[int, list[Fraction]]:
        """Return the instants of a measure's beats, by beat index.

        A measure interrupted by a segment that re-anchors mid-bar can
        state one index more than once — the new segment resumes counting
        from its own anchor while the measure stays open — so every
        instant is kept rather than the last one winning.
        """
        found: dict[int, list[Fraction]] = {}
        for beat in self.iter_beats():
            if beat.measure > measure:
                break
            if beat.measure == measure:
                found.setdefault(beat.beat, []).append(beat.seconds)
        if not found:
            raise ValueError(f"Measure {measure} is not one this grid states")
        return found

    def _instant_of(
        self,
        measure: int,
        beat: int | float | Fraction,
        policy: BeatPolicy | None,
    ) -> Fraction:
        """Resolve a measure and beat to an exact instant in seconds."""
        index = _as_fraction(beat, what="A beat index")
        if index < 1:
            raise ValueError(f"Beat index {index} is below the downbeat")
        whole = int(index)
        remainder = index - whole
        if policy is not None:
            return self._instant_under_policy(measure, whole, remainder, policy)
        instants = self._beats_of_measure(measure)
        start = self._unique_beat(instants, measure, whole)
        if remainder == 0:
            return start
        following = self._unique_beat(instants, measure, whole + 1)
        return start + remainder * (following - start)

    @staticmethod
    def _unique_beat(
        instants: dict[int, list[Fraction]], measure: int, index: int
    ) -> Fraction:
        """Return the one instant a measure states for a beat index.

        Args:
            instants: The measure's beats, by index.
            measure: The measure being addressed, for the error message.
            index: The 1-based beat index.

        Returns:
            The instant that beat sounds at.

        Raises:
            ValueError: If the measure states that index no times or more
                than once. An ambiguous address is not resolved by
                picking one of the candidates.
        """
        candidates = instants.get(index, [])
        if not candidates:
            raise ValueError(
                f"Measure {measure} has no beat {index}. Available beats: "
                f"{sorted(instants)}"
            )
        if len(candidates) > 1:
            spelled = ", ".join(str(candidate) for candidate in candidates)
            raise ValueError(
                f"Measure {measure} states beat {index} at more than one instant: "
                f"{spelled}. A segment re-anchoring inside the measure restarts "
                "its count, so this address names no single beat"
            )
        return candidates[0]

    def _instant_under_policy(
        self,
        measure: int,
        whole: int,
        remainder: Fraction,
        policy: BeatPolicy,
    ) -> Fraction:
        """Resolve a beat index read under a caller's counting policy."""
        downbeat = self._require_downbeat_of(measure)
        offset = policy.offset_for(whole) + remainder * policy.rod_for(whole)
        instant = self._seconds_after(downbeat, offset)
        limit = self._downbeat_of(measure + 1)
        if limit is None:
            limit = self._extent
        if limit is not None and instant >= limit:
            raise ValueError(
                f"Beat {whole + remainder} of measure {measure} falls at {instant}, "
                f"at or beyond {limit}"
            )
        return instant

    def _seconds_after(self, start: Fraction, quarters: Fraction) -> Fraction:
        """Return the instant *quarters* after *start*, across tempo changes.

        The inverse of :meth:`quarters_between` and bounded the same way:
        a result past the grid's extent raises rather than extrapolating,
        because beyond it the grid states no tempo.
        """
        if quarters == 0:
            return start
        remaining = quarters
        cursor = start
        first = self.segment_at(start)
        for segment in self._segments[first:]:
            rate = segment.quarters_per_second
            if segment.end is None:
                return cursor + remaining / rate
            available = (segment.end - cursor) * rate
            if remaining <= available:
                return cursor + remaining / rate
            remaining -= available
            cursor = segment.end
        raise ValueError(
            f"{quarters} quarters after {start} lies beyond the grid's extent "
            f"{self._extent}"
        )
