"""Meter-aware maps for metrical structure.

This module provides maps for handling real-world musical meter:
- MetricMap: Table-based measure boundaries with MC (monotonic count) and MN (measure number)
- MetricalPositionMap: CombinationMap returning {mc: int, beat: Fraction}

These maps handle idiosyncratic measure structures including:
- Anacrusis (incomplete first measure, MN=0)
- Irregular measures (cadenzas with extra notes)
- Repeat endings (MN=1a, MN=1b)
- Non-contiguous measure sequences (da capo, dal segno)

A dataset expressing events in terms of measure number and beats can be modeled
as a concatenation of measure timelines relative to which the events can be added
according to their beat coordinates.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from numpy.typing import NDArray

from timetoalign.core.enums import NumberType, TimeUnit
from timetoalign.core.time import (
    CoordinateValue,
    number_from_json,
    number_to_json,
)
from timetoalign.maps.base import ConversionMap
from timetoalign.maps.combination import CombinationMap

if TYPE_CHECKING:
    from typing_extensions import Self


# region MetricMap


class MetricMap(ConversionMap[int]):
    """Table-based map from quarters to measure count (MC).

    Unlike FloorMap which assumes uniform measure lengths, MetricMap uses
    explicit anchor points for each measure boundary. This correctly handles:

    - **Anacrusis**: Incomplete first measure (MC=1 but MN=0 or "")
    - **Irregular measures**: Cadenzas, fermatas with extra beats
    - **Varying time signatures**: 4/4 → 3/4 → 6/8
    - **Repeat structure**: First/second endings with same MN but different MC

    The map stores (quarterbeat_start, mc, mn, quarters_per_measure) tuples.

    Attributes:
        n_measures: Number of measure boundaries the map holds.
        total_length: Quarter-beat length of the whole span.

    Examples:
        >>> # Standard 4/4, 10 measures starting at MC=1, MN=1
        >>> meter = MetricMap.from_uniform(
        ...     n_measures=10,
        ...     quarters_per_measure=Fraction(4, 1),
        ...     start_mc=1,
        ...     start_mn="1",
        ... )
        >>> meter(0)    # Quarter 0 → MC 1
        1
        >>> meter(4.0)  # Quarter 4 → MC 2
        2

        >>> # With anacrusis (pickup measure)
        >>> meter = MetricMap.from_uniform(
        ...     n_measures=10,
        ...     quarters_per_measure=Fraction(4, 1),
        ...     anacrusis_quarters=Fraction(1, 1),  # 1 beat pickup
        ...     start_mc=1,
        ...     start_mn="0",  # Musicians call anacrusis "measure 0"
        ... )
        >>> meter(0)    # Quarter 0 → MC 1 (the anacrusis)
        1
        >>> meter.get_mn(1)  # MC 1 carries the label "0"
        '0'

        >>> # From explicit boundaries (loaded from TSV)
        >>> meter = MetricMap.from_boundaries(
        ...     boundaries=[
        ...         (Fraction(0, 1), 1, "0", Fraction(1, 1)),    # Anacrusis
        ...         (Fraction(1, 1), 2, "1", Fraction(4, 1)),    # M1
        ...         (Fraction(5, 1), 3, "2", Fraction(4, 1)),    # M2
        ...     ]
        ... )
    """

    # An MC is which measure you are in, not how far into the piece you are:
    # a whole number that indexes the map's own boundaries, whatever
    # representation the measures axis writes positions in.
    _declared_output_number_type = NumberType.int

    def __init__(
        self,
        *,
        starts: Sequence[Fraction],
        mcs: Sequence[int],
        mns: Sequence[str],
        lengths: Sequence[Fraction],
        source_unit: TimeUnit | str = TimeUnit.quarters,
        target_unit: TimeUnit | str | None = None,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a MetricMap from explicit boundary data.

        Use factory methods `from_uniform()` or `from_boundaries()` for convenience.

        Args:
            starts: Quarter-beat start positions for each measure.
            mcs: Measure Count (1-indexed monotonic) for each measure.
            mns: Measure Number labels (strings like "1", "1a", "0") for each measure.
            lengths: Quarter-beat lengths for each measure.
            source_unit: Source coordinate unit (default: quarters).
            target_unit: Target unit. ``None`` by default: an MC is an
                ordinal that indexes this map's own boundaries, not a
                position on any coordinate axis.
            uid: Optional explicit ID.
            name: Human-readable name for this map. Defaults to
                ``"measure_counts"``.

        Raises:
            ValueError: If arrays have different lengths or are empty.
            ValueError: If starts are not strictly increasing.
        """
        super().__init__(
            source_unit=source_unit,
            target_unit=target_unit,
            uid=uid,
            name=name or "measure_counts",
        )

        if not (len(starts) == len(mcs) == len(mns) == len(lengths)):
            raise ValueError("All arrays must have the same length")
        if len(starts) == 0:
            raise ValueError("MetricMap requires at least one measure")

        # Store as numpy arrays for fast lookup
        self._starts = np.array([float(s) for s in starts], dtype=np.float64)
        self._mcs = np.array(mcs, dtype=np.int64)
        self._lengths = np.array([float(ln) for ln in lengths], dtype=np.float64)

        # Store original Fractions for serialization
        self._starts_frac = list(starts)
        self._lengths_frac = list(lengths)
        self._mns = list(mns)

        # Compute end positions
        self._ends = self._starts + self._lengths

        # Build MC → MN mapping
        self._mc_to_mn_map: dict[int, str] = dict(zip(mcs, mns))

        # Validate monotonicity of starts
        if len(self._starts) > 1:
            diffs = np.diff(self._starts)
            if not np.all(diffs > 0):
                raise ValueError("Measure starts must be strictly increasing")

    @property
    def n_measures(self) -> int:
        """Number of measures in this meter map."""
        return len(self._mcs)

    @property
    def total_length(self) -> Fraction:
        """Total length in quarters (end of last measure)."""
        return self._starts_frac[-1] + self._lengths_frac[-1]

    @property
    def is_invertible(self) -> bool:
        """MetricMap is NOT invertible (many quarters map to same MC)."""
        return False

    def get_mn(self, mc: int) -> str | None:
        """Get the measure number label for a given measure count.

        Args:
            mc: Measure Count (1-indexed monotonic).

        Returns:
            The MN label string, or None if MC not found.
        """
        return self._mc_to_mn_map.get(mc)

    def get_measure_info(self, mc: int) -> dict[str, Any] | None:
        """Get full info for a measure by its MC.

        Args:
            mc: Measure Count.

        Returns:
            Dict with 'mc', 'mn', 'start', 'length', 'end', or None if not found.
        """
        idx = np.searchsorted(self._mcs, mc)
        if idx >= len(self._mcs) or self._mcs[idx] != mc:
            return None

        return {
            "mc": int(self._mcs[idx]),
            "mn": self._mns[idx],
            "start": self._starts_frac[idx],
            "length": self._lengths_frac[idx],
            "end": self._starts_frac[idx] + self._lengths_frac[idx],
        }

    def quarters_at(self, mc: int) -> Fraction:
        """Get the downbeat quarter position for a given measure count.

        Args:
            mc: Measure Count (1-indexed monotonic).

        Returns:
            Quarter position of the measure's start (downbeat) as Fraction.

        Raises:
            ValueError: If MC is not found in the meter map.
        """
        info = self.get_measure_info(mc)
        if info is None:
            raise ValueError(f"MC {mc} not found in meter map")
        return info["start"]

    def _convert_scalar(self, value: CoordinateValue, **kwargs: Any) -> int:
        """Find the MC for a given quarter position.

        Uses binary search for O(log n) lookup.
        """
        x = float(value)

        # Binary search: find rightmost start <= x
        idx = np.searchsorted(self._starts, x, side="right") - 1

        # Handle out-of-bounds
        if idx < 0:
            return int(self._mcs[0])
        if idx >= len(self._mcs):
            return int(self._mcs[-1])

        # Check if x is within this measure's bounds
        if x < self._ends[idx]:
            return int(self._mcs[idx])

        # x is beyond last measure
        return int(self._mcs[-1])

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Vectorized MC lookup."""
        x = values.astype(np.float64)

        # Binary search for each value
        indices = np.searchsorted(self._starts, x, side="right") - 1
        indices = np.clip(indices, 0, len(self._mcs) - 1)

        return self._mcs[indices]

    def inverse(self) -> Self:
        """MetricMap is not invertible.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "MetricMap is not invertible: multiple quarters map to the same MC"
        )

    def beat_in_measure(self, quarters: float | Fraction) -> Fraction:
        """Get the beat position within the current measure.

        Beat positions are 1-indexed (beat 1 = downbeat).
        Returns a Fraction for exact representation.

        Args:
            quarters: Position in quarter notes.

        Returns:
            Beat position as Fraction (1-indexed, e.g., Fraction(3, 2) for beat 1.5).
        """
        x = float(quarters)

        # Find measure
        idx = np.searchsorted(self._starts, x, side="right") - 1
        idx = max(0, min(idx, len(self._starts) - 1))

        # Compute offset within measure
        measure_start = self._starts_frac[idx]
        offset = (
            Fraction(quarters) - measure_start
            if isinstance(quarters, Fraction)
            else Fraction(x) - measure_start
        )

        # Convert to beat (1-indexed)
        # Assuming quarter = 1 beat for now; could be generalized with beat_unit
        return offset + Fraction(1, 1)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d = super().to_dict()
        d["starts"] = [number_to_json(s) for s in self._starts_frac]
        d["mcs"] = [int(mc) for mc in self._mcs]
        d["mns"] = self._mns
        d["lengths"] = [number_to_json(ln) for ln in self._lengths_frac]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricMap:
        """Deserialize from dictionary."""
        return cls(
            starts=[Fraction(number_from_json(s)) for s in data["starts"]],
            mcs=data["mcs"],
            mns=data["mns"],
            lengths=[Fraction(number_from_json(ln)) for ln in data["lengths"]],
            source_unit=data.get("source_unit", TimeUnit.quarters),
            target_unit=data.get("target_unit"),
            uid=data.get("id"),
            name=data.get("name"),
        )

    @classmethod
    def from_uniform(
        cls,
        n_measures: int,
        quarters_per_measure: Fraction | int,
        start_mc: int = 1,
        start_mn: str = "1",
        anacrusis_quarters: Fraction | int | None = None,
        uid: str | None = None,
    ) -> MetricMap:
        """Create a MetricMap with uniform measure lengths.

        This is the simple case equivalent to the old FloorMap approach,
        but with proper support for anacrusis.

        Args:
            n_measures: Number of measures (including anacrusis if present).
            quarters_per_measure: Length of each regular measure.
            start_mc: MC of the first measure (default 1).
            start_mn: MN label of the first measure (default "1").
                     Use "0" for anacrusis measures.
            anacrusis_quarters: If set, the first measure is shorter (pickup).
            uid: Optional explicit ID.

        Returns:
            A new MetricMap instance.

        Examples:
            >>> # 10 measures of 4/4
            >>> meter = MetricMap.from_uniform(10, Fraction(4, 1))

            >>> # With 1-beat anacrusis
            >>> meter = MetricMap.from_uniform(
            ...     n_measures=10,
            ...     quarters_per_measure=Fraction(4, 1),
            ...     anacrusis_quarters=Fraction(1, 1),
            ...     start_mn="0",
            ... )
        """
        qpm = Fraction(quarters_per_measure)

        starts: list[Fraction] = []
        mcs: list[int] = []
        mns: list[str] = []
        lengths: list[Fraction] = []

        current_pos = Fraction(0, 1)
        current_mc = start_mc

        # Parse start_mn to determine numbering pattern
        try:
            mn_int = int(start_mn)
            mn_is_int = True
        except ValueError:
            mn_int = 0
            mn_is_int = False

        for i in range(n_measures):
            starts.append(current_pos)
            mcs.append(current_mc)

            # Determine MN
            if i == 0:
                mns.append(start_mn)
            elif mn_is_int:
                mns.append(str(mn_int + i))
            else:
                mns.append(str(current_mc))

            # Determine length
            if i == 0 and anacrusis_quarters is not None:
                length = Fraction(anacrusis_quarters)
            else:
                length = qpm
            lengths.append(length)

            current_pos += length
            current_mc += 1

        return cls(
            starts=starts,
            mcs=mcs,
            mns=mns,
            lengths=lengths,
            uid=uid,
        )

    @classmethod
    def from_boundaries(
        cls,
        boundaries: Sequence[tuple[Fraction, int, str, Fraction]],
        uid: str | None = None,
    ) -> MetricMap:
        """Create a MetricMap from explicit boundary tuples.

        Args:
            boundaries: List of (start_quarters, mc, mn, length_quarters) tuples.
            uid: Optional explicit ID.

        Returns:
            A new MetricMap instance.
        """
        if not boundaries:
            raise ValueError("At least one boundary is required")

        starts = [b[0] for b in boundaries]
        mcs = [b[1] for b in boundaries]
        mns = [b[2] for b in boundaries]
        lengths = [b[3] for b in boundaries]

        return cls(
            starts=starts,
            mcs=mcs,
            mns=mns,
            lengths=lengths,
            uid=uid,
        )

    @classmethod
    def from_verovio_timemap(
        cls,
        path: str | Path,
        *,
        uid: str | None = None,
    ) -> MetricMap:
        """Build a MetricMap from a Verovio timemap JSON file.

        A Verovio timemap is a JSON list of timeline entries, each carrying
        a ``qstamp`` (absolute position in quarter notes).  Entries that
        begin a measure additionally carry a ``measureOn`` key (the xml:id
        of the bar) and a ``meterSig`` value of the form
        ``"<measure_number> <numerator>/<denominator>"`` (e.g.
        ``"1 12/8"``).  This factory walks the list, treats every
        ``measureOn`` entry as a measure-start boundary, and uses the final
        entry's ``qstamp`` as the upper bound of the last measure.

        Each measure's length is the difference between consecutive
        boundary qstamps; the last measure's length is bounded by the final
        timemap entry.  Measure-number labels are taken verbatim from the
        first whitespace token of ``meterSig`` (so ``"1 12/8"`` yields
        measure number ``"1"`` and measure count ``1``).

        Args:
            path: Path to the Verovio timemap JSON file.
            uid: Optional explicit ID.

        Returns:
            A new MetricMap whose boundaries match the timemap.

        Raises:
            ValueError: If the timemap contains no ``measureOn`` boundaries.

        Examples:
            >>> meter = MetricMap.from_verovio_timemap("nocturne.json")
            >>> meter.n_measures
            38
            >>> meter.total_length
            Fraction(425, 2)
        """
        # TODO: A future MetricMap.from_meter_file(path, *, last_measure)
        # factory should ingest the companion ``.meter`` file directly.  A
        # ``.meter`` file only encodes meter *changes* (this specimen lists
        # 4 rows across 38 measures), so it cannot bound the final measure
        # on its own; the caller must supply the last measure's length (or
        # its terminal quarter position) separately.  The Verovio timemap
        # is preferred here precisely because its final ``qstamp`` supplies
        # that upper bound directly.
        timemap = json.loads(Path(path).read_text(encoding="utf-8"))

        starts: list[Fraction] = []
        mcs: list[int] = []
        mns: list[str] = []
        for entry in timemap:
            if "measureOn" not in entry:
                continue
            mn = str(entry["meterSig"]).split()[0]
            starts.append(Fraction(str(entry["qstamp"])))
            mns.append(mn)
            mcs.append(int(mn))

        if not starts:
            raise ValueError(
                "Verovio timemap contains no measure boundaries (no "
                "'measureOn' entries)."
            )

        final_qstamp = Fraction(str(timemap[-1]["qstamp"]))
        # Length of each measure is the gap to the next boundary; the last
        # measure runs to the final timemap entry.
        boundaries = starts + [final_qstamp]
        lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(starts))]

        return cls(
            starts=starts,
            mcs=mcs,
            mns=mns,
            lengths=lengths,
            uid=uid,
        )

    def __repr__(self) -> str:
        return (
            f"MetricMap(n_measures={self.n_measures}, "
            f"total_length={self.total_length})"
        )


# endregion


# region BeatInMeasureMap


class BeatInMeasureMap(ConversionMap[Fraction]):
    """Map from quarters to beat-in-measure as Fraction.

    Uses a MetricMap to determine measure boundaries, then computes
    the beat position within that measure as a proper Fraction.

    Beat positions are 1-indexed (beat 1 = downbeat).

    Examples:
        >>> meter = MetricMap.from_uniform(10, Fraction(4, 1))
        >>> beat_map = BeatInMeasureMap(meter)
        >>> beat_map(0)      # Quarter 0 → beat 1
        Fraction(1, 1)
        >>> beat_map(1)      # Quarter 1 → beat 2
        Fraction(2, 1)
        >>> beat_map(Fraction(3, 2))  # Quarter 1.5 → beat 2.5
        Fraction(5, 2)
    """

    def __init__(
        self,
        meter_map: MetricMap,
        *,
        source_unit: TimeUnit | str = TimeUnit.quarters,
        target_unit: TimeUnit | str | None = None,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a BeatInMeasureMap.

        Args:
            meter_map: The MetricMap providing measure boundaries.
            source_unit: Source coordinate unit (default: quarters).
            target_unit: Target unit. ``None`` by default: a beat count
                inside a bar is a metrical position, not a coordinate.
            uid: Optional explicit ID.
            name: Human-readable name for this map. Defaults to
                ``"beat_in_measure"``.
        """
        super().__init__(
            source_unit=source_unit,
            target_unit=target_unit,
            uid=uid,
            name=name or "beat_in_measure",
        )
        self._meter_map = meter_map

    @property
    def is_invertible(self) -> bool:
        """BeatInMeasureMap is NOT invertible (cyclic/many-to-one)."""
        return False

    def _convert_scalar(self, value: CoordinateValue, **kwargs: Any) -> Fraction:
        """Get beat position for a single quarter value."""
        return self._meter_map.beat_in_measure(value)

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Vectorized beat position lookup.

        Note: Returns float array since numpy doesn't support Fraction.
        For exact Fraction values, use scalar conversion.
        """
        # For array operations, we compute float beats
        # Find measure indices
        indices = np.searchsorted(self._meter_map._starts, values, side="right") - 1
        indices = np.clip(indices, 0, len(self._meter_map._starts) - 1)

        # Compute offset within measure
        measure_starts = self._meter_map._starts[indices]
        offsets = values - measure_starts

        # Convert to 1-indexed beats
        return offsets + 1.0

    def inverse(self) -> Self:
        """BeatInMeasureMap is not invertible.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("BeatInMeasureMap is not invertible: cyclic pattern")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d = super().to_dict()
        d["meter_map"] = self._meter_map.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BeatInMeasureMap:
        """Deserialize from dictionary."""
        meter_map = MetricMap.from_dict(data["meter_map"])
        return cls(
            meter_map,
            source_unit=data.get("source_unit", TimeUnit.quarters),
            target_unit=data.get("target_unit"),
            uid=data.get("id"),
            name=data.get("name"),
        )

    def __repr__(self) -> str:
        return f"BeatInMeasureMap(meter={self._meter_map})"


# endregion


# region MetricalPositionMap


class MetricalPositionMap(CombinationMap):
    """CombinationMap that returns metrical position as {mc: int, beat: Fraction}.

    This is a convenience wrapper combining MetricMap and BeatInMeasureMap.

    Unlike the generic CombinationMap, this class:
    1. Returns proper types: int for MC, Fraction for beat
    2. Provides convenience method for reverse lookup (mc, beat) → quarters

    Examples:
        >>> meter = MetricMap.from_uniform(10, Fraction(4, 1))
        >>> pos_map = MetricalPositionMap(meter)

        Quarter 7.5 is beat 4.5 of the second measure:

        >>> pos_map(7.5)
        {'mc': 2, 'beat': Fraction(9, 2)}

        The reverse lookup returns the position exactly:

        >>> pos_map.quarters_at(mc=2, beat=Fraction(9, 2))
        Fraction(15, 2)
    """

    def __init__(
        self,
        meter_map: MetricMap,
        *,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a MetricalPositionMap.

        Args:
            meter_map: The MetricMap providing measure structure.
            uid: Optional explicit ID.
            name: Human-readable name for this map. Defaults to the map's ID.
        """
        self._meter_map = meter_map
        beat_map = BeatInMeasureMap(meter_map)

        super().__init__(
            maps={"mc": meter_map, "beat": beat_map},
            source_unit=TimeUnit.quarters,
            uid=uid,
            name=name,
        )

    @property
    def meter_map(self) -> MetricMap:
        """The underlying MetricMap."""
        return self._meter_map

    def quarters_at(self, mc: int, beat: Fraction | float = Fraction(1, 1)) -> Fraction:
        """Get quarter position for a given MC and beat.

        Args:
            mc: Measure Count (1-indexed).
            beat: Beat within measure (1-indexed). Default: beat 1 (downbeat).

        Returns:
            Quarter position as Fraction.

        Raises:
            ValueError: If MC is not found in the meter map.
        """
        beat_frac = Fraction(beat) if not isinstance(beat, Fraction) else beat

        # Beat offset from measure start (beat 1 = offset 0)
        beat_offset = beat_frac - Fraction(1, 1)

        return self._meter_map.quarters_at(mc) + beat_offset

    def mn_at(self, quarters: float | Fraction) -> str | None:
        """Get the Measure Number label at a quarter position.

        Args:
            quarters: Position in quarter notes.

        Returns:
            The MN label string, or None if not found.
        """
        mc = self._meter_map(float(quarters))
        return self._meter_map.get_mn(mc)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d = super().to_dict()
        d["meter_map"] = self._meter_map.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricalPositionMap:
        """Deserialize from dictionary."""
        meter_map = MetricMap.from_dict(data["meter_map"])
        position_map = cls(meter_map, uid=data.get("id"), name=data.get("name"))
        position_map._maps = {
            name: ConversionMap.from_dict(map_data)
            for name, map_data in data["maps"].items()
        }
        position_map._names = list(data["names"])
        return position_map

    def __repr__(self) -> str:
        return f"MetricalPositionMap(n_measures={self._meter_map.n_measures})"


# endregion
