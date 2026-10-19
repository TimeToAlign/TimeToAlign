"""Interval-to-constant and quarters-to-measures conversion maps.

This module provides:
- IntervalToConstantMap: Generic step-function map (interval -> constant value)
- QuartersToMeasureNumber: Map quarters -> MN labels (no interpolation, NOT a coordinate map)
- QuartersToFloatingMeasures: Map quarters -> measure coordinates (with interpolation)

These maps enable score timelines to express coordinates in both quarter-beat
and measure terms, supporting cross-domain alignment and human-readable output.
"""

from __future__ import annotations

import bisect
import math
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Generic, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray

from timetoalign.core.enums import ExtrapolationPolicy, InterpolationKind, TimeUnit
from timetoalign.core.ids import mint_id
from timetoalign.core.time import CoordinateValue, number_to_json
from timetoalign.maps.base import ConversionMap, MeasureMapIds
from timetoalign.maps.composite import _piecewise_form
from timetoalign.maps.constant import ConstantMap, _constant_to_json
from timetoalign.maps.table import TableMap, _table_form

if TYPE_CHECKING:
    from typing_extensions import Self

    from timetoalign.loader.score.stores.measures import MeasureData
    from timetoalign.maps.measures import FloatingMeasureMap
    from timetoalign.maps.meter import MetricMap
    from timetoalign.timelines.structure import MeasureMap


def _as_exact(value: CoordinateValue) -> Fraction:
    """Return *value* as the exact ratio it already is, without guessing one."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(float(value))


# Type variable for IntervalToConstantMap output type
T = TypeVar("T")


# region IntervalToConstantMap


class IntervalToConstantMap(ConversionMap[T], Generic[T]):
    """Interval-to-constant map: each interval [x_i, x_{i+1}) maps to value_i.

    Unlike TableMap, IntervalToConstantMap does not interpolate. A coordinate x
    returns the value associated with the interval that contains x.

    Values can be any type (string, int, float, etc.). This is NOT a coordinate
    conversion map when values are non-numeric (e.g., string labels).

    Without an ``end`` the map is total: a position before the first boundary
    takes the first value, and the last interval has no end. With an ``end``
    the map is defined on ``[boundaries[0], end)`` only, the form a manifest
    can hold for labels: a ``PiecewiseMap`` of ``ConstantMap``s. Without an
    ``end``, a map of numbers is written as a ``TableMap`` of kind
    ``previous`` (a ``ConstantMap`` for a single boundary), and a map of
    labels with more than one boundary cannot be written.

    This is useful for:
    - Measure number lookup (quarters -> MN label)
    - Region/section lookup (coordinate -> section name)
    - Any step-function relationship

    Attributes:
        boundaries: Sorted x-coordinates defining interval starts.
        values: Value for each interval (len = len(boundaries)).

    Examples:
        >>> # Map quarters to section names
        >>> section_map = IntervalToConstantMap(
        ...     boundaries=[0, 16, 32, 48],
        ...     values=["Intro", "Verse", "Chorus", "Outro"],
        ...     source_unit="quarters",
        ... )
        >>> section_map(20)
        'Verse'
        >>> section_map(40)
        'Chorus'
    """

    def __init__(
        self,
        *,
        boundaries: Sequence[CoordinateValue],
        values: Sequence[T],
        end: CoordinateValue | None = None,
        source_unit: TimeUnit | str | None = None,
        target_unit: TimeUnit | str | None = None,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize an IntervalToConstantMap.

        Args:
            boundaries: Interval start positions. Must be strictly increasing.
            values: Value for each interval. Must have same length as boundaries.
            end: Where the last interval ends, or None for no end. Given, the
                map is defined on ``[boundaries[0], end)`` only.
            source_unit: The unit of input coordinates (optional).
            target_unit: The unit of output values (optional). Use None for
                non-coordinate outputs like string labels.
            uid: Optional explicit ID.
            name: Human-readable name for this map.

        Raises:
            ValueError: If boundaries and values have different lengths.
            ValueError: If boundaries are not strictly increasing.
            ValueError: If fewer than 1 boundary is provided.
            ValueError: If *end* does not lie after the last boundary.
        """
        super().__init__(
            source_unit=source_unit,
            target_unit=target_unit,
            uid=uid,
            name=name,
        )

        if len(boundaries) != len(values):
            raise ValueError(
                f"boundaries and values must have same length, "
                f"got {len(boundaries)} and {len(values)}"
            )
        if len(boundaries) < 1:
            raise ValueError("IntervalToConstantMap requires at least 1 boundary")

        # Convert to numpy array for fast lookup
        self._boundaries = np.array([float(v) for v in boundaries], dtype=np.float64)
        self._values = list(values)

        # Store original values for serialization (may have Fractions)
        self._boundaries_original = list(boundaries)

        # Validate monotonicity
        if len(self._boundaries) > 1:
            diffs = np.diff(self._boundaries)
            if not np.all(diffs > 0):
                raise ValueError("boundaries must be strictly monotonically increasing")
        if end is not None and not float(end) > self._boundaries[-1]:
            raise ValueError(
                f"end {end} must lie after the last boundary {self._boundaries_original[-1]}"
            )
        self._end = end

    @property
    def end(self) -> CoordinateValue | None:
        """Where the last interval ends, or None when it has no end."""
        return self._end

    def _check_domain(self, low: float, high: float) -> None:
        """Refuse positions outside ``[boundaries[0], end)`` when the map has an end.

        Raises:
            ValueError: If *low* or *high* lies outside the domain.
        """
        if self._end is not None and (
            low < self._boundaries[0] or high >= float(self._end)
        ):
            raise ValueError(
                f"Value outside the intervals [{self._boundaries_original[0]}, "
                f"{self._end}) of {type(self).__name__} {self._id!r}"
            )

    @property
    def boundaries(self) -> NDArray[np.floating[Any]]:
        """The interval start positions."""
        return self._boundaries

    @property
    def values(self) -> list[T]:
        """The values for each interval."""
        return self._values

    @property
    def is_invertible(self) -> bool:
        """IntervalToConstantMap is NOT invertible (many-to-one)."""
        return False

    def _convert_scalar(self, value: CoordinateValue, **kwargs: Any) -> T:
        """Find the value for a given coordinate.

        Uses binary search for O(log n) lookup.
        """
        x = float(value)
        self._check_domain(x, x)

        # Find rightmost boundary <= x
        idx = int(np.searchsorted(self._boundaries, x, side="right")) - 1

        # Clamp to valid range
        idx = max(0, min(idx, len(self._values) - 1))

        return self._values[idx]

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Vectorized lookup.

        Note: Returns object array if values are non-numeric (e.g., strings).
        """
        x = values.astype(np.float64)
        if x.size:
            self._check_domain(float(np.min(x)), float(np.max(x)))

        # Binary search for each value
        indices = np.searchsorted(self._boundaries, x, side="right") - 1
        indices = np.clip(indices, 0, len(self._values) - 1)

        # Convert to numpy array (object dtype for strings)
        return np.array([self._values[i] for i in indices])

    def inverse(self) -> Self:
        """IntervalToConstantMap is not invertible.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "IntervalToConstantMap is not invertible: many coordinates map to same value"
        )

    def _to_manifest(self, *, minimal: bool, entries: MeasureMapIds) -> dict[str, Any]:
        """Write the step function in the form converting exactly as it does.

        With an ``end``, a ``PiecewiseMap`` of ``ConstantMap``s over the
        intervals; without, a ``ConstantMap`` for one boundary, else a
        ``TableMap`` of kind ``previous`` clamping at both ends.

        Raises:
            ValueError: If the map holds labels, several boundaries and no
                ``end``: its last interval has no end a manifest can state.
        """
        if self._end is not None:
            taken = {self._id}
            pieces = []
            for value in self._values:
                piece = ConstantMap(
                    value=value,
                    source_unit=self._source_unit,
                    target_unit=self._target_unit,
                    uid=mint_id(ConstantMap._id_prefix, taken),
                )
                taken.add(piece.id)
                pieces.append(piece._to_manifest(minimal=minimal, entries=entries))
            return _piecewise_form(
                self, [*self._boundaries_original, self._end], pieces, minimal=minimal
            )
        if len(self._values) == 1:
            return self._map_object(
                "ConstantMap",
                {"value": _constant_to_json(self._values[0])},
                minimal=minimal,
            )
        if not all(
            isinstance(v, (int, float, Fraction)) and not isinstance(v, bool)
            for v in self._values
        ):
            raise self._no_form(
                "a step function of labels needs an end for its last interval "
                "(give end=)"
            )
        return _table_form(
            self,
            self._boundaries_original,
            self._values,
            kind=InterpolationKind.previous,
            extrapolate=ExtrapolationPolicy.constant,
            minimal=minimal,
        )

    def to_table_map(
        self, kind: InterpolationKind = InterpolationKind.previous
    ) -> TableMap:
        """Convert to TableMap for numeric values.

        This allows using TableMap operations on interval data when values are
        numeric.

        Args:
            kind: Interpolation kind. Default is "previous" (step function).

        Returns:
            TableMap with the same boundaries and values.

        Raises:
            TypeError: If values are not numeric.
        """
        if not all(isinstance(v, (int, float, Fraction)) for v in self._values):
            raise TypeError("to_table_map() requires numeric values")

        # Need at least 2 points for TableMap; add extrapolation point
        x_values = list(self._boundaries)
        y_values = [float(v) for v in self._values]

        # Add final extrapolation point (extend last interval)
        if len(x_values) >= 1:
            # Use final value extending one unit beyond last boundary
            final_x = x_values[-1] + 1.0
            x_values.append(final_x)
            y_values.append(y_values[-1])

        return TableMap(
            x_values=x_values,
            y_values=y_values,
            kind=kind,
            extrapolate=ExtrapolationPolicy.extrapolate,
            source_unit=self._source_unit,
            target_unit=self._target_unit,
        )

    def __repr__(self) -> str:
        n = len(self._boundaries)
        parts = [f"n_intervals={n}"]
        if self._source_unit:
            parts.append(f"source_unit={self._source_unit}")
        if self._target_unit:
            parts.append(f"target_unit={self._target_unit}")
        return f"IntervalToConstantMap({', '.join(parts)})"


# endregion


def _is_decimal_integer(label: object) -> bool:
    """Whether *label* spells an integer the way ``str(int)`` does."""
    try:
        return isinstance(label, str) and str(int(label)) == label
    except ValueError:
        return False


def _numbers_measures(
    measure_map: MeasureMap, starts: Sequence[Fraction], numbers: Sequence[int]
) -> bool:
    """Whether *measure_map*'s measures start at *starts* and carry *numbers*."""
    records = measure_map.measures
    return len(records) == len(starts) and all(
        record.qstamp == start
        and (record.number if record.number is not None else record.count) == number
        for record, start, number in zip(records, starts, numbers)
    )


# region QuartersToMeasureNumber


class QuartersToMeasureNumber(IntervalToConstantMap[str]):
    """Map quarters to measure number labels (no interpolation).

    Given a quarter position, returns the MN label (e.g., "1", "19a", "0")
    of the measure containing that position.

    This is NOT a coordinate conversion map - it returns string labels,
    not numeric coordinates. Therefore target_unit is None.

    MN labels are strings because they may contain suffixes (e.g., "19a", "19b"
    for split bars, or "0" for anacrusis).

    Examples:
        >>> cmap = QuartersToMeasureNumber(
        ...     boundaries=[0, 4, 8, 12],
        ...     mns=["1", "2", "3", "4"],
        ... )
        >>> cmap(0.0)    # Start of M1
        '1'
        >>> cmap(3.5)    # Still in M1 (assuming 4 quarters/measure)
        '1'
        >>> cmap(4.0)    # Start of M2
        '2'

        >>> # From MetricMap
        >>> cmap = QuartersToMeasureNumber.from_metric_map(meter)
    """

    def __init__(
        self,
        *,
        boundaries: Sequence[CoordinateValue],
        mns: Sequence[str],
        source_unit: TimeUnit | str = TimeUnit.quarters,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a QuartersToMeasureNumber map.

        Args:
            boundaries: Measure start positions in quarters.
            mns: Measure Number labels for each measure.
            source_unit: Source unit (default: quarters).
            uid: Optional explicit ID.
            name: Human-readable name (default: "quarters_to_mn").
        """
        super().__init__(
            boundaries=boundaries,
            values=mns,
            source_unit=source_unit,
            target_unit=None,  # Not a coordinate map - returns labels
            uid=uid,
            name=name or "quarters_to_mn",
        )
        # The default name is derived for display; only a given one is written.
        self._name_given = name is not None

    @property
    def mns(self) -> list[str]:
        """The measure number labels."""
        return self._values

    @classmethod
    def from_metric_map(cls, meter: MetricMap) -> QuartersToMeasureNumber:
        """Create from a MetricMap.

        Args:
            meter: MetricMap with measure boundary information.

        Returns:
            QuartersToMeasureNumber map.
        """
        boundaries = list(meter._starts_frac)
        mns = list(meter._mns)
        return cls(boundaries=boundaries, mns=mns)

    @classmethod
    def from_measure_data(cls, measures: MeasureData) -> QuartersToMeasureNumber:
        """Create from MeasureData (loaded from TSV/JSON).

        Args:
            measures: MeasureData with measure events.

        Returns:
            QuartersToMeasureNumber map.
        """
        if len(measures) == 0:
            raise ValueError("MeasureData is empty")

        # Extract start positions and MN labels from the PyArrow table
        boundaries: list[float] = []
        mns: list[str] = []

        for event in measures:
            start = event.get("start")
            mn = event.get("mn")

            # Handle coordinate struct format
            if isinstance(start, dict) and "value" in start:
                start_val = float(start["value"])
            elif start is not None:
                start_val = float(start)
            else:
                continue

            # Handle MN (may be None, convert to string)
            if mn is None:
                # Fall back to MC if MN is not available
                mc = event.get("mc")
                mn_str = str(mc) if mc is not None else "?"
            else:
                mn_str = str(mn)

            boundaries.append(start_val)
            mns.append(mn_str)

        return cls(boundaries=boundaries, mns=mns)

    def _to_manifest(self, *, minimal: bool, entries: MeasureMapIds) -> dict[str, Any]:
        """Write the ``MeasureNumberMap`` of the measures these labels number.

        Each label must be a measure number spelled as a decimal integer,
        which the view answers as that integer. The view names the
        timeline's measure map when its measures start at the boundaries
        and carry these numbers, and otherwise a measure map of its own:
        one measure per boundary, the first at 0, the last of unknown
        length.

        Raises:
            ValueError: If a label is no decimal integer, the source unit is
                not quarters, or the first boundary is not 0.
        """
        if self._source_unit is not TimeUnit.quarters:
            raise self._no_form("a measure view reads quarters")
        numbers = []
        for label in self._values:
            if not _is_decimal_integer(label):
                raise self._no_form(f"its label {label!r} is not a measure number")
            numbers.append(int(label))
        starts = [_as_exact(b) for b in self._boundaries_original]
        own = entries.own
        if own is not None and _numbers_measures(own, starts, numbers):
            measure_map_id = entries.id_of(own)
        else:
            if starts[0] != 0:
                raise self._no_form(
                    f"its first measure starts at {starts[0]}, not at 0"
                )
            rows: list[dict[str, Any]] = []
            for index, (start, number) in enumerate(zip(starts, numbers), start=1):
                row: dict[str, Any] = {
                    "count": index,
                    "qstamp": number_to_json(start),
                    "number": number,
                }
                if index < len(starts):
                    row["actual_length"] = number_to_json(starts[index] - start)
                rows.append(row)
            measure_map_id = entries.id_of_rows(rows)
        return self._map_object(
            "MeasureNumberMap", {"measure_map": measure_map_id}, minimal=minimal
        )

    def __repr__(self) -> str:
        n = len(self._boundaries)
        return f"QuartersToMeasureNumber(n_measures={n})"


# endregion


# region QuartersToFloatingMeasures


class QuartersToFloatingMeasures(TableMap):
    """Map quarters to floating measures.

    A floating measure (``fm``) writes a position as
    ``<ordinal>.<how far into that bar>`` — the convention behind
    annotation tables that read ``12.5`` for the middle of the twelfth
    bar.  Two properties make it exact rather than approximate:

    * **The fractional part is anchored on the NOMINAL bar.** An
      incomplete bar's content sits where it is notated, so a 1/8 pickup
      in 9/8 onsets at ``0.888`` — eight ninths of the way through the
      bar it belongs to — not at ``0.000``.
    * **Emission truncates to three decimals.** Three decimals is the
      convention's resolution, and truncating (rather than rounding) is
      what makes ``8/9`` read ``0.888``; it is also the rule behind the
      ``.999`` an interval end shows when a bar is read from the right.

    An fm value is a float by definition: the fractional part measures
    an uneven discretisation of logical time — bars vary in length — so
    it is not a ratio of anything, and its resolution is capped at a
    thousandth of a bar.  Converting fm back to quarters is therefore
    exact only to that resolution; the inverse interpolates linearly
    over the same knots and never tries to reconstruct what truncation
    dropped.

    Examples:
        >>> cmap = QuartersToFloatingMeasures(
        ...     x_values=[0, 4, 8, 12],
        ...     y_values=[1.0, 2.0, 3.0, 4.0],
        ... )
        >>> cmap(0.0)    # Start of M1
        1.0
        >>> cmap(2.0)    # Halfway through M1 (4 quarters/measure)
        1.5
        >>> cmap(4.0)    # Start of M2
        2.0
    """

    #: Fractional resolution of a floating measure: one thousandth of a bar.
    RESOLUTION: int = 1000

    def __init__(
        self,
        *,
        x_values: Sequence[CoordinateValue],
        y_values: Sequence[float],
        source_unit: TimeUnit | str = TimeUnit.quarters,
        target_unit: TimeUnit | str = TimeUnit.floating_measures,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a QuartersToFloatingMeasures map.

        Args:
            x_values: Knot positions in quarters — each bar's *nominal*
                downbeat, plus the end of the last bar.
            y_values: The measure ordinal at each knot.
            source_unit: Source unit (default: quarters).
            target_unit: Target unit (default: floating measures).
            uid: Optional explicit ID.
            name: Human-readable name (default: "quarters_to_measures").
        """
        super().__init__(
            x_values=x_values,
            y_values=y_values,
            kind=InterpolationKind.linear,
            extrapolate=ExtrapolationPolicy.extrapolate,
            source_unit=source_unit,
            target_unit=target_unit,
            uid=uid,
        )
        self._name = name or "quarters_to_measures"
        self._name_given = name is not None
        self._x_exact = [_as_exact(v) for v in self._x_original]
        self._y_exact = [_as_exact(v) for v in self._y_original]

    @classmethod
    def from_measure_map(cls, measure_map: MeasureMap) -> FloatingMeasureMap:
        """Build the floating-measure lattice of a measure map.

        One knot per measure record, placed at that bar's **virtual
        nominal downbeat** — its sounding start minus the offset at which
        its content sits inside the notated bar — carrying that bar's
        ordinal.  A final knot closes the lattice at the end of the last
        bar.  Between two knots the map is linear, so the slope inside a
        bar is exactly ``1 / nominal_length``.

        Ordinals count the measure records, never the printed labels:
        the first record is ``0`` when it is an anacrusis (offset
        content, shorter than its nominal bar) and ``1`` otherwise, and
        every following record adds one.  Counting runs monotonically
        through voltas — ``15a`` and ``15b`` are two bars and get two
        consecutive ordinals — and never resets.

        Args:
            measure_map: Measures whose exact starts and lengths define the lattice.

        Returns:
            The fm map for that skeleton: a :class:`FloatingMeasureMap`,
            which remembers the measure map it reads.

        Raises:
            ValueError: If the skeleton has no measures, or if two
                records share a virtual nominal downbeat (a split bar,
                whose two halves cannot both anchor the same fm ordinal).
        """
        from timetoalign.maps.measures import FloatingMeasureMap

        return FloatingMeasureMap(measure_map)

    @staticmethod
    def _lattice(measure_map: MeasureMap) -> tuple[list[Fraction], list[float]]:
        """Return the knots and ordinals of a measure map's lattice.

        Raises:
            ValueError: As :meth:`from_measure_map`.
        """
        records = measure_map.measures
        if not records:
            raise ValueError("A floating-measure lattice requires at least one measure")

        first = records[0]
        first_offset = getattr(first, "offset_within_measure", Fraction(0))
        is_anacrusis = (
            first_offset > 0
            and first.actual_length is not None
            and first.nominal_length is not None
            and first.actual_length < first.nominal_length
        )
        ordinal = 0 if is_anacrusis else 1

        x_values: list[Fraction] = []
        y_values: list[float] = []
        for record in records:
            if record.qstamp is None or record.actual_length is None:
                raise ValueError(
                    "A floating-measure lattice requires exact qstamp and actual_length values"
                )
            offset = getattr(record, "offset_within_measure", Fraction(0))
            x_values.append(record.qstamp - offset)
            y_values.append(float(ordinal))
            ordinal += 1

        last = records[-1]
        assert last.qstamp is not None and last.actual_length is not None
        x_values.append(last.qstamp + last.actual_length)
        y_values.append(float(ordinal))

        for previous, current in zip(x_values, x_values[1:]):
            if current <= previous:
                raise ValueError(
                    f"Two measures have the same nominal downbeat {previous}; a split measure "
                    "cannot anchor two floating-measure ordinals"
                )

        return x_values, y_values

    def _convert_scalar(self, value: CoordinateValue, **kwargs: Any) -> float:
        """Read one quarter position as a floating measure.

        The lattice is walked in exact arithmetic — the knots are bar
        boundaries, which are ratios — and only the final emission
        becomes a float, truncated to the convention's three decimals.

        Args:
            value: A position in quarters.
            **kwargs: Unused.

        Returns:
            The floating-measure reading.
        """
        return self._truncate(self._interpolate_exact(_as_exact(value)))

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Read a column of quarter positions as floating measures.

        The column takes the same exact walk as a single position: the
        knots are ratios, so interpolating in floating point and
        truncating afterwards would drop a whole thousandth wherever the
        float landed a few units below an exact boundary.  Reading a
        column must answer what reading its entries one at a time
        answers, so this walks element by element in exact arithmetic
        and truncates each reading itself.

        Args:
            values: A column of positions in quarters.
            **kwargs: Unused.

        Returns:
            The floating-measure readings.
        """
        readings = [
            self._truncate(self._interpolate_exact(_as_exact(value)))
            for value in np.asarray(values).tolist()
        ]
        return np.asarray(readings, dtype=np.float64)

    def _interpolate_exact(self, value: Fraction) -> Fraction:
        """Interpolate linearly between the exact knots, extrapolating at the ends."""
        xs, ys = self._x_exact, self._y_exact
        index = bisect.bisect_right(xs, value) - 1
        if index < 0:
            index = 0
        elif index >= len(xs) - 1:
            index = len(xs) - 2
        span = xs[index + 1] - xs[index]
        slope = (ys[index + 1] - ys[index]) / span
        return ys[index] + slope * (value - xs[index])

    def _truncate(self, value: Fraction) -> float:
        """Cut an exact reading down to the convention's three decimals."""
        return math.floor(value * self.RESOLUTION) / self.RESOLUTION

    def inverse(self) -> TableMap:
        """Return the inverse map (measures -> quarters).

        Returns:
            A TableMap converting measures to quarters.

        Raises:
            NotImplementedError: If not invertible (should not happen for this map).
        """
        if not self._is_invertible:
            raise NotImplementedError(
                "Cannot invert QuartersToFloatingMeasures: y values are not strictly monotonic"
            )

        # The inverse goes from measures to quarters
        # Use the base TableMap class for the inverse
        if self._y_decreasing:
            return TableMap(
                x_values=list(reversed(self._y_original)),
                y_values=list(reversed(self._x_original)),
                kind=self._kind,
                extrapolate=self._extrapolate,
                source_unit=self._target_unit,
                target_unit=self._source_unit,
            )
        else:
            return TableMap(
                x_values=self._y_original,
                y_values=self._x_original,
                kind=self._kind,
                extrapolate=self._extrapolate,
                source_unit=self._target_unit,
                target_unit=self._source_unit,
            )

    def to_measure_number_map(
        self,
        mns: Sequence[str] | None = None,
    ) -> QuartersToMeasureNumber:
        """Convert to non-interpolating measure number map.

        Args:
            mns: MN labels to use. If None, uses floor(y_value) as string.

        Returns:
            QuartersToMeasureNumber with the same boundaries.
        """
        # Exclude final extrapolation point
        boundaries = list(self._x_original[:-1])

        if mns is None:
            # Generate MN labels from y values
            mns_list = [str(int(y)) for y in self._y[:-1]]
        else:
            mns_list = list(mns)

        return QuartersToMeasureNumber(
            boundaries=boundaries,
            mns=mns_list,
            source_unit=self._source_unit,
        )

    def _to_manifest(self, *, minimal: bool, entries: MeasureMapIds) -> dict[str, Any]:
        """Write the ``FloatingMeasureMap`` of the measure map this lattice reads.

        The view names the timeline's measure map when its lattice is this
        one, and otherwise a measure map of its own, built from the knots:
        one measure per pair of knots, the first an anacrusis when the
        ordinals start at 0.

        Raises:
            ValueError: If the units are not quarters and floating measures,
                or the knots are no measure lattice.
        """
        if (self._source_unit, self._target_unit) != (
            TimeUnit.quarters,
            TimeUnit.floating_measures,
        ):
            raise self._no_form("a measure view reads quarters as floating measures")
        own = entries.own
        if own is not None and self._is_lattice_of(own):
            measure_map_id = entries.id_of(own)
        else:
            measure_map_id = entries.id_of_rows(self._lattice_rows())
        return self._map_object(
            "FloatingMeasureMap", {"measure_map": measure_map_id}, minimal=minimal
        )

    def _is_lattice_of(self, measure_map: MeasureMap) -> bool:
        """Whether *measure_map*'s lattice has this map's knots and ordinals."""
        try:
            x_values, y_values = self._lattice(measure_map)
        except ValueError:
            return False
        return (
            x_values == self._x_exact
            and [_as_exact(y) for y in y_values] == self._y_exact
        )

    def _lattice_rows(self) -> list[dict[str, Any]]:
        """Return the Measure Map rows of the measures whose lattice this is.

        Raises:
            ValueError: If the ordinals do not count up by one from 0 or 1,
                or the first knot is neither 0 under ordinal 1 nor before 0
                under ordinal 0 (an anacrusis, whose bar the second knot ends
                after 0).
        """
        xs, ys = self._x_exact, self._y_exact
        first = ys[0]
        if first not in (0, 1) or ys != [first + index for index in range(len(ys))]:
            raise self._no_form(
                "its ordinals do not count measures up by one from 0 or 1"
            )
        if first == 1 and xs[0] != 0:
            raise self._no_form(f"its first downbeat lies at {xs[0]}, not at 0")
        if first == 0 and not xs[0] < 0 < xs[1]:
            raise self._no_form(
                f"its anacrusis bar [{xs[0]}, {xs[1]}) does not contain 0"
            )
        rows: list[dict[str, Any]] = []
        for index, (start, end) in enumerate(zip(xs, xs[1:]), start=1):
            qstamp = max(start, Fraction(0))
            row: dict[str, Any] = {
                "count": index,
                "qstamp": number_to_json(qstamp),
                "nominal_length": number_to_json(end - start),
                "actual_length": number_to_json(end - qstamp),
            }
            if start < 0:
                row["offset_within_measure"] = number_to_json(-start)
            rows.append(row)
        return rows

    def __repr__(self) -> str:
        n = len(self._x) - 1  # Exclude extrapolation point
        return f"QuartersToFloatingMeasures(n_measures={n})"


# endregion
