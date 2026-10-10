"""MeasureMapLoader: Load MeasureMap JSON files into TimeToAlign!

MeasureMap is a platform-neutral format for capturing measure structure and
flow control information. This loader parses the JSON format (both compressed
and expanded forms) and returns a ScoreStore with MeasureData.

From the MeasureMap paper:
> "We propose the 'Measure Map' for capturing the information about each
> bar-like unit of a symbolic encoding that is essential for the alignment
> of sources."

Key features:
- Parses both compressed and expanded MeasureMap JSON
- Implements expansion rules per MeasureMap paper specification
- Validates MC uniqueness, qstamp monotonicity, next references
- Returns ScoreStore for consistency with other ScoreLoaders
- Enables cross-validation against Ms3Loader (measures.tsv)

MeasureMap JSON Schema Properties:
| Property | Type | Description |
|----------|------|-------------|
| ID | string | Unique identifier (defaults to string of count) |
| count | int >=1 | Position in MM (analogous to MC) |
| qstamp | number >=0 | Symbolic time in quarter notes |
| number | int >=0 | Conventional number (MN as integer) |
| name | string | Label with suffix (e.g., "19a", "19b") |
| time_signature | string | E.g., "3/8", "C", "cadenza" |
| nominal_length | number | Expected duration from time signature |
| actual_length | number >0 | Real duration |
| start_repeat | bool/number | Start repeat marker (||:) |
| end_repeat | bool/number | End repeat marker (:||) |
| next | array | IDs/counts of bars that can follow |
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from timetoalign.core import NumberType, TimeUnit
from timetoalign.timelines.structure import MeasureMap

from .base import ScoreLoader
from .store import ScoreStore
from .stores import AnnotationEventData, ControlEventData, MeasureData, NoteEventData

module_logger = logging.getLogger(__name__)


class MeasureMapLoader(ScoreLoader):
    """Load MeasureMap JSON files into ScoreStore.

    MeasureMapLoader parses the MeasureMap JSON format (as specified in the
    MeasureMap paper) and returns a ScoreStore with populated MeasureData.

    The loader handles:
    - Compressed MeasureMap (minimal entries, defaults inferred)
    - Expanded MeasureMap (all fields explicit)
    - Validation of structure (MC unique, qstamp monotonic, next valid)
    - The standard's anacrusis convention: a first row numbered 0 that is
      shorter than its nominal length is the closing part of nominal bar 0,
      at offset ``nominal_length - actual_length`` within it (see
      :meth:`_apply_anacrusis_convention`)
    - ``-1`` in ``next``, the measure-table spelling of the end of the piece:
      the rows are read without it (it names no row) and the measure map
      leaves it out, while ``expanded_data``, the measure table and
      :meth:`compute_default_traversal` keep it where the file states it

    Known limitation: a ``.mm.json`` file states no offset for the parts of
    an inner split bar, so they load as ``IrregularMeasure``s, whereas an
    ``Ms3Loader`` measure table stating ``mc_offset`` yields
    ``MeasureConstituent``s for the same bar.

    Attributes:
        unit: Always TimeUnit.quarters (MeasureMap uses quarter beats).
        expanded_data: The fully expanded measure list after loading.

    Examples:
        >>> from timetoalign.testdata import ensure_data
        >>> path = (
        ...     ensure_data("score")
        ...     / "couperin_concerts"
        ...     / "c05n05_musete.measures.mm.json"
        ... )
        >>> loader = MeasureMapLoader().load(path)
        >>> len(loader.store.measures)
        58

        The measure data also summarises the flow control it carries:

        >>> summary = loader.store.measures.get_flow_control_summary()
        >>> summary['repeat_starts'], summary['repeat_ends']
        (2, 3)
    """

    _default_unit = TimeUnit.quarters

    def __init__(self, **kwargs: Any) -> None:
        """Initialize MeasureMapLoader."""
        super().__init__(
            unit=TimeUnit.quarters, number_type=NumberType.fraction, **kwargs
        )
        self._expanded_data: list[dict[str, Any]] = []

    @property
    def expanded_data(self) -> list[dict[str, Any]]:
        """The fully expanded measure data after loading."""
        return self._expanded_data

    def _load_source(self, source: Path) -> ScoreStore:
        """Load a MeasureMap JSON file.

        Args:
            source: Path to the .mm.json file.

        Returns:
            ScoreStore with populated MeasureData.

        Raises:
            FileNotFoundError: If source doesn't exist.
            json.JSONDecodeError: If JSON is invalid.
            ValueError: If MeasureMap structure is invalid.
        """
        if not source.exists():
            raise FileNotFoundError(f"MeasureMap file not found: {source}")

        with open(source, encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError(f"MeasureMap must be a JSON array, got {type(raw_data)}")

        if not raw_data:
            return ScoreStore.empty()

        # Measure Map files written from measure tables spell the end of the
        # piece -1, which names no row: the rows are read without it, and the
        # loader's own table and traversal keep it where the file states it.
        rows = [
            (
                {**row, "next": [entry for entry in row["next"] if entry != -1]}
                if isinstance(row, dict) and isinstance(row.get("next"), list)
                else row
            )
            for row in raw_data
        ]
        # Expand through the canonical authoring lane.
        expanded = MeasureMap.expand_rows(rows)
        counts_by_id = {bar["ID"]: bar["count"] for bar in expanded}
        ends = {
            row["count"] if "count" in row else counts_by_id[row["ID"]]
            for row in raw_data
            if -1 in (row.get("next") or ())
        }
        for bar in expanded:
            if bar["count"] in ends:
                bar["next"] = [*bar["next"], -1]
        self._apply_anacrusis_convention(expanded)
        incomplete = next(
            (
                row
                for row in expanded
                if row["actual_length"] is None or row["nominal_length"] is None
            ),
            None,
        )
        if incomplete is not None:
            raise ValueError(
                f"MeasureMap file {source} has no symbolic measure lengths at "
                f"row {incomplete['count']}"
            )
        self._expanded_data = expanded

        # Validate the expanded data
        self._validate_measuremap(expanded)

        # Convert to MeasureData rows
        rows = self._to_event_rows(expanded)

        measures_data = MeasureData.from_dicts(
            rows,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )

        return ScoreStore(
            notes=NoteEventData.empty(),
            measures=measures_data,
            controls=ControlEventData.empty(),
            annotations=AnnotationEventData.empty(),
            metadata={
                "format": "measuremap_json",
                "parser": "MeasureMapLoader",
                "source": str(source),
                "n_measures": len(expanded),
                "flow_control": measures_data.get_flow_control_summary(),
            },
        )

    def _measure_map(self) -> MeasureMap:
        """Build the measure map from the expanded rows, not the measure table.

        The end-of-piece ``-1`` the loader keeps for its traversal names no
        measure, so the map's successors leave it out.
        """
        return MeasureMap.from_expanded_rows(
            [
                {**bar, "next": [entry for entry in bar["next"] if entry != -1]}
                for bar in self._expanded_data
            ]
        )

    @staticmethod
    def _apply_anacrusis_convention(expanded: list[dict[str, Any]]) -> None:
        """Give an anacrusis its offset inside nominal bar 0.

        The Measure Map standard states one convention for a short opening
        bar: a first row numbered 0 whose actual length is shorter than its
        nominal length is an anacrusis, the closing part of nominal bar 0.
        Its offset within that bar is therefore ``nominal - actual``. This
        loader applies the convention to a first row that states no
        ``offset_within_measure`` of its own, before measures are built; no
        other row is ever read as part of a split measure.

        Args:
            expanded: Rows returned by :meth:`MeasureMap.expand_rows`,
                updated in place.
        """
        first = expanded[0]
        nominal = first["nominal_length"]
        actual = first["actual_length"]
        if (
            first["number"] == 0
            and not first["offset_within_measure"]
            and nominal is not None
            and actual is not None
            and actual < nominal
        ):
            first["offset_within_measure"] = nominal - actual

    def _validate_measuremap(self, expanded: list[dict[str, Any]]) -> None:
        """Validate the expanded MeasureMap.

        Checks:
        - MC (count) values are unique
        - qstamp values are monotonically increasing
        - next references point to valid MCs or -1

        Args:
            expanded: The expanded measure list.

        Raises:
            ValueError: If validation fails.
        """
        if not expanded:
            return

        # Check MC uniqueness
        mcs = [bar["count"] for bar in expanded]
        if len(mcs) != len(set(mcs)):
            duplicates = [mc for mc in mcs if mcs.count(mc) > 1]
            raise ValueError(f"Duplicate MC values: {set(duplicates)}")

        # Check qstamp monotonicity
        qstamps = [bar["qstamp"] for bar in expanded]
        for i in range(1, len(qstamps)):
            if qstamps[i] < qstamps[i - 1]:
                raise ValueError(
                    f"qstamp not monotonic: bar {expanded[i]['count']} has "
                    f"qstamp {qstamps[i]} < previous {qstamps[i - 1]}"
                )

        # Check next references
        mc_set = set(mcs)
        for bar in expanded:
            for next_mc in bar["next"]:
                if next_mc != -1 and next_mc not in mc_set:
                    raise ValueError(
                        f"Bar {bar['count']} references invalid next MC: {next_mc}"
                    )

    def _to_event_rows(self, expanded: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert expanded MeasureMap to MeasureData row format.

        Args:
            expanded: The expanded measure list.

        Returns:
            List of row dictionaries for MeasureData.from_dicts().
        """
        from timetoalign.core.time import coordinate_to_struct

        rows: list[dict[str, Any]] = []

        for bar in expanded:
            qstamp = bar["qstamp"]  # Fraction
            actual_length = bar["actual_length"]  # Fraction
            end_qstamp = qstamp + actual_length

            row: dict[str, Any] = {
                # Identity
                "id": f"measure_{bar['count']}",
                "name": f"M{bar['name']}",
                "temporal_type": "interval",
                "event_type": "Measure",
                # MeasureMap fields -> MeasureData mapping
                "mc": bar["count"],
                "mn": bar["name"],
                "mn_int": bar["number"],
                "mm_id": bar["ID"],
                # Temporal - use coordinate_to_struct for base fields
                "start": coordinate_to_struct(qstamp),
                "duration": coordinate_to_struct(actual_length),
                "end": coordinate_to_struct(end_qstamp),
                "nominal_length": float(bar["nominal_length"]),
                "actual_length": float(actual_length),
                # Signature
                "timesig": bar["time_signature"],
                # Flow control
                "start_repeat": bar["start_repeat"],
                "end_repeat": bar["end_repeat"],
                "next": bar["next"],  # Will be converted to string in from_dicts
            }

            rows.append(row)

        return rows

    # ===== Traversal Computation =====

    def compute_default_traversal(self) -> list[int]:
        """Compute the default traversal sequence from the expanded data.

        The algorithm follows the 'next' field, using visit counts to
        choose which branch to take at repeat points.

        From the design spec:
        ```python
        sequence = []
        current_mc = 1
        visit_counts = defaultdict(int)

        while current_mc != -1:
            bar = get_bar_by_mc(mm_data, current_mc)
            sequence.append(bar["ID"])
            next_options = bar["next"]
            visit_counts[current_mc] += 1
            idx = min(visit_counts[current_mc] - 1, len(next_options) - 1)
            current_mc = next_options[idx]

        return TraversalMap(sequence=sequence)
        ```

        Returns:
            List of MC values in traversal order.
        """
        from collections import defaultdict

        if not self._expanded_data:
            return []

        # Build MC -> bar lookup
        mc_to_bar: dict[int, dict[str, Any]] = {
            bar["count"]: bar for bar in self._expanded_data
        }

        sequence: list[int] = []
        visit_counts: dict[int, int] = defaultdict(int)
        current_mc = 1  # Start at MC 1

        # Safety limit to prevent infinite loops
        max_iterations = len(self._expanded_data) * 10

        for _ in range(max_iterations):
            if current_mc == -1 or current_mc not in mc_to_bar:
                break

            bar = mc_to_bar[current_mc]
            sequence.append(current_mc)

            next_options = bar["next"]
            if not next_options:
                break
            visit_counts[current_mc] += 1

            # Choose next based on visit count
            idx = min(visit_counts[current_mc] - 1, len(next_options) - 1)
            current_mc = next_options[idx]

        return sequence

    def get_traversal_summary(self) -> dict[str, Any]:
        """Get summary of the default traversal.

        Returns:
            Dict with traversal statistics.
        """
        traversal = self.compute_default_traversal()
        return {
            "folded_measures": len(self._expanded_data),
            "unfolded_measures": len(traversal),
            "traversal_sequence": traversal[:20],  # First 20 for preview
            "has_repeats": len(traversal) > len(self._expanded_data),
        }
