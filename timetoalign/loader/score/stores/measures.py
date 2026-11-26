"""MeasureData: Storage for measure boundary events.

This module provides the MeasureData class which stores measure information
from both MeasureMap JSON files and measures.tsv files. The schema is designed
to support:

1. **MeasureMap JSON format** (from measure_map paper):
   - ID, count, qstamp, number, name, time_signature, nominal_length, actual_length
   - start_repeat, end_repeat, next (flow control)

2. **measures.tsv format** (ms3/DCML):
   - mc, mn, quarterbeats, duration_qb, timesig, keysig
   - mc_offset, volta, repeats, breaks, next (flow control)

The unified schema allows cross-validation between loaders and enables
derivation of flow-aware maps (MetricMap, MeasureNumberMap, etc.).

Key concepts (from MeasureMap paper):
- **MC (Measure Count)**: Monotonically increasing integer (1-indexed), unique
- **MN (Measure Number)**: Conventional label musicians see (may repeat, have suffixes)
- **Split bars**: Same MN for multiple MCs (e.g., bar split by repeat sign)
- **Anacrusis**: Incomplete first measure, typically MN=0
- **Volta**: Alternative endings (prima/seconda volta)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa
from typing_extensions import Self

from timetoalign.core import IntervalPolicy, NumberType, TimeUnit
from timetoalign.core.fields import ONE_OR_SEVERAL_SUFFIX, split_row_member
from timetoalign.core.ids import IdMinter
from timetoalign.loader._ids import give_event_ids, minter_of
from timetoalign.storage.events import EventData

if TYPE_CHECKING:
    pass

module_logger = logging.getLogger(__name__)


class MeasureData(EventData):
    """EventData for measure boundary events.

    This schema supports both MeasureMap JSON and measures.tsv formats,
    enabling cross-validation and unified measure handling.

    Core Fields (Identity):
        mc: Measure Count - monotonically increasing integer (1-indexed)
        mn: Measure Number label - string like "1", "0", "19a", "19b"
        mn_int: Measure Number as integer (for sorting/grouping)

    Temporal Fields:
        quarterbeats / start: Measure start in continuous logical time
        duration / duration_qb: Measure length in quarter beats
        nominal_length: Expected duration from time signature
        actual_length: Real duration (may differ for anacrusis, irregular bars)
        mc_offset: Offset within split bar (e.g., Fraction(1/4) for second half)

    Signature Fields:
        timesig: Time signature string ("2/4", "4/4", "6/8")
        timesig_num, timesig_den: Numerator and denominator
        keysig: Key signature string
        keysig_fifths: Circle of fifths position (-7 to 7)

    Flow Control Fields:
        start_repeat: True if this bar has a repeat start marker (||:)
        end_repeat: True if this bar has a repeat end marker (:||)
        next: Comma-separated list of MCs that can follow this bar
        volta: Ending number (1, 2, ...) or null
        volta_list: Ending numbers of a bracket that states several
            (``[1, 3]``), or null; a row fills ``volta`` or ``volta_list``,
            as its source states the bracket, never both
        repeats: Legacy field ("start", "end", "firstMeasure")
        breaks: Section boundary marker ("section", etc.)
        dont_count: If true, skip in MN counting

    Context Fields:
        barline: Barline type ("double", etc.)
        part_id: Part identifier for multi-part scores
    """

    _extra_fields: ClassVar[list[pa.Field]] = [
        # ===== Core Identity =====
        pa.field(
            "mc",
            pa.int64(),
            nullable=False,
            metadata={"description": "Measure Count (monotonic, 1-indexed)"},
        ),
        pa.field(
            "mn",
            pa.string(),
            nullable=True,
            metadata={"description": "Measure Number label (may have suffix)"},
        ),
        pa.field(
            "mn_int",
            pa.int64(),
            nullable=True,
            metadata={"description": "Measure Number as integer"},
        ),
        pa.field(
            "mm_id",
            pa.string(),
            nullable=True,
            metadata={"description": "MeasureMap ID (string of count)"},
        ),
        # ===== Temporal - Derived/Explicit =====
        pa.field(
            "nominal_length",
            pa.float64(),
            nullable=True,
            metadata={"description": "Expected duration from timesig"},
        ),
        pa.field(
            "actual_length",
            pa.float64(),
            nullable=True,
            metadata={"description": "Real duration (may differ)"},
        ),
        pa.field(
            "mc_offset",
            pa.string(),
            nullable=True,
            metadata={"description": "Offset in split bar as fraction string"},
        ),
        pa.field(
            "quarterbeats_all_endings",
            pa.string(),
            nullable=True,
            metadata={"description": "Alt qstamp including all endings"},
        ),
        # ===== Signatures =====
        pa.field("timesig", pa.string(), nullable=True),
        pa.field("timesig_num", pa.int64(), nullable=True),
        pa.field("timesig_den", pa.int64(), nullable=True),
        pa.field("keysig", pa.string(), nullable=True),
        pa.field("keysig_fifths", pa.int64(), nullable=True),
        pa.field("keysig_mode", pa.string(), nullable=True),  # "major", "minor"
        # ===== Flow Control =====
        pa.field(
            "start_repeat",
            pa.bool_(),
            nullable=True,
            metadata={"description": "Repeat start marker (||:)"},
        ),
        pa.field(
            "end_repeat",
            pa.bool_(),
            nullable=True,
            metadata={"description": "Repeat end marker (:||)"},
        ),
        pa.field(
            "next",
            pa.string(),
            nullable=True,
            metadata={"description": "Comma-separated list of possible next MCs"},
        ),
        pa.field(
            "volta",
            pa.int64(),
            nullable=True,
            metadata={"description": "Ending number (1, 2, ...)"},
        ),
        pa.field(
            "volta" + ONE_OR_SEVERAL_SUFFIX,
            pa.list_(pa.int64()),
            nullable=True,
            metadata={
                "description": "Ending numbers of a bracket stating several ([1, 3])"
            },
        ),
        pa.field(
            "repeats",
            pa.string(),
            nullable=True,
            metadata={"description": "Legacy: start/end/firstMeasure"},
        ),
        pa.field(
            "breaks",
            pa.string(),
            nullable=True,
            metadata={"description": "Section boundary marker"},
        ),
        pa.field(
            "markers",
            pa.string(),
            nullable=True,
            metadata={"description": "Target marker name (segno, coda, codab, ...)"},
        ),
        pa.field(
            "jump_bwd",
            pa.string(),
            nullable=True,
            metadata={"description": "Backward jump target name (segno, start, ...)"},
        ),
        pa.field(
            "jump_fwd",
            pa.string(),
            nullable=True,
            metadata={
                "description": "Forward jump target name (coda, codab, fine, ...)"
            },
        ),
        pa.field(
            "play_until",
            pa.string(),
            nullable=True,
            metadata={
                "description": "Play-until target name on after-DC/DS pass (coda, fine, ...)"
            },
        ),
        pa.field(
            "dont_count",
            pa.bool_(),
            nullable=True,
            metadata={"description": "Skip in MN counting"},
        ),
        pa.field("numbering_offset", pa.int64(), nullable=True),
        # ===== Context =====
        pa.field("barline", pa.string(), nullable=True),
        pa.field("part_id", pa.string(), nullable=True),
    ]

    @classmethod
    def empty(
        cls,
        unit: TimeUnit = TimeUnit.quarters,
        number_type: NumberType | None = None,
    ) -> Self:
        """Create empty MeasureData."""
        return super().empty(unit, number_type)

    @classmethod
    def from_dicts(
        cls,
        rows: list[dict[str, Any]],
        unit: TimeUnit = TimeUnit.quarters,
        number_type: NumberType | None = None,
        *,
        interval_policy: IntervalPolicy = IntervalPolicy.warn,
        seed: int | None = None,
        minter: IdMinter | None = None,
    ) -> Self:
        """Create from dicts.

        Handles field mapping from both MeasureMap JSON and measures.tsv formats.

        Args:
            rows: List of measure dictionaries.
            unit: Time unit (default: quarters).
            number_type: Number type for coordinates.
            interval_policy: How to handle end/duration inconsistencies.
            seed: The seed of a fresh minter for the IDs of rows without one.
            minter: The minter of those IDs (a loader's); with neither, the
                content minter of the event types
                (:func:`~timetoalign.storage.mint_event_ids`). A stated
                ``id`` is kept.

        Returns:
            MeasureData instance.

        Raises:
            TypeError: If both *seed* and *minter* are given.
        """
        if not rows:
            return cls.empty(unit, number_type)

        from timetoalign.storage.schema import make_table_metadata

        schema = cls.get_schema(unit)
        metadata = make_table_metadata(
            unit, unit.resolve_number_type(number_type), loader_class=cls.__name__
        )
        schema = schema.with_metadata(metadata)

        processed_rows = []
        for row in rows:
            processed = dict(row)

            # ===== Map MeasureMap JSON fields to schema =====
            # MeasureMap uses "count" for MC, "number" for MN as int
            if "count" in processed and "mc" not in processed:
                processed["mc"] = processed.pop("count")
            if "number" in processed:
                processed["mn_int"] = processed.pop("number")
            if "name" in processed and "mn" not in processed:
                processed["mn"] = processed.get("name")  # Keep name, also set mn
            if "ID" in processed:
                processed["mm_id"] = processed.pop("ID")
            if "qstamp" in processed and "start" not in processed:
                processed["start"] = processed.pop("qstamp")
            if "time_signature" in processed and "timesig" not in processed:
                processed["timesig"] = processed.pop("time_signature")

            # ===== Map measures.tsv fields to schema =====
            if "quarterbeats" in processed and "start" not in processed:
                processed["start"] = processed.pop("quarterbeats")
            if "duration_qb" in processed and "duration" not in processed:
                processed["duration"] = processed.pop("duration_qb")
            processed.pop("duration_qb_float", None)
            if "act_dur" in processed and "actual_length" not in processed:
                # act_dur is typically a fraction string like "1/2"
                processed["actual_length"] = processed.pop("act_dur")

            # ===== Handle 'next' field (array in JSON, string in TSV) =====
            if "next" in processed:
                next_val = processed["next"]
                if isinstance(next_val, list):
                    # Convert array to comma-separated string
                    processed["next"] = ", ".join(str(x) for x in next_val)
                # else: already a string from TSV

            # ===== Keep 'volta' as stated: a number, or a list beside it =====
            split_row_member(processed, "volta")

            # ===== Parse timesig into components =====
            if "timesig" in processed and processed["timesig"]:
                ts = str(processed["timesig"])
                if "/" in ts:
                    parts = ts.split("/")
                    try:
                        processed.setdefault("timesig_num", int(parts[0]))
                        processed.setdefault("timesig_den", int(parts[1]))
                    except (ValueError, IndexError):
                        pass

            # ===== Compute nominal_length from timesig if not provided =====
            if processed.get("timesig_num") and processed.get("timesig_den"):
                if (
                    "nominal_length" not in processed
                    or processed["nominal_length"] is None
                ):
                    # quarters_per_measure = (timesig_num / timesig_den) * 4
                    # e.g., 2/4 = 2 quarters, 6/8 = 3 quarters
                    processed["nominal_length"] = (
                        processed["timesig_num"] / processed["timesig_den"] * 4
                    )

            # Remove unused legacy fields
            processed.pop("quarterbeats_float", None)
            # Keep 'name' field - it's part of the base schema and should be preserved
            # (e.g., "M1", "M2" for measure labels)

            # Unified interval normalisation: converts coordinate fields
            # to struct format, fills missing end/duration, and infers
            # temporal_type.
            EventData._normalize_intervals_row(
                processed,
                policy=interval_policy,
                number_type=unit.resolve_number_type(number_type),
            )

            processed_rows.append(processed)

        give_event_ids(processed_rows, minter_of(seed, minter))
        table = pa.Table.from_pylist(processed_rows, schema=schema)
        return cls(table, unit, number_type)

    # ===== Flow Control Helpers =====

    def get_flow_control_summary(self) -> dict[str, Any]:
        """Get summary of flow control elements in this measure data.

        Returns:
            Dict with counts of repeats, voltas, breaks, etc.
        """
        summary: dict[str, Any] = {
            "total_measures": len(self),
            "has_repeats": False,
            "has_voltas": False,
            "has_breaks": False,
            "repeat_starts": 0,
            "repeat_ends": 0,
            "voltas": set(),
            "breaks": set(),
        }

        if len(self) == 0:
            return summary

        # Count repeat starts
        if "start_repeat" in self._table.column_names:
            start_repeat_arr = self._table.column("start_repeat")
            true_count = sum(1 for v in start_repeat_arr.to_pylist() if v is True)
            summary["repeat_starts"] = true_count
            summary["has_repeats"] = true_count > 0

        # Count repeat ends
        if "end_repeat" in self._table.column_names:
            end_repeat_arr = self._table.column("end_repeat")
            true_count = sum(1 for v in end_repeat_arr.to_pylist() if v is True)
            summary["repeat_ends"] = true_count
            summary["has_repeats"] = summary["has_repeats"] or true_count > 0

        # Count voltas: the ending numbers stated, singly or in a list
        voltas: set[int] = set()
        if "volta" in self._table.column_names:
            voltas.update(
                v for v in self._table.column("volta").to_pylist() if v is not None
            )
        list_column = "volta" + ONE_OR_SEVERAL_SUFFIX
        if list_column in self._table.column_names:
            for numbers in self._table.column(list_column).to_pylist():
                voltas.update(numbers or ())
        summary["voltas"] = voltas
        summary["has_voltas"] = len(voltas) > 0

        # Count breaks
        if "breaks" in self._table.column_names:
            breaks_arr = self._table.column("breaks")
            unique_breaks = set(breaks_arr.to_pylist())
            summary["breaks"] = {b for b in unique_breaks if b is not None}
            summary["has_breaks"] = len(summary["breaks"]) > 0

        return summary

    def has_flow_control(self) -> bool:
        """Check if this measure data contains any flow control elements."""
        summary = self.get_flow_control_summary()
        return summary["has_repeats"] or summary["has_voltas"] or summary["has_breaks"]
