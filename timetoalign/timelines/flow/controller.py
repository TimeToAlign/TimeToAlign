"""Flow API: Compute unfolded measure sequences from flow control data.

This module provides classes for computing and representing "flows" — the
sequence of measure visitations that results from following flow control
instructions (repeats, voltas, D.S., D.C., etc.).

From the design spec (measure_handling_design.md Part 14):

    FlowMode: Enum for flow computation modes
    FlowStep: A single step in a Flow sequence
    Flow: A computed flow (sequence of measure visitations)
    FlowMap: Attached to timelines for coordinate transformation
    ScoreFlowController: Compute Flow paths from MeasureData

Terminology:
    - "Flow" is used uniformly instead of "traversal"
    - mc_playthrough: Monotonically increasing index in unfolded sequence
    - mn_playthrough: MN with occurrence suffix (e.g., "19a", "19b")

Gold Standard Conventions (from ms3):
    - mn_playthrough suffix: Always 'a' for first occurrence, 'b' for second
    - Split bars: Same mn_playthrough for all MCs sharing same MN
    - quarterbeats: Cumulative in unfolded, NOT reset at jumps
    - mc_playthrough: Monotonically increasing 1, 2, 3, ...
    - Final row: next = -1 indicates end of flow
"""

from __future__ import annotations

import logging
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Iterator

from timetoalign.core.enums import FlowMode, IncompletePosition
from timetoalign.storage import EventData

from .flowmap import FlowMap
from .measures import (
    CompleteMeasure,
    CompleteMeasureGroup,
    IncompleteGroup,
    IncompleteMeasure,
    MeasureGroup,
    MeasureUnit,
    OverlengthGroup,
    OverlengthMeasure,
    SplitMeasure,
    TypedMeasure,
    VoltaGroup,
)
from .naming import SegmentNameGenerator
from .sections import AtomicSection, Flow, FlowDiagnostic, PlaythroughSection
from .unfolding import compute_qb_sections

if TYPE_CHECKING:
    from timetoalign.display.ascii import Diagram

    from ..flowcontrol import Break, FlowControlRegistry, Jump, Marker

module_logger = logging.getLogger(__name__)


class FlowControllerBase(ABC):
    """Abstract base class for computing flow control transformations.

    A flow controller is a factory for producing Flows and FlowMaps. It operates
    as a background processor that computes flow control transformations but
    is NOT stored on the Timeline itself. Instead, the FlowMaps it produces
    are attached to Timelines.

    Subclasses implement the specific logic for different data sources:
    - ScoreFlowController: Works with MeasureData (mc, mn, next[], volta, etc.)
    - Future: AudioFlowController, VideoFlowController for other media types.

    Design Decisions:
    - A flow controller is a factory, NOT stored on Timeline
    - Sparse sections ARE allowed (atomic sections need not cover entire timeline)
    - Unit independence: Flows are sequences of TimeIntervals in ANY unit

    Public API:
        - iter_atomic_sections(): Iterate atomic (indivisible) sections
        - compute_flow(mode): Compute Flow for the given mode
        - create_flow_map(flow): Create FlowMap from computed Flow
    """

    @abstractmethod
    def iter_atomic_sections(self) -> Iterator[tuple[Fraction, Fraction]]:
        """Iterate over atomic (indivisible) sections.

        Each section is a tuple (start, end) representing a contiguous
        portion that cannot be split by flow control.

        Yields:
            Tuples of (start_coordinate, end_coordinate).
        """
        ...

    @abstractmethod
    def compute_flow(self, mode: FlowMode | None = None) -> Flow:
        """Compute a Flow for the given mode.

        Args:
            mode: The FlowMode to compute. None defaults to ATOMIC.

        Returns:
            The computed Flow.
        """
        ...

    def create_flow_map(self, flow: Flow | None = None) -> FlowMap:
        """Create a FlowMap from a computed Flow.

        If this controller has MeasureUnit data (i.e. is a ScoreFlowController),
        the FlowMap is built in **quarterbeat space** using `compute_qb_sections`.
        Otherwise falls back to MC-number space (legacy behaviour).

        Args:
            flow: The Flow to create a map from. If None, computes DEFAULT flow.

        Returns:
            FlowMap for coordinate transformation.
        """
        if flow is None:
            flow = self.compute_flow(FlowMode.default)
        # Use QB-space if controller has MeasureUnit data
        try:
            qb_sections = compute_qb_sections(flow, self)
            return FlowMap.from_qb_sections(flow, qb_sections, id=flow.mode.value)
        except (AttributeError, ValueError):
            # Fallback to MC-space for controllers without MeasureUnit data
            return FlowMap(flow, id=flow.mode.value)


class ScoreFlowController(FlowControllerBase):
    """Score flow controller specialized for score data (MeasureData).

    ScoreFlowController computes Flow paths from MeasureData, which contains
    measure-level flow control information (mc, mn, next[], volta, etc.).
    A controller instance is intended for single-threaded use.

    The algorithm operates at the section level:
    1. Derives atomic sections from next[] arrays OR accepts from partitura
    2. Uses flow control markers + volta attributes to execute flow logic
    3. Groups atomic sections into playthrough sections per FlowMode

    The algorithm follows the 'next' field in measure data, using visit
    counts to choose which branch to take at repeat points. This matches
    the ms3 unfolding algorithm.

    Attributes:
        measures: The source MeasureData.

    Public API:
        - get_sections(mode=None): Get sections (None=atomic, else playthrough)
        - iter_sections(mode=None): Iterate over sections
        - iter_units(): Iterate over MeasureUnits (folded skeleton)
        - compute_flow(mode): Compute Flow for the given mode

    Examples:
        >>> controller = ScoreFlowController(measure_data)
        >>> flow = controller.compute_flow()
        >>> print(f"Unfolded: {flow.unfolded_length} measures")

        >>> # Get atomic sections (folded structure)
        >>> for sec in controller.get_sections():
        ...     print(f"{sec.id}: MC [{sec.mc_start},{sec.mc_end})")

        >>> # Get playthrough sections for DEFAULT mode
        >>> for sec in controller.get_sections(FlowMode.default):
        ...     print(f"MC [{sec.mc_start},{sec.mc_end})")

        >>> # Iterate over MeasureUnits
        >>> for unit in controller.iter_units():
        ...     print(f"MC {unit.mc}: next={unit.next}, jump_from={unit.jump_from}")
    """

    def __init__(
        self,
        measures: EventData,
        *,
        name_generator: SegmentNameGenerator | None = None,
    ) -> None:
        """Initialize the score flow controller from MeasureData.

        Args:
            measures: MeasureData containing flow control fields.
            name_generator: Strategy for labelling atomic sections. Defaults
                to a ``SegmentNameGenerator`` with the standard alphabet and
                the volta-suffix rule enabled.
        """
        self._measures = measures
        self._name_generator = name_generator or SegmentNameGenerator()
        self._measure_lookup: dict[int, dict[str, Any]] = {}
        self._units: list[MeasureUnit] = []
        self._atomic_sections: list[AtomicSection] = []
        self._traversal_diagnostics: dict[FlowMode, list[FlowDiagnostic]] = {}
        self._build_lookup()
        self._build_units()
        self._build_atomic_sections()

    @classmethod
    def from_atomic_sections(
        cls,
        sections: list[AtomicSection],
        measures: EventData | None = None,
        *,
        name_generator: SegmentNameGenerator | None = None,
    ) -> "ScoreFlowController":
        """Initialize directly from atomic sections (e.g., from partitura).

        Args:
            sections: List of AtomicSection objects.
            measures: Optional MeasureData for detailed step computation.
            name_generator: Strategy for labelling atomic sections. Stored for
                consistency with the regular constructor; the pre-built
                sections are not relabelled. Defaults to a fresh
                ``SegmentNameGenerator``.

        Returns:
            ScoreFlowController with pre-built atomic sections.
        """
        # Create instance without calling __init__
        instance = object.__new__(cls)
        instance._measures = measures
        instance._name_generator = name_generator or SegmentNameGenerator()
        instance._measure_lookup = {}
        instance._units = []
        instance._atomic_sections = list(sections)
        instance._traversal_diagnostics = {}

        # Build lookup and units if measures provided
        if measures is not None:
            instance._build_lookup()
            instance._build_units()

        return instance

    def _build_lookup(self) -> None:
        """Build MC -> measure dict lookup from MeasureData.

        Extracts all available fields including FlowControl markers:
        - Basic: mc, mn, duration, next, quarterbeats
        - FlowControl: volta, timesig, start_repeat, end_repeat, segno, coda, fine
        """
        if len(self._measures) == 0:
            return

        schema_names = self._measures.schema.names

        # Get field data
        mc_vals = self._measures.column_values("mc")
        mn_vals = (
            self._measures.column_values("mn")
            if "mn" in schema_names
            else [str(mc) for mc in mc_vals]
        )

        # Get duration - column_values() decodes the coordinate struct to
        # an exact Fraction (or falls back to actual_length when the
        # measure data carries no dedicated duration column).
        if "duration" in schema_names:
            duration_values = self._measures.column_values(
                "duration", default=Fraction(4)
            )
        elif "actual_length" in schema_names:
            al_vals = self._measures.column_values("actual_length")
            duration_values = [Fraction(d) if d else Fraction(4) for d in al_vals]
        else:
            duration_values = [Fraction(4)] * len(mc_vals)

        # Get 'next' field
        next_vals = self._measures.column_values("next")

        # Get 'start' (quarterbeats) - decoded to an exact Fraction.
        qb_values = self._measures.column_values("start", default=Fraction(0))

        # Get FlowControl fields (with safe defaults)
        volta_vals = self._measures.column_values("volta")
        timesig_vals = self._measures.column_values("timesig")
        start_repeat_vals = self._measures.column_values("start_repeat", default=False)
        end_repeat_vals = self._measures.column_values("end_repeat", default=False)
        segno_vals = self._measures.column_values("segno")
        coda_vals = self._measures.column_values("coda")
        fine_vals = self._measures.column_values("fine", default=False)
        # ms3 fields carrying marker name and jump targets per measure
        markers_vals = self._measures.column_values("markers")
        jump_bwd_vals = self._measures.column_values("jump_bwd")
        jump_fwd_vals = self._measures.column_values("jump_fwd")
        play_until_vals = self._measures.column_values("play_until")
        # section_break: check 'section_break' field first, then 'breaks' field
        if "section_break" in schema_names:
            section_break_vals = self._measures.column_values(
                "section_break", default=False
            )
        elif "breaks" in schema_names:
            # The breaks field may contain compound values like
            # "page & section" or "section & page"; check for substring.
            breaks_vals = self._measures.column_values("breaks")
            section_break_vals = [
                "section" in str(b) if b else False for b in breaks_vals
            ]
        else:
            section_break_vals = [False] * len(mc_vals)

        for i, mc in enumerate(mc_vals):
            # Parse 'next' field
            next_val = next_vals[i]
            if next_val is None or next_val == "":
                # Default: next MC or -1 for last
                if i < len(mc_vals) - 1:
                    next_list = [mc_vals[i + 1]]
                else:
                    next_list = [-1]
            elif isinstance(next_val, str):
                # Parse various string formats:
                # - Comma-separated: "2, 3, 4" or "2,3,4"
                # - Tuple string: "(2,)" or "(2, 3)"
                # - Single value: "2"
                next_list = []
                # Remove tuple parentheses if present
                cleaned = next_val.strip()
                if cleaned.startswith("(") and cleaned.endswith(")"):
                    cleaned = cleaned[1:-1]
                # Split by comma
                for part in cleaned.split(","):
                    part = part.strip()
                    if part:
                        try:
                            next_list.append(int(part))
                        except ValueError:
                            pass
                if not next_list:
                    # If we couldn't parse anything, default to next MC or -1
                    if i < len(mc_vals) - 1:
                        next_list = [mc_vals[i + 1]]
                    else:
                        next_list = [-1]
            elif isinstance(next_val, list):
                next_list = next_val
            else:
                next_list = [-1]

            # Resolve marker name and jump targets — prefer dedicated
            # segno/coda/fine columns when present, fall back to ms3
            # markers/jump_fwd/play_until columns.
            marker = markers_vals[i] if markers_vals[i] else None
            jump_bwd = jump_bwd_vals[i] if jump_bwd_vals[i] else None
            jump_fwd = jump_fwd_vals[i] if jump_fwd_vals[i] else None
            play_until = play_until_vals[i] if play_until_vals[i] else None

            segno_name = segno_vals[i] if segno_vals[i] else None
            coda_name = coda_vals[i] if coda_vals[i] else None
            if marker is not None:
                # ms3 'markers' column carries the target marker NAME.
                if marker.startswith("segno"):
                    segno_name = segno_name or marker
                elif marker.startswith("coda"):
                    # both 'coda' and 'codab' are coda-type markers
                    coda_name = coda_name or marker
                elif marker == "fine":
                    # fine marker can also appear in markers column
                    fine_vals[i] = True

            # 'fine' marker location is signalled by an explicit marker or by
            # jump_fwd='fine'. play_until='fine' is the D.S./D.C.-al-fine
            # *instruction*, which lives elsewhere — not a fine marker.
            fine_flag = bool(fine_vals[i]) or jump_fwd == "fine" or marker == "fine"

            self._measure_lookup[mc] = {
                "mc": mc,
                "mn": str(mn_vals[i]) if mn_vals[i] is not None else str(mc),
                "duration_qb": duration_values[i],
                "quarterbeats": qb_values[i],
                "next": next_list,
                # FlowControl fields
                "volta": volta_vals[i],
                "timesig": timesig_vals[i],
                "start_repeat": bool(start_repeat_vals[i]),
                "end_repeat": bool(end_repeat_vals[i]),
                "segno": segno_name,
                "coda": coda_name,
                "fine": fine_flag,
                "section_break": bool(section_break_vals[i]),
                # Raw ms3 jump-instruction fields (kept for downstream inference)
                "marker": marker,
                "jump_bwd": jump_bwd,
                "jump_fwd": jump_fwd,
                "play_until": play_until,
            }

    def _build_units(self) -> None:
        """Create MeasureUnits from the measure lookup.

        MeasureUnits represent the folded score skeleton - one per MeasureData row.
        Populates all FlowControlElement fields including computed values:
        - jump_from, jump_to derived from next[] patterns
        - timesig_duration_qb computed from timesig string
        - flow_control_types tuple for serialization
        """
        if not self._measure_lookup:
            return

        sorted_mcs = sorted(self._measure_lookup.keys())

        for mc in sorted_mcs:
            info = self._measure_lookup[mc]
            next_list = info.get("next", [-1])

            # Compute jump_from: MC where next[] indicates a jump
            # A jump is when next != [MC+1] (non-contiguous)
            mc_idx = sorted_mcs.index(mc)
            is_jump_from = self._is_jump_from(mc, next_list, sorted_mcs, mc_idx)

            # Compute jump_to: MC that is a jump target (segno, coda, or non-adjacent next target)
            is_jump_to = self._is_jump_to(mc, info, sorted_mcs)

            # Parse timesig to duration in quarterbeats
            timesig = info.get("timesig")
            timesig_duration_qb = self._parse_timesig_duration(timesig)

            # Build flow_control_types tuple
            flow_control_types = self._extract_flow_control_types(
                info, is_jump_from, is_jump_to
            )

            unit = MeasureUnit(
                mc=mc,
                mn=str(info.get("mn", mc)),
                duration_qb=info.get("duration_qb", Fraction(4)),
                next=tuple(next_list),
                volta=info.get("volta"),
                timesig=timesig,
                timesig_duration_qb=timesig_duration_qb,
                start_repeat=bool(info.get("start_repeat")),
                end_repeat=bool(info.get("end_repeat")),
                jump_from=is_jump_from,
                jump_to=is_jump_to,
                segno=info.get("segno"),
                coda=info.get("coda"),
                fine=bool(info.get("fine")),
                section_break=bool(info.get("section_break")),
                flow_control_types=flow_control_types,
                jump_bwd=info.get("jump_bwd"),
                jump_fwd=info.get("jump_fwd"),
                play_until=info.get("play_until"),
            )
            self._units.append(unit)

    def _is_jump_from(
        self, mc: int, next_list: list[int], sorted_mcs: list[int], mc_idx: int
    ) -> bool:
        """Determine if this MC is a jump origin.

        A jump origin is an MC where next[] indicates non-contiguous continuation:
        - next has multiple options (volta/repeat choice)
        - next jumps backward (repeat end)
        - next jumps forward > MC+1 (skip/coda)

        Args:
            mc: Current MC.
            next_list: List of possible next MCs.
            sorted_mcs: Sorted list of all MCs.
            mc_idx: Index of mc in sorted_mcs.

        Returns:
            True if this MC is a jump origin.
        """
        if len(next_list) > 1:
            # Multiple next options (volta or conditional jump)
            return True

        if not next_list or next_list == [-1]:
            # End of piece, not a jump
            return False

        next_mc = next_list[0]

        # Check if next is contiguous
        if mc_idx < len(sorted_mcs) - 1:
            expected_next = sorted_mcs[mc_idx + 1]
            if next_mc != expected_next:
                return True
        else:
            # A final measure may still carry a D.C./D.S. continuation.
            # Only ``-1`` denotes a terminal successor (handled above).
            return True

        return False

    def _is_jump_to(self, mc: int, info: dict[str, Any], sorted_mcs: list[int]) -> bool:
        """Determine if this MC is a jump target.

        A jump target is an MC that:
        - Has a segno marker
        - Has a coda marker
        - Is the target of a non-adjacent next[] from another MC

        Args:
            mc: Current MC.
            info: Measure info dict.
            sorted_mcs: Sorted list of all MCs.

        Returns:
            True if this MC is a jump target.
        """
        # Explicit markers
        if info.get("segno") or info.get("coda"):
            return True

        # Check if any other MC jumps to this MC (non-adjacent)
        mc_idx = sorted_mcs.index(mc)
        for other_mc in sorted_mcs:
            if other_mc == mc:
                continue
            other_info = self._measure_lookup[other_mc]
            other_next = other_info.get("next", [])
            if mc in other_next:
                other_idx = sorted_mcs.index(other_mc)
                # Check if jump is non-adjacent
                if mc_idx != other_idx + 1:
                    return True

        return False

    def _parse_timesig_duration(self, timesig: str | None) -> Fraction | None:
        """Parse time signature to expected duration in quarterbeats.

        Args:
            timesig: Time signature string like "4/4", "3/4", "6/8".

        Returns:
            Expected duration in quarterbeats, or None if unparseable.

        Examples:
            >>> _parse_timesig_duration("4/4")
            Fraction(4, 1)
            >>> _parse_timesig_duration("3/4")
            Fraction(3, 1)
            >>> _parse_timesig_duration("6/8")
            Fraction(3, 1)  # 6 eighth notes = 3 quarterbeats
        """
        if timesig is None:
            return None
        try:
            # Handle compound time signatures
            parts = timesig.split("/")
            if len(parts) != 2:
                return None
            num, denom = int(parts[0]), int(parts[1])
            # Duration in quarterbeats = num * (4 / denom)
            return Fraction(num * 4, denom)
        except (ValueError, ZeroDivisionError):
            return None

    def _extract_flow_control_types(
        self, info: dict[str, Any], is_jump_from: bool, is_jump_to: bool
    ) -> tuple[str, ...]:
        """Extract FlowControlElement values from measure info.

        Builds a tuple of FlowControlElement.value strings for serialization.
        Recognises ms3 jump-instruction columns (``jump_bwd``, ``jump_fwd``,
        ``play_until``) and folds them into the canonical FlowControlElement
        vocabulary (``dal_segno``, ``dal_segno_al_coda``, ``dal_segno_al_fine``,
        ``da_capo``, ``da_capo_al_coda``, ``da_capo_al_fine``, ``to_coda``).

        Args:
            info: Measure info dict.
            is_jump_from: Whether this MC is a jump origin.
            is_jump_to: Whether this MC is a jump target.

        Returns:
            Tuple of FlowControlElement value strings.
        """
        types: list[str] = []

        if info.get("start_repeat"):
            types.append("repeat_start")
        if info.get("end_repeat"):
            types.append("repeat_end")
        if info.get("segno"):
            types.append("segno")
        if info.get("coda"):
            types.append("coda")
        if info.get("fine"):
            types.append("fine")
        if info.get("section_break"):
            types.append("section_break")

        # Derive jump instruction type from ms3 columns
        jump_bwd = info.get("jump_bwd")
        jump_fwd = info.get("jump_fwd")
        play_until = info.get("play_until")
        # 'start' as backward target means D.C. (jump to beginning); a named
        # marker (typically 'segno') means D.S.
        bwd_is_dc = jump_bwd in ("start", "firstMeasure")
        bwd_is_ds = jump_bwd is not None and not bwd_is_dc
        if bwd_is_ds:
            if play_until == "coda":
                types.append("dal_segno_al_coda")
            elif play_until == "fine":
                types.append("dal_segno_al_fine")
            else:
                types.append("dal_segno")
        elif bwd_is_dc:
            if play_until == "coda":
                types.append("da_capo_al_coda")
            elif play_until == "fine":
                types.append("da_capo_al_fine")
            else:
                types.append("da_capo")
        elif jump_fwd is not None and jump_fwd != "fine":
            # Forward-only jump (e.g., 'to coda') with no preceding D.S./D.C.
            types.append("to_coda")

        if is_jump_from:
            types.append("jump_from")
        if is_jump_to:
            types.append("jump_to")

        return tuple(types)

    def _type_measure(self, unit: MeasureUnit) -> TypedMeasure:
        """Create a typed copy of a MeasureUnit (Typing step of the two-step algorithm).

        Compares the measure's actual duration with the expected duration from
        the time signature to classify measures as:
        - IncompleteMeasure: actual < expected (anacrusis, final, split)
        - CompleteMeasure: actual == expected (normal measure)
        - OverlengthMeasure: actual > expected (fermata, cadenza)

        The typed copy inherits all properties from the generating MeasureUnit,
        including FlowControlElements.

        Args:
            unit: The MeasureUnit to type.

        Returns:
            IncompleteMeasure, CompleteMeasure, or OverlengthMeasure.
        """
        base_kwargs = {
            "mc": unit.mc,
            "mn": unit.mn,
            "duration_qb": unit.duration_qb,
            "next": unit.next,
            "volta": unit.volta,
            "timesig": unit.timesig,
            "timesig_duration_qb": unit.timesig_duration_qb,
            "start_repeat": unit.start_repeat,
            "end_repeat": unit.end_repeat,
            "jump_from": unit.jump_from,
            "jump_to": unit.jump_to,
            "segno": unit.segno,
            "coda": unit.coda,
            "fine": unit.fine,
            "section_break": unit.section_break,
            "flow_control_types": unit.flow_control_types,
            "jump_bwd": unit.jump_bwd,
            "jump_fwd": unit.jump_fwd,
            "play_until": unit.play_until,
        }

        # If no time signature info, default to CompleteMeasure
        if unit.timesig_duration_qb is None:
            return CompleteMeasure(**base_kwargs)

        # Compare durations
        if unit.duration_qb < unit.timesig_duration_qb:
            position = self._determine_incomplete_position(unit)
            return IncompleteMeasure(**base_kwargs, position=position)
        elif unit.duration_qb > unit.timesig_duration_qb:
            return OverlengthMeasure(**base_kwargs)
        else:
            return CompleteMeasure(**base_kwargs)

    def _determine_incomplete_position(self, unit: MeasureUnit) -> IncompletePosition:
        """Determine the position of an incomplete measure.

        Uses the unit's position in the score to classify:
        - First measure -> ANACRUSIS
        - Last measure -> FINAL
        - Otherwise -> SPLIT_FIRST (may be refined by the Grouping step)

        Args:
            unit: The incomplete MeasureUnit.

        Returns:
            IncompletePosition classification.
        """
        if not self._units:
            return IncompletePosition.unknown

        # Find index of this unit
        idx = next((i for i, u in enumerate(self._units) if u.mc == unit.mc), -1)
        if idx == -1:
            return IncompletePosition.unknown

        if idx == 0:
            return IncompletePosition.anacrusis
        elif idx == len(self._units) - 1:
            return IncompletePosition.final
        else:
            # Check if this might be part of a split measure
            # For now, classify as SPLIT_FIRST (the Grouping step will refine)
            return IncompletePosition.split_first

    # ========================================================================
    # Grouping step: Grouping methods
    # ========================================================================

    def _build_groups(
        self, typed_measures: tuple[TypedMeasure, ...]
    ) -> tuple[MeasureGroup, ...]:
        """Group typed measures into MeasureGroups (Grouping step of the two-step algorithm).

        Algorithm for complete coverage (every measure belongs to exactly one group):
        1. Identify VoltaGroups (consecutive measures with same volta number)
        2. Identify SplitMeasures (IncompleteMeasures with same mn that sum to time sig)
        3. Identify isolated IncompleteMeasures -> IncompleteGroup
        4. Identify OverlengthMeasures -> OverlengthGroup
        5. Group remaining CompleteMeasures into CompleteMeasureGroups

        Args:
            typed_measures: Tuple of typed measures from the Typing step.

        Returns:
            Tuple of MeasureGroup objects covering all measures.
        """
        if not typed_measures:
            return ()

        # Track which measures have been grouped
        grouped_mcs: set[int] = set()
        groups: list[MeasureGroup] = []

        # Step 1: VoltaGroups (consecutive measures with same volta number)
        volta_groups = self._group_voltas(typed_measures, grouped_mcs)
        groups.extend(volta_groups)

        # Step 2: SplitMeasures (IncompleteMeasures with same mn)
        split_groups = self._group_splits(typed_measures, grouped_mcs)
        groups.extend(split_groups)

        # Step 3: Remaining IncompleteMeasures -> IncompleteGroup
        incomplete_groups = self._group_incompletes(typed_measures, grouped_mcs)
        groups.extend(incomplete_groups)

        # Step 4: OverlengthMeasures -> OverlengthGroup
        overlength_groups = self._group_overlengths(typed_measures, grouped_mcs)
        groups.extend(overlength_groups)

        # Step 5: Remaining CompleteMeasures -> CompleteMeasureGroup
        complete_groups = self._group_completes(typed_measures, grouped_mcs)
        groups.extend(complete_groups)

        # Sort groups by mc_start for consistent ordering
        groups.sort(key=lambda g: g.mc_start)

        return tuple(groups)

    def _group_voltas(
        self, typed_measures: tuple[TypedMeasure, ...], grouped_mcs: set[int]
    ) -> list[VoltaGroup]:
        """Group consecutive measures with the same volta number.

        Volta groups are identified by the volta attribute (1, 2, 3...).
        A VoltaGroup is contained within ONE AtomicSection - multiple voltas
        are separated by Breaks and cannot be in the same section.

        Args:
            typed_measures: Tuple of typed measures.
            grouped_mcs: Set of MCs already grouped (will be updated).

        Returns:
            List of VoltaGroup objects.
        """
        groups: list[VoltaGroup] = []
        current_volta: int | None = None
        current_members: list[TypedMeasure] = []

        for m in typed_measures:
            if m.volta is not None and m.mc not in grouped_mcs:
                if m.volta == current_volta:
                    # Continue current volta group
                    current_members.append(m)
                else:
                    # Start new volta group (save previous if exists)
                    if current_members and current_volta is not None:
                        groups.append(
                            VoltaGroup(
                                members=tuple(current_members),
                                volta_number=current_volta,
                            )
                        )
                        for member in current_members:
                            grouped_mcs.add(member.mc)
                    current_volta = m.volta
                    current_members = [m]
            else:
                # Non-volta measure: finalize current group if exists
                if current_members and current_volta is not None:
                    groups.append(
                        VoltaGroup(
                            members=tuple(current_members),
                            volta_number=current_volta,
                        )
                    )
                    for member in current_members:
                        grouped_mcs.add(member.mc)
                current_volta = None
                current_members = []

        # Finalize last group
        if current_members and current_volta is not None:
            groups.append(
                VoltaGroup(
                    members=tuple(current_members),
                    volta_number=current_volta,
                )
            )
            for member in current_members:
                grouped_mcs.add(member.mc)

        return groups

    def _group_splits(
        self, typed_measures: tuple[TypedMeasure, ...], grouped_mcs: set[int]
    ) -> list[SplitMeasure]:
        """Group IncompleteMeasures that together form a complete metrical unit.

        Detection strategy: Measures with the same mn (measure number) are
        candidates for split measures. Their durations should sum to the
        time signature duration.

        Args:
            typed_measures: Tuple of typed measures.
            grouped_mcs: Set of MCs already grouped (will be updated).

        Returns:
            List of SplitMeasure objects.
        """
        groups: list[SplitMeasure] = []

        # Group IncompleteMeasures by mn
        by_mn: dict[str, list[IncompleteMeasure]] = defaultdict(list)
        for m in typed_measures:
            if isinstance(m, IncompleteMeasure) and m.mc not in grouped_mcs:
                by_mn[m.mn].append(m)

        # Check each group with same mn
        for mn, measures in by_mn.items():
            if len(measures) > 1:
                # Check if they sum to time signature duration
                total = sum((m.duration_qb for m in measures), Fraction(0))
                # Get expected duration from first measure
                expected = measures[0].timesig_duration_qb
                if expected is not None and total == expected:
                    # Valid split measure
                    groups.append(SplitMeasure(members=tuple(measures)))
                    for m in measures:
                        grouped_mcs.add(m.mc)

        return groups

    def _group_incompletes(
        self, typed_measures: tuple[TypedMeasure, ...], grouped_mcs: set[int]
    ) -> list[IncompleteGroup]:
        """Group isolated IncompleteMeasures that don't form splits.

        These are measures that haven't been grouped yet - typically anacrusis
        or final measures that will be paired with their complement when the
        flow brings them together in PlaythroughSections.

        Args:
            typed_measures: Tuple of typed measures.
            grouped_mcs: Set of MCs already grouped (will be updated).

        Returns:
            List of IncompleteGroup objects (one per isolated incomplete).
        """
        groups: list[IncompleteGroup] = []

        for m in typed_measures:
            if isinstance(m, IncompleteMeasure) and m.mc not in grouped_mcs:
                # Create singleton IncompleteGroup
                groups.append(IncompleteGroup(members=(m,)))
                grouped_mcs.add(m.mc)

        return groups

    def _group_overlengths(
        self, typed_measures: tuple[TypedMeasure, ...], grouped_mcs: set[int]
    ) -> list[OverlengthGroup]:
        """Group OverlengthMeasures.

        Currently creates individual groups for each overlength measure.
        Future: could group adjacent overlength measures (cadenzas).

        Args:
            typed_measures: Tuple of typed measures.
            grouped_mcs: Set of MCs already grouped (will be updated).

        Returns:
            List of OverlengthGroup objects.
        """
        groups: list[OverlengthGroup] = []

        for m in typed_measures:
            if isinstance(m, OverlengthMeasure) and m.mc not in grouped_mcs:
                groups.append(OverlengthGroup(members=(m,)))
                grouped_mcs.add(m.mc)

        return groups

    def _group_completes(
        self, typed_measures: tuple[TypedMeasure, ...], grouped_mcs: set[int]
    ) -> list[CompleteMeasureGroup]:
        """Group adjacent CompleteMeasures together.

        Creates CompleteMeasureGroup for runs of consecutive CompleteMeasures
        that haven't been grouped yet.

        Args:
            typed_measures: Tuple of typed measures.
            grouped_mcs: Set of MCs already grouped (will be updated).

        Returns:
            List of CompleteMeasureGroup objects.
        """
        groups: list[CompleteMeasureGroup] = []
        current_run: list[CompleteMeasure] = []

        for m in typed_measures:
            if isinstance(m, CompleteMeasure) and m.mc not in grouped_mcs:
                # Check if adjacent to current run
                if current_run and m.mc == current_run[-1].mc + 1:
                    current_run.append(m)
                else:
                    # Save current run and start new one
                    if current_run:
                        groups.append(CompleteMeasureGroup(members=tuple(current_run)))
                        for member in current_run:
                            grouped_mcs.add(member.mc)
                    current_run = [m]
            else:
                # Non-complete or already grouped: save current run
                if current_run:
                    groups.append(CompleteMeasureGroup(members=tuple(current_run)))
                    for member in current_run:
                        grouped_mcs.add(member.mc)
                    current_run = []

        # Finalize last run
        if current_run:
            groups.append(CompleteMeasureGroup(members=tuple(current_run)))
            for member in current_run:
                grouped_mcs.add(member.mc)

        return groups

    def iter_units(self) -> Iterator[MeasureUnit]:
        """Iterate over MeasureUnits (the folded score skeleton).

        Yields:
            MeasureUnit objects in MC order.

        Examples:
            >>> for unit in controller.iter_units():
            ...     print(f"MC {unit.mc}: {unit.mn}, next={unit.next}")
        """
        yield from self._units

    # region Flow control object accessors

    def _qb_at_mc(self, mc: int) -> Fraction:
        """Quarterbeat coordinate at the start of an MC, as a Fraction."""
        info = self._measure_lookup.get(mc, {})
        qb = info.get("quarterbeats", Fraction(0))
        return qb if isinstance(qb, Fraction) else Fraction(qb)

    def _qb_at_mc_end(self, mc: int) -> Fraction:
        """Quarterbeat coordinate at the END of an MC (start + duration)."""
        info = self._measure_lookup.get(mc, {})
        qb = info.get("quarterbeats", Fraction(0))
        dur = info.get("duration_qb", Fraction(0))
        if not isinstance(qb, Fraction):
            qb = Fraction(qb)
        if not isinstance(dur, Fraction):
            dur = Fraction(dur)
        return qb + dur

    def get_breaks(self) -> list["Break"]:
        """Return all `Break` events derived from the score's flow control.

        Currently emits one `Break` per measure flagged as a section break.
        Each `Break` is positioned at the END coordinate of its MC and
        carries a label indicating the source MC.

        Returns:
            List of `timetoalign.timelines.flowcontrol.Break` objects, sorted
            by coordinate.
        """
        from timetoalign.core import Coordinate, TimeUnit
        from timetoalign.core.enums import ActivationCondition, FlowControlElement

        from ..flowcontrol import Break

        breaks: list[Break] = []
        for unit in self._units:
            if unit.section_break:
                breaks.append(
                    Break(
                        coordinate=Coordinate(
                            self._qb_at_mc_end(unit.mc), TimeUnit.quarters
                        ),
                        control_type=FlowControlElement.section_break,
                        condition=ActivationCondition.always,
                        label=f"section break after MC {unit.mc}",
                    )
                )
            if unit.fine:
                breaks.append(
                    Break(
                        coordinate=Coordinate(
                            self._qb_at_mc_end(unit.mc), TimeUnit.quarters
                        ),
                        control_type=FlowControlElement.fine,
                        condition=ActivationCondition.after_dc_ds,
                        name="fine",
                        label=f"fine at MC {unit.mc}",
                    )
                )
        return breaks

    def get_jumps(self) -> list["Jump"]:
        """Return all `Jump` events derived from the score's flow control.

        Emits one `Jump` per repeat-end, plus one per D.S./D.C./to-coda
        instruction (including the ``_al_coda`` and ``_al_fine`` variants).
        Each jump's `from_coordinate` is the END of the originating MC and
        its `to_coordinate` is the START of the destination MC.

        Returns:
            List of `timetoalign.timelines.flowcontrol.Jump` objects, sorted
            by their from-coordinate.
        """
        from timetoalign.core import Coordinate, TimeUnit
        from timetoalign.core.enums import ActivationCondition, FlowControlElement

        from ..flowcontrol import Jump

        jumps: list[Jump] = []
        sorted_mcs = sorted(self._measure_lookup)

        # Find the MC and the marker id of any named target marker (segno,
        # coda, codab, fine); the first marker of a name is the target.
        marker_mc: dict[str, int] = {}
        for unit in self._units:
            if unit.segno:
                marker_mc.setdefault(unit.segno, unit.mc)
            if unit.coda:
                marker_mc.setdefault(unit.coda, unit.mc)
            if unit.fine:
                marker_mc.setdefault("fine", unit.mc)
        marker_ids: dict[str, str] = {}
        for marker in self.get_markers():
            marker_ids.setdefault(marker.name, marker.id)

        # Resolve every repeat-end to its back-jump target once, via the
        # nearest-first candidate generator (see _repeat_end_resolution).
        repeat_targets, _ = self._repeat_end_resolution()

        for unit in self._units:
            from_qb = self._qb_at_mc_end(unit.mc)
            from_coord = Coordinate(from_qb, TimeUnit.quarters)

            # Repeat: an end-repeat jumps back to its resolved candidate
            # target. A repeat-end with no candidate is dangling and emits no
            # jump (the diagnostic is surfaced by flow_diagnostics()).
            if unit.end_repeat:
                target_mc = repeat_targets.get(unit.mc)
                if target_mc is not None:
                    jumps.append(
                        Jump(
                            from_coordinate=from_coord,
                            to_coordinate=Coordinate(
                                self._qb_at_mc(target_mc), TimeUnit.quarters
                            ),
                            control_type=FlowControlElement.repeat_end,
                            condition=ActivationCondition.first_n,
                            repeat_count=1,
                            label=f"MC {unit.mc} → MC {target_mc}",
                        )
                    )

            # D.S./D.C. (and -al-coda/-al-fine) — fire after the first pass.
            fct = unit.flow_control_types
            jump_type: FlowControlElement | None = None
            target_name: str
            for cand in (
                FlowControlElement.dal_segno_al_coda,
                FlowControlElement.dal_segno_al_fine,
                FlowControlElement.da_capo_al_coda,
                FlowControlElement.da_capo_al_fine,
                FlowControlElement.dal_segno,
                FlowControlElement.da_capo,
                FlowControlElement.to_coda,
            ):
                if cand.value in fct:
                    jump_type = cand
                    break
            if jump_type is None:
                continue

            if jump_type in (
                FlowControlElement.da_capo,
                FlowControlElement.da_capo_al_coda,
                FlowControlElement.da_capo_al_fine,
            ):
                if not sorted_mcs:
                    continue
                to_mc = sorted_mcs[0]
                target_name = "start"
            elif jump_type == FlowControlElement.to_coda:
                target_name = unit.jump_fwd or "coda"
                to_mc = marker_mc.get(target_name)
            else:
                # dal_segno / dal_segno_al_coda / dal_segno_al_fine
                target_name = unit.jump_bwd or "segno"
                to_mc = marker_mc.get(target_name)

            if to_mc is None:
                continue

            jumps.append(
                Jump(
                    from_coordinate=from_coord,
                    to_coordinate=Coordinate(self._qb_at_mc(to_mc), TimeUnit.quarters),
                    control_type=jump_type,
                    condition=ActivationCondition.after_first,
                    target=marker_ids.get(target_name),
                    label=f"MC {unit.mc} → {target_name} (MC {to_mc})",
                )
            )

        return jumps

    def get_markers(self) -> list["Marker"]:
        """Return all named target markers (segno / coda / fine instances).

        Returns:
            List of `timetoalign.timelines.flowcontrol.Marker` objects in MC
            order, with minted ids ``marker_1``, ``marker_2``, ... ``name``
            is the marker instance name (e.g., ``"segno"``, ``"coda"``,
            ``"codab"``, ``"fine"``); ``coordinate`` is the start of the MC.
        """
        from timetoalign.core import Coordinate, TimeUnit

        from ..flowcontrol import FlowControlRegistry, Marker

        registry = FlowControlRegistry()
        for unit in self._units:
            coord = Coordinate(self._qb_at_mc(unit.mc), TimeUnit.quarters)
            for name in (unit.segno, unit.coda, "fine" if unit.fine else None):
                if name:
                    registry.add_marker(Marker(coord, name=name))
        return registry.markers

    def get_flow_control_registry(self) -> "FlowControlRegistry":
        """Return a `FlowControlRegistry` populated from this controller.

        Bundles the breaks, jumps, and named markers into a single object,
        which is the canonical registry shape used elsewhere in the library.
        """
        from ..flowcontrol import FlowControlRegistry

        registry = FlowControlRegistry()
        for brk in self.get_breaks():
            registry.add_break(brk)
        for marker in self.get_markers():
            registry.add_marker(marker)
        for jump in self.get_jumps():
            registry.add_jump(jump)
        return registry

    @property
    def breaks(self) -> list["Break"]:
        """All `Break` events derived from the score (see `get_breaks()`)."""
        return self.get_breaks()

    @property
    def jumps(self) -> list["Jump"]:
        """All `Jump` events derived from the score (see `get_jumps()`)."""
        return self.get_jumps()

    @property
    def markers(self) -> list["Marker"]:
        """All named target markers (see `get_markers()`)."""
        return self.get_markers()

    # endregion

    def _build_atomic_sections(self) -> None:
        """Derive atomic sections from next[] arrays.

        Note:
            Uses right-open interval convention [mc_start, mc_end).

        Algorithm:
        1. Find section boundaries where next != [MC+1]
        2. Create AtomicSection for each contiguous run
        3. Set section_type based on flow control patterns
        4. Build to[] arrays from grouped next[] values

        Section boundaries occur when:
        - next contains multiple options (volta/repeat)
        - next jumps backward (repeat end)
        - next jumps forward > MC+1 (skip/coda)
        - next is [-1] (end of piece)
        """
        if not self._measure_lookup:
            return

        sorted_mcs = sorted(self._measure_lookup.keys())
        if not sorted_mcs:
            return

        # Find segment boundaries
        # Boundaries occur at:
        # 1. Start of piece (MC 1)
        # 2. Jump targets (any MC that is destination of a non-adjacent next[])
        #    - This includes repeat_start when repeat_end jumps back to it
        # 3. MCs after jump sources (when next[] indicates non-sequential continuation)
        # 4. After explicit section_break markers
        # Note: Uses derived flow info (next column). When principal FlowControlElements
        # are available, they should be preferred for boundary detection.
        boundaries: set[int] = {sorted_mcs[0]}  # Start of first segment
        volta_endings: set[int] = set()  # Track volta ending MCs

        # First pass: identify volta endings
        # Voltas are mutually exclusive paths. A volta situation occurs when:
        # - One target's next jumps backward (volta 1 goes back to repeat)
        # - One target's next continues forward (volta 2 continues to next section)
        for mc in sorted_mcs:
            bar = self._measure_lookup[mc]
            next_list = bar["next"]
            if len(next_list) > 1:
                # Check what each target does
                target_goes_back = False
                target_continues = False
                for target_mc in next_list:
                    if target_mc != -1 and target_mc in self._measure_lookup:
                        target_bar = self._measure_lookup[target_mc]
                        target_next = target_bar.get("next", [])
                        if target_next and target_next[0] != -1:
                            if target_next[0] < target_mc:
                                target_goes_back = True
                            else:
                                target_continues = True
                # Volta: one target goes back, one continues
                if target_goes_back and target_continues:
                    for target_mc in next_list:
                        if target_mc != -1:
                            volta_endings.add(target_mc)

        # Second pass: find boundaries
        for i, mc in enumerate(sorted_mcs):
            bar = self._measure_lookup[mc]
            next_list = bar["next"]

            # NOTE: We do NOT add boundaries at repeat_start markers by themselves.
            # A repeat_start is just a target marker - it only creates a boundary
            # if something actually jumps to it, which is handled by the jump target
            # detection below.

            # Boundary at all jump targets (volta alternatives, conditional paths)
            if len(next_list) > 1:
                for target_mc in next_list:
                    if target_mc != -1 and target_mc in self._measure_lookup:
                        boundaries.add(target_mc)
                # Also add the MC after this one (end of pre-jump section)
                if i < len(sorted_mcs) - 1:
                    boundaries.add(sorted_mcs[i + 1])

            # Boundary at jump targets (when next doesn't continue sequentially)
            elif next_list and next_list[0] != -1:
                target_mc = next_list[0]
                if i < len(sorted_mcs) - 1:
                    sequential_next = sorted_mcs[i + 1]
                    if target_mc != sequential_next:
                        # This is a jump - add boundary at target
                        if target_mc in self._measure_lookup:
                            boundaries.add(target_mc)
                        # Add boundary at the sequential next MC too
                        boundaries.add(sequential_next)

            # Boundary after section_break markers (a section break voids contiguity).
            # TimeIntervals cannot span a coordinate containing a Break.
            if bar.get("section_break"):
                # The section break is AT this MC, so the new section starts at next MC
                if i < len(sorted_mcs) - 1:
                    boundaries.add(sorted_mcs[i + 1])

            # Boundary at the start of every volta bracket. Each ending must
            # occupy its own section so it reads on the diagram as a ┌N volta
            # bracket. A prima/seconda pair already gets boundaries via its
            # jump targets, but a first ending with no following alternative
            # produces no jump target at the volta MC and would otherwise be
            # absorbed into the preceding section, hiding the volta.
            volta_here = bar.get("volta")
            if volta_here is not None:
                prev_volta = (
                    self._measure_lookup[sorted_mcs[i - 1]].get("volta")
                    if i > 0
                    else None
                )
                if prev_volta != volta_here:
                    boundaries.add(mc)

        # Convert to sorted list
        boundaries_list: list[int] = sorted(boundaries)

        # Create atomic sections from boundaries
        sections: list[AtomicSection] = []

        # Build lookup from mc to MeasureUnit for typed_measures
        unit_lookup: dict[int, MeasureUnit] = {u.mc: u for u in self._units}

        # Assign labels to every boundary section in one pass. A section opens
        # a volta bracket when its first measure carries a volta number (the
        # same criterion the diagram uses). The name generator turns these
        # flags into the per-section labels, applying the volta-suffix rule.
        volta_flags: list[bool] = [
            (start_mc in unit_lookup and unit_lookup[start_mc].volta is not None)
            for start_mc in boundaries_list
        ]
        labels = self._name_generator.generate(volta_flags)

        for i, start_mc in enumerate(boundaries_list):
            # Find end MC (last MC before next boundary, or last MC)
            if i + 1 < len(boundaries_list):
                end_mc = boundaries_list[i + 1] - 1
            else:
                end_mc = sorted_mcs[-1]

            # Ensure end_mc is valid
            while end_mc not in self._measure_lookup and end_mc > start_mc:
                end_mc -= 1

            if end_mc < start_mc:
                end_mc = start_mc

            # Determine segment type
            end_bar = self._measure_lookup.get(end_mc, {})
            next_list = end_bar.get("next", [])

            section_type = "default"
            if len(next_list) > 1 or (next_list and next_list[0] < start_mc):
                section_type = "leap_end"
            elif start_mc > 1:
                # Check if this is a jump target
                for mc in sorted_mcs:
                    bar = self._measure_lookup[mc]
                    if start_mc in bar.get("next", []) and mc != start_mc - 1:
                        section_type = "leap_start"
                        break

            # Build to[] array
            #
            # next=-1 in ms3 means "stop here", but a `fine` only stops the
            # piece on the after-DC/DS pass. On the first pass, a fine
            # measure followed by more music continues sequentially. So if
            # next contains -1 and the section's end MC is NOT the absolute
            # last MC of the score, treat -1 as "continue to the next
            # sequential MC" for the purposes of the to[] list.
            end_mc_idx = sorted_mcs.index(end_mc)
            is_absolute_end = end_mc_idx == len(sorted_mcs) - 1

            def _section_letter_for_mc(target_mc: int) -> str | None:
                for j, bnd in enumerate(boundaries_list):
                    if j + 1 < len(boundaries_list):
                        if bnd <= target_mc < boundaries_list[j + 1]:
                            return labels[j]
                    else:
                        if bnd <= target_mc:
                            return labels[j]
                return None

            to_sections: list[str] = []
            for next_mc in next_list or ():
                if next_mc == -1:
                    if is_absolute_end:
                        # Genuine end of piece — no successor.
                        continue
                    # First-pass continuation past a fine marker.
                    next_mc = sorted_mcs[end_mc_idx + 1]
                letter = _section_letter_for_mc(next_mc)
                if letter is not None and letter not in to_sections:
                    to_sections.append(letter)

            # Build typed_measures for this section (Typing step)
            # Collect MeasureUnits in the range [start_mc, end_mc+1) (right-open)
            section_units: list[MeasureUnit] = []
            for mc in range(start_mc, end_mc + 1):
                if mc in unit_lookup:
                    section_units.append(unit_lookup[mc])

            # Type each unit (Typing step)
            typed_measures: tuple[TypedMeasure, ...] | None = None
            groups: tuple[MeasureGroup, ...] | None = None
            if section_units:
                typed_list = [self._type_measure(u) for u in section_units]
                typed_measures = tuple(typed_list)
                # Build groups (Grouping step)
                groups = self._build_groups(typed_measures)

            sections.append(
                AtomicSection(
                    id=labels[i],
                    mc_start=start_mc,
                    mc_end=end_mc + 1,  # Right-open: end is exclusive
                    to=tuple(to_sections),
                    section_type=section_type,
                    typed_measures=typed_measures,
                    groups=groups,
                )
            )

        self._atomic_sections = sections

    def check_invariants(self) -> list[FlowDiagnostic]:
        """Check structural invariants of the atomic flow graph.

        The controller's posture toward a malformed flow is detect-and-report,
        not crash: this method returns a list of :class:`FlowDiagnostic`
        describing every violation it finds, and an empty list when the flow is
        well-formed.

        Currently checks the **volta-follows-volta** invariant: in the atomic
        flow graph a volta section can never have a ``to`` edge to another volta
        section. A prima volta's only out-edge is the repeat back-edge (to the
        repeat-start, a non-volta section); a seconda volta is reached only from
        the repeat-start and continues into the music that follows the bracket.
        Two flow-adjacent voltas therefore indicate a malformed ``next`` array —
        most often a jump target that resolved to the wrong ending. This is the
        ``to`` (flow) edge relation, NOT score-order adjacency: volta sections
        are naturally adjacent in MC order, which is correct; they must not be
        connected by a ``to`` edge.

        Returns:
            One ``FlowDiagnostic(kind="volta_follows_volta", ...)`` per
            offending edge, naming the source and destination section ids; an
            empty list when no invariant is violated.
        """
        unit_lookup: dict[int, MeasureUnit] = {u.mc: u for u in self._units}

        def _is_volta_section(section: AtomicSection) -> bool:
            unit = unit_lookup.get(section.mc_start)
            return unit is not None and unit.volta is not None

        section_by_id: dict[str, AtomicSection] = {
            section.id: section for section in self._atomic_sections
        }

        diagnostics: list[FlowDiagnostic] = []
        for section in self._atomic_sections:
            if not _is_volta_section(section):
                continue
            for target_id in section.to:
                target = section_by_id.get(target_id)
                if target is not None and _is_volta_section(target):
                    diagnostics.append(
                        FlowDiagnostic(
                            kind="volta_follows_volta",
                            message=(
                                f"volta section {section.id!r} flows directly to "
                                f"volta section {target_id!r}; a volta's only "
                                f"out-edge is the repeat back-edge (prima) or the "
                                f"music after the bracket (seconda), so the "
                                f"source section's next/jump target is likely "
                                f"mis-resolved"
                            ),
                            section_id=section.id,
                            mc=section.mc_start,
                        )
                    )
        return diagnostics

    def _repeat_end_resolution(
        self,
    ) -> tuple[dict[int, int], list[FlowDiagnostic]]:
        """Resolve every repeat-end to its back-jump target by candidate search.

        Candidates are the open explicit repeat-start scopes, ordered nearest
        first. Segno and coda markers are not repeat candidates. When no
        explicit scope is open, an encoded edge to the beginning of the piece
        supplies the conventional implicit-repeat fallback.

        - exactly one candidate resolves automatically;
        - several candidates resolve to the nearest and record the ambiguity,
          listing the alternatives;
        - no candidate and no backward edge leaves the repeat-end dangling.

        The controller never raises on a malformed repeat structure; it reports.

        Returns:
            A ``(target_by_end_mc, diagnostics)`` pair. ``target_by_end_mc``
            maps each resolvable repeat-end MC to its back-jump target;
            dangling ends are absent from it and named in ``diagnostics``.
        """
        sorted_mcs = sorted(self._measure_lookup.keys())
        if not sorted_mcs:
            return {}, []

        unit_lookup: dict[int, MeasureUnit] = {u.mc: u for u in self._units}
        first_mc = sorted_mcs[0]
        target_by_end: dict[int, int] = {}
        diagnostics: list[FlowDiagnostic] = []

        # Open explicit repeat scopes, innermost last. Marker instants belong to
        # D.S./D.C. and coda instructions and do not enter this stack.
        scopes: list[int] = []
        for mc in sorted_mcs:
            unit = unit_lookup.get(mc)
            if unit is None:
                continue

            if unit.start_repeat:
                scopes.append(mc)

            if unit.end_repeat:
                candidates = list(reversed(scopes))
                if candidates:
                    target_mc = candidates[0]
                    target_by_end[mc] = target_mc
                    if len(candidates) > 1:
                        alternatives = "; ".join(
                            f"MC {candidate}" for candidate in candidates
                        )
                        diagnostics.append(
                            FlowDiagnostic(
                                kind="ambiguous_repeat_end",
                                message=(
                                    f"repeat-end at MC {mc} has several open explicit "
                                    f"repeat starts; resolving to the nearest, MC "
                                    f"{target_mc}. Candidates (nearest first): "
                                    f"{alternatives}"
                                ),
                                section_id=self._section_id_at(mc),
                                mc=mc,
                            )
                        )
                    scopes.pop()
                    continue

                backward_targets = {
                    target
                    for target in unit.next
                    if target in unit_lookup and target <= mc
                }
                if first_mc in backward_targets:
                    target_by_end[mc] = first_mc
                elif not backward_targets:
                    diagnostics.append(
                        FlowDiagnostic(
                            kind="dangling_repeat_end",
                            message=(
                                f"repeat-end at MC {mc} has no open explicit repeat "
                                f"start and no encoded backward edge; no back-jump "
                                f"is produced"
                            ),
                            section_id=self._section_id_at(mc),
                            mc=mc,
                        )
                    )

        return target_by_end, diagnostics

    def _section_id_at(self, mc: int) -> str | None:
        """Return the atomic-section id covering ``mc``, if any."""
        ids = self._find_atomic_ids(mc, mc + 1)
        return ids[0] if ids else None

    def flow_diagnostics(
        self, mode: FlowMode = FlowMode.default
    ) -> list[FlowDiagnostic]:
        """Report flow-control problems found while resolving and traversing.

        Complements :meth:`check_invariants` (which checks the folded atomic
        graph) with the problems surfaced by executing the flow: repeat-ends
        that are dangling or ambiguous, and traversal cycles that would never
        terminate. Like every diagnostic surface on this controller, it detects
        and reports — it never raises on malformed input.

        Args:
            mode: The flow mode to traverse for cycle detection. Defaults to the
                default (unfolding) flow.

        Returns:
            The repeat-end resolution diagnostics followed by the traversal
            diagnostics; an empty list when the flow is well-formed.
        """
        _, repeat_diagnostics = self._repeat_end_resolution()
        flow = self.compute_flow(mode)
        traversal_diagnostics = self._traversal_diagnostics.get(flow.mode, [])
        return [*repeat_diagnostics, *traversal_diagnostics]

    def get_sections(
        self, mode: FlowMode | None = None
    ) -> list[AtomicSection] | list[PlaythroughSection]:
        """Get list of sections.

        This is the unified API for retrieving sections. Replaces the old
        get_atomic_sections() method with added support for playthrough sections.

        Args:
            mode: If None, returns AtomicSections (default).
                  If specified, returns PlaythroughSections for that mode.

        Returns:
            List of AtomicSection if mode is None, otherwise list of PlaythroughSection.

        Examples:
            >>> # Get atomic sections (folded structure)
            >>> for sec in controller.get_sections():
            ...     print(f"{sec.id}: MC [{sec.mc_start},{sec.mc_end})")

            >>> # Get playthrough sections for DEFAULT mode (unfolded)
            >>> for sec in controller.get_sections(FlowMode.default):
            ...     print(f"MC [{sec.mc_start},{sec.mc_end})")
        """
        if mode is None:
            return list(self._atomic_sections)
        else:
            flow = self.compute_flow(mode)
            return list(flow.sections)

    def iter_sections(
        self, mode: FlowMode | None = None
    ) -> Iterator[AtomicSection | PlaythroughSection]:
        """Iterate over sections.

        Args:
            mode: If None, iterates over AtomicSections (default).
                  If specified, iterates over PlaythroughSections for that mode.

        Yields:
            AtomicSection objects if mode is None, otherwise PlaythroughSection objects.

        Examples:
            >>> # Iterate over atomic sections (folded structure)
            >>> for sec in controller.iter_sections():
            ...     print(f"{sec.id}: MC [{sec.mc_start},{sec.mc_end})")

            >>> # Iterate over playthrough sections (unfolded traversal)
            >>> for sec in controller.iter_sections(FlowMode.default):
            ...     print(f"MC [{sec.mc_start},{sec.mc_end})")
        """
        if mode is None:
            yield from self._atomic_sections
        else:
            flow = self.compute_flow(mode)
            yield from flow.sections

    def _compute_playthrough_sections(
        self, mc_sequence: list[int]
    ) -> list[PlaythroughSection]:
        """Convert MC sequence to PlaythroughSection list with groups.

        Groups consecutive MCs into sections. A new section starts when:
        - There's a non-consecutive MC jump
        - The MC sequence reverses (repeat)

        Includes flow-aware SplitMeasure detection: when a section boundary
        falls between two IncompleteGroups that together form a complete
        measure (e.g., anacrusis at start + final at end of previous section),
        they are merged into a SplitMeasure.

        Note:
            Uses right-open interval convention [mc_start, mc_end).

        Args:
            mc_sequence: List of MC values in traversal order.

        Returns:
            List of PlaythroughSection objects with typed_measures and groups.
        """
        if not mc_sequence:
            return []

        # Build lookup from mc to MeasureUnit for typed_measures
        unit_lookup: dict[int, MeasureUnit] = {u.mc: u for u in self._units}

        sections: list[PlaythroughSection] = []
        current_start = mc_sequence[0]
        current_end = mc_sequence[0]

        def _build_section(start_mc: int, end_mc: int) -> PlaythroughSection:
            """Build a PlaythroughSection with typed_measures and groups."""
            atomic_ids = self._find_atomic_ids(start_mc, end_mc + 1)

            # Collect and type MeasureUnits in the range (Typing step)
            typed_list: list[TypedMeasure] = []
            for mc in range(start_mc, end_mc + 1):
                if mc in unit_lookup:
                    typed_list.append(self._type_measure(unit_lookup[mc]))

            typed_measures = tuple(typed_list) if typed_list else None

            # Build groups (Grouping step)
            groups: tuple[MeasureGroup, ...] | None = None
            if typed_measures:
                groups = self._build_groups(typed_measures)

            return PlaythroughSection(
                mc_start=start_mc,
                mc_end=end_mc + 1,  # Right-open: end is exclusive
                atomic_section_ids=tuple(atomic_ids),
                typed_measures=typed_measures,
                groups=groups,
            )

        for i, mc in enumerate(mc_sequence):
            # Check if this continues the current section
            if i > 0:
                prev_mc = mc_sequence[i - 1]
                # Determine if we need a new section:
                # 1. Non-consecutive or backward jump (repeat, D.S., etc.)
                # 2. Previous MC has section_break=True (breaks=section in TSV)
                #    A PlaythroughSection can NEVER span a section break.
                is_non_consecutive = mc != prev_mc + 1
                has_section_break = (
                    prev_mc in self._measure_lookup
                    and self._measure_lookup[prev_mc].get("section_break", False)
                )
                if is_non_consecutive or has_section_break:
                    # Save current section
                    sections.append(_build_section(current_start, current_end))
                    # Start new section
                    current_start = mc
                    current_end = mc
                else:
                    # Continue current section
                    current_end = mc
            else:
                current_end = mc

        # Don't forget the last section
        sections.append(_build_section(current_start, current_end))

        # Flow-aware SplitMeasure detection: check section boundaries
        sections = self._detect_boundary_splits(sections)

        return sections

    def _detect_boundary_splits(
        self, sections: list[PlaythroughSection]
    ) -> list[PlaythroughSection]:
        """Detect SplitMeasures at section boundaries.

        When a section ends with an IncompleteGroup and the next section
        starts with an IncompleteGroup, check if they form a complete measure
        (e.g., final 3/4 + anacrusis 1/4 = 4/4).

        If so, create a SplitMeasure that spans the boundary.

        Args:
            sections: List of PlaythroughSections.

        Returns:
            Updated list with boundary SplitMeasures detected.
        """
        if len(sections) < 2:
            return sections

        result: list[PlaythroughSection] = []

        for i, sec in enumerate(sections):
            if i == 0:
                result.append(sec)
                continue

            prev_sec = result[-1]

            # Check if previous section ends with IncompleteGroup
            prev_end_incomplete = self._get_end_incomplete(prev_sec)
            # Check if current section starts with IncompleteGroup
            curr_start_incomplete = self._get_start_incomplete(sec)

            if prev_end_incomplete and curr_start_incomplete:
                # Check if they form a complete measure
                total = (
                    prev_end_incomplete.total_duration_qb
                    + curr_start_incomplete.total_duration_qb
                )
                expected = None
                # Get expected duration from any member
                for m in prev_end_incomplete.members:
                    if m.timesig_duration_qb:
                        expected = m.timesig_duration_qb
                        break
                if expected is None:
                    for m in curr_start_incomplete.members:
                        if m.timesig_duration_qb:
                            expected = m.timesig_duration_qb
                            break

                if expected is not None and total == expected:
                    # They form a complete measure! Create SplitMeasure
                    # For now, we note this in the groups but don't restructure
                    # (full restructuring would require re-building groups)
                    module_logger.debug(
                        f"Boundary SplitMeasure detected: "
                        f"section {i-1} end + section {i} start = {total}"
                    )

            result.append(sec)

        return result

    def _get_end_incomplete(self, sec: PlaythroughSection) -> IncompleteGroup | None:
        """Get the IncompleteGroup at the end of a section, if any."""
        if not sec.groups:
            return None
        last_group = sec.groups[-1]
        if isinstance(last_group, IncompleteGroup):
            return last_group
        return None

    def _get_start_incomplete(self, sec: PlaythroughSection) -> IncompleteGroup | None:
        """Get the IncompleteGroup at the start of a section, if any."""
        if not sec.groups:
            return None
        first_group = sec.groups[0]
        if isinstance(first_group, IncompleteGroup):
            return first_group
        return None

    def _find_atomic_ids(self, mc_start: int, mc_end: int) -> list[str]:
        """Find which atomic sections cover the given MC range.

        Args:
            mc_start: First MC of the range (inclusive).
            mc_end: First MC AFTER the range (exclusive, right-open).

        Returns:
            List of atomic section IDs that overlap with the range.
        """
        ids = []
        for sec in self._atomic_sections:
            # Check for overlap (right-open intervals)
            if sec.mc_end > mc_start and sec.mc_start < mc_end:
                ids.append(sec.id)
        return ids

    def compute_flow(self, mode: FlowMode | None = None) -> Flow:
        """Compute a single Flow using the specified mode.

        Args:
            mode: The FlowMode to use. None is equivalent to ATOMIC.

        Returns:
            Computed Flow object.
        """
        # mode=None is equivalent to ATOMIC (the default for AtomicSections)
        if mode is None:
            mode = FlowMode.atomic
        self._traversal_diagnostics[mode] = []

        if mode == FlowMode.atomic:
            return self._compute_atomic_flow()
        elif mode == FlowMode.printed:
            return self._compute_printed_flow()
        elif mode == FlowMode.single:
            return self._compute_single_pass_flow()
        elif mode in (FlowMode.default, FlowMode.ms3):
            return self._compute_default_flow(mode)
        else:
            # TODO: Implement other modes (MUSIC21, etc.)
            module_logger.warning(
                f"FlowMode.{mode.name} not yet implemented, using DEFAULT"
            )
            return self._compute_default_flow(FlowMode.default)

    def _compute_printed_flow(self) -> Flow:
        """Compute flow for printed score (no unfolding).

        Returns:
            Flow with each MC visited exactly once.
        """
        # For printed flow, just use sorted MCs in order
        sorted_mcs = sorted(self._measure_lookup.keys())

        # Convert MC sequence to sections
        sections = self._compute_playthrough_sections(sorted_mcs)

        return Flow(
            sections=sections,
            mode=FlowMode.printed,
            folded_length=len(sorted_mcs),
            _controller_ref=weakref.ref(self),
        )

    def _compute_single_pass_flow(self) -> Flow:
        """Compute single-pass flow (no repeats, skip volta 1 endings).

        SINGLE_PASS mode traverses the score once without taking any repeats.
        For volta alternatives:
        - Skip volta 1 endings (those that jump backward to repeat)
        - Include only volta 2+ endings (those that continue forward)

        This ensures musically impossible transitions (volta 1 → volta 2)
        are never included in the flow.

        Returns:
            Flow with single traversal, volta 1 endings excluded.
        """
        sorted_mcs = sorted(self._measure_lookup.keys())

        # Filter out volta 1 endings that jump backward
        # Volta 1 endings are identified by:
        # 1. volta field = 1
        # 2. Their next[] jumps backward (to repeat start)
        single_pass_mcs: list[int] = []
        for mc in sorted_mcs:
            bar = self._measure_lookup[mc]
            volta = bar.get("volta")

            # Skip volta 1 endings (they jump back, can't be in single pass)
            if volta == 1:
                next_list = bar.get("next", [])
                if next_list and next_list[0] != -1 and next_list[0] < mc:
                    # This volta 1 jumps backward - skip it
                    continue

            single_pass_mcs.append(mc)

        # Convert MC sequence to sections
        sections = self._compute_playthrough_sections(single_pass_mcs)

        return Flow(
            sections=sections,
            mode=FlowMode.single,
            folded_length=len(self._measure_lookup),
            _controller_ref=weakref.ref(self),
        )

    def _compute_atomic_flow(self) -> Flow:
        """Compute atomic flow from AtomicSections.

        The atomic flow represents the canonical segment structure (A, B, C, ...)
        without any unfolding. Each AtomicSection becomes a PlaythroughSection.

        This is the default flow mode (mode=None) and defines the canonical
        segment IDs used for mapping all other flow modes.

        Returns:
            Flow with one PlaythroughSection per AtomicSection.
        """
        sections = [
            PlaythroughSection(
                mc_start=sec.mc_start,
                mc_end=sec.mc_end,
                atomic_section_ids=(sec.id,),
                typed_measures=sec.typed_measures,
                groups=sec.groups,
            )
            for sec in self._atomic_sections
        ]

        return Flow(
            sections=sections,
            mode=FlowMode.atomic,
            folded_length=len(self._measure_lookup),
            _controller_ref=weakref.ref(self),
        )

    def _compute_default_flow(self, mode: FlowMode) -> Flow:
        """Compute the default flow purely from flow-control elements.

        Simulates a state machine over the MCs in score order, driven
        entirely by each MeasureUnit's flow-control annotations and the
        raw marker names recorded in the measure lookup:

        - ``start_repeat`` / ``end_repeat`` delimit a repeat block. The
          natural pass count is 2 (the standard sheet-music convention);
          higher voltas (volta 3, 4, ...) are *alternative* endings
          reached only by al-Fine / al-Coda passes.
        - ``volta`` marks an ending. On pass *k* through the block, the
          ``target_volta`` is *k* — MCs marked with a different volta are
          skipped. When al-Fine is armed and the block contains a volta
          whose number exceeds the natural limit and that holds the
          ``fine`` marker, the algorithm jumps directly to that volta
          instead of cycling through earlier ones.
        - ``jump_from`` with ``jump_bwd=segno`` resets the segno's
          enclosing-block pass count so the post-jump pass plays fresh;
          ``jump_bwd=start`` does not reset anything (the natural pass's
          counters persist and naturally suppress already-exhausted
          repeats).
        - ``play_until=X`` arms a generic "play until trigger" mode. The
          trigger fires when an MC is visited whose marker name (segno,
          coda, fine, or raw ms3 ``markers`` value) matches X. For
          ``play_until="coda"`` in a score that lacks an explicit
          ``coda`` marker, the trigger falls back to the next
          atomic-section boundary (per the convention that a "to coda"
          point is always a section boundary). ``jump_fwd`` resolves to
          the MC carrying that marker; an empty ``jump_fwd`` terminates
          the traversal when the trigger fires.

        This algorithm does not generally follow the ms3 ``next`` field as its
        traversal path. It uses encoded backward edges to resolve repeat ends
        with no explicit start and a same-MC successor to retain a one-bar
        ``startend`` repeat when a loader flattened away both repeat flags.
        """
        self._traversal_diagnostics[mode] = []
        if not self._units:
            return Flow(
                sections=[],
                mode=mode,
                folded_length=0,
                _controller_ref=weakref.ref(self),
            )

        sorted_units = sorted(self._units, key=lambda u: u.mc)
        sorted_mcs = [u.mc for u in sorted_units]
        unit_by_mc: dict[int, MeasureUnit] = {u.mc: u for u in sorted_units}
        first_mc = sorted_mcs[0]
        mc_index = {mc: i for i, mc in enumerate(sorted_mcs)}

        # Marker name -> MC (raw ms3 `markers` value, plus segno/coda
        # destinations parsed onto the unit).
        marker_mc_by_name: dict[str, int] = {}
        for mc, bar in self._measure_lookup.items():
            raw = bar.get("marker")
            if raw and raw not in marker_mc_by_name:
                marker_mc_by_name[raw] = mc
        for u in sorted_units:
            if u.segno and u.segno not in marker_mc_by_name:
                marker_mc_by_name[u.segno] = u.mc
            if u.coda and u.coda not in marker_mc_by_name:
                marker_mc_by_name[u.coda] = u.mc

        segno_mc = next((u.mc for u in sorted_units if u.segno), first_mc)

        # Registry construction and every plain executable repeat consume the
        # same resolved targets. Loader-specific unresolved structures retain
        # their established execution inference below without becoming
        # registry jumps.
        resolved_repeat_targets, _ = self._repeat_end_resolution()

        # Enclosing block start for volta routing. Coda subsections remain
        # execution blocks, but coda markers never enter plain repeat-end
        # candidate resolution.
        enclosing_start: dict[int, int] = {}
        current_block_start = first_mc
        for u in sorted_units:
            if u.start_repeat or u.coda is not None:
                current_block_start = u.mc
            enclosing_start[u.mc] = current_block_start

        # Fill execution-only targets for unresolved loader structures. This
        # retains coda-volta blocks and one-bar inferred repeats while all
        # resolved plain repeats keep the registry's target.
        atomic_start_by_mc: dict[int, int] = {}
        for section in self._atomic_sections:
            for section_mc in range(section.mc_start, section.mc_end):
                atomic_start_by_mc[section_mc] = section.mc_start

        repeat_start_for_end = dict(resolved_repeat_targets)
        repeat_stack: list[int] = [first_mc]
        inferred_self_repeats: set[int] = set()
        for i, u in enumerate(sorted_units):
            if u.coda is not None and repeat_stack[-1:] != [u.mc]:
                repeat_stack.append(u.mc)
            if u.start_repeat and repeat_stack[-1:] != [u.mc]:
                repeat_stack.append(u.mc)

            if u.end_repeat:
                fallback = (
                    repeat_stack[-1]
                    if repeat_stack
                    else atomic_start_by_mc.get(u.mc, u.mc)
                )
                repeat_start_for_end.setdefault(u.mc, fallback)
                if repeat_stack:
                    repeat_stack.pop()
            elif not u.start_repeat and u.mc in u.next:
                sequential_next = sorted_mcs[i + 1] if i + 1 < len(sorted_mcs) else -1
                if sequential_next in u.next:
                    repeat_start_for_end[u.mc] = u.mc
                    inferred_self_repeats.add(u.mc)

        repeat_starts = set(repeat_start_for_end.values())

        def _reset_nested_repeats(block_start: int, block_end: int) -> None:
            """Reset repeat counters strictly nested in a restarting block."""
            start_i = mc_index[block_start]
            end_i = mc_index[block_end]
            for nested_start in repeat_starts:
                nested_i = mc_index[nested_start]
                if start_i < nested_i <= end_i:
                    pass_count[nested_start] = 0

        # Volta-containing-fine per block (used to route the al-Fine pass
        # through alternative endings beyond the natural 2-pass limit).
        fine_volta_per_block: dict[int, int] = {}
        for u in sorted_units:
            if u.fine and u.volta is not None:
                blk = enclosing_start[u.mc]
                cur = fine_volta_per_block.get(blk)
                if cur is None or u.volta > cur:
                    fine_volta_per_block[blk] = u.volta

        section_end_mcs: set[int] = {sec.mc_end - 1 for sec in self._atomic_sections}

        natural_limit = 2  # Standard "play once, repeat once" convention.

        def _trigger_fires(unit: MeasureUnit, kind: str, mc: int) -> bool:
            """Does this MC satisfy a 'play until' trigger?"""
            if kind == "fine":
                return unit.fine
            if unit.segno == kind or unit.coda == kind:
                return True
            raw = self._measure_lookup.get(mc, {}).get("marker")
            if raw == kind:
                return True
            # Fallback for play_until="coda" in scores that don't carry
            # an explicit "coda" marker: fire at the next atomic-section
            # boundary (a "to coda" point is always a section boundary).
            if (
                kind == "coda"
                and "coda" not in marker_mc_by_name
                and mc in section_end_mcs
            ):
                return True
            return False

        mc_sequence: list[int] = []
        pass_count: dict[int, int] = defaultdict(int)
        play_until_kind: str | None = None
        play_until_dest: int | None = None
        # How many times each al-coda jump origin has armed so far. Destination
        # selection remains indexed by that per-origin pass count, while the
        # source format currently supplies one scalar ``jump_fwd`` target.
        coda_arm_count: dict[int, int] = defaultdict(int)
        pc: int | None = first_mc

        # The traversal is a guarded transition system. At each visited MC a
        # fixed total order picks which eligible out-edge fires:
        #
        #   1. armed Fine stop       (a terminating play-until)
        #   2. repeat back-edge      (an unexhausted end-repeat)
        #   3. armed coda/segno jump (a play-until trigger)
        #   4. D.S./D.C. jump
        #   5. sequential successor
        #
        # The Fine is a hard stop and outranks everything: an al-Fine pass ends
        # at the Fine even if a repeat would otherwise still run. Below it, the
        # repeat back-edge outranks an un-armed coda/segno jump so that an
        # atomic section still owing a repeat plays it before any al-coda exit
        # fires. Termination is guarded by a lasso check over the whole
        # traversal state (position, per-block pass counts, and the armed-jump
        # configuration): revisiting a state means some guard-disabled edge
        # closes a loop, so the traversal stops and reports the cycle instead
        # of spinning or silently truncating.
        diagnostics: list[FlowDiagnostic] = []
        visited_states: dict[tuple[Any, ...], int] = {}
        state_mcs: list[int] = []
        state_edges: list[str] = []
        last_jump: tuple[str, int, int] | None = None
        # Defensive convergence ceiling. The lasso check terminates every
        # genuine cycle; this only bounds pathological growth of the per-block
        # pass counters and, if ever reached, is itself reported — never a
        # silent truncation.
        step_ceiling = (len(sorted_units) + 1) * 64

        while pc is not None and pc in unit_by_mc:
            if len(state_mcs) > step_ceiling:
                if last_jump is None:
                    last_jump_description = "no jump has been taken"
                else:
                    edge, origin_mc, destination_mc = last_jump
                    last_jump_description = (
                        f"the last jump taken was the {edge} at MC {origin_mc} "
                        f"to MC {destination_mc}"
                    )
                diagnostics.append(
                    FlowDiagnostic(
                        kind="flow_nonconvergence",
                        message=(
                            "default-flow traversal did not converge within "
                            f"{step_ceiling} steps; the flow-control graph likely "
                            f"contains an unguarded jump or repeat; {last_jump_description}"
                        ),
                        mc=pc,
                    )
                )
                break

            state = (
                pc,
                tuple(sorted(pass_count.items())),
                play_until_kind,
                play_until_dest,
                tuple(
                    sorted(
                        (origin, min(count, 1))
                        for origin, count in coda_arm_count.items()
                    )
                ),
            )
            if state in visited_states:
                diagnostics.append(
                    self._describe_flow_cycle(
                        state_mcs,
                        visited_states[state],
                        pc,
                        state_edges[-1] if state_edges else "unknown",
                    )
                )
                break
            visited_states[state] = len(state_mcs)
            state_mcs.append(pc)

            unit = unit_by_mc[pc]

            # Volta-mismatch skip.
            if unit.volta is not None:
                blk = enclosing_start[pc]
                # On the al-Fine pass, route directly to a volta beyond
                # the natural limit that carries the fine marker.
                fv = fine_volta_per_block.get(blk)
                if play_until_kind == "fine" and fv is not None and fv > natural_limit:
                    target_volta = fv
                else:
                    target_volta = pass_count[blk] + 1
                if unit.volta != target_volta:
                    skip_to: int | None = None
                    for j in range(mc_index[pc] + 1, len(sorted_mcs)):
                        other = unit_by_mc[sorted_mcs[j]]
                        if enclosing_start[other.mc] != blk:
                            skip_to = other.mc
                            break
                        if other.volta is None or other.volta == target_volta:
                            skip_to = other.mc
                            break
                    state_edges.append("volta skip")
                    pc = skip_to
                    continue

            mc_sequence.append(pc)

            # 1. Armed Fine stop. The Fine terminates an al-Fine pass and
            #    outranks a still-owed repeat.
            if play_until_kind == "fine" and _trigger_fires(unit, play_until_kind, pc):
                dest = play_until_dest
                play_until_kind = None
                play_until_dest = None
                state_edges.append("fine stop")
                if dest is None:
                    break
                pc = dest
                continue

            # 2. Repeat back-edge. An unexhausted end-repeat takes precedence
            #    over an armed coda/segno jump.
            if pc in repeat_start_for_end or pc in inferred_self_repeats:
                blk = repeat_start_for_end[pc]
                pass_count[blk] = min(pass_count[blk] + 1, natural_limit)
                if pass_count[blk] < natural_limit:
                    _reset_nested_repeats(blk, pc)
                    state_edges.append("repeat back-edge")
                    last_jump = ("repeat back-edge", pc, blk)
                    pc = blk
                    continue

            # 3. Armed coda/segno jump. Once no repeat back-edge remains, a
            #    play-until trigger fires and jumps to its pass-selected
            #    destination.
            if play_until_kind is not None and _trigger_fires(
                unit, play_until_kind, pc
            ):
                dest = play_until_dest
                play_until_kind = None
                play_until_dest = None
                state_edges.append("coda jump")
                if dest is None:
                    break
                last_jump = ("coda jump", pc, dest)
                pc = dest
                continue

            # 4. D.S./D.C. jump.
            if unit.jump_from:
                target: int | None = None
                if unit.jump_bwd == "segno":
                    target = segno_mc
                elif unit.jump_bwd == "start":
                    target = first_mc
                elif unit.jump_fwd:
                    target = marker_mc_by_name.get(unit.jump_fwd)
                if target is not None:
                    if unit.play_until:
                        play_until_kind = unit.play_until
                        coda_arm_count[pc] += 1
                        targets = self._coda_targets(unit.jump_fwd, marker_mc_by_name)
                        if targets:
                            play_until_dest = targets[
                                min(coda_arm_count[pc] - 1, len(targets) - 1)
                            ]
                        else:
                            play_until_dest = None
                    # A jump_bwd=segno restarts the segno-home-block's
                    # pass count so the post-jump pass plays it fresh.
                    # A jump_bwd=start (D.C.) leaves counts intact: any
                    # blocks the post-jump pass crosses inherit whatever
                    # pass count the natural traversal left.
                    if unit.jump_bwd == "segno":
                        seg_blk = enclosing_start.get(segno_mc)
                        if seg_blk is not None:
                            pass_count[seg_blk] = 0
                            _reset_nested_repeats(seg_blk, pc)
                    state_edges.append("D.S./D.C. jump")
                    last_jump = ("D.S./D.C. jump", pc, target)
                    pc = target
                    continue

            # 5. Sequential successor.
            i = mc_index[pc]
            if i + 1 >= len(sorted_mcs):
                break
            state_edges.append("sequential")
            pc = sorted_mcs[i + 1]

        self._traversal_diagnostics[mode] = diagnostics
        sections = self._compute_playthrough_sections(mc_sequence)
        return Flow(
            sections=sections,
            mode=mode,
            folded_length=len(self._measure_lookup),
            _controller_ref=weakref.ref(self),
        )

    def _coda_targets(
        self, jump_fwd: str | None, marker_mc_by_name: dict[str, int]
    ) -> tuple[int, ...]:
        """Resolve an al-coda origin's scalar destination marker.

        ``jump_fwd`` is the scalar MuseScore ``continueAt`` field. Its full
        value is one literal marker name, including any comma in that name.

        Args:
            jump_fwd: The origin's forward-target marker name.
            marker_mc_by_name: Marker-name to MC lookup.

        Returns:
            A one-element destination tuple, or an empty tuple when unresolved.
        """
        if not jump_fwd:
            return ()
        destination = marker_mc_by_name.get(jump_fwd)
        return (destination,) if destination is not None else ()

    def _describe_flow_cycle(
        self,
        state_mcs: list[int],
        loop_start: int,
        head_mc: int,
        closing_edge: str,
    ) -> FlowDiagnostic:
        """Build the diagnostic for a cycle found by the lasso check.

        Args:
            state_mcs: The MCs visited, in traversal order.
            loop_start: Index in ``state_mcs`` of the loop head's first visit.
            head_mc: The MC the closing edge re-enters (the loop head).
            closing_edge: The edge kind that closes the loop.

        Returns:
            A ``FlowDiagnostic(kind="flow_cycle", ...)`` naming the looping
            sections and the guard-disabled edge that closes them.
        """
        loop_mcs = state_mcs[loop_start:]
        last_mc = loop_mcs[-1] if loop_mcs else head_mc

        section_ids: list[str] = []
        for mc in loop_mcs:
            for sid in self._find_atomic_ids(mc, mc + 1):
                if sid not in section_ids:
                    section_ids.append(sid)
        section_label = ", ".join(section_ids) if section_ids else "?"

        return FlowDiagnostic(
            kind="flow_cycle",
            message=(
                f"default-flow traversal cycles through section(s) {section_label} "
                f"and never terminates: the {closing_edge} edge at MC {last_mc} "
                f"re-enters MC {head_mc} with its repeat/jump guard exhausted; "
                f"check the flow control that closes this loop"
            ),
            section_id=section_ids[0] if section_ids else None,
            mc=head_mc,
        )

    def _occurrence_to_suffix(self, occurrence: int) -> str:
        """Convert occurrence number to letter suffix.

        1 -> 'a', 2 -> 'b', ..., 26 -> 'z', 27 -> 'aa', etc.

        Args:
            occurrence: 1-indexed occurrence number.

        Returns:
            Letter suffix.
        """
        if occurrence <= 0:
            return "a"

        result = []
        n = occurrence
        while n > 0:
            n -= 1
            result.append(chr(ord("a") + (n % 26)))
            n //= 26
        return "".join(reversed(result))

    def compute_all_flows(self) -> list[Flow]:
        """Compute all possible Flows.

        For scores with optional repeats, this computes all valid paths.

        Returns:
            List of Flow objects.
        """
        # For now, just return default and printed
        return [
            self.compute_flow(FlowMode.default),
            self.compute_flow(FlowMode.printed),
        ]

    def create_flow_map(self, flow: Flow | None = None) -> FlowMap:
        """Create a FlowMap in QB-space from a computed Flow.

        Overrides the base class to always use QB-space coordinates,
        since ScoreFlowController has MeasureUnit data.

        Args:
            flow: The Flow to create a map from. If None, computes DEFAULT flow.

        Returns:
            FlowMap with QB-space source coordinates.
        """
        if flow is None:
            flow = self.compute_flow(FlowMode.default)
        qb_sections = compute_qb_sections(flow, self)
        return FlowMap.from_qb_sections(flow, qb_sections, id=flow.id)

    def create_flow_map_for_mode(self, mode: FlowMode = FlowMode.default) -> FlowMap:
        """Convenience method to create a FlowMap for a specific mode.

        Args:
            mode: The FlowMode to use.

        Returns:
            FlowMap wrapping the computed Flow.
        """
        flow = self.compute_flow(mode)
        return self.create_flow_map(flow)

    def iter_atomic_sections(self) -> Iterator[tuple[Fraction, Fraction]]:
        """Iterate over atomic (indivisible) sections.

        Each section is a tuple (start, end) representing a contiguous
        portion that cannot be split by flow control. For ScoreFlowController,
        these are MC-based coordinates from AtomicSections.

        Yields:
            Tuples of (start_mc, end_mc) as Fractions.
        """
        for sec in self._atomic_sections:
            yield (Fraction(sec.mc_start), Fraction(sec.mc_end))

    def get_section_boundary_coordinates(self) -> list[Fraction]:
        """Return quarterbeat coordinates where section breaks occur.

        A *section break* (as opposed to repeat/volta/jump boundaries) is a
        structural marker that separates large-scale sections such as
        movements.  Each returned coordinate is the ``start`` (quarterbeat)
        of the first MC **after** the break, i.e. the point where the new
        section begins.

        The list does **not** include 0 (start of the piece) nor the end.

        Returns:
            Sorted list of quarterbeat boundary coordinates.

        Raises:
            RuntimeError: If the measure lookup has not been built
                (controller created without MeasureData).

        Examples:
            >>> controller = ScoreFlowController(measures)
            >>> controller.get_section_boundary_coordinates()
            [Fraction(305, 1), Fraction(1291, 1), Fraction(3125, 2)]
        """
        if not self._measure_lookup:
            raise RuntimeError(
                "Section boundary coordinates require a measure lookup. "
                "Ensure the controller was created with MeasureData."
            )
        sorted_mcs = sorted(self._measure_lookup.keys())
        boundaries: list[Fraction] = []
        for i, mc in enumerate(sorted_mcs):
            if self._measure_lookup[mc].get("section_break"):
                # The break is AT this MC; the new section starts at next MC
                if i + 1 < len(sorted_mcs):
                    next_mc = sorted_mcs[i + 1]
                    qb = self._measure_lookup[next_mc]["quarterbeats"]
                    boundaries.append(Fraction(qb))
        return boundaries

    def get_atomic_section_coordinates(
        self, flow: "Flow | None" = None
    ) -> dict[str, Fraction]:
        """Return a mapping of atomic section IDs to their start coordinates.

        Each key is the section's label (e.g. ``"A"``, ``"B"``, …) and the
        value is the quarterbeat coordinate of the section's first measure.

        When *flow* is provided the coordinates are **unfolded** (i.e. the
        running quarterbeat position in the playthrough order), which is
        required for cross-group coordinate transfer via an
        ``AlignmentBundle`` whose WarpMaps are built from unfolded note
        matches.  Without *flow*, the folded quarterbeats from the measure
        lookup are returned (the coordinate resets at every repeat start).

        Args:
            flow: A ``Flow`` computed from this controller (e.g. via
                ``compute_flow(FlowMode.default)``).  If given, unfolded
                coordinates are returned.

        Returns:
            Ordered dict mapping section ID to quarterbeat start coordinate.

        Raises:
            RuntimeError: If the measure lookup has not been built
                (controller created without MeasureData).

        Examples:
            >>> controller = ScoreFlowController(measures)
            >>> controller.get_atomic_section_coordinates()
            {'A': Fraction(0, 1), 'B': Fraction(32, 1), ...}
            >>> flow = controller.compute_flow(FlowMode.default)
            >>> controller.get_atomic_section_coordinates(flow=flow)
            {'A': Fraction(0, 1), 'B': Fraction(64, 1), ...}
        """
        if not self._measure_lookup:
            raise RuntimeError(
                "Atomic section coordinates require a measure lookup. "
                "Ensure the controller was created with MeasureData."
            )

        if flow is not None:
            return self._unfolded_section_coordinates(flow)

        result: dict[str, Fraction] = {}
        for sec in self._atomic_sections:
            qb_val = self._measure_lookup.get(sec.mc_start, {}).get("quarterbeats")
            if qb_val is not None:
                result[sec.id] = Fraction(qb_val)
        return result

    def _unfolded_section_coordinates(self, flow: "Flow") -> dict[str, Fraction]:
        """Compute unfolded quarterbeat positions for each atomic section.

        Walks the flow's playthrough sections, accumulating quarterbeats
        from the measure lookup.  For compound playthrough sections that
        span multiple atomic sections, the internal boundaries are resolved
        by mapping each MC to the atomic section it belongs to.

        Args:
            flow: A ``Flow`` computed from this controller.

        Returns:
            Ordered dict mapping section ID to unfolded quarterbeat start.
        """
        mc_durations: dict[int, Fraction] = {
            mc: info["duration_qb"] for mc, info in self._measure_lookup.items()
        }
        # Build MC → atomic section ID mapping
        atomic_by_mc: dict[int, str] = {}
        for asec in self._atomic_sections:
            for mc in range(asec.mc_start, asec.mc_end):
                if mc not in atomic_by_mc:
                    atomic_by_mc[mc] = asec.id

        running_qb = Fraction(0)
        result: dict[str, Fraction] = {}

        for ps in flow.sections:
            prev_atomic = None
            for mc in ps.to_mc_sequence():
                atomic_id = atomic_by_mc.get(mc)
                if atomic_id and atomic_id != prev_atomic:
                    if atomic_id not in result:
                        result[atomic_id] = running_qb
                    prev_atomic = atomic_id
                dur = mc_durations.get(mc, Fraction(4))
                running_qb += dur

        return result

    # region Display

    def diagram(
        self,
        width: int = 70,
        unicode: bool = True,
        show_graph: bool = True,
        show_legend: bool = True,
        mode: str = "auto",
    ) -> "Diagram":
        """Show folded score map with atomic sections and flow control markers.

        Args:
            width: Total width of the diagram in characters.
            unicode: Use Unicode characters (True) or ASCII fallback (False).
            show_graph: Whether to show section transition graph.
            show_legend: Whether to show flow control event legend.
            mode: Rendering mode — ``"auto"``, ``"full"``, ``"sections"``,
                or ``"table"``. See :func:`flow_control_diagram` for details.

        Returns:
            Diagram object (displays as ASCII in terminal, rich HTML in Jupyter).
        """
        from timetoalign.display.ascii import flow_control_diagram

        return flow_control_diagram(
            self,
            width=width,
            unicode=unicode,
            show_graph=show_graph,
            show_legend=show_legend,
            mode=mode,
        )

    def __str__(self) -> str:
        return str(self.diagram())

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        return self.diagram()._repr_html_()
