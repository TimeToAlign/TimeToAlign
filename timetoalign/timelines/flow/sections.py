"""Represent score sections and computed musical flows."""

from __future__ import annotations

import logging
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Iterator

from timetoalign.core.enums import FlowMode
from timetoalign.core.events import Gap

from .measures import (
    IncompleteMeasure,
    MeasureGroup,
    MeasureUnit,
    OverlengthMeasure,
    TypedMeasure,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from timetoalign.display.ascii import Diagram
    from timetoalign.timelines.structure import MeasureMap

    from .controller import ScoreFlowController

module_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AtomicSection:
    """Smallest indivisible traversal unit (similar to partitura's segment model).

    Atomic sections are derived from:
    - partitura's add_segments()/get_segments() for MusicXML/MEI
    - next[] array analysis for TSV/MeasureMap

    Section IDs (A, B, C, ...) form a canonical reference for mapping all
    flow modes. The atomic flow mode defines these canonical sections.

    Note:
        MC ranges use the **right-open interval convention** [mc_start, mc_end),
        consistent with partitura and TimeToAlign's TimeInterval model. For
        example, mc_start=1 and mc_end=5 means measures 1, 2, 3, 4 (four
        measures total).

    Attributes:
        id: Letter identifier (A, B, C...) from partitura or generated.
        mc_start: First MC of this section (inclusive).
        mc_end: First MC AFTER this section (exclusive, right-open).
        to: List of possible next section IDs.
        section_type: "default", "leap_end", or "leap_start".

    Examples:
        >>> sec = AtomicSection(
        ...     id="A",
        ...     mc_start=1,
        ...     mc_end=5,  # Right-open: includes MCs 1,2,3,4
        ...     to=("A", "B"),
        ...     section_type="leap_end",
        ... )
        >>> sec.mc_range
        (1, 5)
        >>> sec.mc_count
        4
    """

    id: str
    mc_start: int
    mc_end: int
    to: tuple[str, ...] = ()
    section_type: str = "default"  # "default" | "leap_end" | "leap_start"
    typed_measures: tuple["TypedMeasure", ...] | None = None  # Typing-step output
    groups: tuple["MeasureGroup", ...] | None = None  # Grouping-step output
    measure_map: "MeasureMap | None" = None
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate section configuration."""
        if self.mc_end < self.mc_start:
            raise ValueError(
                f"AtomicSection '{self.id}': mc_end ({self.mc_end}) "
                f"cannot be before mc_start ({self.mc_start})"
            )
        if self.section_type not in ("default", "leap_end", "leap_start"):
            raise ValueError(
                f"AtomicSection '{self.id}': invalid section_type '{self.section_type}'"
            )

    @property
    def mc_range(self) -> tuple[int, int]:
        """Return (mc_start, mc_end) tuple (right-open interval)."""
        return (self.mc_start, self.mc_end)

    @property
    def mc_count(self) -> int:
        """Number of MCs in this section (right-open: end - start)."""
        return self.mc_end - self.mc_start

    @property
    def n_measures(self) -> int:
        """Number of measures carried by this leaf section."""
        return len(self.measure_map) if self.measure_map is not None else self.mc_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "id": self.id,
            "mc_start": self.mc_start,
            "mc_end": self.mc_end,
            "to": list(self.to),
            "section_type": self.section_type,
        }
        if self.typed_measures is not None:
            result["typed_measures_count"] = len(self.typed_measures)
            result["incomplete_count"] = sum(
                1 for m in self.typed_measures if isinstance(m, IncompleteMeasure)
            )
            result["overlength_count"] = sum(
                1 for m in self.typed_measures if isinstance(m, OverlengthMeasure)
            )
        if self.groups is not None:
            result["groups_count"] = len(self.groups)
            result["group_types"] = [type(g).__name__ for g in self.groups]
        return result

    def __repr__(self) -> str:
        n = self.mc_count
        measure_word = "measure" if n == 1 else "measures"
        bits = [f"{n} {measure_word}", f"to=[{', '.join(self.to)}]"]
        descriptor = self._flow_control_descriptor()
        if descriptor:
            bits.append(descriptor)
        return (
            f"AtomicSection({self.id}: MC [{self.mc_start},{self.mc_end}), "
            f"{', '.join(bits)})"
        )

    def _flow_control_descriptor(self) -> str:
        """One-liner describing this section's flow-control role.

        Lists the marker that begins a leap_start section and the
        instruction(s) that close a leap_end section, using short labels:
        DSaC / DSaF / DCaC / DCaF / DS / DC / →coda / repeat_end / fine.
        The internal branching mechanic — a leap_end with multiple
        ``next`` targets but no recognised jump instruction on its
        boundary measures — surfaces musicologically as alternative
        endings, so it is rendered as "ends with alternative endings"
        rather than as the implementation-level term "branching".
        """
        if not self.typed_measures:
            return ""
        first = self.typed_measures[0]
        last = self.typed_measures[-1]
        parts: list[str] = []

        if self.section_type == "leap_start":
            if first.coda:
                parts.append(f"target '{first.coda}'")
            elif first.segno:
                parts.append(f"target '{first.segno}'")
            else:
                parts.append("leap target")

        end_parts: list[str] = []
        if last.end_repeat:
            end_parts.append("repeat_end")
        # Scan every measure in the section so an instruction sitting on a
        # non-final measure (a "to coda" mid-section, for instance) still
        # surfaces in the descriptor.
        section_fct: set[str] = set()
        for m in self.typed_measures:
            fct = m.flow_control_types
            section_fct.update(fct)
            # First-coda marker — a coda marker co-located with a jump
            # origin — is the implicit "to coda" trigger in loaders that
            # encode the jump via the marker name rather than emitting a
            # dedicated to_coda instruction.
            if "coda" in fct and "jump_from" in fct:
                section_fct.add("to_coda")
        for short, fc_type in (
            ("DSaC", "dal_segno_al_coda"),
            ("DSaF", "dal_segno_al_fine"),
            ("DCaC", "da_capo_al_coda"),
            ("DCaF", "da_capo_al_fine"),
            ("DS", "dal_segno"),
            ("DC", "da_capo"),
            ("→coda", "to_coda"),
        ):
            if fc_type in section_fct:
                end_parts.append(short)
        if last.fine:
            end_parts.append("fine")
        if end_parts:
            parts.append(f"ends with {'+'.join(end_parts)}")
        elif self.section_type == "leap_end":
            parts.append("ends with alternative endings")
        return ", ".join(parts)


# endregion

# region FlowDiagnostic


@dataclass(frozen=True)
class FlowDiagnostic:
    """A structural-invariant violation found in an atomic flow graph.

    Diagnostics are produced by ``ScoreFlowController.check_invariants`` and
    describe a way in which the folded section graph departs from the rules a
    well-formed score must satisfy. The controller never raises on a malformed
    flow — it reports. Each diagnostic names the kind of violation and, where
    applicable, the section(s) and measure involved.

    Attributes:
        kind: Short machine-readable violation tag (e.g. ``"volta_follows_volta"``).
        message: Human-readable explanation, including a remediation hint.
        section_id: The offending source section's id, if the violation is
            attributable to a single section. ``None`` otherwise.
        mc: The measure count most relevant to the violation, if any.

    Examples:
        >>> diag = FlowDiagnostic(
        ...     kind="volta_follows_volta",
        ...     message="section 'B' flows directly to volta section 'C'",
        ...     section_id="B",
        ...     mc=5,
        ... )
        >>> diag.kind
        'volta_follows_volta'
    """

    kind: str
    message: str
    section_id: str | None = None
    mc: int | None = None

    def __repr__(self) -> str:
        bits = [f"kind={self.kind!r}"]
        if self.section_id is not None:
            bits.append(f"section={self.section_id!r}")
        if self.mc is not None:
            bits.append(f"mc={self.mc}")
        bits.append(f"message={self.message!r}")
        return f"FlowDiagnostic({', '.join(bits)})"


# endregion

# region PlaythroughSection


@dataclass(frozen=True)
class PlaythroughSection:
    """A contiguous group of atomic sections in a specific traversal.

    This is what gets written to .flow.csv and compared via is_equivalent().
    Each PlaythroughSection represents a contiguous run of MCs in the unfolded
    sequence.

    Note:
        MC ranges use the **right-open interval convention** [mc_start, mc_end),
        consistent with partitura and TimeToAlign's TimeInterval model. For
        example, mc_start=1 and mc_end=9 means measures 1, 2, 3, 4, 5, 6, 7, 8
        (eight measures total).

    Attributes:
        mc_start: First MC of this playthrough section (inclusive).
        mc_end: First MC AFTER this playthrough section (exclusive, right-open).
        atomic_section_ids: Which atomic sections this covers.

    Examples:
        >>> sec = PlaythroughSection(mc_start=1, mc_end=9, atomic_section_ids=("A", "B"))
        >>> sec.mc_range
        (1, 9)
        >>> sec.mc_count
        8
    """

    mc_start: int
    mc_end: int
    atomic_section_ids: tuple[str, ...] = ()
    typed_measures: tuple["TypedMeasure", ...] | None = None  # Typing-step output
    groups: tuple["MeasureGroup", ...] | None = None  # Grouping-step output

    def __post_init__(self) -> None:
        """Validate section configuration."""
        if self.mc_end < self.mc_start:
            raise ValueError(
                f"PlaythroughSection: mc_end ({self.mc_end}) "
                f"cannot be before mc_start ({self.mc_start})"
            )

    @property
    def mc_range(self) -> tuple[int, int]:
        """Return (mc_start, mc_end) tuple (right-open interval)."""
        return (self.mc_start, self.mc_end)

    @property
    def mc_count(self) -> int:
        """Number of MCs in this section (right-open: end - start)."""
        return self.mc_end - self.mc_start

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "mc_start": self.mc_start,
            "mc_end": self.mc_end,
            "atomic_sections": ";".join(self.atomic_section_ids),
        }
        if self.typed_measures is not None:
            result["typed_measures_count"] = len(self.typed_measures)
        if self.groups is not None:
            result["groups_count"] = len(self.groups)
            result["group_types"] = [type(g).__name__ for g in self.groups]
        return result

    def to_mc_sequence(self) -> list[int]:
        """Return list of all MCs in this section.

        Returns:
            List of MC values from mc_start to mc_end (right-open, excludes mc_end).
        """
        return list(range(self.mc_start, self.mc_end))

    def __repr__(self) -> str:
        secs = ", ".join(self.atomic_section_ids) if self.atomic_section_ids else "?"
        n = self.mc_count
        measure_word = "measure" if n == 1 else "measures"
        return (
            f"PlaythroughSection(MC [{self.mc_start},{self.mc_end}), "
            f"{n} {measure_word}, atomic=[{secs}])"
        )


# endregion

# region Flow


@dataclass
class Flow:
    """A computed flow (sequence of measure visitations).

    A Flow represents one possible path through a score, accounting for
    repeats, jumps, and voltas. It can be:
    - Computed by ScoreFlowController from MeasureData
    - Loaded from .flow.csv ground truth
    - Compared using is_equivalent()

    Flows are section-based, using `sections` (list of PlaythroughSection)
    for .flow.csv serialization and is_equivalent() comparison.

    Flows computed by ScoreFlowController have a controller reference, allowing
    access to MeasureUnits via iter_units(). Flows loaded from CSV are
    "detached" and do not have controller access.

    Note:
        MC ranges use the **right-open interval convention** [mc_start, mc_end),
        consistent with partitura and TimeToAlign's TimeInterval model.

    Attributes:
        sections: The sequence of PlaythroughSection objects.
        mode: The FlowMode used to compute this flow.
        folded_length: Number of unique MCs (measures in printed score).
        id: Identifier for this flow (defaults to mode.value).
        source_metadata: Optional metadata from the source MeasureData.
    """

    sections: list[PlaythroughSection] = field(default_factory=list)
    mode: FlowMode = FlowMode.default
    folded_length: int = 0
    id: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)
    _controller_ref: "weakref.ref[ScoreFlowController] | None" = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Initialize id from mode if not set."""
        if not self.id:
            object.__setattr__(self, "id", self.mode.value)

    @property
    def controller(self) -> "ScoreFlowController | None":
        """Get the ScoreFlowController that created this Flow, if still alive.

        Returns:
            ScoreFlowController if this Flow was computed and controller is alive,
            None if Flow was loaded from CSV or controller was garbage collected.
        """
        if self._controller_ref is None:
            return None
        return self._controller_ref()

    def iter_units(self) -> Iterator[MeasureUnit]:
        """Iterate over MeasureUnits via the controller.

        This provides access to the folded score skeleton (one MeasureUnit
        per MeasureData row).

        Yields:
            MeasureUnit objects in MC order.

        Raises:
            ValueError: If Flow is detached from controller (e.g., loaded from CSV).

        Examples:
            >>> flow = controller.compute_flow()
            >>> for unit in flow.iter_units():
            ...     print(f"MC {unit.mc}: next={unit.next}")
        """
        ctrl = self.controller
        if ctrl is None:
            raise ValueError(
                "Flow is detached from controller. "
                "iter_units() is only available for flows computed by ScoreFlowController."
            )
        yield from ctrl.iter_units()

    # === Class Methods (Constructors) ===

    @classmethod
    def from_sections(
        cls,
        sections: list[PlaythroughSection],
        mode: FlowMode,
        folded_length: int | None = None,
    ) -> "Flow":
        """Create a Flow from PlaythroughSections.

        Args:
            sections: List of PlaythroughSection objects.
            mode: The FlowMode for this flow.
            folded_length: Number of unique MCs. If None, computed from sections.

        Returns:
            New Flow instance with sections populated.
        """
        if folded_length is None:
            # Estimate from section ranges (may not be accurate for repeated sections)
            # Note: Right-open interval, so range(start, end) is correct
            all_mcs = set()
            for sec in sections:
                all_mcs.update(range(sec.mc_start, sec.mc_end))
            folded_length = len(all_mcs)

        return cls(
            sections=sections,
            mode=mode,
            folded_length=folded_length,
        )

    @classmethod
    def from_records(cls, records: list[dict], mode: FlowMode) -> "Flow":
        """Create Flow from list of dicts with mc_start, mc_end, atomic_sections.

        Note:
            MC ranges use right-open interval convention [mc_start, mc_end).

        Args:
            records: List of dicts, each with keys:
                - mc_start: int (inclusive)
                - mc_end: int (exclusive, right-open)
                - atomic_sections: str (semicolon-separated, e.g., "A;B");
                  the persisted flow-CSV format names this column
                  "atomic_segments", which is accepted as a fallback key
            mode: The FlowMode for this flow.

        Returns:
            New Flow instance with sections populated.
        """
        sections = []
        for rec in records:
            # The persisted flow-CSV format names this column "atomic_segments".
            atomic_ids_str = rec.get("atomic_sections", rec.get("atomic_segments", ""))
            atomic_ids = tuple(
                s.strip() for s in atomic_ids_str.split(";") if s.strip()
            )
            sections.append(
                PlaythroughSection(
                    mc_start=int(rec["mc_start"]),
                    mc_end=int(rec["mc_end"]),
                    atomic_section_ids=atomic_ids,
                )
            )
        return cls.from_sections(sections, mode)

    @classmethod
    def from_dataframe(cls, df: "pd.DataFrame", mode: FlowMode) -> "Flow":
        """Create Flow from DataFrame with mc_start, mc_end, atomic_sections fields.

        Note:
            MC ranges use right-open interval convention [mc_start, mc_end).

        Args:
            df: DataFrame with fields: mc_start, mc_end, atomic_sections
                (or its flow-CSV spelling "atomic_segments").
            mode: The FlowMode for this flow.

        Returns:
            New Flow instance with sections populated.
        """
        records = df.to_dict("records")
        return cls.from_records(records, mode)

    @classmethod
    def from_csv(cls, path: "Path | str", mode: FlowMode) -> "Flow":
        """Load Flow for specific mode from .flow.csv file.

        Filters CSV to rows matching the given flow_mode.

        Note:
            MC ranges use right-open interval convention [mc_start, mc_end).

        Args:
            path: Path to the .flow.csv file.
            mode: The FlowMode to load.

        Returns:
            New Flow instance with sections populated.

        Raises:
            ValueError: If no entries found for the given mode.
        """
        from pathlib import Path as _Path

        import pandas as pd

        if isinstance(path, str):
            path = _Path(path)

        df = pd.read_csv(path)

        # Filter to matching flow_mode
        mode_df = df[df["flow_mode"] == mode.value]

        if len(mode_df) == 0:
            available_modes = df["flow_mode"].unique().tolist()
            raise ValueError(
                f"No entries for flow_mode '{mode.value}' in {path}. "
                f"Available modes: {available_modes}"
            )

        # Skip ERROR entries
        mode_df = mode_df[mode_df["mc_start"] != "ERROR"]

        # Convert to DataFrame explicitly to satisfy type checker
        return cls.from_dataframe(pd.DataFrame(mode_df), mode)

    # === Serialization Methods ===

    def to_records(self) -> list[dict]:
        """Export as list of dicts (section-based).

        Returns:
            List of dicts with keys: mc_start, mc_end, atomic_sections.
        """
        return [sec.to_dict() for sec in self.sections]

    def to_csv_rows(self, source_file: str, software_version: str) -> list[dict]:
        """Export as .flow.csv format rows.

        Returns list of dicts with keys:
            flow_mode, source_file, software_version, mc_start, mc_end, atomic_sections

        Args:
            source_file: The file that was parsed to produce this flow.
            software_version: Software name and version for reproducibility.

        Returns:
            List of dicts ready for CSV writing.
        """
        rows = []
        for sec in self.sections:
            rows.append(
                {
                    "flow_mode": self.mode.value,
                    "source_file": source_file,
                    "software_version": software_version,
                    "mc_start": sec.mc_start,
                    "mc_end": sec.mc_end,
                    "atomic_sections": ";".join(sec.atomic_section_ids),
                }
            )
        return rows

    # === Comparison Methods ===

    def is_equivalent(self, other: "Flow") -> bool:
        """Compare by zipped (mc_start, mc_end) ranges.

        Two flows are equivalent if they have the same number of sections
        and each corresponding section has matching mc_start and mc_end.

        Note: atomic_section_ids are NOT compared - only MC ranges matter.

        Args:
            other: Another Flow to compare against.

        Returns:
            True if flows are equivalent, False otherwise.
        """
        if len(self.sections) != len(other.sections):
            return False
        return all(
            (a.mc_start, a.mc_end) == (b.mc_start, b.mc_end)
            for a, b in zip(self.sections, other.sections)
        )

    # === Properties ===

    @property
    def unfolded_length(self) -> int:
        """Number of measure visitations in the unfolded sequence."""
        # Compute from sections (right-open: mc_count = end - start)
        return sum(sec.mc_count for sec in self.sections)

    @property
    def total_quarterbeats(self) -> Fraction:
        """Total duration of the unfolded sequence in quarter beats.

        Note:
            This requires controller access to get measure durations.
            Returns 0 for detached flows (loaded from CSV).
        """
        ctrl = self.controller
        if ctrl is None:
            return Fraction(0)
        # Sum durations from MeasureUnits for all MCs in the flow
        total = Fraction(0)
        mc_sequence = self.to_mc_sequence()
        unit_lookup = {u.mc: u for u in ctrl.iter_units()}
        for mc in mc_sequence:
            if mc in unit_lookup:
                total += unit_lookup[mc].duration_qb
        return total

    @property
    def has_repeats(self) -> bool:
        """Whether the flow contains repeated measures."""
        return self.unfolded_length > self.folded_length

    # === Convenience Methods ===

    def to_mc_sequence(self) -> list[int]:
        """Return the sequence of MCs in traversal order.

        Returns:
            List of MC values in the order they are visited (right-open intervals).
        """
        # Compute from sections (right-open: range(start, end) excludes end)
        result = []
        for sec in self.sections:
            result.extend(range(sec.mc_start, sec.mc_end))
        return result

    def to_atomic_sequence(self) -> list[str]:
        """Return flattened sequence of atomic section IDs.

        This provides a canonical representation of the flow as a sequence
        of atomic section traversals. Useful for comparing flows and debugging.

        Returns:
            List of atomic section IDs in traversal order.

        Examples:
            >>> flow = Flow.from_sections([
            ...     PlaythroughSection(1, 17, ("A", "B")),
            ...     PlaythroughSection(17, 32, ("C",)),
            ...     PlaythroughSection(6, 17, ("B",)),
            ... ], FlowMode.default)
            >>> flow.to_atomic_sequence()
            ['A', 'B', 'C', 'B']
        """
        result = []
        for sec in self.sections:
            result.extend(sec.atomic_section_ids)
        return result

    def diff_flows(self, other: "Flow") -> str:
        """Show differences between this flow and another using sequence alignment.

        Uses difflib to produce a human-readable diff of atomic sequences.

        Args:
            other: Another Flow to compare against.

        Returns:
            String showing the alignment/diff between the two flows.
        """
        import difflib

        self_seq = self.to_atomic_sequence()
        other_seq = other.to_atomic_sequence()

        # Use unified diff for readability
        diff = difflib.unified_diff(
            [f"{i+1}: {s}" for i, s in enumerate(other_seq)],
            [f"{i+1}: {s}" for i, s in enumerate(self_seq)],
            fromfile="other",
            tofile="self",
            lineterm="",
        )
        return "\n".join(diff)

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert to pandas DataFrame with section information.

        Returns:
            DataFrame with fields: mc_start, mc_end, atomic_sections
            for each PlaythroughSection.

        Note:
            This returns section-level data. For per-MC data, use
            iter_units() with controller access.
        """
        import pandas as pd

        rows = [sec.to_dict() for sec in self.sections]
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        ratio = self.unfolded_length / self.folded_length if self.folded_length else 0
        sec_info = f", {len(self.sections)} sections" if self.sections else ""
        return (
            f"Flow({self.mode.value}: {self.folded_length} folded -> "
            f"{self.unfolded_length} unfolded, ratio={ratio:.2f}{sec_info})"
        )

    def __str__(self) -> str:
        return str(self.diagram())

    def diagram(
        self,
        width: int = 70,
        unicode: bool = True,
        show_mcs: bool = False,
        show_reasons: bool = True,
    ) -> "Diagram":
        """Show playthrough section sequence for this flow.

        Args:
            width: Total width of the diagram in characters.
            unicode: Use Unicode characters (True) or ASCII fallback (False).
            show_mcs: Whether to expand MC sequences per section.
            show_reasons: Whether to annotate why each section starts.

        Returns:
            Diagram object (displays as ASCII in terminal, rich HTML in Jupyter).
        """
        from timetoalign.display.ascii import flow_diagram

        return flow_diagram(
            self,
            width=width,
            unicode=unicode,
            show_mcs=show_mcs,
            show_reasons=show_reasons,
        )

    def diff_diagram(
        self,
        other: "Flow",
        width: int = 80,
        unicode: bool = True,
    ) -> "Diagram":
        """Show side-by-side comparison of this flow with another.

        Args:
            other: Another Flow to compare with.
            width: Total width of the diagram in characters.
            unicode: Use Unicode characters (True) or ASCII fallback (False).

        Returns:
            Diagram object (displays as ASCII in terminal, rich HTML in Jupyter).
        """
        from timetoalign.display.ascii import flow_comparison_diagram

        return flow_comparison_diagram(self, other, width=width, unicode=unicode)

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        return self.diagram()._repr_html_()


def load_valid_flows(csv_path: "Path | str") -> dict[FlowMode, "Flow"]:
    """Load all valid flows from a .flow.csv file, grouped by flow_mode.

    Args:
        csv_path: Path to the .flow.csv file.

    Returns:
        Dict mapping FlowMode to Flow for each unique flow_mode in the CSV.
        Skips unknown flow_modes and ERROR entries.

    Examples:
        >>> flows = load_valid_flows(Path("tests/data/target_flows/specimen.flow.csv"))
        >>> default_flow = flows[FlowMode.default]
        >>> for mode, flow in flows.items():
        ...     print(f"{mode.value}: {len(flow.sections)} sections")
    """
    from pathlib import Path as _Path

    import pandas as pd

    if isinstance(csv_path, str):
        csv_path = _Path(csv_path)

    df = pd.read_csv(csv_path)

    # Skip ERROR entries
    df = df[df["mc_start"] != "ERROR"]

    flows: dict[FlowMode, Flow] = {}
    for mode_str, group in df.groupby("flow_mode"):
        try:
            mode = FlowMode(str(mode_str))
            # Ensure group is a DataFrame
            group_df = pd.DataFrame(group)
            flows[mode] = Flow.from_dataframe(group_df, mode)
        except ValueError:
            # Skip unknown flow modes
            module_logger.debug(f"Skipping unknown flow_mode: {mode_str}")
            pass

    return flows


# endregion

# region Interval coercion


def _interval_value(x: object) -> Fraction:
    """Coerce a single coordinate-like value to an exact ``Fraction``.

    Args:
        x: A coordinate-like object exposing a ``value`` attribute
            (``Coordinate`` / ``IdCoordinate``) or a raw ``int`` / ``float`` /
            ``Fraction``.

    Returns:
        The position as an exact ``Fraction``.
    """
    return Fraction(x.value if hasattr(x, "value") else x)


def _as_interval(
    obj: object,
    *,
    resolve: Callable[[str], object] | None = None,
) -> tuple[Fraction, Fraction, str | None]:
    """Coerce a single interval-like descriptor to a ``(start, end, label)`` triple.

    Descriptors are recognised in strict priority order, and each carries a
    best-effort identity ``label`` (``None`` when the descriptor names nothing):

    1. ``str`` — a region id resolved through *resolve* (typically a
       timeline's ``get_region``). The string itself is the label. A ``str``
       with no resolver is an error; a string is never iterated character by
       character.
    2. Any object exposing ``start`` and ``end`` — a ``Region``, an interval
       event (``Note`` / ``Measure``), or a ``TimeIntervalStamp``. The label is
       ``getattr(obj, "id", None) or getattr(obj, "name", None)`` (a
       ``Region`` is labelled by its ``id``).
    3. Any object exposing ``origin`` and ``length`` — a ``Timeline`` is
       itself an interval running from its origin to its length. The label is
       ``getattr(obj, "id", None) or getattr(obj, "name", None)``.
    4. A two-element sequence — an explicit ``(start, end)`` coordinate pair,
       which names nothing, so the label is ``None``.

    Args:
        obj: The descriptor to coerce.
        resolve: Callable mapping a region id to an interval-like object.
            Required only when *obj* is a ``str``.

    Returns:
        The ``(start, end, label)`` triple: the bounds as exact ``Fraction``
        values, and the identity label (a ``str`` or ``None``).

    Raises:
        ValueError: If *obj* is a ``str`` and no *resolve* is given, or if a
            sequence does not hold exactly two elements.
        TypeError: If *obj* matches none of the recognised descriptor shapes.
    """
    if isinstance(obj, str):
        if resolve is None:
            raise ValueError(f"Cannot resolve interval name {obj!r} without a resolver")
        start, end, _ = _as_interval(resolve(obj), resolve=resolve)
        return (start, end, obj)
    if hasattr(obj, "start") and hasattr(obj, "end"):
        label = getattr(obj, "id", None) or getattr(obj, "name", None)
        return (_interval_value(obj.start), _interval_value(obj.end), label)
    if hasattr(obj, "origin") and hasattr(obj, "length"):
        label = getattr(obj, "id", None) or getattr(obj, "name", None)
        return (_interval_value(obj.origin), _interval_value(obj.length), label)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        if len(obj) != 2:
            raise ValueError(
                f"A coordinate-pair sequence must hold exactly two elements, "
                f"got {len(obj)}: {obj!r}"
            )
        start, end = obj
        return (_interval_value(start), _interval_value(end), None)
    raise TypeError(f"Cannot interpret {obj!r} as an interval")


def _coerce_intervals(
    spec: object,
    *,
    resolve: Callable[[str], object] | None = None,
) -> list[tuple[Fraction, Fraction, str | None]]:
    """Coerce a singleton-or-collection spec to ``(start, end, label)`` triples.

    One positional argument carries both a single interval-like descriptor and
    a collection of them. Disambiguation tries the whole *spec* as a single
    interval first; only if that coercion fails is *spec* iterated as a
    collection. A ``str`` is always a singleton — it is never iterated
    character by character.

    Args:
        spec: One interval-like descriptor (see :func:`_as_interval`) or an
            iterable of them.
        resolve: Callable mapping a region id to an interval-like object,
            forwarded to :func:`_as_interval`.

    Returns:
        The list of ``(start, end, label)`` triples, one per interval.

    Raises:
        ValueError: If any resulting interval has ``end < start``.
    """
    if isinstance(spec, str):
        intervals = [_as_interval(spec, resolve=resolve)]
    else:
        try:
            intervals = [_as_interval(spec, resolve=resolve)]
        except (TypeError, ValueError):
            intervals = [_as_interval(x, resolve=resolve) for x in spec]

    for start, end, _label in intervals:
        if end < start:
            raise ValueError(f"Interval end ({end}) cannot be before start ({start})")
    return intervals


# endregion


def _coerce_flow_entries(
    spec: object,
    *,
    resolve: Callable[[str], object] | None = None,
) -> list["tuple[Fraction, Fraction, str | None] | Gap"]:
    """Coerce a flow spec that may contain Gaps into an ordered entry list.

    Extends :func:`_coerce_intervals` with :class:`Gap` entries. Every
    non-``Gap`` element is coerced by :func:`_as_interval` exactly as before,
    so a spec holding no gaps yields the same ``(start, end, label)`` triples
    that :func:`_coerce_intervals` produces.

    A spec that *is* a single ``Gap`` — or that holds nothing but gaps —
    describes no played material and is rejected: a flow has to play something.

    Args:
        spec: One interval-like descriptor (see :func:`_as_interval`), a
            ``Gap``, or an iterable mixing the two.
        resolve: Callable mapping a region id to an interval-like object,
            forwarded to :func:`_as_interval`.

    Returns:
        The entries in target order: ``(start, end, label)`` triples for played
        spans and :class:`Gap` objects for holes.

    Raises:
        ValueError: If any played interval has ``end < start``, or if the spec
            holds no played span at all.
    """

    def _played(item: object) -> tuple[Fraction, Fraction, str | None]:
        start, end, label = _as_interval(item, resolve=resolve)
        if end < start:
            raise ValueError(f"Interval end ({end}) cannot be before start ({start})")
        return (start, end, label)

    entries: list["tuple[Fraction, Fraction, str | None] | Gap"] = []
    if isinstance(spec, Gap):
        entries.append(spec)
    elif isinstance(spec, str):
        entries.append(_played(spec))
    else:
        # Same singleton-or-collection disambiguation as `_coerce_intervals`:
        # try the whole spec as one interval, and only iterate if that fails.
        try:
            entries.append(_played(spec))
        except (TypeError, ValueError):
            for item in spec:  # type: ignore[attr-defined]
                entries.append(item if isinstance(item, Gap) else _played(item))

    if not any(not isinstance(entry, Gap) for entry in entries):
        raise ValueError(
            "A flow spec must contain at least one played span; "
            "gaps alone describe no material to play"
        )
    return entries


# endregion
