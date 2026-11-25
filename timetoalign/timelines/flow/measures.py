"""Represent measures and related measure groups."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from timetoalign.core.enums import IncompletePosition


@dataclass(frozen=True)
class MeasureUnit:
    """Single measure from the folded score (one MeasureData row).

    This is the fundamental building block from which AtomicSections
    are constructed. MeasureUnit represents the "folded" skeleton of
    the score before any unfolding takes place.

    Attributes:
        mc: Measure Count (monotonic, 1-indexed).
        mn: Measure Number label (may have suffix like "19a").
        duration_qb: Duration in quarter beats.
        next: Tuple of possible next MCs. (-1 = end of piece)
        volta: Ending number (1, 2, ...) or None.
        timesig: Time signature string ("4/4", "3/4") or None.
        timesig_duration_qb: Expected duration from time signature (computed).
        start_repeat: True if has repeat start marker (||:).
        end_repeat: True if has repeat end marker (:||).
        jump_from: True if this MC is a jump origin (D.C., D.S., to_coda).
        jump_to: True if this MC is a jump target (segno, coda destination).
        segno: Segno marker name if present (e.g., "segno", "segno2").
        coda: Coda marker name if present (e.g., "coda", "codab").
        fine: True if Fine marker present.
        section_break: True if section break at this MC.
        flow_control_types: Tuple of FlowManifestation.value strings for serialization.
        jump_bwd: Name of backward-jump target (e.g., "segno", "start") or None.
        jump_fwd: Name of forward-jump target (e.g., "coda", "codab", "fine") or None.
        play_until: Name of play-until target on after-DC/DS pass
            (e.g., "coda", "fine") or None.

    Examples:
        >>> unit = MeasureUnit(
        ...     mc=1,
        ...     mn="1",
        ...     duration_qb=Fraction(4),
        ...     next=(2,),
        ... )
        >>> unit.mc
        1
        >>> unit.next
        (2,)
    """

    mc: int
    mn: str
    duration_qb: Fraction
    next: tuple[int, ...]
    volta: int | None = None
    timesig: str | None = None
    timesig_duration_qb: Fraction | None = None
    start_repeat: bool = False
    end_repeat: bool = False
    jump_from: bool = False
    jump_to: bool = False
    segno: str | None = None
    coda: str | None = None
    fine: bool = False
    section_break: bool = False
    flow_control_types: tuple[str, ...] = ()
    # Raw ms3 jump-instruction fields (target NAMES, may be None)
    jump_bwd: str | None = None
    jump_fwd: str | None = None
    play_until: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame serialization.

        Returns:
            Dict with all MeasureUnit fields, suitable for DataFrame conversion.
            The flow_control_types tuple is joined with semicolons for CSV storage.
        """
        return {
            "mc": self.mc,
            "mn": self.mn,
            "duration_qb": float(self.duration_qb),
            "next": list(self.next),
            "volta": self.volta,
            "timesig": self.timesig,
            "timesig_duration_qb": (
                float(self.timesig_duration_qb) if self.timesig_duration_qb else None
            ),
            "start_repeat": self.start_repeat,
            "end_repeat": self.end_repeat,
            "jump_from": self.jump_from,
            "jump_to": self.jump_to,
            "segno": self.segno,
            "coda": self.coda,
            "fine": self.fine,
            "section_break": self.section_break,
            "flow_control_types": ";".join(self.flow_control_types),
            "jump_bwd": self.jump_bwd,
            "jump_fwd": self.jump_fwd,
            "play_until": self.play_until,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MeasureUnit":
        """Create MeasureUnit from dictionary (e.g., DataFrame row).

        Args:
            d: Dictionary with MeasureUnit field values.

        Returns:
            New MeasureUnit instance.
        """
        # Parse flow_control_types from semicolon-separated string
        fct = d.get("flow_control_types", "")
        if isinstance(fct, str):
            flow_control_types = tuple(fct.split(";")) if fct else ()
        else:
            flow_control_types = tuple(fct) if fct else ()

        # Parse next field
        next_val = d.get("next", [-1])
        if isinstance(next_val, str):
            # Handle string representation like "[1, 2]" or "1, 2"
            cleaned = next_val.strip().strip("[]()").replace(" ", "")
            next_tuple = (
                tuple(int(x) for x in cleaned.split(",") if x) if cleaned else (-1,)
            )
        elif isinstance(next_val, (list, tuple)):
            next_tuple = tuple(next_val)
        else:
            next_tuple = (int(next_val),) if next_val is not None else (-1,)

        # Parse duration
        duration = d.get("duration_qb", 4)
        if isinstance(duration, Fraction):
            duration_qb = duration
        else:
            duration_qb = Fraction(duration) if duration else Fraction(4)

        # Parse timesig_duration_qb
        ts_dur = d.get("timesig_duration_qb")
        timesig_duration_qb = Fraction(ts_dur) if ts_dur is not None else None

        return cls(
            mc=int(d["mc"]),
            mn=str(d.get("mn", d["mc"])),
            duration_qb=duration_qb,
            next=next_tuple,
            volta=d.get("volta"),
            timesig=d.get("timesig"),
            timesig_duration_qb=timesig_duration_qb,
            start_repeat=bool(d.get("start_repeat")),
            end_repeat=bool(d.get("end_repeat")),
            jump_from=bool(d.get("jump_from")),
            jump_to=bool(d.get("jump_to")),
            segno=d.get("segno"),
            coda=d.get("coda"),
            fine=bool(d.get("fine")),
            section_break=bool(d.get("section_break")),
            flow_control_types=flow_control_types,
            jump_bwd=d.get("jump_bwd"),
            jump_fwd=d.get("jump_fwd"),
            play_until=d.get("play_until"),
        )

    def __repr__(self) -> str:
        volta_str = f", volta={self.volta}" if self.volta else ""
        return f"MeasureUnit(MC {self.mc}: {self.mn}, next={self.next}{volta_str})"


# endregion

# region Typed MeasureUnit Subclasses


@dataclass(frozen=True)
class IncompleteMeasure(MeasureUnit):
    """A MeasureUnit that does not metrically complete on its own.

    NOT a group — this is a typed copy of a MeasureUnit created during
    the Typing step of the two-step typing/grouping algorithm.

    IncompleteMeasure inherits all MeasureUnit properties including
    FlowManifestations, enabling serialization round-trip.

    Examples:
        - Anacrusis (pickup): First measure shorter than time signature
        - Final incomplete: Last measure shorter than time signature
        - Split part: One half of a split measure

    Note:
        An IncompleteMeasure can later form part of a SplitMeasure group
        (e.g., anacrusis + final measure after repeat = SplitMeasure).

    Attributes:
        position: Classification of why this measure is incomplete.
    """

    position: IncompletePosition = IncompletePosition.unknown


@dataclass(frozen=True)
class CompleteMeasure(MeasureUnit):
    """A MeasureUnit that metrically completes on its own.

    NOT a group — this is a typed copy of a MeasureUnit created during
    the Typing step of the two-step typing/grouping algorithm.

    The simple, default case: duration_qb == timesig_duration_qb.

    CompleteMeasure inherits all MeasureUnit properties including
    FlowManifestations, enabling serialization round-trip.
    """

    pass


@dataclass(frozen=True)
class OverlengthMeasure(MeasureUnit):
    """A MeasureUnit that exceeds the expected metrical length.

    NOT a group — this is a typed copy of a MeasureUnit created during
    the Typing step of the two-step typing/grouping algorithm.

    OverlengthMeasure inherits all MeasureUnit properties including
    FlowManifestations, enabling serialization round-trip.

    Examples:
        - Fermata: Written duration > time signature
        - Cadenza: Extended passage notated in single measure
        - Ad lib: Performer-determined length
    """

    pass


# Type alias for any typed measure
TypedMeasure = IncompleteMeasure | CompleteMeasure | OverlengthMeasure

# endregion

# region MeasureGroup


@dataclass(frozen=True)
class MeasureGroup:
    """Base class for groupings of typed MeasureUnits.

    Groups are constructed in the Grouping step, AFTER MeasureUnits have been
    typed as IncompleteMeasure/CompleteMeasure/OverlengthMeasure in the
    Typing step.

    MeasureGroup is an abstract base - use SplitMeasure, IncompleteGroup,
    Volta, or CompleteMeasureGroup for concrete groupings.

    Attributes:
        members: Tuple of typed MeasureUnits in this group.
    """

    members: tuple["TypedMeasure", ...]

    def __post_init__(self) -> None:
        """Validate group has at least one member."""
        if not self.members:
            raise ValueError("MeasureGroup must have at least one member")

    @property
    def mc_start(self) -> int:
        """First MC in this group."""
        return min(m.mc for m in self.members)

    @property
    def mc_end(self) -> int:
        """First MC AFTER this group (right-open)."""
        return max(m.mc for m in self.members) + 1

    @property
    def mc_range(self) -> tuple[int, int]:
        """Return (mc_start, mc_end) tuple (right-open interval)."""
        return (self.mc_start, self.mc_end)

    @property
    def total_duration_qb(self) -> Fraction:
        """Total duration in quarterbeats."""
        return sum((m.duration_qb for m in self.members), Fraction(0))

    def __len__(self) -> int:
        """Return number of measures in this group."""
        return len(self.members)

    def __repr__(self) -> str:
        return f"MeasureGroup(MC [{self.mc_start},{self.mc_end}), {len(self.members)} measures)"


@dataclass(frozen=True)
class SplitMeasure(MeasureGroup):
    """Multiple IncompleteMeasures that together form a complete metrical unit.

    This is the complex case where IncompleteMeasures combine to form a
    complete measure duration. Split measures share the same measure number (mn).

    Examples:
        - Anacrusis (1/4) + Final (3/4) after repeat = complete 4/4
        - Split bar: MC 11 (1/4) + MC 12 (1/4) = complete 2/4 (same mn="10")

    Note:
        All members must be IncompleteMeasure instances. The detection
        algorithm uses shared mn as a strong signal for split measure pairs.
    """

    def __post_init__(self) -> None:
        """Validate all members are IncompleteMeasure."""
        super().__post_init__()
        for m in self.members:
            if not isinstance(m, IncompleteMeasure):
                raise TypeError(
                    f"SplitMeasure members must be IncompleteMeasure, got {type(m).__name__}"
                )

    def __repr__(self) -> str:
        mcs = [m.mc for m in self.members]
        return f"SplitMeasure(MC {mcs}, total={self.total_duration_qb})"


@dataclass(frozen=True)
class IncompleteGroup(MeasureGroup):
    """Isolated IncompleteMeasure(s) that don't yet form a split.

    Used for anacrusis or final measures that haven't been paired with their
    complement. In AtomicSections, these may later merge into SplitMeasures
    when the flow brings them together in PlaythroughSections.

    Note:
        All members must be IncompleteMeasure instances.
    """

    def __post_init__(self) -> None:
        """Validate all members are IncompleteMeasure."""
        super().__post_init__()
        for m in self.members:
            if not isinstance(m, IncompleteMeasure):
                raise TypeError(
                    f"IncompleteGroup members must be IncompleteMeasure, got {type(m).__name__}"
                )

    def __repr__(self) -> str:
        # Safe to access position since __post_init__ validates all are IncompleteMeasure
        positions = [
            m.position.value for m in self.members if isinstance(m, IncompleteMeasure)
        ]
        return f"IncompleteGroup(MC [{self.mc_start},{self.mc_end}), positions={positions})"


@dataclass(frozen=True)
class VoltaGroup(MeasureGroup):
    """Typed MeasureUnits under the same volta bracket.

    A VoltaGroup contains measures that share the same volta number (1, 2, 3...).
    Each different volta number creates a separate VoltaGroup.

    Note:
        A VoltaGroup is contained within ONE AtomicSection. Multiple voltas
        (e.g., volta 1 and volta 2) are separated by Breaks and thus cannot
        be in the same section.

    Attributes:
        volta_number: The volta bracket number (1, 2, 3, ...).
    """

    volta_number: int = 1

    def __post_init__(self) -> None:
        """Validate volta_number is positive."""
        super().__post_init__()
        if self.volta_number < 1:
            raise ValueError(f"volta_number must be >= 1, got {self.volta_number}")

    def __repr__(self) -> str:
        return (
            f"VoltaGroup(volta={self.volta_number}, MC [{self.mc_start},{self.mc_end}), "
            f"{len(self.members)} measures)"
        )


@dataclass(frozen=True)
class CompleteMeasureGroup(MeasureGroup):
    """Adjacent CompleteMeasures grouped together.

    Useful for identifying contiguous "normal" passages between special
    structures (voltas, splits, incomplete measures, etc.).

    Note:
        All members must be CompleteMeasure instances. OverlengthMeasures
        are NOT included in CompleteMeasureGroup (they're a special case).
    """

    def __post_init__(self) -> None:
        """Validate all members are CompleteMeasure."""
        super().__post_init__()
        for m in self.members:
            if not isinstance(m, CompleteMeasure):
                raise TypeError(
                    f"CompleteMeasureGroup members must be CompleteMeasure, got {type(m).__name__}"
                )

    def __repr__(self) -> str:
        return f"CompleteMeasureGroup(MC [{self.mc_start},{self.mc_end}), {len(self.members)} measures)"


@dataclass(frozen=True)
class OverlengthGroup(MeasureGroup):
    """One or more OverlengthMeasures grouped together.

    For overlength measures like fermatas, cadenzas, or ad lib passages.

    Note:
        All members must be OverlengthMeasure instances.
    """

    def __post_init__(self) -> None:
        """Validate all members are OverlengthMeasure."""
        super().__post_init__()
        for m in self.members:
            if not isinstance(m, OverlengthMeasure):
                raise TypeError(
                    f"OverlengthGroup members must be OverlengthMeasure, got {type(m).__name__}"
                )

    def __repr__(self) -> str:
        return f"OverlengthGroup(MC [{self.mc_start},{self.mc_end}), {len(self.members)} measures)"
