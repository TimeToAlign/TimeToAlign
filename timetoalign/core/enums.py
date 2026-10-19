"""Enumerations for the TTA model.

This module defines the fundamental categorical types used throughout
the library: temporal domains, time units, number types, and event types.

The FancyStrEnum base class provides:
- Lowercase member names (via auto() from StrEnum)
- Abbreviation aliases (e.g., q = quarters, ms = milliseconds)
- Flexible instantiation from any alias
- get_abbreviations() for documentation
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum, StrEnum, auto
from fractions import Fraction


class FancyStrEnum(StrEnum):
    """A StrEnum with support for abbreviation aliases and flexible instantiation.

    Features:
        * Can be instantiated from any alias: FancyStrEnum("abbr") == FancyStrEnum.abbreviation
        * list(FancyStrEnum) returns only non-aliases (canonical members)
        * FancyStrEnum.get_abbreviations() returns a mapping from names to abbreviations

    Example:
        class Vocabulary(FancyStrEnum):
            abbreviation = auto()  # assigns the name as value (lowercase per StrEnum)
            abbr = abbreviation    # alias 1
            abb = abbreviation     # alias 2
    """

    @classmethod
    def _missing_(cls, value: object) -> "FancyStrEnum | None":
        """Allow instantiation from values, including aliases.

        Args:
            value: The value or name string to look up.

        Returns:
            The corresponding enum member, or None if not found.

        Raises:
            ValueError: If the value does not match any member or alias.
        """
        if isinstance(value, str):
            lower_value = value.lower()
            if lower_value in cls.__members__:
                name = cls.__members__[lower_value]
                return cls(name)
        abbrv = cls.get_abbreviations(string=True)
        raise ValueError(
            f"'{value}' is not a valid {cls.__name__}. Available values are: {abbrv}"
        )

    @classmethod
    def get_abbreviations(cls, string: bool = False) -> dict[str, list[str]] | str:
        """Returns a mapping from enum names/values to abbreviated alias values.

        Args:
            string: If True, return a formatted string instead of a dict.

        Returns:
            A dict mapping canonical names to lists of aliases, or a formatted string.
        """
        name2values: dict[str, list[str]] = defaultdict(list)
        for value, name in cls.__members__.items():
            name2values[name].append(value)
        abbreviations: dict[str, list[str]] = {}
        for name, values in name2values.items():
            # Sort by length descending, skip the first (canonical name)
            abbreviations[name] = sorted(values, key=lambda x: len(x), reverse=True)[1:]
        if not string:
            return abbreviations
        str_components = []
        for name, values in abbreviations.items():
            if not values:
                str_components.append(name)
                continue
            abbrev_str = ", ".join(values)
            str_components.append(f"{name} ({abbrev_str})")
        return ", ".join(str_components)

    def __repr__(self) -> str:
        return f'"{self.name}"'

    def __str__(self) -> str:
        return self.name


class Domain(FancyStrEnum):
    """The temporal domain of a timeline.

    TimeToAlign distinguishes three temporal domains:
    - logical: Logical time domain (conceptualizing, reading)
    - physical: Physical time domain (hearing, seeing dynamic)
    - graphical: Graphical time domain (seeing static, spatial)

    See the Conceptual Model documentation
    (https://timetoalign.github.io/concepts.html).
    """

    logical = auto()
    """Logical time domain for symbolic/musical data (beats, measures, ticks)."""
    lo = logical

    physical = auto()
    """Physical time domain for audio/time data (seconds, samples)."""
    ph = physical

    graphical = auto()
    """Graphical time domain for visual/spatial data (pixels, coordinates)."""
    gr = graphical


class NumberType(Enum):
    """The numeric type used for coordinate values.

    Members can be instantiated both via NumberType("name") and NumberType(value).

    Example:
        NumberType(int) is NumberType("int")
        # True
        NumberType(int).value(1.4)
        # 1
    """

    int = int
    float = float
    fraction = Fraction

    @classmethod
    def _missing_(cls, value: object) -> "NumberType | None":
        if isinstance(value, str):
            for member in cls:
                if member.name == value:
                    return member
        return None

    @classmethod
    def from_number(cls, number: int | float | Fraction) -> "NumberType":
        """Create NumberType from a number instance."""
        return cls(type(number))

    @property
    def python_type(self) -> type:
        """Return the corresponding Python type."""
        return self.value

    def __str__(self) -> str:
        return self.name


# Domain mappings for TimeUnit (module-level for efficiency)
_LOGICAL_UNITS: frozenset[str] = frozenset(
    {"floating_measures", "whole_notes", "quarters", "ticks", "number"}
)
_PHYSICAL_UNITS: frozenset[str] = frozenset(
    {"milliseconds", "seconds", "minutes", "samples", "frames"}
)
_GRAPHICAL_UNITS: frozenset[str] = frozenset(
    {"pixels", "meters", "centimeters", "millimeters", "inches", "points"}
)

# Units whose values are inherently integral: a fractional tick or a
# fractional pixel does not denote anything.  These are the ONLY discrete
# units, and this set is the single source of that fact for the whole
# library -- read it through ``TimeUnit.is_discrete``, never by
# re-listing unit names.
_DISCRETE_UNITS: frozenset[str] = frozenset({"ticks", "samples", "frames", "pixels"})

# Continuous units whose values are most faithfully written as exact
# ratios: notated rhythm is rational by construction (a triplet eighth is
# 1/3 of a quarter, not 0.333...).
_RATIONAL_DEFAULT_UNITS: frozenset[str] = frozenset({"whole_notes", "quarters"})


class TimeUnit(FancyStrEnum):
    """Units of measurement for coordinates.

    Organized by domain. Each unit belongs to a specific domain.
    Aliases are provided for common abbreviations.
    """

    # generic
    number = auto()

    # logical domain
    floating_measures = auto()
    """measures"""
    fm = floating_measures
    """measures"""

    whole_notes = auto()
    """whole notes"""
    w = whole_notes
    """whole notes"""

    quarters = auto()
    """quarter notes"""
    q = quarters
    """quarter notes"""

    ticks = auto()
    """ticks (MIDI's time unit)"""
    pulses = ticks
    """ticks (MIDI's time unit)"""
    divs = ticks
    """ticks (MIDI's time unit)"""

    # physical domain
    milliseconds = auto()
    """milliseconds"""
    ms = milliseconds
    """milliseconds"""

    seconds = auto()
    """seconds"""
    s = seconds
    """seconds"""

    minutes = auto()
    """minutes"""

    samples = auto()
    """samples"""

    frames = auto()
    """frames"""

    # graphical domain
    pixels = auto()
    """pixels"""
    px = pixels
    """pixels"""

    meters = auto()
    """meters"""

    centimeters = auto()
    """centimeters"""
    cm = centimeters
    """centimeters"""

    millimeters = auto()
    """millimeters"""
    mm = millimeters
    """millimeters"""

    inches = auto()
    """inches"""

    points = auto()
    """points"""
    pt = points
    """points"""

    @property
    def domain(self) -> Domain:
        """Return the domain this unit belongs to."""
        if self.name in _LOGICAL_UNITS:
            return Domain.logical
        elif self.name in _PHYSICAL_UNITS:
            return Domain.physical
        elif self.name in _GRAPHICAL_UNITS:
            return Domain.graphical
        raise ValueError(f"Unknown domain for unit {self.name}")  # pragma: no cover

    @property
    def is_discrete(self) -> bool:
        """Whether this unit is inherently discrete (integer-valued).

        This property is the library's single source of discreteness. No
        other module may re-list the discrete unit names.
        """
        return self.name in _DISCRETE_UNITS

    @property
    def default_number_type(self) -> NumberType:
        """The numeric representation this unit uses unless told otherwise.

        A unit knows how its values are written, so nothing downstream --
        timeline, loader, or field -- needs to carry its own default:

        * discrete units are ``int``;
        * ``quarters`` is ``fraction``, because notated rhythm is
          rational by construction;
        * every other continuous unit is ``float``.
        """
        if self.is_discrete:
            return NumberType.int
        if self.name in _RATIONAL_DEFAULT_UNITS:
            return NumberType.fraction
        return NumberType.float

    @property
    def allowed_number_types(self) -> frozenset[NumberType]:
        """The numeric representations this unit accepts.

        Discrete units admit ``int`` only: a fractional tick has no
        referent, so offering the choice would only let callers record
        nonsense. Continuous units admit ``float`` and ``fraction``, the
        caller choosing whether exactness or speed matters more.
        :attr:`TimeUnit.number` admits all three.
        """
        if self.is_discrete:
            return frozenset({NumberType.int})
        if self is TimeUnit.number:
            # TODO: ``number`` is the generic escape hatch for values that
            # are not time at all, so it currently constrains nothing. If a
            # clearer role emerges for it, tighten this to match.
            return frozenset(NumberType)
        return frozenset({NumberType.float, NumberType.fraction})

    def resolve_number_type(
        self, requested: NumberType | str | None = None
    ) -> NumberType:
        """Return the numeric representation to use for this unit.

        Args:
            requested: An explicit choice, or ``None`` to take
                :attr:`default_number_type`.

        Returns:
            The resolved :class:`NumberType`.

        Raises:
            ValueError: If *requested* is not among
                :attr:`allowed_number_types`.
        """
        if requested is None:
            return self.default_number_type
        resolved = NumberType(requested) if isinstance(requested, str) else requested
        allowed = self.allowed_number_types
        if resolved not in allowed:
            permitted = ", ".join(sorted(member.name for member in allowed))
            raise ValueError(
                f"Unit '{self}' does not admit number_type '{resolved}'. "
                f"Allowed number types: {permitted}"
            )
        return resolved


class AgentType(FancyStrEnum):
    """The kind of agent that authored a match claim.

    Distinguishes a human annotator from a software aligner.  The
    ``identifier`` on the paired :class:`~timetoalign.alignment.claims.Agent`
    is interpreted accordingly: a URI for a human (e.g. an ORCID or homepage)
    and a version string for software (e.g. ``"v0.20.0"``).
    """

    human = auto()
    """A human annotator; the agent's identifier is a URI."""

    software = auto()
    """A software aligner; the agent's identifier is a version string."""


class ClaimType(FancyStrEnum):
    """The semantic kind of a MatchClaim, derived from its structure.

    A claim's kind is never stored; it is read from whether the claim is
    explicit, whether it is synchronous, and how many of its two sides name an
    event. The members span every shape a pairwise claim can take:

    Members:
        event_match: Two identified events on different timelines correspond,
            temporally anchored.
        projection: One identified event corresponds to a bare coordinate on
            the other timeline (which names no event of its own).
        nomatch: An identified event has no counterpart on the other timeline.
        anchor: Two anonymous coordinates correspond, with no event identity on
            either side.
        implicit: A correspondence inferred by graph extension rather than
            directly asserted.
        conceptual: A structural correspondence with no temporal commitment —
            no anchors and no single orphaned event.
    """

    event_match = auto()
    """Two identified events correspond, temporally anchored."""
    event = event_match
    """Alias for event_match."""

    projection = auto()
    """One identified event corresponds to a bare coordinate on the other timeline."""

    nomatch = auto()
    """An identified event has no counterpart on the other timeline."""
    nomat = nomatch
    """Alias for nomatch."""

    anchor = auto()
    """Two anonymous coordinates correspond (no event identity)."""
    anon = anchor
    """Alias for anchor."""

    implicit = auto()
    """An inferred (not directly asserted) correspondence."""

    conceptual = auto()
    """A structural correspondence with no temporal commitment (no anchors)."""


class EventType(FancyStrEnum):
    """Whether an event is an instant or interval.

    - instant: Zero duration, associated with a single coordinate
    - interval: Has duration, defined by start and end coordinates
    """

    instant = auto()
    inst = instant

    interval = auto()
    intv = interval


class IntervalPolicy(FancyStrEnum):
    """Policy for normalising interval events (end/duration consistency).

    When loading events, interval events may carry ``end``, ``duration``,
    or both.  This enum controls how the loader fills in missing values
    and what happens when both are present but inconsistent.

    Members:
        warn: (default) Prefer ``end``; compute ``duration = end - start``.
            If both are present and inconsistent, log a warning and
            recompute ``duration`` from ``end``.
        prefer_end: Silently prefer ``end``; recompute ``duration`` from
            ``end - start``, ignoring any supplied ``duration``.
        prefer_duration: Silently prefer ``duration``; recompute ``end``
            from ``start + duration``, ignoring any supplied ``end``.
        strict: Raise ``ValueError`` if both are present and inconsistent.
            Otherwise behave like ``warn`` (fill whichever is missing).
    """

    warn = auto()
    """Prefer end; warn on inconsistency."""

    prefer_end = auto()
    """Silently prefer end; always recompute duration."""

    prefer_duration = auto()
    """Silently prefer duration; always recompute end."""

    strict = auto()
    """Raise ValueError on inconsistency."""


class ColumnNaming(FancyStrEnum):
    """How to name columns in timestamp DataFrames.

    Controls the column header naming strategy for timestamp tables.

    Members:
        name: Use the human-readable name property (e.g., "Musical Holes Region").
            Falls back to ID if name is not set.
        id: Use the unique identifier (e.g., "dgt_holes").
    """

    name = auto()
    """Use human-readable name (with id fallback)."""

    id = auto()
    """Use unique identifier."""


class FlowControlElement(FancyStrEnum):
    """Canonical taxonomy of flow control markers in musical scores.

    This enum provides a unified vocabulary for flow control across all loaders
    (MeasureMapLoader, Ms3Loader, PartituraLoader, Music21Loader).

    **Design Principles**:

    1. **Types vs Names**: This enum defines marker TYPES, not instance names.
       Jump target markers (segno, coda, fine) are TYPES. Each marker INSTANCE
       has a NAME attribute (e.g., "coda", "codab", "segno2") that differentiates
       multiple markers of the same type. Jump instructions reference targets
       by NAME, which defaults to the type name but can be customized.

    2. **Structural vs Flow Control**: Some markers are purely structural (barlines),
       while others affect traversal (jumps, breaks). The `is_structural_marker`
       property distinguishes these.

    3. **Volta Model**: Volta (alternative ending) information is a MEASURE ATTRIBUTE,
       not a separate flow control type. Each measure has a `volta` field (1, 2, None)
       indicating which ending it belongs to. The repeat_end jump evaluates which
       volta to take based on pass count.

    **Taxonomy** (aligned with the Conceptual Model documentation and
    MeasureMap paper):

    **Repeat Markers**:
    - repeat_start: Structural marker (||:) - target for repeat_end jumps
    - repeat_end: Jump instruction (:||) - triggers backward jump; MAY also
      constitute a section_break when it coincides with an end-type barline

    **Jump Instructions** (triggers navigation):
    - da_capo: Jump to beginning (D.C.)
    - dal_segno: Jump to marker named "segno" by default (D.S.)
    - dal_segno_al_coda: D.S., play until "coda" marker, then jump to coda section
    - dal_segno_al_fine: D.S., play until "fine" marker
    - da_capo_al_coda: D.C., play until "coda" marker, then jump to coda section
    - da_capo_al_fine: D.C., play until "fine" marker
    - to_coda: Jump to marker named "coda" by default

    **Jump Target Markers** (destinations - TYPE, instances have names):
    - segno: Target for dal_segno (instance name defaults to "segno")
    - coda: Target for to_coda (instance name defaults to "coda")
    - fine: End marker (instance name defaults to "fine")

    **Structural Breaks** (void contiguity - single instant, not start/end pair):
    - section_break: Voids contiguity at this instant (e.g., section boundary)

    **Structural Markers** (do NOT void contiguity, mark boundaries):
    - double_barline: Double bar line (structural boundary, not a break)
    - final_barline: Final bar line (end of piece)

    **Boundary Markers** (reference markers for addressing):
    - first_measure: First measure of piece (ms3 convention)
    - last_measure: Last measure of piece (ms3 convention)

    **Abstract Jump Roles** (super-categories carried alongside concrete types):
    - jump_from: Abstract origin of any jump (super-category of repeat_end,
      da_capo, dal_segno, to_coda, ...). Emitted by loaders into
      `flow_control_types` so consumers can filter jump-origin positions
      without enumerating every concrete instruction type.
    - jump_to: Abstract destination of any jump (super-category of repeat_start,
      segno, coda). Symmetric counterpart of `jump_from`.

    **Loader Mappings**:

    | FlowControlElement | MeasureMap | ms3/TSV | partitura | music21 |
    |-----------------|------------|---------|-----------|---------|
    | repeat_start | start_repeat=true | repeats="start" | Repeat.start | leftBarline.type="heavy-light" |
    | repeat_end | end_repeat=true | repeats="end" | Repeat.end | rightBarline.type="light-heavy" |
    | da_capo | next=[1] pattern | jump_bwd="dacapo" | DaCapo | DaCapo |
    | dal_segno | next=[segno_mc] | jump_bwd="dalsegno" | DalSegno | DalSegno* |
    | to_coda | - | jump_fwd="tocoda" | ToCoda | Coda(type="to") |
    | fine | next=[-1] + marker | play_until="fine" | Fine | Fine |
    | segno | marker in JSON | marker="segno" | Segno | Segno |
    | coda | marker in JSON | marker="coda" | Coda | Coda |
    | section_break | - | breaks="section" | - | - |
    | double_barline | - | breaks="double" | - | barline.type="double" |
    | final_barline | - | - | - | barline.type="final" |
    | jump_from | derived | derived | derived | derived |
    | jump_to | derived | derived | derived | derived |
    """

    # Repeat markers
    repeat_start = auto()
    """Structural marker (||:) - target for repeat_end jumps."""
    repeat_end = auto()
    """Jump instruction (:||) - may also be a section_break when coinciding with end barline."""

    # Jump instructions
    da_capo = auto()
    """Jump to beginning (D.C.)"""
    dc = da_capo
    """Alias for da_capo"""

    dal_segno = auto()
    """Jump to segno marker by name (D.S.)"""
    ds = dal_segno
    """Alias for dal_segno"""

    dal_segno_al_coda = auto()
    """Jump to segno, play until coda, then jump to coda section (D.S. al Coda)"""
    dsac = dal_segno_al_coda
    """Alias for dal_segno_al_coda"""

    dal_segno_al_fine = auto()
    """Jump to segno, play until fine (D.S. al Fine)"""
    dsaf = dal_segno_al_fine
    """Alias for dal_segno_al_fine"""

    da_capo_al_coda = auto()
    """Jump to beginning, play until coda, then jump to coda section (D.C. al Coda)"""
    dcac = da_capo_al_coda
    """Alias for da_capo_al_coda"""

    da_capo_al_fine = auto()
    """Jump to beginning, play until fine (D.C. al Fine)"""
    dcaf = da_capo_al_fine
    """Alias for da_capo_al_fine"""

    to_coda = auto()
    """Jump to coda marker by name (typically to "codab", the second coda marker)"""

    # Jump target markers (TYPE - instances have names like "coda", "codab", "segno2")
    segno = auto()
    """Target marker TYPE for dal_segno (instance name defaults to "segno")"""

    coda = auto()
    """Target marker TYPE for coda symbols.

    In "D.S./D.C. al Coda" structures, there are typically TWO coda markers:
    - First coda marker (name="coda"): Where `to_coda` instruction is placed (JumpFrom).
      This marker becomes active after the D.S./D.C. jump.
    - Second coda marker (name="codab"): Where the jump lands (JumpTo).
      This is the beginning of the coda section. MuseScore uses "codab" as default name.

    Both markers have TYPE `coda` but different instance NAMES.
    """

    fine = auto()
    """End marker TYPE (instance name defaults to "fine")"""

    # Structural breaks (void contiguity - single instant)
    section_break = auto()
    """Single instant that voids contiguity (e.g., section boundary)"""

    # Structural markers (do NOT void contiguity)
    double_barline = auto()
    """Double bar line - structural boundary marker, does NOT void contiguity"""

    final_barline = auto()
    """Final bar line - end of piece marker"""

    # Boundary markers (for reference/addressing)
    first_measure = auto()
    """First measure of piece (ms3 convention)"""

    last_measure = auto()
    """Last measure of piece (ms3 convention)"""

    # Abstract jump roles (super-categories of the concrete jump/target types)
    jump_from = auto()
    """Abstract origin role: this position is the source of some jump.

    Emitted by loaders alongside the concrete instruction type (da_capo,
    dal_segno, to_coda, repeat_end, ...) so that downstream consumers can
    filter all jump origins without enumerating every concrete subtype.
    """

    jump_to = auto()
    """Abstract destination role: this position is the target of some jump.

    Emitted by loaders alongside the concrete target type (repeat_start,
    segno, coda) so that downstream consumers can filter all jump
    destinations without enumerating every concrete subtype.
    """

    @classmethod
    def from_ms3_repeats(cls, value: str | None) -> "FlowControlElement | None":
        """Convert ms3 'repeats' column value to FlowControlElement.

        Args:
            value: Value from ms3 repeats column ("start", "end", "firstMeasure", etc.)

        Returns:
            Corresponding FlowControlElement or None.
        """
        if not value:
            return None
        mapping = {
            "start": cls.repeat_start,
            "end": cls.repeat_end,
            "firstMeasure": cls.first_measure,
            "lastMeasure": cls.last_measure,
        }
        return mapping.get(value)

    @classmethod
    def from_ms3_breaks(cls, value: str | None) -> "FlowControlElement | None":
        """Convert ms3 'breaks' column value to FlowControlElement.

        Args:
            value: Value from ms3 breaks column ("section", "double", etc.)

        Returns:
            Corresponding FlowControlElement or None.
        """
        if not value:
            return None
        mapping = {
            "section": cls.section_break,
            "double": cls.double_barline,
        }
        return mapping.get(value)

    @classmethod
    def from_measuremap(
        cls, start_repeat: bool = False, end_repeat: bool = False
    ) -> list["FlowControlElement"]:
        """Convert MeasureMap flow control fields to FlowControlElement list.

        Args:
            start_repeat: Whether start_repeat is true in MeasureMap.
            end_repeat: Whether end_repeat is true in MeasureMap.

        Returns:
            List of corresponding FlowControlElements.
        """
        result = []
        if start_repeat:
            result.append(cls.repeat_start)
        if end_repeat:
            result.append(cls.repeat_end)
        return result

    @property
    def is_jump(self) -> bool:
        """Whether this marker triggers a jump to another location.

        True for every concrete jump instruction AND for the abstract
        ``jump_from`` role.
        """
        return self in {
            FlowControlElement.repeat_end,
            FlowControlElement.da_capo,
            FlowControlElement.dal_segno,
            FlowControlElement.dal_segno_al_coda,
            FlowControlElement.dal_segno_al_fine,
            FlowControlElement.da_capo_al_coda,
            FlowControlElement.da_capo_al_fine,
            FlowControlElement.to_coda,
            FlowControlElement.jump_from,
        }

    @property
    def is_target(self) -> bool:
        """Whether this marker TYPE is a jump destination.

        True for every concrete target marker AND for the abstract
        ``jump_to`` role.

        Note: Actual target resolution uses instance NAMES, not just types.
        Multiple markers of the same type (e.g., two coda markers named
        "coda" and "codab") are differentiated by name.
        """
        return self in {
            FlowControlElement.repeat_start,
            FlowControlElement.segno,
            FlowControlElement.coda,
            FlowControlElement.jump_to,
        }

    @property
    def is_break(self) -> bool:
        """Whether this marker voids contiguity.

        Note: repeat_end MAY also void contiguity when it coincides with
        an end-type barline, but this is context-dependent and not
        intrinsic to the type.
        """
        return self in {
            FlowControlElement.fine,
            FlowControlElement.section_break,
        }

    @property
    def is_structural_marker(self) -> bool:
        """Whether this is a structural marker that does NOT void contiguity."""
        return self in {
            FlowControlElement.repeat_start,
            FlowControlElement.double_barline,
            FlowControlElement.final_barline,
            FlowControlElement.first_measure,
            FlowControlElement.last_measure,
        }


class InterpolationKind(FancyStrEnum):
    """Interpolation methods for TableMap.

    Attributes:
        linear: Linear interpolation between points.
        nearest: Nearest-neighbor interpolation.
        previous: Use the previous point's value (step function, left).
        next: Use the next point's value (step function, right).
    """

    linear = auto()
    nearest = auto()
    previous = auto()
    next = auto()


class ExtrapolationPolicy(FancyStrEnum):
    """How to handle values outside the table bounds.

    Attributes:
        error: Raise an error for out-of-bounds values.
        extrapolate: Extend the interpolation beyond bounds.
        constant: Use the boundary value (clamp).
        nan: Return NaN for out-of-bounds values.
    """

    error = auto()
    extrapolate = auto()
    constant = auto()
    nan = auto()


class SupportPolicy(FancyStrEnum):
    """How to treat a transferred coordinate that falls outside alignment support.

    A coordinate transferred to another timeline is *out-of-support* when the
    entering coordinate lies outside the transferring WarpMap's source-anchor
    hull, or when the produced coordinate would fall outside the target
    timeline's ``[0, length]`` span. This policy decides what happens to such a
    timeline when a cross-timeline stamp is assembled.

    Attributes:
        omit: Drop the out-of-support timeline from the stamp entirely. This
            is the default: a coordinate below the first alignment anchor or
            beyond the last one has no defined position, so the timeline simply
            does not appear.
        clamp: Clamp the entering coordinate to the nearest hull boundary
            before warping (yielding that boundary anchor's target), then clip
            the result to ``[0, length]``. The timeline stays present.
        extrapolate: Keep the linear extrapolation beyond the hull, then clip
            the result to ``[0, length]``. The timeline stays present.

    No policy ever emits a negative coordinate or one exceeding the target
    timeline's length.
    """

    omit = auto()
    clamp = auto()
    extrapolate = auto()


class ActivationCondition(FancyStrEnum):
    """When a flow control event becomes active.

    This determines on which traversal pass the event takes effect. An
    element's manifestation decides its condition: a ``repeat_end`` acts on
    its first N passes, a ``to_coda`` after a D.C. or D.S., every other
    element always.
    """

    always = auto()
    """Always active (e.g., section end)."""

    first_n = auto()
    """Active on first N passes (e.g., repeat 2x)."""

    after_dc_ds = auto()
    """Active after Da Capo or Dal Segno."""


class ColumnRole(FancyStrEnum):
    """Semantic role of a column in the table.

    Used for automatic role inference when columns are not explicitly specified.
    """

    # Core event identity
    id = auto()
    name = auto()
    event_type = auto()

    # Coordinates (primary timeline)
    start = auto()
    end = auto()
    duration = auto()
    instant = auto()

    # Conversion (C-Map target)
    cmap_target = auto()

    # Structure
    partition = auto()
    parent_id = auto()
    child_id = auto()
    segment_name = auto()
    region = auto()

    # Alignment
    match_ref = auto()

    # Generic data
    extra = auto()


class PartitionMode(FancyStrEnum):
    """How partition columns create timelines.

    separate: Each unique value creates an independent timeline with its own
              coordinate system. Coordinates are NOT comparable across partitions.
              Use for: Different recordings, different performers.

    children: Each unique value creates a child timeline that shares the parent's
              coordinate system. Coordinates ARE comparable (offset by parent position).
              Use for: Voices, instruments, staves within a score.
    """

    separate = auto()
    """Disparate coordinate systems."""

    children = auto()
    """Same coordinate system, parent-child relationship."""


class FlowMode(FancyStrEnum):
    """Flow computation modes.

    Different contexts require different unfoldings:

    Deterministic modes (must be identical across all loaders):
    - atomic: True atomic sections from ScoreFlowController (= mode=None)
    - printed: All bars as printed (no unfolding)
    - single: Single playthrough (last volta only)

    Contingent modes (may have alternatives if divergent):
    - default: Most complete flow (all repeats taken), equivalent to MS3
    - music21: music21's expandRepeats() - only if diverges from default
    - partitura_maximal: partitura's unfold_part_maximal() - only if diverges from default
    - partitura_minimal: partitura's atomic segments - only if diverges from atomic

    Other modes:
    - ms3: From ms3's *_unfolded.measures.tsv (gold standard for default)
    - custom: User-provided flow sequence
    """

    atomic = auto()
    default = auto()
    ms3 = auto()
    partitura_minimal = auto()
    partitura_maximal = auto()
    music21 = auto()
    printed = auto()
    single = auto()
    custom = auto()


class IncompletePosition(FancyStrEnum):
    """Position of an incomplete measure within the score.

    Used by IncompleteMeasure to classify why a measure is incomplete:
    - anacrusis: Pickup measure at the start of the piece
    - final: Final incomplete measure (often pairs with anacrusis)
    - split_first: First part of a split measure
    - split_second: Second part of a split measure
    - unknown: Position not yet determined
    """

    anacrusis = auto()
    final = auto()
    split_first = auto()
    split_second = auto()
    unknown = auto()
