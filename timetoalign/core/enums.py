"""Enumerations for the TTA model.

This module defines the fundamental categorical types used throughout
the library: temporal domains, time units, number types, and event types.

The FancyStrEnum base class provides:
- Lowercase member values (via auto() from StrEnum)
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
        * A member whose Python name would shadow a ``str`` or ``Enum`` attribute takes
          another name and an explicit value (``label = "name"``); ``str()`` and
          ``repr()`` give the value; lookup by string takes a member's value (any
          case) or an alias name, never a canonical member's Python name

    Example:
        class Vocabulary(FancyStrEnum):
            abbreviation = auto()  # assigns the name as value (lowercase per StrEnum)
            abbr = abbreviation    # alias 1
            abb = abbreviation     # alias 2
    """

    @classmethod
    def _missing_(cls, value: object) -> FancyStrEnum:
        """Allow instantiation from values, including aliases.

        Args:
            value: A member's value or an alias name, in any case.

        Returns:
            The member with that value, or the member the alias names.

        Raises:
            ValueError: If ``value`` is not a string, or matches no member's value
                and no alias name.
        """
        if isinstance(value, str):
            lower_value = value.lower()
            for member in cls:
                if member.value == lower_value:
                    return member
            alias = cls.__members__.get(lower_value)
            if alias is not None and alias.name != lower_value:
                return alias
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
        str_components: list[str] = []
        for name, values in abbreviations.items():
            if not values:
                str_components.append(name)
                continue
            abbrev_str = ", ".join(values)
            str_components.append(f"{name} ({abbrev_str})")
        return ", ".join(str_components)

    def __repr__(self) -> str:
        return f'"{self.value}"'


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
    {
        "pixels",
        "meters",
        "centimeters",
        "millimeters",
        "inches",
        "points",
        "css_pixels",
        "user_units",
    }
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

    css_pixels = auto()
    """CSS pixels (1/96 in), the unit of an SVG's outermost viewport and of
    HTML/CSS layouts; distinct from the discrete :attr:`pixels`"""

    user_units = auto()
    """a document's own coordinate system, such as an SVG viewBox; its size
    is known only per document, so it converts only through that document's
    conversion map"""
    uu = user_units
    """a document's own coordinate system, such as an SVG viewBox"""

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


class ClaimType(FancyStrEnum):
    """The semantic kind of a MatchClaim, derived from its structure.

    A claim's kind is never stored; it is read from whether the claim is
    synchronous and how many of its two sides name an event. A MatchClaim
    always names two events, or one event and the NOMATCH sentinel on the
    other side; a side without an event of its own is given a Projection event
    first, never a bare coordinate. What a MatchGraph infers through groups is
    the graph's own and has no kind here.

    Members:
        event_match: Two events on different timelines correspond, temporally
            anchored.
        nomatch: An event has no counterpart on the other timeline.
        conceptual: Two events correspond structurally, with no temporal
            commitment (no anchors).
    """

    event_match = auto()
    """Two events correspond, temporally anchored."""
    event = event_match
    """Alias for event_match."""

    nomatch = auto()
    """An event has no counterpart on the other timeline."""
    nomat = nomatch
    """Alias for nomatch."""

    conceptual = auto()
    """Two events correspond with no temporal commitment (no anchors)."""


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
        label: Use the human-readable name property (e.g., "Musical Holes Region").
            Falls back to ID if name is not set. Its value is ``"name"``.
        id: Use the unique identifier (e.g., "dgt_holes").
    """

    label = "name"
    """Use human-readable name (with id fallback); value ``"name"``."""

    id = auto()
    """Use unique identifier."""


class FlowManifestation(FancyStrEnum):
    """What a flow-control element looks like in the score.

    A manifestation names the sign a source prints for an element or for a
    destination of a Jump. It is optional everywhere and nothing computes
    with it: the flow follows from the elements' classes, places and
    conditions alone. A word outside this vocabulary is kept as given.

    Which words each class of
    :mod:`timetoalign.timelines.flowcontrol` admits:

    - a destination of a ``Jump``: ``repeat_end``, ``da_capo``,
      ``da_capo_al_fine``, ``da_capo_al_coda``, ``dal_segno``,
      ``dal_segno_al_fine``, ``dal_segno_al_coda``, ``to_coda``, ``volta``,
      ``ossia``;
    - a ``JumpTarget``: ``repeat_start``, ``segno``, ``coda``, ``volta``,
      ``ossia``;
    - a ``Break``: ``fine``, ``volta``;
    - a ``Barline``: ``double_barline``, ``final_barline``;
    - a ``SectionBreak``: none.
    """

    repeat_start = auto()
    """A repeat start sign (``|:``), the place a repeat end goes back to."""
    repeat_end = auto()
    """A repeat end sign (``:|``)."""

    da_capo = auto()
    """Back to the beginning (D.C.)."""
    dc = da_capo
    """Alias for da_capo"""

    dal_segno = auto()
    """Back to the segno (D.S.)."""
    ds = dal_segno
    """Alias for dal_segno"""

    dal_segno_al_coda = auto()
    """Back to the segno, then from the To Coda sign to the coda (D.S. al Coda)."""
    dsac = dal_segno_al_coda
    """Alias for dal_segno_al_coda"""

    dal_segno_al_fine = auto()
    """Back to the segno, then up to the Fine (D.S. al Fine)."""
    dsaf = dal_segno_al_fine
    """Alias for dal_segno_al_fine"""

    da_capo_al_coda = auto()
    """Back to the beginning, then from the To Coda sign to the coda (D.C. al Coda)."""
    dcac = da_capo_al_coda
    """Alias for da_capo_al_coda"""

    da_capo_al_fine = auto()
    """Back to the beginning, then up to the Fine (D.C. al Fine)."""
    dcaf = da_capo_al_fine
    """Alias for da_capo_al_fine"""

    to_coda = auto()
    """The To Coda sign, from which the flow goes on at the coda after a leap."""

    segno = auto()
    """The segno sign, the place a dal segno goes back to."""

    coda = auto()
    """The coda sign at the start of the coda, the place a To Coda goes on at."""

    fine = auto()
    """The Fine, where the flow ends after a leap."""

    volta = auto()
    """A bracket of alternative endings: its Jump's destination, the place
    the Jump lands on, or the Break between two brackets."""

    ossia = auto()
    """An alternative text nobody can choose by rule: its Jump's destination
    or the place the Jump lands on."""

    double_barline = auto()
    """A double bar line."""

    final_barline = auto()
    """A final bar line."""


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


class ColumnRole(FancyStrEnum):
    """Semantic role of a column in the table.

    Used for automatic role inference when columns are not explicitly specified.

    Members whose Python name differs from their value: ``label`` (value
    ``"name"``) and ``partition_by`` (value ``"partition"``).
    """

    # Core event identity
    id = auto()
    label = "name"
    event_type = auto()

    # Coordinates (primary timeline)
    start = auto()
    end = auto()
    duration = auto()
    instant = auto()

    # Conversion (C-Map target)
    cmap_target = auto()

    # Structure
    partition_by = "partition"
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


class FlowConvention(FancyStrEnum):
    """How flow control is played into a flow.

    - default: every Jump and Break acts as its conditions say.
    - single: once through: nothing goes back, and every Jump is arrived at
      on the pass it would have after the repeats around it.
    - printed: the axis from its start to its end, flow control ignored.
    """

    default = auto()
    single = auto()
    printed = auto()


class FlowFindingKind(FancyStrEnum):
    """What playing flow control found."""

    flow_ends_at_break = auto()
    """The flow ended at a Break that states no condition, before the end."""

    never_played = auto()
    """A stretch of the axis lies in no run."""

    endless_loop = auto()
    """A Jump was taken in a state the flow had been in before."""

    ambiguous_targets = auto()
    """Two destinations of one Jump, of the same kind, held at once."""

    undetermined_target = auto()
    """A destination nobody can check was passed and not taken."""


class MeasureFindingKind(FancyStrEnum):
    """What a look at a score's measures found about their flow control."""

    repeat_end_behind_brackets = auto()
    """A repeat end without a printed start, where the last sign before it
    that closes anything is a last bracket of alternative endings: the
    passage it repeats begins right behind that bracket."""
