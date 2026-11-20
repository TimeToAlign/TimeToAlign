"""Protocols for semantic type classification in TimeToAlign!

This module defines the structural typing contracts (Protocols) that unify
scalar-level types (e.g., Coordinate) and field-level types (e.g.,
CoordinateField) under a single interface.

Hierarchy Overview::

    SemanticTypeLike                       Root protocol
    ├── CoordinateLike[V]                  Coordinate-bearing objects
    ├── TimedObjectLike                    Objects with temporal position
    │   ├── InstantEventLike              Single-instant events
    │   └── IntervalEventLike             Interval [start, end) events
    │       ├── NoteLike                  Notes/rests
    │       └── MeasureLike               Measures
    ├── HarmonyLabelLike                   Harmonic content (no temporal binding)
    │   └── PitchBasedHarmonyLike          (modelled after OHR)
    │       └── WesternTertianHarmonyLike
    │           └── RomanNumeralHarmonyLike
    │               └── DcmlHarmonyLike
    └── GenericPitchLike                  Pitch class only
        ├── SpecificPitchClassLike         + spelling (step, alter, fifths)
        └── EnharmonicPitchLike           + octave (midi_number) [EP]
            └── SpecificPitchLike         + spelling (step, alter, fifths, cents) [SP]

The ``TwelveTETPitchMixin`` is a concrete mixin (not a Protocol) that adds
12-TET pitch methods (``pitch_class``, ``to()``, ``get()``) to scalar classes.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Protocol, TypeVar, runtime_checkable

from .enums import Domain, NumberType, TimeUnit
from .fields import ScalarVocabulary

V = TypeVar("V", covariant=True)


# region Root Protocol


@runtime_checkable
class SemanticTypeLike(Protocol):
    """Protocol for objects that carry semantic type metadata.

    Any object with a ``semantic_type`` property and a
    ``metadata_dict()`` method satisfies this protocol.  This is the
    root contract shared by both scalar types (``Coordinate``) and
    field types (future ``SemanticField`` subclasses).

    Attributes:
        semantic_type: Canonical name of the semantic type
            (e.g., ``"Coordinate"``).

    Examples:
        >>> from timetoalign.core.time import Coordinate
        >>> from timetoalign.core.enums import TimeUnit
        >>> coord = Coordinate(1.5, TimeUnit.seconds)
        >>> isinstance(coord, SemanticTypeLike)
        True
        >>> coord.semantic_type
        'Coordinate'
    """

    @property
    def semantic_type(self) -> str:
        """The canonical SemanticType name."""
        ...

    def metadata_dict(self) -> dict[str, str]:
        """Return metadata dict matching the Parquet storage contract."""
        ...


# endregion Root Protocol

# region Coordinate Protocol


@runtime_checkable
class CoordinateLike(SemanticTypeLike, Protocol[V]):
    """Protocol for coordinate-bearing objects.

    Extends ``SemanticTypeLike`` with properties that describe a
    coordinate's value, unit, domain, and numeric representation.
    The type parameter ``V`` is covariant and unbound -- it may be a
    ``CoordinateValue`` (for scalars) or a ``StructField`` (for
    field-level wrappers).

    The existing ``Coordinate`` frozen dataclass satisfies this
    protocol structurally with no modifications.

    Attributes:
        value: The underlying value (scalar or field-level).
        unit: The time unit of this coordinate.
        domain: The temporal domain (logical, physical, graphical).
        number_type: The numeric representation (int, float, fraction).
    """

    @property
    def value(self) -> V:
        """The underlying value."""
        ...

    @property
    def unit(self) -> TimeUnit:
        """The time unit of this coordinate."""
        ...

    @property
    def domain(self) -> Domain:
        """The temporal domain this coordinate belongs to."""
        ...

    @property
    def number_type(self) -> NumberType:
        """The numeric type used for this coordinate's values."""
        ...


# endregion Coordinate Protocol

# region TimeScalar Protocol


@runtime_checkable
class TimeScalarLike(SemanticTypeLike, Protocol):
    """Protocol satisfied by ``Coordinate``, ``Duration``, and their Id-variants.

    Structural typing — any object exposing a ``unit`` of type
    :class:`TimeUnit` and the inherited ``semantic_type`` /
    ``metadata_dict`` from :class:`SemanticTypeLike` satisfies the
    protocol.  This is the grouping protocol used by
    ``get_fields_satisfying(TimeScalarLike)`` to discover ALL time-like
    columns on an EventData regardless of which sibling scalar
    (``Coordinate``, ``Duration``, ``IdCoordinate``, ``IdDuration``)
    they hold.
    """

    @property
    def unit(self) -> TimeUnit:
        """The time unit of this scalar."""
        ...


# endregion TimeScalar Protocol

# region Temporal Protocols


@runtime_checkable
class TimedObjectLike(SemanticTypeLike, Protocol):
    """Protocol for any object with a temporal position.

    Uses the canonical TTA model names: ``start`` (StartInstant)
    and ``end`` (EndInstant), matching the storage schema in
    ``EventData`` (fields ``start``, ``end``, ``duration``).
    """

    @property
    def start(self) -> CoordinateLike:  # type: ignore[type-arg]
        """The temporal position (StartInstant)."""
        ...


@runtime_checkable
class InstantEventLike(TimedObjectLike, Protocol):
    """A timed object at a single Instant (no duration).

    Only ``start`` is required (inherited from ``TimedObjectLike``).
    """

    pass


@runtime_checkable
class IntervalEventLike(TimedObjectLike, Protocol):
    """A timed object spanning an interval [start, end).

    Uses the canonical TTA model names from AGENTS.md Section 1.3:
    StartInstant -> ``start``, EndInstant -> ``end``.
    Intervals are left-inclusive, right-exclusive ``[start, end)``.
    """

    @property
    def end(self) -> CoordinateLike | None:  # type: ignore[type-arg]
        """The end position (EndInstant), or ``None``."""
        ...

    @property
    def duration(self) -> CoordinateLike | None:  # type: ignore[type-arg]
        """The duration as a ``Coordinate``, or ``None``."""
        ...


# endregion Temporal Protocols

# region Pitch Protocols


@runtime_checkable
class GenericPitchLike(SemanticTypeLike, Protocol):
    """Pitch class only (no octave, no spelling).

    The minimal 12-TET pitch representation: just the pitch class
    (chroma, 0-11).
    """

    @property
    def pitch_class(self) -> int:
        """Pitch class (0-11, C=0)."""
        ...


@runtime_checkable
class SpecificPitchClassLike(GenericPitchLike, Protocol):
    """Pitch class with spelling (e.g., C\u266f vs D\u266d).

    Adds enharmonic identity to the generic pitch class.
    """

    @property
    def step(self) -> str:
        """Generic pitch class as letter (``"C"``, ``"D"``, etc.)."""
        ...

    @property
    def alter(self) -> int:
        """Accidental in semitones (-1=flat, 0=natural, +1=sharp)."""
        ...

    @property
    def fifths(self) -> int:
        """Position on the line of fifths."""
        ...


@runtime_checkable
class EnharmonicPitchLike(GenericPitchLike, Protocol):
    """Enharmonic pitch: MIDI-level representation with octave.

    Called "enharmonic" because it **equates** enharmonic equivalents
    (C\u266f4 and D\u266d4 share MIDI number 61).

    The ``MidiPitch`` scalar (alias ``EnharmonicPitch``) satisfies this.
    """

    @property
    def midi_number(self) -> int:
        """MIDI note number (0-127)."""
        ...

    @property
    def octave(self) -> int:
        """Octave number (C4 = octave 4)."""
        ...


@runtime_checkable
class SpecificPitchLike(EnharmonicPitchLike, Protocol):
    """Specific pitch: full spelling with octave (C\u266f4 \u2260 D\u266d4).

    Called "specific" because it preserves the *specific* enharmonic
    spelling.

    The ``SpecificPitch`` scalar (alias ``SpecificPitch``) satisfies
    this protocol.
    """

    @property
    def step(self) -> str:
        """Generic pitch class as letter."""
        ...

    @property
    def alter(self) -> int:
        """Accidental in semitones."""
        ...

    @property
    def fifths(self) -> int:
        """Position on the line of fifths."""
        ...

    @property
    def cents(self) -> float:
        """Cents value."""
        ...


# endregion Pitch Protocols

# region Note Protocol


@runtime_checkable
class NoteLike(IntervalEventLike, Protocol):
    """Protocol for note/rest events.

    Ties pitch information to an interval event.  Temporal fields
    (``start``, ``end``, ``duration``) come from ``IntervalEventLike``.

    Attributes:
        pitch: The pitch of the note, or ``None`` for rests.
        voice: Voice number, or ``None``.
        staff: Staff number, or ``None``.
        velocity: MIDI velocity (0-127), or ``None``.
        instrument: Instrument name/identifier, or ``None``.
        is_rest: ``True`` if this event is a rest.
    """

    @property
    def pitch(self) -> GenericPitchLike | None:
        """The pitch of the note, or ``None`` for rests."""
        ...

    @property
    def voice(self) -> int | None:
        """Voice number, or ``None``."""
        ...

    @property
    def staff(self) -> int | None:
        """Staff number, or ``None``."""
        ...

    @property
    def velocity(self) -> int | None:
        """MIDI velocity (0-127), or ``None``."""
        ...

    @property
    def instrument(self) -> str | None:
        """Instrument name/identifier, or ``None``."""
        ...

    @property
    def is_rest(self) -> bool:
        """``True`` if this event is a rest (no pitch)."""
        ...


# endregion Note Protocol

# region Measure Protocol


@runtime_checkable
class MeasureLike(Protocol):
    """Structural protocol for Measure Map measure scalars."""

    id: str | None
    count: int | None
    qstamp: Fraction | None
    number: int | None
    name: str | None
    time_signature: str | None
    nominal_length: Fraction | None
    actual_length: Fraction | None
    start_repeat: bool
    end_repeat: bool
    next: tuple[str, ...] | None
    volta: int | None


# endregion Measure Protocol

# region Harmony Protocols


@runtime_checkable
class HarmonyLabelLike(SemanticTypeLike, Protocol):
    """Abstract root for all harmony annotations.

    Represents harmonic content (label + standard).  Temporal placement
    is NOT part of this protocol -- it belongs to the EventData row
    that contains the harmony.  A harmony label describes *what* the
    harmony is, not *when* it occurs.

    The ``.to_ohr()`` method (future) bridges to FlexOHR's rich
    harmonic model.
    """

    @property
    def label(self) -> str:
        """The full harmony label string."""
        ...

    @property
    def standard(self) -> str:
        """Codec identifier (e.g., ``"dcml"``, ``"chord_symbol"``)."""
        ...


@runtime_checkable
class PitchBasedHarmonyLike(HarmonyLabelLike, Protocol):
    """Harmony grounded in pitch, modelled after OHR.

    An OHR (Object of Harmonic Reference) has three components:
    - **reference component**: the root pitch
    - **reference OHR**: the bass (which may differ from root in inversions)
    - **body**: the chord quality describing the intervallic structure

    This protocol captures the minimal pitch-based properties.
    """

    @property
    def root(self) -> GenericPitchLike | None:
        """Root pitch (reference component), or ``None``."""
        ...

    @property
    def bass(self) -> GenericPitchLike | None:
        """Bass note (reference OHR), or ``None``."""
        ...


@runtime_checkable
class WesternTertianHarmonyLike(PitchBasedHarmonyLike, Protocol):
    """Western tertian chord model.

    Minimal schema adds: chord type and inversion.  Everything else
    (which notes are in the chord, voicing, etc.) can be inferred from
    root + chord_type + inversion.
    """

    @property
    def chord_type(self) -> str:
        """Chord type (``"M"``, ``"m"``, ``"o"``, ``"+"``, ``"Mm7"``, etc.)."""
        ...

    @property
    def inversion(self) -> int | None:
        """Inversion number, or ``None``."""
        ...


@runtime_checkable
class RomanNumeralHarmonyLike(WesternTertianHarmonyLike, Protocol):
    """Roman-numeral analysis.

    Minimal schema adds: the numeral itself, plus localkey and globalkey.
    """

    @property
    def numeral(self) -> str:
        """Roman numeral (``"I"``, ``"ii"``, ``"V"``, etc.)."""
        ...

    @property
    def localkey(self) -> str:
        """Local key at this position (e.g., ``"IV"``)."""
        ...

    @property
    def globalkey(self) -> str:
        """Global key of the piece (e.g., ``"C"``)."""
        ...


@runtime_checkable
class DcmlHarmonyLike(RomanNumeralHarmonyLike, Protocol):
    """DCML codec specifics.

    Named ``DcmlHarmonyLike`` (not ``DcmlLabelLike``) because this
    represents the DCML-specific harmonic annotation model.

    The ``pedal`` concept is intentionally minimal here; full
    pedal-tone semantics (embedding chord progressions over a pedal
    as a horizontal sequence of OHRs) will co-evolve with FlexOHR.
    """

    @property
    def tonicized_key(self) -> str | None:
        """Tonicized key (DCML ``relativeroot``), or ``None``."""
        ...

    @property
    def pedal(self) -> str | None:
        """Pedal tone, or ``None``."""
        ...


# endregion Harmony Protocols

# region Pitch Mixin


class TwelveTETPitchMixin(ScalarVocabulary):
    """Concrete mixin adding 12-TET pitch methods.

    Not a Protocol -- a mixin that provides a unified ``.to()`` dispatch
    method and a ``.get()`` method with ``format`` support.  Scalar pitch
    classes (``EnharmonicPitchClass``, ``MidiPitch``, ``SpecificPitch``,
    etc.) compose this mixin alongside ``pydantic.BaseModel``.

    Note on ``pitch_class``: the mixin intentionally does NOT declare
    ``pitch_class`` as a property here.  Concrete scalar classes own the
    attribute — either as a pydantic field (``EnharmonicPitchClass``,
    ``GenericPitchClass``) or as a ``@property`` (``EnharmonicPitch``,
    ``SpecificPitch``, …).  Declaring it on the mixin as a raising
    property would conflict with pydantic's field-descriptor handling on
    subclasses that name a same-name field (V2 emits a "field shadows
    attribute in parent" warning and uses the parent property's
    descriptor, breaking the field).

    Representation contract: ``repr()`` is the SHORT typed form
    ``ABBR(token)`` (each concrete scalar sets ``_REPR_ABBR``); ``str()``
    is the PRETTY human token (``get()``).  Concrete scalars whose pretty
    ``get()`` token differs from the desired repr token (e.g. the dual
    enharmonic spelling) override ``__repr__``.
    """

    #: Short repr prefix.  No sensible default -- each concrete scalar sets it.
    _REPR_ABBR: str

    def __str__(self) -> str:
        return self.get()

    def __repr__(self) -> str:
        return f"{self._REPR_ABBR}({self.get()})"

    def to(
        self, target_type: type, *, format: str | None = None
    ) -> "TwelveTETPitchMixin":
        """Convert to a different pitch representation.

        Unified dispatch method for pitch conversion.  Replaces
        individual ``to_midi_number()``, ``to_generic()``,
        ``to_specific()`` methods.

        Args:
            target_type: The target pitch type (e.g., ``EnharmonicPitchClass``,
                ``MidiPitch``, ``SpecificPitch``).
            format: Optional format specifier controlling output
                representation (e.g., for string formatting).

        Returns:
            A ``TwelveTETPitchMixin`` instance of the target type.

        Raises:
            TypeError: If conversion to *target_type* is not supported.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support conversion "
            f"to {target_type.__name__}"
        )

    def get(self, *, format: str | None = None) -> str:
        """Return a string representation of this pitch.

        Args:
            format: Format specifier (e.g., ``"midi"``, ``"specific"``,
                ``"lily"``, ``"kern"``, ``"abc"``).  Default uses the
                most natural format for this pitch type.

        Returns:
            Formatted string representation.
        """
        raise NotImplementedError


# endregion Pitch Mixin
