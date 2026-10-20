"""Flow Control Events: Break, Jump and Marker for timeline traversal.

This module provides the three kinds of flow-control element a timeline
holds, each written in a TimeSkeleton manifest as one element of the
timeline's ``flow_control`` array:

**Break** (a ``flow_marker``): a control event that voids the contiguity at
the Instant where it is located: a Fine, a section break, a double or a
final barline.

**Jump** (a ``jump_from``): a control event defined by two Instants, a
JumpFrom and a JumpTo Instant. When active, it makes any event located or
starting at JumpTo contiguous with any event ending at JumpFrom: a repeat
end, a D.C. or D.S. (al Fine, al Coda) or a To Coda. It names the marker it
lands on, or none when it lands at the timeline's start.

**Marker** (a ``jump_to``): a position a jump lands on: a repeat start, a
segno or a coda.

Each element's ``control_type`` is its manifestation, which decides when it
acts (:attr:`Jump.condition`): a repeat end on its first ``repeat_count``
passes, a To Coda only after a D.C. or D.S., every other element always.

Common music notation mappings:
| Notation        | Element | Notes                                    |
|-----------------|---------|-------------------------------------------|
| Repeat End (:∥) | Jump    | Jump back to its repeat start, or 0       |
| Repeat Start    | Marker  | Target of a repeat end                    |
| First Ending    | -       | Volta info is a measure attribute         |
| Fine            | Break   | End of piece after an al-Fine jump        |
| Da Capo         | Jump    | Jump to coordinate 0                      |
| Dal Segno       | Jump    | Jump to a Segno marker by id              |
| Section Break   | Break   | Single instant voiding contiguity         |
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any, ClassVar

from timetoalign.core import Coordinate, IdCoordinate, NumberType, TimeUnit
from timetoalign.core.enums import ActivationCondition, FlowControlElement
from timetoalign.core.ids import _admit_id, _IdentifiedElement, mint_id
from timetoalign.core.time import express_scalar_as

from .manifest import (
    axis_value_from_json,
    axis_value_to_json,
    read_id,
    read_meta,
    read_name,
    require_object,
)

module_logger = logging.getLogger(__name__)

# region Manifestations

_FLOW_MARKERS = frozenset(
    {
        FlowControlElement.fine,
        FlowControlElement.section_break,
        FlowControlElement.double_barline,
        FlowControlElement.final_barline,
    }
)
"""The manifestations of a ``flow_marker``: a :class:`Break`'s control types."""

_JUMP_ORIGINS = frozenset(
    {
        FlowControlElement.repeat_end,
        FlowControlElement.da_capo,
        FlowControlElement.da_capo_al_fine,
        FlowControlElement.da_capo_al_coda,
        FlowControlElement.dal_segno,
        FlowControlElement.dal_segno_al_fine,
        FlowControlElement.dal_segno_al_coda,
        FlowControlElement.to_coda,
    }
)
"""The manifestations of a ``jump_from``: a :class:`Jump`'s control types."""

_JUMP_TARGETS = frozenset(
    {FlowControlElement.repeat_start, FlowControlElement.segno, FlowControlElement.coda}
)
"""The manifestations of a ``jump_to``: a :class:`Marker`'s control types."""

_TO_START = frozenset(
    {
        FlowControlElement.repeat_end,
        FlowControlElement.da_capo,
        FlowControlElement.da_capo_al_fine,
        FlowControlElement.da_capo_al_coda,
    }
)
"""The jumps that may name no target, landing at the timeline's start."""

_LANDING = {
    FlowControlElement.repeat_end: FlowControlElement.repeat_start,
    FlowControlElement.dal_segno: FlowControlElement.segno,
    FlowControlElement.dal_segno_al_fine: FlowControlElement.segno,
    FlowControlElement.dal_segno_al_coda: FlowControlElement.segno,
    FlowControlElement.to_coda: FlowControlElement.coda,
    FlowControlElement.da_capo_al_coda: FlowControlElement.coda,
}
"""The marker a jump lands on, by the jump's manifestation."""


def _condition(control_type: FlowControlElement) -> ActivationCondition:
    """When an element of *control_type* acts."""
    if control_type is FlowControlElement.repeat_end:
        return ActivationCondition.first_n
    if control_type is FlowControlElement.to_coda:
        return ActivationCondition.after_dc_ds
    return ActivationCondition.always


def _is_active(
    condition: ActivationCondition,
    repeat_count: int,
    pass_number: int,
    after_dc_ds: bool,
) -> bool:
    """Whether an element acting under *condition* acts on this pass."""
    if condition is ActivationCondition.first_n:
        return pass_number <= repeat_count
    if condition is ActivationCondition.after_dc_ds:
        return after_dc_ds
    return True


def _check_control_type(
    element: str, control_type: FlowControlElement, allowed: frozenset[Any]
) -> None:
    """Refuse a control type that is not among an element's manifestations.

    Raises:
        ValueError: If *control_type* is not in *allowed*.
    """
    if control_type not in allowed:
        raise ValueError(
            f"A {element} is one of {sorted(member.value for member in allowed)}, "
            f"not {control_type.value!r}"
        )


# endregion

# region Reading elements


def _read_element(
    data: Any, element_type: str, allowed: frozenset[Any]
) -> tuple[str, str, FlowControlElement]:
    """Read an element's id and manifestation.

    Returns:
        The element's id, its description for messages and its
        manifestation.

    Raises:
        ValueError: If the element lacks a required key or its manifestation
            is none of *element_type*'s.
    """
    require_object(
        data, "a flow-control element", ("id", "type", "manifestation", "coordinate")
    )
    element_id = read_id(data["id"], "a flow-control element")
    what = f"{element_type} {element_id!r}"
    manifestation = data["manifestation"]
    allowed_values = {member.value for member in allowed}
    if manifestation not in allowed_values:
        raise ValueError(
            f"{what} has the manifestation {manifestation!r}; a {element_type} is "
            f"one of {sorted(allowed_values)}"
        )
    return element_id, what, FlowControlElement(manifestation)


def _refuse_keys(data: Mapping[str, Any], what: str, keys: tuple[str, ...]) -> None:
    """Refuse keys another manifestation or element type holds.

    Raises:
        ValueError: If *data* holds one of *keys*.
    """
    held = [key for key in keys if key in data]
    if held:
        raise ValueError(f"{what} holds {held}, which only a repeat end or a jump has")


def _read_label(data: Mapping[str, Any], what: str) -> str | None:
    """Read an optional ``label``, null when absent.

    Raises:
        ValueError: If the label is neither a string nor null.
    """
    label = data.get("label")
    if label is not None and not isinstance(label, str):
        raise ValueError(f"{what} has a non-string label {label!r}")
    return label


def _read_coordinate(
    data: Mapping[str, Any], element_id: str, unit: TimeUnit, number_type: NumberType
) -> Coordinate:
    """Read an element's ``coordinate``, a bare value on the timeline's axis."""
    return axis_value_from_json(
        data["coordinate"],
        key="coordinate",
        owner=element_id,
        unit=unit,
        number_type=number_type,
    )


# endregion

# region Break


@dataclass(frozen=True)
class Break(_IdentifiedElement):
    """A control event that voids contiguity at a specific coordinate.

    A break voids contiguity at the Instant where it is located: a Fine, a
    section break, a double or a final barline. A manifest writes it as a
    ``flow_marker`` element whose manifestation is its control type.

    Constraints:
    - A Break cannot be added at coordinates already spanned by a
      TimeInterval (unless the spanning element "adopts" the break)
    - TimeIntervals cannot span a coordinate containing a Break

    Note on Voltas (Alternative Endings):
        Volta information is a MEASURE ATTRIBUTE, not a Break type.
        Each measure has a `volta` field (1, 2, None) indicating which
        alternative ending it belongs to. The traversal logic evaluates
        the volta attribute when processing repeat_end jumps.

    Attributes:
        coordinate: The coordinate where contiguity is voided.
        control_type: The type of break (``section_break``, ``fine``,
            ``double_barline``, ``final_barline``).
        label: Optional human-readable label (e.g., "End of Var. XI").
        name: For target markers, the instance name (e.g., "fine", "fine2").
        meta: Additional metadata.
        id: The break's id; minted as ``break_{N}`` when not given.

    Examples:
        >>> from timetoalign.core import Coordinate, TimeUnit
        >>> # Section break
        >>> brk = Break(
        ...     coordinate=Coordinate(100.0, TimeUnit.quarters),
        ...     control_type=FlowControlElement.section_break,
        ... )

        >>> # Fine
        >>> fine = Break(
        ...     coordinate=Coordinate(200.0, TimeUnit.quarters),
        ...     control_type=FlowControlElement.fine,
        ...     name="fine",
        ... )
    """

    _id_prefix: ClassVar[str] = "flow_marker"
    _dict_keys: ClassVar[tuple[str, ...]] = (
        "id",
        "type",
        "manifestation",
        "coordinate",
        "name",
        "label",
        "meta",
    )
    _dict_defaults: ClassVar[Mapping[str, Any]] = {
        "name": None,
        "label": None,
        "meta": {},
    }

    coordinate: Coordinate
    control_type: FlowControlElement = FlowControlElement.section_break
    label: str | None = None
    name: str | None = None  # Instance name for target markers
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        unit: TimeUnit,
        number_type: NumberType,
    ) -> Break:
        """Restore a break from its ``flow_marker`` element.

        ``name``, ``label`` and ``meta`` may be left out; keys the library
        does not know are kept and written back. *unit* and *number_type*
        describe the timeline's axis.

        Raises:
            ValueError: If the element is malformed, its manifestation is no
                ``flow_marker``'s, it holds a ``target`` or ``repeat_count``,
                or its coordinate is no value of the axis.
        """
        break_id, what, manifestation = _read_element(
            data, "flow_marker", _FLOW_MARKERS
        )
        _refuse_keys(data, what, ("target", "repeat_count"))
        brk = cls(
            id=break_id,
            coordinate=_read_coordinate(data, break_id, unit, number_type),
            control_type=manifestation,
            label=_read_label(data, what),
            name=read_name(data, what),
            meta=read_meta(data, what),
        )
        brk._keep_unknown_keys(data)
        return brk

    def __post_init__(self) -> None:
        """Mint or validate the id, then refuse a control type no break has."""
        super().__post_init__()
        _check_control_type("Break", self.control_type, _FLOW_MARKERS)

    @property
    def condition(self) -> ActivationCondition:
        """When the break acts: always."""
        return _condition(self.control_type)

    @property
    def position(self) -> float | Fraction:
        """The coordinate value where the break occurs."""
        return self.coordinate.value

    @property
    def unit(self) -> TimeUnit:
        """The time unit of the break coordinate."""
        return self.coordinate.unit

    def is_active(self, pass_number: int = 1, after_dc_ds: bool = False) -> bool:
        """Check if this break is active on a given traversal pass.

        Args:
            pass_number: The current pass number (1-indexed).
            after_dc_ds: Whether we're after a Da Capo or Dal Segno.

        Returns:
            True if the break should void contiguity on this pass.
        """
        return _is_active(self.condition, 1, pass_number, after_dc_ds)

    def to_dict(self, *, minimal: bool = False) -> dict[str, Any]:
        """Return this break as its ``flow_marker`` element.

        Args:
            minimal: Leave out every key at its default.
        """
        return self._dict_form(
            {
                "id": self.id,
                "type": "flow_marker",
                "manifestation": self.control_type.value,
                "coordinate": axis_value_to_json(self.coordinate),
                "name": self.name,
                "label": self.label,
                "meta": deepcopy(self.meta),
            },
            minimal=minimal,
        )

    def __repr__(self) -> str:
        type_str = self.control_type.name
        pos = self.position
        if self.name:
            return f"Break({type_str} '{self.name}' @ {pos})"
        return f"Break({type_str} @ {pos})"


# endregion

# region Jump


@dataclass(frozen=True)
class Jump(_IdentifiedElement):
    """A control event that creates contiguity between non-adjacent coordinates.

    A jump is defined by two Instants: a JumpFrom and a JumpTo Instant. When
    active, it makes any event located or starting at JumpTo contiguous with
    any event ending at JumpFrom. A manifest writes it as a ``jump_from``
    element at its from coordinate, naming the ``jump_to`` it lands on.

    Target Resolution:
        A jump to a marker references it by the marker's ID (``target``),
        so several markers of one type ("segno", "segno2") coexist in a
        score. ``target`` is ``None`` for a jump without a marker target
        (a plain repeat, a jump to the start); ``to_coordinate`` always
        carries the destination.

    Attributes:
        from_coordinate: The coordinate where the jump originates.
        to_coordinate: The coordinate where the jump lands.
        control_type: The type of jump (``repeat_end``, ``da_capo``,
            ``dal_segno``, ``to_coda``, and the al-Fine and al-Coda forms of
            a D.C. or D.S.).
        repeat_count: For a ``repeat_end``, how many times to take the
            jump; 1 for every other jump.
        target: The id of the target marker of the same timeline, or None.
        name: Optional display name.
        label: Optional human-readable label.
        meta: Additional metadata.
        id: The jump's id; minted as ``jump_{N}`` when not given.

    Examples:
        >>> from timetoalign.core import Coordinate, TimeUnit
        >>> # Simple repeat: jump from measure 32 back to measure 8
        >>> repeat_jump = Jump(
        ...     from_coordinate=Coordinate(64.0, TimeUnit.quarters),
        ...     to_coordinate=Coordinate(16.0, TimeUnit.quarters),
        ...     control_type=FlowControlElement.repeat_end,
        ...     repeat_count=1,  # Take jump once (play section twice)
        ... )

        >>> # Da Capo: jump to beginning
        >>> dc_jump = Jump(
        ...     from_coordinate=Coordinate(200.0, TimeUnit.quarters),
        ...     to_coordinate=Coordinate(0.0, TimeUnit.quarters),
        ...     control_type=FlowControlElement.da_capo,
        ... )

        >>> # Dal Segno to the marker with id "segno2"
        >>> ds_jump = Jump(
        ...     from_coordinate=Coordinate(300.0, TimeUnit.quarters),
        ...     to_coordinate=Coordinate(50.0, TimeUnit.quarters),
        ...     control_type=FlowControlElement.dal_segno,
        ...     target="segno2",
        ... )
    """

    _id_prefix: ClassVar[str] = "jump_from"
    _dict_keys: ClassVar[tuple[str, ...]] = (
        "id",
        "type",
        "manifestation",
        "coordinate",
        "target",
        "repeat_count",
        "name",
        "label",
        "meta",
    )
    _dict_defaults: ClassVar[Mapping[str, Any]] = {
        "repeat_count": 1,
        "name": None,
        "label": None,
        "meta": {},
    }

    from_coordinate: Coordinate
    to_coordinate: Coordinate
    control_type: FlowControlElement = FlowControlElement.repeat_end
    repeat_count: int = 1
    target: str | None = None
    label: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    name: str | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        unit: TimeUnit,
        number_type: NumberType,
        markers: Mapping[str, Marker],
    ) -> Jump:
        """Restore a jump from its ``jump_from`` element.

        The jump lands on its target's coordinate, or at 0 when a
        ``repeat_end`` or a D.C. names no target. ``repeat_count`` (only on
        a ``repeat_end``), ``name``, ``label`` and ``meta`` may be left out;
        keys the library does not know are kept and written back.

        Args:
            data: The ``jump_from`` element.
            unit: The axis's unit.
            number_type: The axis's number type.
            markers: The timeline's markers by id, which a target names.

        Raises:
            ValueError: If the element is malformed, its manifestation is no
                ``jump_from``'s, its target names no ``jump_to`` of the
                timeline or is missing where one is required, it holds a
                ``repeat_count`` though no ``repeat_end``, or its coordinate
                is no value of the axis.
        """
        jump_id, what, manifestation = _read_element(data, "jump_from", _JUMP_ORIGINS)
        if manifestation is not FlowControlElement.repeat_end:
            _refuse_keys(data, what, ("repeat_count",))
        repeat_count = data.get("repeat_count", 1)
        if (
            isinstance(repeat_count, bool)
            or not isinstance(repeat_count, int)
            or repeat_count < 1
        ):
            raise ValueError(
                f"{what} has a repeat_count {repeat_count!r}, which is no integer of "
                "at least 1"
            )
        target = data.get("target")
        if target is None:
            if manifestation not in _TO_START:
                raise ValueError(
                    f"{what} ({manifestation.value}) names no target: only a repeat "
                    "end or a D.C. lands at the timeline's start"
                )
            to_coordinate = Coordinate(0, unit, number_type=number_type)
        elif not isinstance(target, str) or target not in markers:
            raise ValueError(
                f"{what} targets {target!r}, which names no jump_to of this "
                f"timeline. Available jump_to elements: {list(markers)}"
            )
        else:
            to_coordinate = markers[target].coordinate
        jump = cls(
            id=jump_id,
            from_coordinate=_read_coordinate(data, jump_id, unit, number_type),
            to_coordinate=to_coordinate,
            control_type=manifestation,
            repeat_count=repeat_count,
            target=target,
            label=_read_label(data, what),
            meta=read_meta(data, what),
            name=read_name(data, what),
        )
        jump._keep_unknown_keys(data)
        return jump

    def __post_init__(self) -> None:
        """Mint or validate the id, then validate the jump configuration."""
        super().__post_init__()
        _check_control_type("Jump", self.control_type, _JUMP_ORIGINS)
        if self.from_coordinate.unit != self.to_coordinate.unit:
            raise ValueError(
                f"Jump coordinates must have same unit: "
                f"{self.from_coordinate.unit} != {self.to_coordinate.unit}"
            )
        if (
            self.repeat_count != 1
            and self.control_type is not FlowControlElement.repeat_end
        ):
            raise ValueError(
                f"A {self.control_type.value} jump has no repeat_count: only a "
                "repeat_end is taken several times"
            )
        if self.repeat_count < 1:
            raise ValueError(f"A jump's repeat_count {self.repeat_count} is below 1")

    @property
    def condition(self) -> ActivationCondition:
        """When the jump acts, which its control type decides.

        A ``repeat_end`` on its first ``repeat_count`` passes, a ``to_coda``
        only after a D.C. or D.S., every other jump always.
        """
        return _condition(self.control_type)

    @property
    def from_position(self) -> float | Fraction:
        """The coordinate value where the jump originates."""
        return self.from_coordinate.value

    @property
    def to_position(self) -> float | Fraction:
        """The coordinate value where the jump lands."""
        return self.to_coordinate.value

    @property
    def unit(self) -> TimeUnit:
        """The time unit of the jump coordinates."""
        return self.from_coordinate.unit

    @property
    def is_backward(self) -> bool:
        """Whether this jump goes to an earlier coordinate (repeat)."""
        return self.to_position < self.from_position

    @property
    def is_forward(self) -> bool:
        """Whether this jump goes to a later coordinate (skip)."""
        return self.to_position > self.from_position

    @property
    def distance(self) -> float | Fraction:
        """The absolute distance covered by the jump."""
        diff = self.to_position - self.from_position
        return abs(diff)

    def is_active(self, pass_number: int = 1, after_dc_ds: bool = False) -> bool:
        """Check if this jump is active on a given traversal pass.

        Args:
            pass_number: The current pass number (1-indexed).
            after_dc_ds: Whether we're after a Da Capo or Dal Segno.

        Returns:
            True if the jump should be taken on this pass.
        """
        return _is_active(self.condition, self.repeat_count, pass_number, after_dc_ds)

    def remaining_activations(self, pass_number: int) -> int:
        """How many more times this jump can be taken.

        Args:
            pass_number: The current pass number (1-indexed).

        Returns:
            Number of remaining activations (0 if no longer active).
        """
        if self.condition is ActivationCondition.first_n:
            return max(0, self.repeat_count - pass_number + 1)
        elif self.condition is ActivationCondition.always:
            return float("inf")  # type: ignore
        return 0

    def to_dict(self, *, minimal: bool = False) -> dict[str, Any]:
        """Return this jump as its ``jump_from`` element.

        ``target`` is written when the jump names one, ``repeat_count`` on a
        ``repeat_end`` only. A jump to a position no marker of its timeline
        lies on is written by its registry, which names a ``jump_to`` there.

        Args:
            minimal: Leave out every key at its default.
        """
        fields: dict[str, Any] = {
            "id": self.id,
            "type": "jump_from",
            "manifestation": self.control_type.value,
            "coordinate": axis_value_to_json(self.from_coordinate),
        }
        if self.target is not None:
            fields["target"] = self.target
        if self.control_type is FlowControlElement.repeat_end:
            fields["repeat_count"] = self.repeat_count
        fields.update(name=self.name, label=self.label, meta=deepcopy(self.meta))
        return self._dict_form(fields, minimal=minimal)

    def __repr__(self) -> str:
        type_str = self.control_type.name
        from_pos = self.from_position
        to_pos = self.to_position
        direction = "←" if self.is_backward else "→"
        return f"Jump({type_str}: {from_pos} {direction} {to_pos})"


# endregion

# region Marker


@dataclass(frozen=True)
class Marker(_IdentifiedElement):
    """A position a jump lands on: a repeat start, a segno or a coda.

    Jumps reference a marker by its ``id``; ``name`` is the display name
    and the source's marker instance name (``"segno"``, ``"codab"``). What
    the marker is, ``control_type``, is stated, never guessed from the name.
    A manifest writes it as a ``jump_to`` element whose manifestation is its
    control type.

    Attributes:
        coordinate: The marker's position.
        name: Optional display name.
        control_type: ``repeat_start``, ``segno`` or ``coda``.
        label: Optional human-readable label.
        meta: Additional metadata.
        id: The marker's id; minted as ``marker_{N}`` when not given.

    Examples:
        >>> from timetoalign.core import Coordinate, TimeUnit
        >>> Marker(
        ...     Coordinate(16, TimeUnit.quarters),
        ...     name="segno",
        ...     control_type=FlowControlElement.segno,
        ... ).id
        'jump_to_1'
    """

    _id_prefix: ClassVar[str] = "jump_to"
    _dict_keys: ClassVar[tuple[str, ...]] = (
        "id",
        "type",
        "manifestation",
        "coordinate",
        "name",
        "label",
        "meta",
    )
    _dict_defaults: ClassVar[Mapping[str, Any]] = {
        "name": None,
        "label": None,
        "meta": {},
    }

    coordinate: Coordinate
    name: str | None = None
    control_type: FlowControlElement = field(kw_only=True)
    label: str | None = field(default=None, kw_only=True)
    meta: dict[str, Any] = field(default_factory=dict, kw_only=True)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        unit: TimeUnit,
        number_type: NumberType,
    ) -> Marker:
        """Restore a marker from its ``jump_to`` element.

        ``name``, ``label`` and ``meta`` may be left out; keys the library
        does not know are kept and written back. *unit* and *number_type*
        describe the timeline's axis.

        Raises:
            ValueError: If the element is malformed, its manifestation is no
                ``jump_to``'s, it holds a ``target`` or ``repeat_count``, or
                its coordinate is no value of the axis.
        """
        marker_id, what, manifestation = _read_element(data, "jump_to", _JUMP_TARGETS)
        _refuse_keys(data, what, ("target", "repeat_count"))
        marker = cls(
            id=marker_id,
            name=read_name(data, what),
            control_type=manifestation,
            label=_read_label(data, what),
            meta=read_meta(data, what),
            coordinate=_read_coordinate(data, marker_id, unit, number_type),
        )
        marker._keep_unknown_keys(data)
        return marker

    def __post_init__(self) -> None:
        """Mint or validate the id, then refuse a control type no marker has."""
        super().__post_init__()
        _check_control_type("Marker", self.control_type, _JUMP_TARGETS)

    @property
    def position(self) -> float | Fraction:
        """The coordinate value of the marker."""
        return self.coordinate.value

    def to_dict(self, *, minimal: bool = False) -> dict[str, Any]:
        """Return this marker as its ``jump_to`` element.

        Args:
            minimal: Leave out every key at its default.
        """
        return self._dict_form(
            {
                "id": self.id,
                "type": "jump_to",
                "manifestation": self.control_type.value,
                "coordinate": axis_value_to_json(self.coordinate),
                "name": self.name,
                "label": self.label,
                "meta": deepcopy(self.meta),
            },
            minimal=minimal,
        )

    def __repr__(self) -> str:
        label = (
            repr(self.id) if self.name is None else f"{self.id!r}, name={self.name!r}"
        )
        return f"Marker({label} @ {self.position})"


# endregion

# region FlowControlRegistry


def _ordered(elements: list[Any]) -> list[Any]:
    """Sort flow-control elements by position, keeping insertion order on ties."""
    return sorted(elements, key=lambda element: element.position)


@dataclass
class FlowControlRegistry:
    """Registry of all flow control events for a timeline.

    This class manages the collection of Breaks, Jumps and Markers for a
    timeline, providing methods to query, add, and validate them. Every
    element carries an id unique within the registry and, once the registry
    belongs to a timeline, within that timeline's whole manifest: an element
    whose minted id is taken is stored under a freshly minted id, and one
    whose given id is taken is refused. A manifest writes the registry as
    the timeline's ``flow_control`` array.

    Attributes:
        breaks: List of Break events, ordered by coordinate.
        jumps: List of Jump events, ordered by from coordinate.
        markers: List of Marker targets, ordered by coordinate.

    Examples:
        >>> from timetoalign.core import Coordinate, TimeUnit
        >>> registry = FlowControlRegistry(unit=TimeUnit.quarters)
        >>> registry.add_break(Break(Coordinate(64, TimeUnit.quarters)))
        Break(section_break @ 64)
        >>> segno = FlowControlElement.segno
        >>> registry.add_marker(
        ...     Marker(Coordinate(16, TimeUnit.quarters), name="segno", control_type=segno)
        ... )
        Marker('jump_to_1', name='segno' @ 16)
        >>> coda = FlowControlElement.coda
        >>> registry.add_marker(
        ...     Marker(Coordinate(32, TimeUnit.quarters), name="coda", control_type=coda)
        ... ).id
        'jump_to_2'
    """

    breaks: list[Break] = field(default_factory=list)
    jumps: list[Jump] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    unit: TimeUnit | None = field(default=None, kw_only=True)
    number_type: NumberType | None = field(default=None, kw_only=True)
    # Set by the owning timeline: admits an id into its whole manifest.
    _admit: Callable[..., str] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def from_dict(
        cls,
        data: list[dict[str, Any]],
        *,
        unit: TimeUnit | str,
        number_type: NumberType | str | None = None,
    ) -> FlowControlRegistry:
        """Restore a registry bound to a timeline's axis from its ``flow_control``.

        Each element of the array is a ``flow_marker`` (a :class:`Break`),
        a ``jump_from`` (a :class:`Jump`) or a ``jump_to`` (a
        :class:`Marker`), its coordinate a bare value on the axis that
        *unit* and *number_type* (the unit's own when None) describe.
        Elements are added through the public add methods, markers before
        jumps, in any order the array lists them. Keys the library does not
        know are kept on each element and written back.

        Args:
            data: The ``flow_control`` array.
            unit: The axis's unit.
            number_type: The axis's number type, or None for the unit's own.

        Raises:
            ValueError: If the array or an element is malformed, an element's
                type is unknown, or a jump's target names no ``jump_to``.
        """
        resolved_unit = TimeUnit(unit)
        resolved_number_type = (
            resolved_unit.default_number_type
            if number_type is None
            else NumberType(number_type)
        )
        if not isinstance(data, list):
            raise ValueError("the flow_control must be an array")
        axis = {"unit": resolved_unit, "number_type": resolved_number_type}
        by_type: dict[str, list[dict[str, Any]]] = {
            "flow_marker": [],
            "jump_to": [],
            "jump_from": [],
        }
        for item in data:
            require_object(item, "a flow-control element", ("type",))
            if item["type"] not in by_type:
                raise ValueError(
                    f"a flow-control element has the unknown type {item['type']!r}; "
                    f"an element is one of {sorted(by_type)}"
                )
            by_type[item["type"]].append(item)
        registry = cls(unit=resolved_unit, number_type=resolved_number_type)
        for item in by_type["flow_marker"]:
            registry.add_break(Break.from_dict(item, **axis))
        for item in by_type["jump_to"]:
            registry.add_marker(Marker.from_dict(item, **axis))
        markers = {marker.id: marker for marker in registry.markers}
        for item in by_type["jump_from"]:
            registry.add_jump(Jump.from_dict(item, **axis, markers=markers))
        return registry

    def __post_init__(self) -> None:
        """Normalize binding enum spellings and any constructor content."""
        if isinstance(self.unit, str):
            self.unit = TimeUnit(self.unit)
        if isinstance(self.number_type, str):
            self.number_type = NumberType(self.number_type)
        original_breaks = list(self.breaks)
        original_jumps = list(self.jumps)
        original_markers = list(self.markers)
        self.breaks = []
        self.jumps = []
        self.markers = []
        for brk in original_breaks:
            self.add_break(brk)
        for marker in original_markers:
            self.add_marker(marker)
        for jump in original_jumps:
            self.add_jump(jump)

    def _on_axis(self, coordinate: Coordinate) -> Coordinate:
        """Validate and re-express a coordinate on this registry's axis."""
        if self.unit is not None and coordinate.unit is not self.unit:
            raise ValueError(
                f"Flow-control coordinate unit {coordinate.unit} does not match "
                f"registry unit {self.unit}"
            )
        if self.number_type is not None:
            coordinate = express_scalar_as(coordinate, self.number_type)
        return coordinate

    def _coordinate(
        self, coordinate: int | float | Fraction | Coordinate | IdCoordinate
    ) -> Coordinate:
        """Coerce one public coordinate input for storage.

        An unbound registry stores a given coordinate object unchanged; a raw
        number requires a bound registry unit.
        """
        if not isinstance(coordinate, Coordinate):
            if self.unit is None:
                raise ValueError(
                    "A raw marker coordinate requires a bound registry unit"
                )
            coordinate = Coordinate(coordinate, self.unit)
        elif self.unit is None and self.number_type is None:
            return coordinate
        if isinstance(coordinate, IdCoordinate):
            coordinate = Coordinate(
                coordinate.value,
                coordinate.unit,
                number_type=coordinate.number_type,
            )
        return self._on_axis(coordinate)

    def _value(
        self,
        coordinate: int | float | Fraction | Coordinate | IdCoordinate,
    ) -> int | float | Fraction:
        """Return the numeric value of one accepted query coordinate.

        Raises:
            ValueError: If a coordinate object's unit differs from the unit of
                a bound registry.
        """
        if not isinstance(coordinate, Coordinate):
            return coordinate
        if self.unit is not None and coordinate.unit is not self.unit:
            raise ValueError(
                f"Query coordinate unit {coordinate.unit} does not match "
                f"registry unit {self.unit}"
            )
        return coordinate.value

    def _elements(self) -> list[Break | Jump | Marker]:
        """Return every stored element: breaks, then markers, then jumps."""
        return [*self.breaks, *self.markers, *self.jumps]

    def _admitted(self, element: Any) -> Any:
        """Return *element* under the id it may be stored with.

        Raises:
            ValueError: If the element's given id is already taken.
        """
        prefix = element._id_prefix
        description = f"{prefix} {element.id!r}"
        if self._admit is not None:
            new_id = self._admit(
                element.id, minted=element._minted, prefix=prefix, element=description
            )
        else:
            taken = {
                held.id: f"{held._id_prefix} {held.id!r}" for held in self._elements()
            }
            new_id = _admit_id(
                element.id,
                minted=element._minted,
                prefix=prefix,
                taken=taken,
                element=description,
            )
        if new_id == element.id:
            return element
        return replace(element, id=new_id, _minted=True)

    def _reassign_id(self, old_id: str, new_id: str) -> None:
        """Store the element with id *old_id* under the minted id *new_id*.

        A marker's jumps follow it to the new id.
        """
        for elements in (self.breaks, self.jumps, self.markers):
            for index, element in enumerate(elements):
                if element.id == old_id:
                    elements[index] = replace(element, id=new_id, _minted=True)
        self.jumps = [
            replace(jump, target=new_id) if jump.target == old_id else jump
            for jump in self.jumps
        ]

    def _absorb(self, other: FlowControlRegistry) -> None:
        """Add every element of *other*, jumps following re-minted markers."""
        for brk in other.breaks:
            self.add_break(brk)
        renamed = {marker.id: self.add_marker(marker).id for marker in other.markers}
        for jump in other.jumps:
            target = renamed.get(jump.target, jump.target)
            self.add_jump(
                jump if target == jump.target else replace(jump, target=target)
            )

    def add_break(self, brk: Break) -> Break:
        """Add a break to the registry.

        Args:
            brk: The Break to add.

        Returns:
            The stored break, with its coordinate on the registry's axis and
            its id as admitted.

        Raises:
            ValueError: If the break's given id is already taken.
        """
        stored = (
            brk
            if self.unit is None and self.number_type is None
            else replace(brk, coordinate=self._on_axis(brk.coordinate))
        )
        stored = self._admitted(stored)
        self.breaks = _ordered([*self.breaks, stored])
        return stored

    def add_jump(self, jump: Jump) -> Jump:
        """Add a jump to the registry.

        Args:
            jump: The Jump to add.

        Returns:
            The stored jump, with its coordinates on the registry's axis and
            its id as admitted.

        Raises:
            ValueError: If the jump's given id is already taken, or its
                ``target`` names no marker of this registry.
        """
        if jump.target is not None and jump.target not in {
            marker.id for marker in self.markers
        }:
            raise ValueError(
                f"Jump target {jump.target!r} names no marker of this registry. "
                f"Available markers: {[marker.id for marker in self.markers]}"
            )
        stored = (
            jump
            if self.unit is None and self.number_type is None
            else replace(
                jump,
                from_coordinate=self._on_axis(jump.from_coordinate),
                to_coordinate=self._on_axis(jump.to_coordinate),
            )
        )
        stored = self._admitted(stored)
        self.jumps = sorted([*self.jumps, stored], key=lambda j: j.from_position)
        return stored

    def add_marker(self, marker: Marker) -> Marker:
        """Add a marker (Segno, Coda, etc.).

        Args:
            marker: The Marker to add.

        Returns:
            The stored marker, with its coordinate on the registry's axis and
            its id as admitted.

        Raises:
            ValueError: If the marker's given id is already taken.
        """
        coordinate = self._coordinate(marker.coordinate)
        stored = (
            marker
            if coordinate is marker.coordinate
            else replace(marker, coordinate=coordinate)
        )
        stored = self._admitted(stored)
        self.markers = _ordered([*self.markers, stored])
        return stored

    def breaks_at(
        self, coordinate: int | float | Fraction | Coordinate | IdCoordinate
    ) -> list[Break]:
        """Get all breaks at a specific coordinate.

        Args:
            coordinate: The coordinate to query.

        Returns:
            List of breaks at that coordinate.
        """
        value = self._value(coordinate)
        return [b for b in self.breaks if b.position == value]

    def jumps_from(
        self, coordinate: int | float | Fraction | Coordinate | IdCoordinate
    ) -> list[Jump]:
        """Get all jumps originating from a specific coordinate.

        Args:
            coordinate: The coordinate to query.

        Returns:
            List of jumps from that coordinate.
        """
        value = self._value(coordinate)
        return [j for j in self.jumps if j.from_position == value]

    def jumps_to(
        self, coordinate: int | float | Fraction | Coordinate | IdCoordinate
    ) -> list[Jump]:
        """Get all jumps landing at a specific coordinate.

        Args:
            coordinate: The coordinate to query.

        Returns:
            List of jumps to that coordinate.
        """
        value = self._value(coordinate)
        return [j for j in self.jumps if j.to_position == value]

    def has_break_at(
        self, coordinate: int | float | Fraction | Coordinate | IdCoordinate
    ) -> bool:
        """Check if there's a break at the coordinate.

        Args:
            coordinate: The coordinate to check.

        Returns:
            True if a break exists at that coordinate.
        """
        value = self._value(coordinate)
        return any(b.position == value for b in self.breaks)

    def has_flow_control(self) -> bool:
        """Check if any flow control events exist.

        Returns:
            True if there are any breaks, jumps, or markers.
        """
        return bool(self.breaks or self.jumps or self.markers)

    @property
    def has_repeats(self) -> bool:
        """Check if there are any repeat jumps."""
        return any(j.control_type == FlowControlElement.repeat_end for j in self.jumps)

    @property
    def has_da_capo(self) -> bool:
        """Check if there's a Da Capo jump."""
        return any(j.control_type == FlowControlElement.da_capo for j in self.jumps)

    @property
    def has_dal_segno(self) -> bool:
        """Check if there's a Dal Segno jump."""
        return any(j.control_type == FlowControlElement.dal_segno for j in self.jumps)

    def clear(self) -> None:
        """Clear all flow control events."""
        self.breaks.clear()
        self.jumps.clear()
        self.markers.clear()

    def copy(self) -> FlowControlRegistry:
        """Return an independent registry with the same binding and content.

        The copy belongs to no timeline until one adopts it.
        """
        owner, self._admit = self._admit, None
        try:
            return deepcopy(self)
        finally:
            self._admit = owner

    def to_dict(
        self, *, minimal: bool = False, taken: Collection[str] = ()
    ) -> list[dict[str, Any]]:
        """Return all control events as the ``flow_control`` array.

        Breaks are written as ``flow_marker``, markers as ``jump_to`` and
        jumps as ``jump_from`` elements, in canonical order whatever order
        they were added in: by coordinate, then by id, values compared
        exactly and ids by Unicode code point. A jump names its target; a
        jump without one names the marker at its destination, or, where
        none lies, a ``jump_to`` written there whose manifestation follows
        from the jump (a repeat end lands on a repeat start, a D.S. on a
        segno, a To Coda and a D.C. al Coda on a coda), with an id minted
        around *taken* and the registry's own ids; a repeat end or a D.C.
        landing at 0 names none. The registry's own lists are not changed.

        Args:
            minimal: Leave out every key at its default.
            taken: Ids held beside the registry's own (its timeline's
                manifest), which a minted ``jump_to`` avoids.

        Raises:
            ValueError: If a jump's target lies elsewhere than the jump
                lands, or a D.C. or D.C. al Fine lands elsewhere than 0.
        """
        occupied = {*taken, *(element.id for element in self._elements())}
        landings: dict[tuple[Any, FlowControlElement], dict[str, Any]] = {}
        elements: list[tuple[Any, str, dict[str, Any]]] = [
            (element.position, element.id, element.to_dict(minimal=minimal))
            for element in (*self.breaks, *self.markers)
        ]
        for jump in self.jumps:
            target = self._target_of(jump, landings, occupied, minimal=minimal)
            if target != jump.target:
                jump = replace(jump, target=target)
            elements.append(
                (jump.from_position, jump.id, jump.to_dict(minimal=minimal))
            )
        elements.extend(
            (landing_key[0], landing["id"], landing)
            for landing_key, landing in landings.items()
        )
        return [written for _, _, written in sorted(elements, key=lambda e: e[:2])]

    def _target_of(
        self,
        jump: Jump,
        landings: dict[tuple[Any, FlowControlElement], dict[str, Any]],
        occupied: set[str],
        *,
        minimal: bool,
    ) -> str | None:
        """Return the id of the ``jump_to`` *jump* lands on, writing one if needed.

        Raises:
            ValueError: As :meth:`to_dict`.
        """
        if jump.target is not None:
            marker = next(m for m in self.markers if m.id == jump.target)
            if marker.position != jump.to_position:
                raise ValueError(
                    f"jump {jump.id!r} lands at {jump.to_position}, but its target "
                    f"{marker.id!r} lies at {marker.position}"
                )
            return jump.target
        if jump.to_position == 0 and jump.control_type in _TO_START:
            return None
        landing = _LANDING.get(jump.control_type)
        if landing is None:
            raise ValueError(
                f"jump {jump.id!r} ({jump.control_type.value}) lands at "
                f"{jump.to_position}, not at the timeline's start"
            )
        here = sorted(
            (m for m in self.markers if m.position == jump.to_position),
            key=lambda m: (m.control_type is not landing, m.id),
        )
        if here:
            return here[0].id
        key = (jump.to_position, landing)
        if key not in landings:
            marker_id = mint_id(Marker._id_prefix, occupied)
            occupied.add(marker_id)
            landings[key] = Marker(
                jump.to_coordinate, control_type=landing, id=marker_id
            ).to_dict(minimal=minimal)
        return landings[key]["id"]

    def __repr__(self) -> str:
        return (
            f"FlowControlRegistry({len(self.breaks)} breaks, "
            f"{len(self.jumps)} jumps, {len(self.markers)} markers)"
        )


# endregion
