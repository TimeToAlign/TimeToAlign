"""Flow Control Events: Break and Jump for timeline traversal.

This module provides the Break and Jump classes that control the flow of
time through a timeline.

**Break**: A break is a control event that voids the contiguity at the
Instant where it is located.
- Prevents TimeIntervals from spanning its coordinate
- Cannot be inserted at coordinates already spanned by a TimeInterval
  (unless the spanning element "adopts" the break)

**Jump**: A jump is a control event defined by two Instants: a JumpFrom
and a JumpTo Instant. When active, it makes any event located or starting
at JumpTo contiguous with any event ending at JumpFrom.
- Creates contiguity between non-adjacent coordinates
- Has activation conditions (e.g., "on Nth pass")

Common music notation mappings:
| Notation        | Break?  | Jump?  | Notes                           |
|-----------------|---------|--------|----------------------------------|
| Repeat End (:∥) | Maybe   | Yes    | Jump back; may be break if at section end |
| Repeat Start    | No      | -      | Structural marker, target for repeat_end |
| First Ending    | -       | -      | Volta info is measure attribute, not break |
| Fine            | Yes     | No     | End of piece (conditional)      |
| Da Capo         | No      | Yes    | Jump to coordinate 0            |
| Dal Segno       | No      | Yes    | Jump to a Segno marker by id    |
| Section Break   | Yes     | No     | Single instant voiding contiguity |
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any, ClassVar

from timetoalign.core import Coordinate, IdCoordinate, NumberType, TimeUnit
from timetoalign.core.enums import ActivationCondition, FlowControlElement
from timetoalign.core.ids import _admit_id, _IdentifiedElement, _OpenObject
from timetoalign.core.time import express_scalar_as

from .manifest import (
    axis_value_from_json,
    axis_value_to_json,
    read_id,
    read_list,
    read_meta,
    read_name,
    require_object,
)

module_logger = logging.getLogger(__name__)

# region Enums


# endregion

# region Break


@dataclass(frozen=True)
class Break(_IdentifiedElement):
    """A control event that voids contiguity at a specific coordinate.

    A break voids contiguity at the Instant where it is located.

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
        control_type: The type of break (section_break, fine).
        condition: When the break becomes active.
        repeat_count: For FINE, how many DC/DS must occur before active.
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

        >>> # Fine marker (conditional break after DC/DS)
        >>> fine = Break(
        ...     coordinate=Coordinate(200.0, TimeUnit.quarters),
        ...     control_type=FlowControlElement.fine,
        ...     condition=ActivationCondition.after_dc_ds,
        ...     name="fine",  # Default name, can be customized
        ... )
    """

    _id_prefix: ClassVar[str] = "break"
    _dict_keys: ClassVar[tuple[str, ...]] = (
        "id",
        "coordinate",
        "control_type",
        "condition",
        "repeat_count",
        "label",
        "name",
        "meta",
    )
    _dict_defaults: ClassVar[Mapping[str, Any]] = {
        "control_type": FlowControlElement.section_break.value,
        "condition": ActivationCondition.always.value,
        "repeat_count": 1,
        "label": None,
        "name": None,
        "meta": {},
    }

    coordinate: Coordinate
    control_type: FlowControlElement = FlowControlElement.section_break
    condition: ActivationCondition = ActivationCondition.always
    repeat_count: int = 1
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
        """Restore a break from its object, ``coordinate`` a bare axis value.

        Every key but ``id`` and ``coordinate`` may be left out and reads as
        its default; keys the library does not know are kept and written
        back. *unit* and *number_type* describe the timeline's axis.

        Raises:
            ValueError: If the object is malformed or the coordinate is
                represented otherwise than the axis selects.
        """
        require_object(data, "a break", ("id", "coordinate"))
        break_id = read_id(data["id"], "a break")
        what = f"break {break_id!r}"
        brk = cls(
            id=break_id,
            coordinate=axis_value_from_json(
                data["coordinate"],
                key="coordinate",
                owner=break_id,
                unit=unit,
                number_type=number_type,
            ),
            control_type=_read_enum(
                data, "control_type", FlowControlElement, cls, what
            ),
            condition=_read_enum(data, "condition", ActivationCondition, cls, what),
            repeat_count=_read_repeat_count(data, what),
            label=_read_label(data, what),
            name=read_name(data, what),
            meta=read_meta(data, what),
        )
        brk._keep_unknown_keys(data)
        return brk

    def __post_init__(self) -> None:
        """Mint or validate the id, then validate the break configuration."""
        super().__post_init__()
        # Validate that control_type is actually a break type
        if not self.control_type.is_break:
            module_logger.warning(
                f"Break created with non-break type: {self.control_type}"
            )

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
        if self.condition == ActivationCondition.always:
            return True
        elif self.condition == ActivationCondition.first_n:
            return pass_number <= self.repeat_count
        elif self.condition == ActivationCondition.after_first:
            return pass_number > 1
        elif self.condition == ActivationCondition.after_dc_ds:
            return after_dc_ds
        return False

    def to_dict(self, *, minimal: bool = False) -> dict[str, Any]:
        """Return this break as its JSON object, the coordinate a bare value.

        Args:
            minimal: Leave out every key at its default.
        """
        return self._dict_form(
            {
                "id": self.id,
                "coordinate": axis_value_to_json(self.coordinate),
                "control_type": self.control_type.value,
                "condition": self.condition.value,
                "repeat_count": self.repeat_count,
                "label": self.label,
                "name": self.name,
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
    any event ending at JumpFrom.

    Target Resolution:
        A jump to a marker references it by the marker's ID (``target``),
        so several markers of one type ("segno", "segno2") coexist in a
        score. ``target`` is ``None`` for a jump without a marker target
        (a plain repeat, a jump to the start); ``to_coordinate`` always
        carries the destination.

    Attributes:
        from_coordinate: The coordinate where the jump originates.
        to_coordinate: The coordinate where the jump lands.
        control_type: The type of jump (repeat_end, da_capo, dal_segno, to_coda).
        condition: When the jump becomes active.
        repeat_count: For repeat_end, how many times to take the jump.
        target: The id of the target marker of the same timeline, or None.
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

    _id_prefix: ClassVar[str] = "jump"
    _dict_keys: ClassVar[tuple[str, ...]] = (
        "id",
        "from_coordinate",
        "to_coordinate",
        "control_type",
        "condition",
        "repeat_count",
        "target",
        "label",
        "meta",
    )
    _dict_defaults: ClassVar[Mapping[str, Any]] = {
        "control_type": FlowControlElement.repeat_end.value,
        "condition": ActivationCondition.first_n.value,
        "repeat_count": 1,
        "target": None,
        "label": None,
        "meta": {},
    }

    from_coordinate: Coordinate
    to_coordinate: Coordinate
    control_type: FlowControlElement = FlowControlElement.repeat_end
    condition: ActivationCondition = ActivationCondition.first_n
    repeat_count: int = 1
    target: str | None = None
    label: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        unit: TimeUnit,
        number_type: NumberType,
    ) -> Jump:
        """Restore a jump from its object, its coordinates bare axis values.

        Every key but ``id``, ``from_coordinate`` and ``to_coordinate`` may
        be left out and reads as its default; keys the library does not know
        are kept and written back. *unit* and *number_type* describe the
        timeline's axis.

        Raises:
            ValueError: If the object is malformed or a coordinate is
                represented otherwise than the axis selects.
        """
        require_object(data, "a jump", ("id", "from_coordinate", "to_coordinate"))
        jump_id = read_id(data["id"], "a jump")
        what = f"jump {jump_id!r}"
        coordinates = {
            key: axis_value_from_json(
                data[key], key=key, owner=jump_id, unit=unit, number_type=number_type
            )
            for key in ("from_coordinate", "to_coordinate")
        }
        target = data.get("target")
        if target is not None and not isinstance(target, str):
            raise ValueError(f"{what} has a non-string target {target!r}")
        jump = cls(
            id=jump_id,
            control_type=_read_enum(
                data, "control_type", FlowControlElement, cls, what
            ),
            condition=_read_enum(data, "condition", ActivationCondition, cls, what),
            repeat_count=_read_repeat_count(data, what),
            target=target,
            label=_read_label(data, what),
            meta=read_meta(data, what),
            **coordinates,
        )
        jump._keep_unknown_keys(data)
        return jump

    def __post_init__(self) -> None:
        """Mint or validate the id, then validate the jump configuration."""
        super().__post_init__()
        if self.from_coordinate.unit != self.to_coordinate.unit:
            raise ValueError(
                f"Jump coordinates must have same unit: "
                f"{self.from_coordinate.unit} != {self.to_coordinate.unit}"
            )

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
        if self.condition == ActivationCondition.always:
            return True
        elif self.condition == ActivationCondition.first_n:
            return pass_number <= self.repeat_count
        elif self.condition == ActivationCondition.after_first:
            return pass_number > 1
        elif self.condition == ActivationCondition.after_dc_ds:
            return after_dc_ds
        return False

    def remaining_activations(self, pass_number: int) -> int:
        """How many more times this jump can be taken.

        Args:
            pass_number: The current pass number (1-indexed).

        Returns:
            Number of remaining activations (0 if no longer active).
        """
        if self.condition == ActivationCondition.first_n:
            return max(0, self.repeat_count - pass_number + 1)
        elif self.condition == ActivationCondition.always:
            return float("inf")  # type: ignore
        return 0

    def to_dict(self, *, minimal: bool = False) -> dict[str, Any]:
        """Return this jump as its JSON object, the coordinates bare values.

        Args:
            minimal: Leave out every key at its default.
        """
        return self._dict_form(
            {
                "id": self.id,
                "from_coordinate": axis_value_to_json(self.from_coordinate),
                "to_coordinate": axis_value_to_json(self.to_coordinate),
                "control_type": self.control_type.value,
                "condition": self.condition.value,
                "repeat_count": self.repeat_count,
                "target": self.target,
                "label": self.label,
                "meta": deepcopy(self.meta),
            },
            minimal=minimal,
        )

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
    """A named target position of a timeline (Segno, Coda, ...).

    Jumps reference a marker by its ``id``; ``name`` is the display name
    and the source's marker instance name (``"segno"``, ``"codab"``).

    Attributes:
        coordinate: The marker's position.
        name: Optional display name.
        id: The marker's id; minted as ``marker_{N}`` when not given.

    Examples:
        >>> from timetoalign.core import Coordinate, TimeUnit
        >>> Marker(Coordinate(16, TimeUnit.quarters), name="segno").id
        'marker_1'
    """

    _id_prefix: ClassVar[str] = "marker"
    _dict_keys: ClassVar[tuple[str, ...]] = ("id", "name", "coordinate")
    _dict_defaults: ClassVar[Mapping[str, Any]] = {"name": None}

    coordinate: Coordinate
    name: str | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        unit: TimeUnit,
        number_type: NumberType,
    ) -> Marker:
        """Restore a marker from its object, ``coordinate`` a bare axis value.

        ``name`` may be left out; keys the library does not know are kept
        and written back. *unit* and *number_type* describe the timeline's
        axis.

        Raises:
            ValueError: If the object is malformed or the coordinate is
                represented otherwise than the axis selects.
        """
        require_object(data, "a marker", ("id", "coordinate"))
        marker_id = read_id(data["id"], "a marker")
        marker = cls(
            id=marker_id,
            name=read_name(data, f"marker {marker_id!r}"),
            coordinate=axis_value_from_json(
                data["coordinate"],
                key="coordinate",
                owner=marker_id,
                unit=unit,
                number_type=number_type,
            ),
        )
        marker._keep_unknown_keys(data)
        return marker

    @property
    def position(self) -> float | Fraction:
        """The coordinate value of the marker."""
        return self.coordinate.value

    def to_dict(self, *, minimal: bool = False) -> dict[str, Any]:
        """Return this marker as its JSON object, the coordinate a bare value.

        Args:
            minimal: Leave out ``name`` at its default.
        """
        return self._dict_form(
            {
                "id": self.id,
                "name": self.name,
                "coordinate": axis_value_to_json(self.coordinate),
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


def _read_enum(
    data: Mapping[str, Any], key: str, enum: type[Any], element: type[Any], what: str
) -> Any:
    """Read an optional enum member by value, its default when absent.

    Raises:
        ValueError: If the value names no member.
    """
    value = data.get(key, element._dict_defaults[key])
    try:
        return enum(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{what} has an unknown {key} {value!r}") from error


def _read_repeat_count(data: Mapping[str, Any], what: str) -> int:
    """Read an optional ``repeat_count``, 1 when absent.

    Raises:
        ValueError: If the value is not an integer.
    """
    value = data.get("repeat_count", 1)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} has a non-integer repeat_count {value!r}")
    return value


def _read_label(data: Mapping[str, Any], what: str) -> str | None:
    """Read an optional ``label``, null when absent.

    Raises:
        ValueError: If the label is neither a string nor null.
    """
    label = data.get("label")
    if label is not None and not isinstance(label, str):
        raise ValueError(f"{what} has a non-string label {label!r}")
    return label


def _ordered(elements: list[Any]) -> list[Any]:
    """Sort flow-control elements by position, keeping insertion order on ties."""
    return sorted(elements, key=lambda element: element.position)


@dataclass
class FlowControlRegistry(_OpenObject):
    """Registry of all flow control events for a timeline.

    This class manages the collection of Breaks, Jumps and Markers for a
    timeline, providing methods to query, add, and validate them. Every
    element carries an id unique within the registry and, once the registry
    belongs to a timeline, within that timeline's whole manifest: an element
    whose minted id is taken is stored under a freshly minted id, and one
    whose given id is taken is refused.

    Attributes:
        breaks: List of Break events, ordered by coordinate.
        jumps: List of Jump events, ordered by from coordinate.
        markers: List of Marker targets, ordered by coordinate.

    Examples:
        >>> from timetoalign.core import Coordinate, TimeUnit
        >>> registry = FlowControlRegistry(unit=TimeUnit.quarters)
        >>> registry.add_break(Break(Coordinate(64, TimeUnit.quarters)))
        Break(section_break @ 64)
        >>> registry.add_marker(Marker(Coordinate(16, TimeUnit.quarters), name="segno"))
        Marker('marker_1', name='segno' @ 16)
        >>> registry.add_marker(Marker(Coordinate(32, TimeUnit.quarters), name="coda")).id
        'marker_2'
    """

    _dict_keys: ClassVar[tuple[str, ...]] = ("breaks", "jumps", "markers")
    _dict_defaults: ClassVar[Mapping[str, Any]] = {
        "breaks": [],
        "jumps": [],
        "markers": [],
    }

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
        data: dict[str, Any],
        *,
        unit: TimeUnit | str,
        number_type: NumberType | str | None = None,
    ) -> FlowControlRegistry:
        """Restore a registry bound to a timeline's axis from its object.

        The coordinates are bare values on the axis that *unit* and
        *number_type* (the unit's own when None) describe; ``breaks``,
        ``jumps`` and ``markers`` may each be left out. Every element is
        added through the public add methods, markers before jumps. Keys the
        library does not know are kept and written back.

        Args:
            data: The ``flow_control`` object.
            unit: The axis's unit.
            number_type: The axis's number type, or None for the unit's own.

        Raises:
            ValueError: If the object is malformed or a coordinate is
                represented otherwise than the axis selects.
        """
        resolved_unit = TimeUnit(unit)
        resolved_number_type = (
            resolved_unit.default_number_type
            if number_type is None
            else NumberType(number_type)
        )
        require_object(data, "a flow_control")
        axis = {"unit": resolved_unit, "number_type": resolved_number_type}
        registry = cls(unit=resolved_unit, number_type=resolved_number_type)
        for brk in read_list(data, "breaks", "the flow_control"):
            registry.add_break(Break.from_dict(brk, **axis))
        for marker in read_list(data, "markers", "the flow_control"):
            registry.add_marker(Marker.from_dict(marker, **axis))
        for jump in read_list(data, "jumps", "the flow_control"):
            registry.add_jump(Jump.from_dict(jump, **axis))
        registry._keep_unknown_keys(data)
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

    def to_dict(self, *, minimal: bool = False) -> dict[str, Any]:
        """Return all control events as the ``flow_control`` object.

        The entries are written in canonical order, whatever order they were
        added in: ``breaks`` by coordinate, ``jumps`` by from coordinate and
        then to coordinate, ``markers`` by coordinate, each then by id,
        values compared exactly and ids by Unicode code point. The
        registry's own lists are not reordered.

        Args:
            minimal: Leave out every key at its default, in the elements too.
        """
        return self._dict_form(
            {
                "breaks": [
                    brk.to_dict(minimal=minimal)
                    for brk in sorted(
                        self.breaks, key=lambda brk: (brk.position, brk.id)
                    )
                ],
                "jumps": [
                    jump.to_dict(minimal=minimal)
                    for jump in sorted(
                        self.jumps,
                        key=lambda jump: (
                            jump.from_position,
                            jump.to_position,
                            jump.id,
                        ),
                    )
                ],
                "markers": [
                    marker.to_dict(minimal=minimal)
                    for marker in sorted(
                        self.markers, key=lambda marker: (marker.position, marker.id)
                    )
                ],
            },
            minimal=minimal,
        )

    def __repr__(self) -> str:
        return (
            f"FlowControlRegistry({len(self.breaks)} breaks, "
            f"{len(self.jumps)} jumps, {len(self.markers)} markers)"
        )


# endregion
