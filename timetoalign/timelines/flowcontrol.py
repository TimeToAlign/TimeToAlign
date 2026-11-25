"""Flow control: the elements that say how a timeline is played through.

A timeline holds its flow control as one list of elements, written in a
TimeToShare manifest as the timeline's ``flow_control`` array. Each element
is one object of one of five classes, all deriving from
:class:`FlowControlElement`; the manifest element is the object's dictionary
form, its ``role`` naming the class and every other key an attribute:

| Class | ``role`` | Own keys | When the flow plays up to it |
|---|---|---|---|
| :class:`Jump` | ``jump`` | ``targets`` | a destination that holds is taken |
| :class:`JumpTarget` | ``jump_target`` | ``manifestation`` | nothing: a place a destination names |
| :class:`Break` | ``break`` | ``manifestation``, ``ends`` | the flow never goes on across it |
| :class:`SectionBreak` | ``section_break`` | ``manifestation`` | the flow goes on; a section ends |
| :class:`Barline` | ``barline`` | ``manifestation`` | nothing |

A :class:`Jump` lies where the flow may leave the axis, and holds one
:class:`JumpDestination` for every place it may go on at: a repeat end, a
D.C., a D.S., a To Coda and the brackets of alternative endings are all
destinations of the Jump at their coordinate. A destination names its place
(``to`` an element by ID, ``by`` a distance, or neither for the start of the
section) and the conditions under which it is taken (``passes``,
``repeat_count``, ``after``, ``determined``). A :class:`Break` ends the flow
unless a Jump at its coordinate is taken; with ``ends`` it does so only
while that condition holds (a Fine).

What an element looks like in the source is its ``manifestation``
(:class:`~timetoalign.core.enums.FlowManifestation`), which is optional and
decides nothing: the flow follows from places and conditions alone
(:mod:`timetoalign.timelines.flow.engine`).

Common music notation:

| Notation | Elements |
|---|---|
| Repeat start | a :class:`JumpTarget` |
| Repeat end | a destination ``{to: <repeat start>, repeat_count: 1}`` |
| D.C., D.S. | a destination ``{repeat_count: 1, reset: false}``, a D.S. ``to`` its segno |
| To Coda | a destination ``{to: <coda>, after: {jump: <the leap's Jump>}}`` |
| Fine | a :class:`Break` with ``ends: {after: {jump: <the leap's Jump>}}`` |
| Brackets 1, 2 | a Jump before them: ``{by: 0, passes: [1]}``, ``{to: <bracket 2>, passes: [2]}``; a Break between |
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any, ClassVar, Self

from timetoshare import DEFAULT_BELIEFS, BeliefState

from timetoalign.core import Coordinate, Duration, IdCoordinate, NumberType, TimeUnit
from timetoalign.core.enums import FlowManifestation
from timetoalign.core.ids import (
    _IdentifiedElement,
    _OpenObject,
    _plan_admissions,
    quote_id,
)
from timetoalign.core.provenance import Provenance
from timetoalign.core.time import express_scalar_as, number_to_json

from .manifest import (
    axis_value_from_json,
    axis_value_to_json,
    read_id,
    read_meta,
    read_name,
    require_object,
)

module_logger = logging.getLogger(__name__)

_Number = int | float | Fraction

# region Manifestations

_KNOWN_MANIFESTATIONS: dict[str, FlowManifestation] = {
    member.value: member for member in FlowManifestation
}

_DESTINATION_MANIFESTATIONS = frozenset(
    {
        FlowManifestation.repeat_end,
        FlowManifestation.da_capo,
        FlowManifestation.da_capo_al_fine,
        FlowManifestation.da_capo_al_coda,
        FlowManifestation.dal_segno,
        FlowManifestation.dal_segno_al_fine,
        FlowManifestation.dal_segno_al_coda,
        FlowManifestation.to_coda,
        FlowManifestation.volta,
        FlowManifestation.ossia,
    }
)
"""What a :class:`JumpDestination` may look like in the source."""


def _manifestation(
    value: object, admitted: frozenset[FlowManifestation], what: str
) -> FlowManifestation | str | None:
    """Return a manifestation as an object holds it.

    A word of the vocabulary is held as its
    :class:`~timetoalign.core.enums.FlowManifestation` and must be one
    *what* admits; any other string is kept as written.

    Raises:
        ValueError: If *value* is no string, or a word of the vocabulary
            that *what* does not admit.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{what} has a non-string manifestation {value!r}")
    member = _KNOWN_MANIFESTATIONS.get(str(value))
    if member is None:
        return value
    if member not in admitted:
        raise ValueError(
            f"{what} is one of {sorted(m.value for m in admitted)} or a word "
            f"outside the vocabulary, not {member.value!r}"
        )
    return member


def _written(manifestation: FlowManifestation | str) -> str:
    """A manifestation as a manifest writes it."""
    return (
        manifestation.value
        if isinstance(manifestation, FlowManifestation)
        else manifestation
    )


# endregion

# region Conditions


@dataclass(frozen=True)
class PassesFrom:
    """The passes from the *first* on: ``{from: n}`` in a manifest.

    Attributes:
        first: The first pass that counts, at least 1.
    """

    first: int

    def __post_init__(self) -> None:
        if isinstance(self.first, bool) or not isinstance(self.first, int):
            raise ValueError(f"passes from {self.first!r}: a pass is an integer")
        if self.first < 1:
            raise ValueError(f"passes from {self.first}: passes are counted from 1")


def _bound_distance(by: Any, coordinate: Coordinate, what: str) -> Duration:
    """Return the distance *by* on the axis of *coordinate*.

    A bare number is read on that axis; a :class:`Duration` must have its
    unit and is rewritten in its number type.

    Raises:
        ValueError: If a duration's unit is not the axis's.
    """
    if not isinstance(by, Duration):
        return Duration(by, coordinate.unit, number_type=coordinate.number_type)
    if by.unit is not coordinate.unit:
        raise ValueError(
            f"{what} is a distance in {by.unit} on an axis in {coordinate.unit}"
        )
    return express_scalar_as(by, coordinate.number_type)


def _read_distance(
    data: Any, *, owner: str, unit: TimeUnit, number_type: NumberType
) -> Duration:
    """Read ``by``, a bare signed value on the timeline's own axis."""
    read = axis_value_from_json(
        data, key="by", owner=owner, unit=unit, number_type=number_type
    )
    return Duration(read.value, unit, number_type=number_type)


def _distance_to_json(by: Any) -> int | float | list[int]:
    return number_to_json(by.value if isinstance(by, Duration) else by)


def _refuse_two_places(to: object, by: object, what: str) -> None:
    """Refuse a place stated twice.

    Raises:
        ValueError: If both *to* and *by* are given, or *to* is no string.
    """
    if to is not None and by is not None:
        raise ValueError(
            f"{what} states both `to` ({to!r}) and `by`: a place is one of them"
        )
    if to is not None and not isinstance(to, str):
        raise ValueError(f"{what} names the place {to!r}, which is no ID")


@dataclass(frozen=True, kw_only=True)
class DestinationReference(_OpenObject):
    """One destination of a Jump, named by that Jump and its place.

    The value of ``after``. The place keys are left out for the destination
    without place, or when the Jump has one destination only.

    Attributes:
        jump: The ID of the Jump.
        to: The ``to`` of the destination meant, if it has one.
        by: The ``by`` of the destination meant, if it has one.
    """

    _dict_keys: ClassVar[tuple[str, ...]] = ("jump", "to", "by")

    jump: str
    to: str | None = None
    by: Duration | _Number | None = None
    _unknown_keys: Mapping[str, Any] = field(
        default_factory=dict[str, Any], repr=False, compare=False
    )

    @classmethod
    def from_dict(
        cls, data: Any, what: str, *, unit: TimeUnit, number_type: NumberType
    ) -> DestinationReference:
        """Read an ``after`` object.

        Raises:
            ValueError: If it is malformed.
        """
        require_object(data, f"the `after` of {what}", ("jump",))
        reference = cls(
            jump=read_id(data["jump"], f"the `after` of {what}: a jump"),
            to=data.get("to"),
            by=(
                _read_distance(
                    data["by"], owner=what, unit=unit, number_type=number_type
                )
                if "by" in data
                else None
            ),
        )
        reference._keep_unknown_keys(data)
        return reference

    def __post_init__(self) -> None:
        if not isinstance(self.jump, str):
            raise ValueError(f"`after` names the jump {self.jump!r}, which is no ID")
        _refuse_two_places(self.to, self.by, f"`after` naming jump {self.jump!r}")

    @property
    def place(self) -> tuple[str, Any] | None:
        """The place named: ``("to", ID)``, ``("by", value)`` or None."""
        return _place_key(self.to, self.by)

    def to_dict(self) -> dict[str, Any]:
        """Return the reference as its JSON object, absent keys left out."""
        written: dict[str, Any] = {"jump": self.jump}
        if self.to is not None:
            written["to"] = self.to
        if self.by is not None:
            written["by"] = _distance_to_json(self.by)
        written.update(deepcopy(dict(self._unknown_keys)))
        return written

    def _bound_to(self, coordinate: Coordinate, what: str) -> DestinationReference:
        if self.by is None:
            return self
        by = _bound_distance(self.by, coordinate, f"the `after` of {what}")
        return self if by is self.by else replace(self, by=by)

    def _renamed(self, renamed: Mapping[str, str]) -> DestinationReference:
        jump = renamed.get(self.jump, self.jump)
        to = None if self.to is None else renamed.get(self.to, self.to)
        if jump == self.jump and to == self.to:
            return self
        return replace(self, jump=jump, to=to)


def _place_key(to: str | None, by: Any) -> tuple[str, Any] | None:
    """A place as destinations of one Jump are told apart."""
    if to is not None:
        return ("to", to)
    if by is not None:
        return ("by", by.value if isinstance(by, Duration) else by)
    return None


@dataclass(frozen=True, kw_only=True)
class FlowCondition(_OpenObject):
    """When a destination is taken, or a Break ends the flow.

    The value of a Break's ``ends``, and the part a :class:`JumpDestination`
    shares with it. Every condition stated must hold.

    Attributes:
        passes: The arrivals of the flow at the element that count: distinct
            positive integers (the n-th arrival is pass n), or
            :class:`PassesFrom`. Beyond the highest number the element
            names in a list, the list naming it holds.
        after: A destination of another Jump: the condition holds once that
            destination has been taken, and is used up when it has served.
    """

    _dict_keys: ClassVar[tuple[str, ...]] = ("passes", "after")

    passes: tuple[int, ...] | PassesFrom | None = None
    after: DestinationReference | None = None
    _unknown_keys: Mapping[str, Any] = field(
        default_factory=dict[str, Any], repr=False, compare=False
    )

    @classmethod
    def from_dict(
        cls, data: Any, what: str, *, unit: TimeUnit, number_type: NumberType
    ) -> Self:
        """Read a condition object.

        Raises:
            ValueError: If it is malformed.
        """
        require_object(data, what)
        try:
            condition = cls(**cls._read(data, what, unit, number_type))
        except ValueError as error:
            if what in str(error):
                raise
            raise ValueError(f"{what}: {error}") from error
        condition._keep_unknown_keys(data)
        return condition

    @classmethod
    def _read(
        cls,
        data: Mapping[str, Any],
        what: str,
        unit: TimeUnit,
        number_type: NumberType,
    ) -> dict[str, Any]:
        """Read the constructor arguments the object *data* states."""
        read: dict[str, Any] = {}
        if "passes" in data:
            passes = data["passes"]
            if isinstance(passes, dict) and set(passes) == {"from"}:
                read["passes"] = PassesFrom(passes["from"])
            elif isinstance(passes, list):
                read["passes"] = tuple(passes)
            else:
                raise ValueError(
                    f"passes {passes!r} is neither a list of pass numbers nor "
                    "{from: n}"
                )
        if "after" in data:
            read["after"] = DestinationReference.from_dict(
                data["after"], what, unit=unit, number_type=number_type
            )
        return read

    def __post_init__(self) -> None:
        passes = self.passes
        if passes is not None and not isinstance(passes, PassesFrom):
            passes = tuple(passes)
            object.__setattr__(self, "passes", passes)
            if not passes:
                raise ValueError("passes lists at least one pass")
            for number in passes:
                if isinstance(number, bool) or not isinstance(number, int):
                    raise ValueError(f"passes {list(passes)}: a pass is an integer")
                if number < 1:
                    raise ValueError(
                        f"passes {list(passes)}: passes are counted from 1"
                    )
            if len(set(passes)) != len(passes):
                raise ValueError(f"passes {list(passes)} names a pass twice")
        if self.after is not None and not isinstance(self.after, DestinationReference):
            raise ValueError(
                f"`after` is a DestinationReference, not {type(self.after).__name__}"
            )

    @property
    def is_stated(self) -> bool:
        """Whether any condition is stated."""
        return self.passes is not None or self.after is not None

    def to_dict(self) -> dict[str, Any]:
        """Return the condition as its JSON object, absent keys left out."""
        written = self._fields()
        written.update(deepcopy(dict(self._unknown_keys)))
        return written

    def _fields(self) -> dict[str, Any]:
        written: dict[str, Any] = {}
        if isinstance(self.passes, PassesFrom):
            written["passes"] = {"from": self.passes.first}
        elif self.passes is not None:
            written["passes"] = list(self.passes)
        if self.after is not None:
            written["after"] = self.after.to_dict()
        return written

    def _bound_to(self, coordinate: Coordinate, what: str) -> Self:
        """Return the condition with its distances on the axis of *coordinate*."""
        if self.after is None:
            return self
        after = self.after._bound_to(coordinate, what)
        return self if after is self.after else replace(self, after=after)

    def _renamed(self, renamed: Mapping[str, str]) -> Self:
        """Return the condition naming the new ID of every element renamed."""
        if self.after is None:
            return self
        after = self.after._renamed(renamed)
        return self if after is self.after else replace(self, after=after)


@dataclass(frozen=True, kw_only=True)
class JumpDestination(FlowCondition):
    """One place a Jump may go on at, and when it does.

    The place is ``to``, ``by`` or neither. The conditions are ``passes``,
    ``repeat_count``, ``after`` and ``determined``; a destination stating
    none is taken at every arrival.

    Attributes:
        to: The ID of the place: a :class:`JumpTarget` of the timeline, a
            Region of it, a child timeline of it (its start) or the timeline
            itself (its start).
        by: A signed distance from the Jump on its axis. ``by: 0`` is going
            on: it is written where the source prints it (the first
            bracket's numbers) and is never taken.
        passes: See :class:`FlowCondition`.
        repeat_count: The destination is taken at most that many times.
        after: See :class:`FlowCondition`.
        determined: False where nobody can check the choice (an ossia): the
            destination is never taken.
        reset: Whether taking a destination lying before its Jump restores
            every Jump and Break lying wholly inside the stretch gone back
            over, as a repeat does. False on a D.C. or D.S. played without
            its repeats.
        manifestation: What the destination looks like in the source.
        label: A human-readable label.

    Examples:
        >>> JumpDestination(to="segno", repeat_count=1, reset=False).to_dict()
        {'to': 'segno', 'repeat_count': 1, 'reset': False}
        >>> JumpDestination(by=0, passes=[1, 3], manifestation="volta").passes
        (1, 3)
    """

    _dict_keys: ClassVar[tuple[str, ...]] = (
        "to",
        "by",
        "passes",
        "repeat_count",
        "after",
        "determined",
        "reset",
        "manifestation",
        "label",
    )

    to: str | None = None
    by: Duration | _Number | None = None
    repeat_count: int | None = None
    determined: bool = True
    reset: bool = True
    manifestation: FlowManifestation | str | None = None
    label: str | None = None

    @classmethod
    def _read(
        cls,
        data: Mapping[str, Any],
        what: str,
        unit: TimeUnit,
        number_type: NumberType,
    ) -> dict[str, Any]:
        read = super()._read(data, what, unit, number_type)
        if "by" in data:
            read["by"] = _read_distance(
                data["by"], owner=what, unit=unit, number_type=number_type
            )
        for key in ("to", "repeat_count", "manifestation", "label"):
            if key in data:
                read[key] = data[key]
        for key in ("determined", "reset"):
            if key in data:
                if not isinstance(data[key], bool):
                    raise ValueError(f"{key} {data[key]!r} is neither true nor false")
                read[key] = data[key]
        return read

    def __post_init__(self) -> None:
        super().__post_init__()
        _refuse_two_places(self.to, self.by, "a destination")
        count = self.repeat_count
        if count is not None:
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f"repeat_count {count!r} is no integer of at least 1")
        if self.label is not None and not isinstance(self.label, str):
            raise ValueError(f"a destination has a non-string label {self.label!r}")
        object.__setattr__(
            self,
            "manifestation",
            _manifestation(
                self.manifestation, _DESTINATION_MANIFESTATIONS, "a destination"
            ),
        )

    @property
    def place(self) -> tuple[str, Any] | None:
        """The place stated: ``("to", ID)``, ``("by", value)`` or None."""
        return _place_key(self.to, self.by)

    @property
    def goes_on(self) -> bool:
        """Whether the destination is ``by: 0``: going on, never taken."""
        return self.by is not None and _place_key(None, self.by) == ("by", 0)

    def _fields(self) -> dict[str, Any]:
        written: dict[str, Any] = {}
        condition = super()._fields()
        if self.to is not None:
            written["to"] = self.to
        if self.by is not None:
            written["by"] = _distance_to_json(self.by)
        if "passes" in condition:
            written["passes"] = condition["passes"]
        if self.repeat_count is not None:
            written["repeat_count"] = self.repeat_count
        if "after" in condition:
            written["after"] = condition["after"]
        if not self.determined:
            written["determined"] = False
        if not self.reset:
            written["reset"] = False
        if self.manifestation is not None:
            written["manifestation"] = _written(self.manifestation)
        if self.label is not None:
            written["label"] = self.label
        return written

    def _bound_to(self, coordinate: Coordinate, what: str) -> Self:
        bound = super()._bound_to(coordinate, what)
        if bound.by is None:
            return bound
        by = _bound_distance(bound.by, coordinate, f"a `by` of {what}")
        return bound if by is bound.by else replace(bound, by=by)

    def _renamed(self, renamed: Mapping[str, str]) -> Self:
        result = super()._renamed(renamed)
        if result.to is not None and result.to in renamed:
            result = replace(result, to=renamed[result.to])
        return result


# endregion

# region Elements

_OLD_ROLES = {
    "jump_from": "a `jump` with `targets`",
    "jump_to": "a `jump_target`",
    "flow_marker": "a `break`, a `section_break` or a `barline`",
}


def _refuse_old_form(data: Mapping[str, Any]) -> None:
    """Refuse an element written with ``type``, ``target`` or its own ``repeat_count``.

    Raises:
        ValueError: Naming the form to write instead.
    """
    named = f" {quote_id(data['id'])}" if isinstance(data.get("id"), str) else ""
    what = f"flow-control element{named}"
    old = data.get("type", data.get("role"))
    if "type" in data or old in _OLD_ROLES:
        instead = (
            f": a {old} is written as {_OLD_ROLES[old]}" if old in _OLD_ROLES else ""
        )
        raise ValueError(
            f"{what} is a {old!r}: an element states its `role`, one of "
            f"{sorted(_ROLES)}{instead}"
        )
    if "target" in data:
        raise ValueError(
            f"{what} holds `target`: a jump names where it goes on in `targets`, "
            "a list of destinations such as {to: <ID>}"
        )
    if "repeat_count" in data:
        raise ValueError(
            f"{what} holds `repeat_count`: it is a key of a destination in the "
            "`targets` of a jump"
        )


def _read_label(data: Mapping[str, Any], what: str) -> str | None:
    """Read an optional ``label``, null when absent.

    Raises:
        ValueError: If the label is neither a string nor null.
    """
    label = data.get("label")
    if label is not None and not isinstance(label, str):
        raise ValueError(f"{what} has a non-string label {label!r}")
    return label


def _read_provenance(data: Mapping[str, Any], what: str) -> Provenance | None:
    """Read an optional ``provenance``, one provenance entry.

    Raises:
        ValueError: If it is no provenance entry.
    """
    if data.get("provenance") is None:
        return None
    entry = data["provenance"]
    if not isinstance(entry, dict):
        raise ValueError(f"the provenance of {what} must be an object")
    try:
        return Provenance.from_dict(entry)
    except ValueError as error:
        raise ValueError(f"the provenance of {what}: {error}") from error


@dataclass(frozen=True, repr=False)
class FlowControlElement(_IdentifiedElement):
    """What the five kinds of flow-control element share.

    An element lies at one coordinate of its timeline's axis and is written
    in a manifest as one object of the timeline's ``flow_control``, its
    ``role`` naming its class. Keys the library does not know are kept and
    written back.

    Attributes:
        coordinate: Where the element lies.
        name: A display name.
        label: A human-readable label.
        meta: Additional metadata.
        provenance: How the element was made and who believes it. An
            element is never changed or deleted to repair flow control: it
            is rejected by a belief, and another one added.
        id: The element's ID; minted from its role when not given.
    """

    role: ClassVar[str]
    """The element's ``role`` in a manifest, also the prefix of a minted ID."""
    _own_keys: ClassVar[tuple[str, ...]] = ()
    _manifest_path: ClassVar[tuple[str, ...]] = ("timelines", "flow_control")

    coordinate: Coordinate
    name: str | None = field(default=None, kw_only=True)
    label: str | None = field(default=None, kw_only=True)
    meta: dict[str, Any] = field(default_factory=dict[str, Any], kw_only=True)
    provenance: Provenance | None = field(default=None, kw_only=True)

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._dict_keys = (
            "id",
            "role",
            "coordinate",
            *cls._own_keys,
            "name",
            "label",
            "meta",
            "provenance",
        )

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, unit: TimeUnit, number_type: NumberType
    ) -> Self:
        """Restore an element from its object in a ``flow_control`` array.

        Called on :class:`FlowControlElement`, the element's ``role`` picks
        the class; called on one of the five classes, the role must be that
        class's. ``coordinate`` and every ``by`` are bare values on the
        timeline's axis, which *unit* and *number_type* describe. ``name``,
        ``label``, ``meta`` and ``provenance`` may be left out; keys the
        library does not know are kept and written back.

        Raises:
            ValueError: If the element is malformed, is written in the form
                with ``type`` and ``target``, has another role than the
                class's, or a value is no value of the axis.
        """
        require_object(data, "a flow-control element")
        _refuse_old_form(data)
        require_object(data, "a flow-control element", ("id", "role", "coordinate"))
        element_id = read_id(data["id"], "a flow-control element")
        role = data["role"]
        element_class = _ROLES.get(role) if isinstance(role, str) else None
        if element_class is None:
            raise ValueError(
                f"flow-control element {quote_id(element_id)} has the unknown role "
                f"{role!r}; an element is one of {sorted(_ROLES)}"
            )
        if not issubclass(element_class, cls):
            raise ValueError(
                f"flow-control element {quote_id(element_id)} is a {role}, "
                f"not a {cls.role}"
            )
        what = f"{role} {quote_id(element_id)}"
        try:
            element = element_class(
                axis_value_from_json(
                    data["coordinate"],
                    key="coordinate",
                    owner=element_id,
                    unit=unit,
                    number_type=number_type,
                ),
                id=element_id,
                name=read_name(data, what),
                label=_read_label(data, what),
                meta=read_meta(data, what),
                provenance=_read_provenance(data, what),
                **element_class._read_own(data, what, unit, number_type),
            )
        except ValueError as error:
            if what in str(error) or quote_id(element_id) in str(error):
                raise
            raise ValueError(f"{what}: {error}") from error
        element._keep_unknown_keys(data)
        assert isinstance(element, cls)
        return element

    @classmethod
    def _read_own(
        cls,
        data: Mapping[str, Any],
        what: str,
        unit: TimeUnit,
        number_type: NumberType,
    ) -> dict[str, Any]:
        """Read the constructor arguments of the class's own keys."""
        return {}

    def __post_init__(self) -> None:
        """Mint or validate the ID, then check the coordinate."""
        super().__post_init__()
        if not isinstance(self.coordinate, Coordinate):
            raise TypeError(
                f"a {self.role} lies at a Coordinate, not at "
                f"{type(self.coordinate).__name__} {self.coordinate!r}"
            )
        if isinstance(self.coordinate, IdCoordinate):
            object.__setattr__(self, "coordinate", self.coordinate.to_coordinate())
        if isinstance(self.provenance, Mapping):
            object.__setattr__(
                self, "provenance", Provenance.from_dict(dict(self.provenance))
            )

    @property
    def is_rejected(self) -> bool:
        """Whether any actor holds the belief ``"false"`` in the element."""
        return self.provenance is not None and any(
            belief.belief == "false" for belief in self.provenance.beliefs
        )

    def _own_fields(self) -> dict[str, Any]:
        """Return the class's own keys as written, absent ones left out."""
        return {}

    def _dict_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "id": self.id,
            "role": self.role,
            "coordinate": axis_value_to_json(self.coordinate),
            **self._own_fields(),
            "name": self.name,
            "label": self.label,
            "meta": deepcopy(self.meta),
        }
        if self.provenance is not None:
            fields["provenance"] = self.provenance.to_dict()
        return fields

    def _renamed(self, renamed: Mapping[str, str]) -> Self:
        """Return the element naming the new ID of every element renamed."""
        return self

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id!r} @ {self.coordinate})"


@dataclass(frozen=True, repr=False)
class Jump(FlowControlElement):
    """Where the flow may leave the axis, and every place it may go on at.

    One Jump lies at a coordinate; several signs at one place (a repeat end
    and a D.S. on one barline, the brackets of alternative endings) are the
    destinations of that Jump. When the flow plays up to the Jump, the
    destinations that hold are found and one of them is taken; when none
    holds, the flow goes on.

    Attributes:
        coordinate: Where the flow leaves: the end of the measure before.
        targets: The destinations, at least one, each with a place of its
            own: no two with the same ``to``, the same ``by``, or both
            without place.
        name: A display name.
        label: A human-readable label.
        meta: Additional metadata.
        provenance: How the Jump was made and who believes it.
        id: The Jump's ID; minted as ``jump_{N}`` when not given.

    Examples:
        >>> from timetoalign.core import Coordinate, TimeUnit
        >>> repeat = Jump(
        ...     Coordinate(64, TimeUnit.quarters),
        ...     [JumpDestination(repeat_count=1, manifestation="repeat_end")],
        ... )
        >>> repeat.id, repeat.manifestations
        ('jump_1', ("repeat_end",))
        >>> repeat.to_dict()["targets"]
        [{'repeat_count': 1, 'manifestation': 'repeat_end'}]
    """

    role: ClassVar[str] = "jump"
    _id_prefix: ClassVar[str] = role
    _own_keys: ClassVar[tuple[str, ...]] = ("targets",)

    targets: tuple[JumpDestination, ...]

    @classmethod
    def _read_own(
        cls,
        data: Mapping[str, Any],
        what: str,
        unit: TimeUnit,
        number_type: NumberType,
    ) -> dict[str, Any]:
        targets = data.get("targets")
        if not isinstance(targets, list):
            raise ValueError(f"{what} lacks `targets`, the list of its destinations")
        return {
            "targets": tuple(
                JumpDestination.from_dict(
                    item,
                    f"destination {index + 1} of {what}",
                    unit=unit,
                    number_type=number_type,
                )
                for index, item in enumerate(targets)
            )
        }

    def __post_init__(self) -> None:
        """Check the destinations and put their distances on the Jump's axis."""
        super().__post_init__()
        what = f"jump {quote_id(self.id)}"
        if isinstance(self.targets, JumpDestination):
            targets: tuple[Any, ...] = (self.targets,)
        else:
            targets = tuple(self.targets)
        if not targets:
            raise ValueError(f"{what} has no destination: `targets` lists at least one")
        for destination in targets:
            if not isinstance(destination, JumpDestination):
                raise ValueError(
                    f"{what} has the destination {destination!r}, which is no "
                    "JumpDestination"
                )
        bound = tuple(
            destination._bound_to(self.coordinate, what) for destination in targets
        )
        places = [destination.place for destination in bound]
        for place in places:
            if places.count(place) > 1:
                stated = "no place" if place is None else f"`{place[0]}: {place[1]}`"
                raise ValueError(
                    f"{what} has {places.count(place)} destinations with {stated}: "
                    "the destinations of one jump are distinct places"
                )
        object.__setattr__(self, "targets", bound)

    @property
    def manifestations(self) -> tuple[FlowManifestation | str, ...]:
        """What the Jump looks like in the source: its destinations' manifestations."""
        return tuple(
            destination.manifestation
            for destination in self.targets
            if destination.manifestation is not None
        )

    def _own_fields(self) -> dict[str, Any]:
        return {"targets": [destination.to_dict() for destination in self.targets]}

    def _renamed(self, renamed: Mapping[str, str]) -> Self:
        targets = tuple(destination._renamed(renamed) for destination in self.targets)
        if all(new is old for new, old in zip(targets, self.targets, strict=True)):
            return self
        return replace(self, targets=targets)

    def __repr__(self) -> str:
        count = len(self.targets)
        return (
            f"Jump({self.id!r} @ {self.coordinate}, "
            f"{count} destination{'' if count == 1 else 's'})"
        )


@dataclass(frozen=True, repr=False)
class _ManifestedElement(FlowControlElement):
    """An element that states itself what it looks like in the source."""

    _admitted: ClassVar[frozenset[FlowManifestation]] = frozenset()
    _own_keys: ClassVar[tuple[str, ...]] = ("manifestation",)

    manifestation: FlowManifestation | str | None = field(default=None, kw_only=True)

    @classmethod
    def _read_own(
        cls,
        data: Mapping[str, Any],
        what: str,
        unit: TimeUnit,
        number_type: NumberType,
    ) -> dict[str, Any]:
        return {"manifestation": data.get("manifestation")}

    def __post_init__(self) -> None:
        """Hold the manifestation as a word of the vocabulary where it is one."""
        super().__post_init__()
        object.__setattr__(
            self,
            "manifestation",
            _manifestation(
                self.manifestation,
                self._admitted,
                f"{self.role} {quote_id(self.id)}",
            ),
        )

    def _own_fields(self) -> dict[str, Any]:
        if self.manifestation is None:
            return {}
        return {"manifestation": _written(self.manifestation)}

    def __repr__(self) -> str:
        shown = (
            "" if self.manifestation is None else f", {_written(self.manifestation)}"
        )
        return f"{type(self).__name__}({self.id!r}{shown} @ {self.coordinate})"


@dataclass(frozen=True, repr=False)
class JumpTarget(_ManifestedElement):
    """A place a destination names: a repeat start, a segno, a coda, a bracket.

    Nothing happens when the flow plays up to it.

    Attributes:
        coordinate: The place.
        manifestation: ``repeat_start``, ``segno``, ``coda``, ``volta``,
            ``ossia`` or a word outside the vocabulary.
        name: A display name, such as the source's name of the sign.
        label: A human-readable label.
        meta: Additional metadata.
        provenance: How the element was made and who believes it.
        id: The element's ID; minted as ``jump_target_{N}`` when not given.

    Examples:
        >>> from timetoalign.core import Coordinate, TimeUnit
        >>> JumpTarget(Coordinate(16, TimeUnit.quarters), manifestation="segno")
        JumpTarget('jump_target_1', segno @ 16 quarters)
    """

    role: ClassVar[str] = "jump_target"
    _id_prefix: ClassVar[str] = role
    _admitted: ClassVar[frozenset[FlowManifestation]] = frozenset(
        {
            FlowManifestation.repeat_start,
            FlowManifestation.segno,
            FlowManifestation.coda,
            FlowManifestation.volta,
            FlowManifestation.ossia,
        }
    )


@dataclass(frozen=True, repr=False)
class Break(_ManifestedElement):
    """A place the flow never goes on across.

    When the flow plays up to a Break, a Jump at the same coordinate with a
    destination that holds is taken; otherwise the flow ends there. With
    ``ends`` it is a Break only while that condition holds: a Fine ends the
    flow after the D.C. A Break lies between two brackets of alternative
    endings.

    Attributes:
        coordinate: Where the flow ends.
        manifestation: ``fine``, ``volta`` or a word outside the vocabulary.
        ends: When the Break ends the flow; None for always. A condition
            that states nothing is refused.
        name: A display name.
        label: A human-readable label.
        meta: Additional metadata.
        provenance: How the element was made and who believes it.
        id: The element's ID; minted as ``break_{N}`` when not given.

    Examples:
        >>> from timetoalign.core import Coordinate, TimeUnit
        >>> fine = Break(
        ...     Coordinate(32, TimeUnit.quarters),
        ...     manifestation="fine",
        ...     ends=FlowCondition(after=DestinationReference(jump="jump_1")),
        ... )
        >>> fine.to_dict()["ends"]
        {'after': {'jump': 'jump_1'}}
    """

    role: ClassVar[str] = "break"
    _id_prefix: ClassVar[str] = role
    _admitted: ClassVar[frozenset[FlowManifestation]] = frozenset(
        {FlowManifestation.fine, FlowManifestation.volta}
    )
    _own_keys: ClassVar[tuple[str, ...]] = ("manifestation", "ends")

    ends: FlowCondition | None = field(default=None, kw_only=True)

    @classmethod
    def _read_own(
        cls,
        data: Mapping[str, Any],
        what: str,
        unit: TimeUnit,
        number_type: NumberType,
    ) -> dict[str, Any]:
        read = super()._read_own(data, what, unit, number_type)
        if data.get("ends") is not None:
            read["ends"] = FlowCondition.from_dict(
                data["ends"],
                f"the `ends` of {what}",
                unit=unit,
                number_type=number_type,
            )
        return read

    def __post_init__(self) -> None:
        """Refuse an ``ends`` that states no condition."""
        super().__post_init__()
        if self.ends is None:
            return
        what = f"break {quote_id(self.id)}"
        if type(self.ends) is not FlowCondition:
            raise ValueError(
                f"the `ends` of {what} is a FlowCondition, not "
                f"{type(self.ends).__name__}"
            )
        if not self.ends.is_stated:
            raise ValueError(
                f"the `ends` of {what} states no condition: leave `ends` out for "
                "a break that always ends the flow"
            )
        object.__setattr__(self, "ends", self.ends._bound_to(self.coordinate, what))

    def _own_fields(self) -> dict[str, Any]:
        fields = super()._own_fields()
        if self.ends is not None:
            fields["ends"] = self.ends.to_dict()
        return fields

    def _renamed(self, renamed: Mapping[str, str]) -> Self:
        if self.ends is None:
            return self
        ends = self.ends._renamed(renamed)
        return self if ends is self.ends else replace(self, ends=ends)


@dataclass(frozen=True, repr=False)
class SectionBreak(_ManifestedElement):
    """Where a section ends: a movement boundary, a section of a piece.

    The flow goes on across it. A destination without place lands at the
    nearest section break before its Jump, and a flow ended by a Break
    begins again at the next section break.

    Attributes:
        coordinate: Where the section ends.
        manifestation: A word outside the vocabulary, which defines none
            for a section break.
        name: A display name.
        label: A human-readable label.
        meta: Additional metadata.
        provenance: How the element was made and who believes it.
        id: The element's ID; minted as ``section_break_{N}`` when not given.
    """

    role: ClassVar[str] = "section_break"
    _id_prefix: ClassVar[str] = role


@dataclass(frozen=True, repr=False)
class Barline(_ManifestedElement):
    """A barline the source prints: a double or a final barline.

    It changes nothing in the flow.

    Attributes:
        coordinate: Where the barline lies.
        manifestation: ``double_barline``, ``final_barline`` or a word
            outside the vocabulary.
        name: A display name.
        label: A human-readable label.
        meta: Additional metadata.
        provenance: How the element was made and who believes it.
        id: The element's ID; minted as ``barline_{N}`` when not given.
    """

    role: ClassVar[str] = "barline"
    _id_prefix: ClassVar[str] = role
    _admitted: ClassVar[frozenset[FlowManifestation]] = frozenset(
        {FlowManifestation.double_barline, FlowManifestation.final_barline}
    )


_ROLES: dict[str, type[FlowControlElement]] = {
    element_class.role: element_class
    for element_class in (Jump, JumpTarget, Break, SectionBreak, Barline)
}
"""The class of each ``role``."""

# endregion

# region FlowControlRegistry

_Entry = tuple[str, bool, str, str]


def _canonical(elements: Iterable[FlowControlElement]) -> list[FlowControlElement]:
    """Sort elements by coordinate, then by ID."""
    return sorted(elements, key=lambda element: (element.coordinate.value, element.id))


@dataclass
class FlowControlRegistry:
    """The flow control of one timeline: its elements in canonical order.

    One list holds every :class:`FlowControlElement`, ordered by coordinate
    and then by ID; ``jumps``, ``jump_targets``, ``breaks``,
    ``section_breaks`` and ``barlines`` are views of it by class. Every
    element carries an ID unique within the registry and, once the registry
    belongs to a timeline, within that timeline's whole manifest: an element
    whose minted ID is taken is stored under a freshly minted ID, and one
    whose given ID is taken is refused. A manifest writes the registry as
    the timeline's ``flow_control`` array.

    The registry refuses what cannot be played: a second Jump at a
    coordinate (unless one of the two is rejected by a belief), a ``to``
    naming neither a :class:`JumpTarget` of the registry nor a place of its
    timeline, and an ``after`` that does not name exactly one destination
    of another Jump of the registry.

    Attributes:
        elements: Every element, by coordinate and then by ID.
        unit: The unit of the axis the registry is bound to, if any.
        number_type: The number type of that axis, if any.

    Examples:
        >>> from timetoalign.core import Coordinate, TimeUnit
        >>> registry = FlowControlRegistry(unit=TimeUnit.quarters)
        >>> registry.add(JumpTarget(Coordinate(16, "quarters"), id="segno"))
        JumpTarget('segno' @ 16 quarters)
        >>> registry.add(
        ...     Jump(
        ...         Coordinate(64, "quarters"),
        ...         [JumpDestination(to="segno", repeat_count=1, reset=False)],
        ...     )
        ... ).id
        'jump_1'
        >>> [element.role for element in registry.elements]
        ['jump_target', 'jump']
        >>> registry.place_of(registry.jumps[0].targets[0], registry.jumps[0])
        Coordinate(Fraction(16, 1), quarters)
    """

    elements: list[FlowControlElement] = field(default_factory=list[FlowControlElement])
    unit: TimeUnit | None = field(default=None, kw_only=True)
    number_type: NumberType | None = field(default=None, kw_only=True)
    # Set by the owning timeline: admits IDs into its whole manifest, and
    # gives the places of the timeline a destination may name beside the
    # registry's own jump targets.
    _admit: Callable[..., list[str]] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _places: Callable[[], Mapping[str, Coordinate]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def from_dict(
        cls,
        data: list[dict[str, Any]],
        *,
        unit: TimeUnit | str,
        number_type: NumberType | str | None = None,
        places: Mapping[str, Coordinate] | None = None,
    ) -> FlowControlRegistry:
        """Restore a registry bound to a timeline's axis from its ``flow_control``.

        Each element of the array states its ``role`` (``jump``,
        ``jump_target``, ``break``, ``section_break`` or ``barline``), its
        coordinate a bare value on the axis that *unit* and *number_type*
        (the unit's own when None) describe. The elements are added
        together, in any order the array lists them. Keys the library does
        not know are kept on each element and written back.

        Args:
            data: The ``flow_control`` array.
            unit: The axis's unit.
            number_type: The axis's number type, or None for the unit's own.
            places: The places of the timeline a destination may name beside
                the array's jump targets: the starts of its Regions and of
                its child timelines and its own, by ID.

        Raises:
            ValueError: If the array or an element is malformed, an element
                is written in the form with ``type`` and ``target``, or the
                registry refuses the elements.
        """
        resolved_unit = TimeUnit(unit)
        resolved_number_type = (
            resolved_unit.default_number_type
            if number_type is None
            else NumberType(number_type)
        )
        if not isinstance(data, list):
            raise ValueError("the flow_control must be an array")
        registry = cls(unit=resolved_unit, number_type=resolved_number_type)
        if places is not None:
            table = dict(places)
            registry._places = lambda: table
        registry.add_all(
            FlowControlElement.from_dict(
                item, unit=resolved_unit, number_type=resolved_number_type
            )
            for item in data
        )
        return registry

    def __post_init__(self) -> None:
        """Normalize binding enum spellings and any constructor content."""
        if isinstance(self.unit, str):
            self.unit = TimeUnit(self.unit)
        if isinstance(self.number_type, str):
            self.number_type = NumberType(self.number_type)
        given, self.elements = list(self.elements), []
        self.add_all(given)

    # -- views --------------------------------------------------------------

    def _of(self, element_class: type[Any]) -> list[Any]:
        return [element for element in self.elements if type(element) is element_class]

    @property
    def jumps(self) -> list[Jump]:
        """The Jumps, by coordinate and then by ID."""
        return self._of(Jump)

    @property
    def jump_targets(self) -> list[JumpTarget]:
        """The JumpTargets, by coordinate and then by ID."""
        return self._of(JumpTarget)

    @property
    def breaks(self) -> list[Break]:
        """The Breaks, by coordinate and then by ID."""
        return self._of(Break)

    @property
    def section_breaks(self) -> list[SectionBreak]:
        """The SectionBreaks, by coordinate and then by ID."""
        return self._of(SectionBreak)

    @property
    def barlines(self) -> list[Barline]:
        """The Barlines, by coordinate and then by ID."""
        return self._of(Barline)

    # -- coordinates --------------------------------------------------------

    def _coordinate(
        self, coordinate: _Number | Coordinate | IdCoordinate
    ) -> Coordinate:
        """Return one public coordinate input on the registry's axis.

        Raises:
            ValueError: If a bare number is given to an unbound registry, or
                a coordinate's unit is not the registry's.
        """
        if not isinstance(coordinate, Coordinate):
            if self.unit is None:
                raise ValueError("A bare coordinate requires a bound registry unit")
            return Coordinate(
                coordinate,
                self.unit,
                number_type=self.number_type or self.unit.default_number_type,
            )
        if isinstance(coordinate, IdCoordinate):
            coordinate = coordinate.to_coordinate()
        if self.unit is not None and coordinate.unit is not self.unit:
            raise ValueError(
                f"Flow-control coordinate unit {coordinate.unit} does not match "
                f"registry unit {self.unit}"
            )
        if self.number_type is not None:
            coordinate = express_scalar_as(coordinate, self.number_type)
        return coordinate

    def _on_axis(self, element: FlowControlElement) -> FlowControlElement:
        """Return *element* with its coordinate on the registry's axis."""
        if not isinstance(element, FlowControlElement):
            raise TypeError(f"a registry holds flow-control elements, not {element!r}")
        coordinate = self._coordinate(element.coordinate)
        if coordinate is element.coordinate:
            return element
        return replace(element, coordinate=coordinate)

    def _place_table(self) -> Mapping[str, Coordinate]:
        return {} if self._places is None else self._places()

    # -- adding -------------------------------------------------------------

    def add(self, element: FlowControlElement) -> Any:
        """Add one element.

        An element another one names (a :class:`JumpTarget` named by a
        ``to``, a :class:`Jump` named by an ``after``) is added before the
        one naming it; :meth:`add_all` adds elements naming one another
        together.

        Args:
            element: The element.

        Returns:
            The stored element, with its coordinate on the registry's axis
            and its ID as admitted.

        Raises:
            ValueError: As :meth:`add_all`.
        """
        return self.add_all([element])[0]

    def add_all(
        self, elements: Iterable[FlowControlElement]
    ) -> list[FlowControlElement]:
        """Add several elements together, or none of them.

        The elements may name one another in any order. An element whose
        minted ID is taken is stored under a freshly minted ID, and the
        elements added with it name it by that ID.

        Args:
            elements: The elements.

        Returns:
            The stored elements in the order given, each with its
            coordinate on the registry's axis and its ID as admitted.

        Raises:
            ValueError: If an element's given ID is already taken, its unit
                is not the registry's, a coordinate holds a second Jump
                (unless one of the two is rejected by a belief), a ``to``
                names neither a JumpTarget of the registry nor a place of
                its timeline, or an ``after`` does not name exactly one
                destination of another Jump of the registry. Nothing has
                changed then.
        """
        staged = [self._on_axis(element) for element in elements]
        if not staged:
            return []
        entries: list[_Entry] = [
            (
                element.id,
                element._minted,
                element._id_prefix,
                f"{element.role} {quote_id(element.id)}",
            )
            for element in staged
        ]
        ids = self._admitted_ids(entries, check_only=True)
        kept = set(ids)
        renamed: dict[str, str] = {}
        for element, new_id in zip(staged, ids, strict=True):
            if new_id != element.id and element.id not in kept:
                renamed.setdefault(element.id, new_id)
        stored = [
            (
                element
                if new_id == element.id
                else replace(element, id=new_id, _minted=True)
            )._renamed(renamed)
            for element, new_id in zip(staged, ids, strict=True)
        ]
        self._check(stored, [*self.elements, *stored])
        self._admitted_ids(entries, check_only=False)
        self.elements = _canonical([*self.elements, *stored])
        return stored

    def _admitted_ids(self, entries: list[_Entry], *, check_only: bool) -> list[str]:
        """Return the ID each joining element is stored under.

        Raises:
            ValueError: If a given ID is already taken.
        """
        if self._admit is not None:
            return self._admit(entries, check_only=check_only)
        taken = {held.id: f"{held.role} {quote_id(held.id)}" for held in self.elements}
        plan = _plan_admissions(entries, taken)
        return [plan.get(index, entry[0]) for index, entry in enumerate(entries)]

    def _check(
        self, new: Sequence[FlowControlElement], every: Sequence[FlowControlElement]
    ) -> None:
        """Refuse *new* elements that cannot be played among *every* element.

        Raises:
            ValueError: As :meth:`add_all`.
        """
        jumps = {element.id: element for element in every if isinstance(element, Jump)}
        targets = {element.id for element in every if isinstance(element, JumpTarget)}
        places: Mapping[str, Coordinate] | None = None
        for element in new:
            what = f"{element.role} {quote_id(element.id)}"
            if isinstance(element, Jump):
                if not element.is_rejected:
                    for other in jumps.values():
                        if (
                            other is not element
                            and not other.is_rejected
                            and other.coordinate.value == element.coordinate.value
                        ):
                            raise ValueError(
                                f"{what} lies at {element.coordinate}, where jump "
                                f"{quote_id(other.id)} lies already: one jump lies "
                                "at a coordinate, and several signs there are its "
                                "destinations"
                            )
                for destination in element.targets:
                    if destination.to is not None and destination.to not in targets:
                        if places is None:
                            places = self._place_table()
                        if destination.to not in places:
                            raise ValueError(
                                f"{what} goes on at {quote_id(destination.to)}, which "
                                "names no jump target of this registry and no place "
                                "of its timeline. Available jump targets: "
                                f"{sorted(targets)}. Available places: {list(places)}"
                            )
                    if destination.after is not None:
                        self._destination_among(
                            jumps, destination.after, what, holder=element.id
                        )
            elif isinstance(element, Break) and element.ends is not None:
                if element.ends.after is not None:
                    self._destination_among(jumps, element.ends.after, what)

    @staticmethod
    def _destination_among(
        jumps: Mapping[str, Jump],
        reference: DestinationReference,
        what: str,
        *,
        holder: str | None = None,
    ) -> tuple[Jump, int]:
        """Return the Jump and the index of the destination *reference* names.

        Raises:
            ValueError: If it does not name exactly one destination of a
                Jump of *jumps* other than *holder*.
        """
        jump = jumps.get(reference.jump)
        if jump is None:
            raise ValueError(
                f"the `after` of {what} names {quote_id(reference.jump)}, which is "
                f"no jump of this registry. Available jumps: {list(jumps)}"
            )
        if jump.id == holder:
            raise ValueError(
                f"the `after` of {what} names a destination of the same jump: it "
                "names one of another jump"
            )
        place = reference.place
        if place is None and len(jump.targets) == 1:
            return jump, 0
        named = [
            index
            for index, destination in enumerate(jump.targets)
            if destination.place == place
        ]
        if len(named) != 1:
            stated = (
                "without place" if place is None else f"with `{place[0]}: {place[1]}`"
            )
            raise ValueError(
                f"the `after` of {what} names the destination of jump "
                f"{quote_id(jump.id)} {stated}, which names {len(named)} of its "
                f"{len(jump.targets)} destinations instead of one"
            )
        return jump, named[0]

    # -- queries ------------------------------------------------------------

    def jump_at(self, coordinate: _Number | Coordinate | IdCoordinate) -> Jump | None:
        """Return the Jump at *coordinate*, if one lies there.

        Where a rejected Jump and its replacement lie together, the one not
        rejected is returned.

        Raises:
            ValueError: If a coordinate object's unit is not the registry's.
        """
        value = (
            self._coordinate(coordinate).value
            if isinstance(coordinate, Coordinate)
            else coordinate
        )
        here = [jump for jump in self.jumps if jump.coordinate.value == value]
        standing = [jump for jump in here if not jump.is_rejected]
        return (standing or here or [None])[0]

    def destination_of(self, reference: DestinationReference) -> tuple[Jump, int]:
        """Return the Jump an ``after`` names and the index of the destination.

        Raises:
            ValueError: If *reference* does not name exactly one destination
                of a Jump of the registry.
        """
        jumps = {jump.id: jump for jump in self.jumps}
        return self._destination_among(jumps, reference, "a condition")

    def place_of(
        self,
        destination: JumpDestination,
        jump: Jump,
        *,
        elements: Sequence[FlowControlElement] | None = None,
    ) -> Coordinate:
        """Return where *destination* of *jump* goes on.

        ``by`` is added to the Jump's coordinate. ``to`` is the coordinate
        of the JumpTarget of that ID, else the place of that ID the
        registry's timeline gives (the start of a Region or of a child
        timeline, or the timeline's own start). A destination without place
        goes on at the nearest section break before the Jump, or at 0.

        Args:
            destination: The destination.
            jump: The Jump holding it.
            elements: The elements to find jump targets and section breaks
                among; every element of the registry when None.

        Raises:
            KeyError: If ``to`` names nothing.
        """
        among = self.elements if elements is None else elements
        axis = jump.coordinate
        if isinstance(destination.by, Duration):
            return Coordinate(
                axis.value + destination.by.value,
                axis.unit,
                number_type=axis.number_type,
            )
        if destination.to is not None:
            for element in among:
                if isinstance(element, JumpTarget) and element.id == destination.to:
                    return element.coordinate
            places = self._place_table()
            if destination.to in places:
                return self._coordinate(places[destination.to])
            raise KeyError(
                f"{destination.to!r} names no jump target and no place. Available "
                f"jump targets: {[e.id for e in among if isinstance(e, JumpTarget)]}. "
                f"Available places: {list(places)}"
            )
        before = [
            element.coordinate
            for element in among
            if isinstance(element, SectionBreak)
            and element.coordinate.value < axis.value
        ]
        if before:
            return max(before, key=lambda coordinate: coordinate.value)
        return Coordinate(0, axis.unit, number_type=axis.number_type)

    def active_elements(
        self,
        *,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
        context: Mapping[str, Any] | None = None,
    ) -> list[FlowControlElement]:
        """Return the elements *believer* does not reject, in canonical order.

        The belief filter of
        :meth:`Provenance.passes_belief_filter
        <timetoalign.core.provenance.Provenance.passes_belief_filter>`,
        read on each element's provenance: an element without provenance, or
        without a belief of the believer's, is one the believer has not
        decided on. Without a believer nothing is filtered.

        Args:
            believer: The actor whose beliefs filter; None keeps all.
            beliefs: The belief states kept; every state but ``"false"`` by
                default.
            context: The JSON-LD context under which actors compare as IRIs.

        Raises:
            ValueError: If two of the elements kept are Jumps at one
                coordinate: a rejected Jump and its replacement, read
                without the believer who rejected it.
        """
        kept = [
            element
            for element in self.elements
            if believer is None
            or (
                "undecided" in beliefs
                if element.provenance is None
                else element.provenance.passes_belief_filter(
                    believer, beliefs, context=context
                )
            )
        ]
        seen: dict[Any, Jump] = {}
        for element in kept:
            if not isinstance(element, Jump):
                continue
            other = seen.setdefault(element.coordinate.value, element)
            if other is not element:
                raise ValueError(
                    f"jumps {quote_id(other.id)} and {quote_id(element.id)} both lie "
                    f"at {element.coordinate}: name the believer who rejected one "
                    "of them"
                )
        return kept

    def has_flow_control(self) -> bool:
        """Whether the registry holds any element."""
        return bool(self.elements)

    # -- maintenance --------------------------------------------------------

    def _reassign_id(self, old_id: str, new_id: str) -> None:
        """Store the element with ID *old_id* under the minted ID *new_id*.

        Every destination and condition naming it follows.
        """
        self.elements = [
            (
                replace(element, id=new_id, _minted=True)
                if element.id == old_id
                else element
            )
            for element in self.elements
        ]
        self._rename_references({old_id: new_id})

    def _rename_references(self, renamed: Mapping[str, str | None]) -> None:
        """Name the new ID of every renamed element or place; keep the others."""
        names = {old: new for old, new in renamed.items() if new is not None}
        if names:
            self.elements = _canonical(
                element._renamed(names) for element in self.elements
            )

    def _absorb(self, other: FlowControlRegistry) -> None:
        """Add every element of *other* together."""
        self.add_all(other.elements)

    def clear(self) -> None:
        """Remove every element."""
        self.elements = []

    def copy(self) -> FlowControlRegistry:
        """Return an independent registry with the same binding and content.

        The copy belongs to no timeline until one adopts it.
        """
        owner = (self._admit, self._places)
        self._admit = self._places = None
        try:
            return deepcopy(self)
        finally:
            self._admit, self._places = owner

    def to_dict(self, *, minimal: bool = True) -> list[dict[str, Any]]:
        """Return the elements as the ``flow_control`` array.

        Each element is written as its own object, in canonical order
        whatever order the elements were added in: by coordinate, then by
        ID, values compared exactly and IDs by Unicode code point. Nothing
        is written that the registry does not hold.

        Args:
            minimal: Write each element in the minimal form (the default):
                every key at its default value left out. With False, the
                complete form of the element's own keys. A destination and
                a condition are written alike in both: a key at its default
                is absent.
        """
        return [element.to_dict(minimal=minimal) for element in self.elements]

    def __repr__(self) -> str:
        counts = ", ".join(
            f"{len(self._of(element_class))} {element_class.role}s"
            for element_class in _ROLES.values()
        )
        return f"FlowControlRegistry({counts})"


# endregion
