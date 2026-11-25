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

import datetime
import logging
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from timetoshare import DEFAULT_BELIEFS, BeliefState, check_belief_filter

from timetoalign.core import Coordinate, Duration, IdCoordinate, NumberType, TimeUnit
from timetoalign.core.enums import FlowManifestation
from timetoalign.core.ids import (
    _IdentifiedElement,
    _OpenObject,
    _plan_admissions,
    quote_id,
)
from timetoalign.core.provenance import (
    Belief,
    BeliefValue,
    Provenance,
    belief_mask,
    provenance_array,
)
from timetoalign.core.time import express_scalar_as, number_to_json

from .manifest import (
    axis_value_from_json,
    axis_value_to_json,
    read_id,
    read_meta,
    read_name,
    require_object,
)
from .statements import _day

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


def _by_place(destination: JumpDestination) -> tuple[int, Any, str]:
    """The canonical order of a Jump's destinations.

    The destination without place first, then those with ``by``, by value,
    then those with ``to``, by ID (by Unicode code point).
    """
    place = destination.place
    if place is None:
        return (0, 0, "")
    if place[0] == "by":
        return (1, place[1], "")
    return (2, 0, place[1])


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
            :class:`PassesFrom`. On a Break the passes listed are exactly
            those arrivals. Among the destinations of a Jump, the one
            naming the highest number any of them lists also holds at
            every later arrival (the last ending is played again after a
            D.C.).
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

    def to_dict(self, *, minimal: bool = True) -> dict[str, Any]:
        """Return the condition as its JSON object.

        Args:
            minimal: Write the minimal form (the default): every key at its
                default value left out. With False, the complete form: the
                keys that have a default (``determined``, ``reset`` and
                ``label`` of a destination) are written whatever they hold.
                A key without a default is written only when stated.
        """
        written = self._fields(minimal=minimal)
        written.update(deepcopy(dict(self._unknown_keys)))
        return written

    def _fields(self, *, minimal: bool) -> dict[str, Any]:
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
        >>> leap = JumpDestination(to="segno", repeat_count=1, reset=False)
        >>> leap.to_dict()
        {'to': 'segno', 'repeat_count': 1, 'reset': False}
        >>> leap.to_dict(minimal=False)
        {'to': 'segno', 'repeat_count': 1, 'determined': True, 'reset': False, 'label': None}
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

    def _fields(self, *, minimal: bool) -> dict[str, Any]:
        written: dict[str, Any] = {}
        condition = super()._fields(minimal=minimal)
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
        if not (minimal and self.determined):
            written["determined"] = self.determined
        if not (minimal and self.reset):
            written["reset"] = self.reset
        if self.manifestation is not None:
            written["manifestation"] = _written(self.manifestation)
        if not (minimal and self.label is None):
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


def _read_provenance(data: Mapping[str, Any], what: str) -> list[Any] | None:
    """Read an optional ``provenance``, a list of provenance entries.

    Raises:
        ValueError: If it is no list.
    """
    entries = data.get("provenance")
    if entries is not None and not isinstance(entries, list):
        raise ValueError(
            f"the provenance of {what} is a list of provenance entries, not "
            f"{entries!r}"
        )
    return entries


class _OpenBelief(Belief):
    """A belief as a manifest states it, with the keys the library does not know."""

    model_config = ConfigDict(frozen=True, extra="allow", title="Belief")

    def to_dict(self) -> dict[str, Any]:
        """The belief as it was read: an unknown key keeps its value, null too."""
        return {**super().to_dict(), **_null_extras(self)}


class _OpenProvenance(Provenance):
    """A provenance entry as the format admits it on an element.

    The format's entry is an open object whose keys are all optional: an
    entry may state nothing (``{}``) and may hold keys the library does not
    know, as may each of its beliefs. Such an entry is held as read and
    written back unchanged; the keys the library knows are checked as on
    every entry. The entries of flow-control elements and FlowMaps are read
    this way (:func:`_read_provenance_entry`).
    """

    model_config = ConfigDict(frozen=True, extra="allow", title="Provenance")

    @field_validator("beliefs", mode="before")
    @classmethod
    def _read_null_beliefs(cls, value: Any) -> Any:
        if value is None:
            return ()
        if isinstance(value, str) or not isinstance(value, Iterable):
            return value
        return tuple(
            (
                _OpenBelief.model_validate(dict(belief))
                if isinstance(belief, Mapping)
                else belief
            )
            for belief in value
        )

    @model_validator(mode="after")
    def _check_stated(self) -> Provenance:
        return self

    def to_dict(self) -> dict[str, Any]:
        """The entry as it was read: an unknown key keeps its value, null too."""
        data = super().to_dict()
        beliefs = data.pop("beliefs", None)
        data.update(_null_extras(self))
        if beliefs is not None:
            data["beliefs"] = beliefs
        return data


def _null_extras(model: BaseModel) -> dict[str, None]:
    """The unknown keys of *model* that hold null, which ``to_dict`` leaves out."""
    return {
        key: None for key, value in (model.model_extra or {}).items() if value is None
    }


def _read_provenance_entry(entry: Mapping[str, Any]) -> Provenance:
    """Read one provenance entry of a flow-control element or a FlowMap.

    An entry holding only keys the library knows, and at least one, is the
    plain :class:`~timetoalign.core.provenance.Provenance`. An entry that
    states nothing, or holds a key the library does not know (its own or a
    belief's), is valid in the format: it is held with those keys and
    written back unchanged.

    Raises:
        ValueError: If a key the library knows holds an invalid value.
    """
    data = dict(entry)
    try:
        return Provenance.from_dict(data)
    except ValueError:
        return _OpenProvenance.model_validate(data)


def _provenance_entries(value: object, what: str) -> tuple[Provenance, ...] | None:
    """Return the provenance of an element as it holds it: its entries, or None.

    One entry (a :class:`~timetoalign.core.provenance.Provenance` or its
    dictionary) is the list of that entry.

    Raises:
        ValueError: If *value* lists no entry, or holds something that is no
            provenance entry.
    """
    if value is None:
        return None
    if isinstance(value, Provenance | Mapping):
        value = (value,)
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValueError(
            f"the provenance of {what} is a list of provenance entries, not {value!r}"
        )
    entries: list[Provenance] = []
    for entry in value:
        if isinstance(entry, Mapping):
            try:
                entry = _read_provenance_entry(entry)
            except ValueError as error:
                raise ValueError(f"the provenance of {what}: {error}") from error
        if not isinstance(entry, Provenance):
            raise ValueError(
                f"the provenance of {what} holds {entry!r}, which is no "
                "provenance entry"
            )
        entries.append(entry)
    if not entries:
        raise ValueError(
            f"the provenance of {what} lists no entry: leave `provenance` out for "
            "an element that states none"
        )
    return tuple(entries)


def _belief_state(
    entries: Sequence[Provenance], believer: str, context: Mapping[str, Any] | None
) -> BeliefState:
    """What *believer* believes of an element holding the provenance *entries*.

    The most critical belief the believer holds in any entry (``"false"``,
    then ``"unlikely"``, then ``"true"``), or ``"undecided"`` where the
    believer holds none; each entry is read by
    :func:`~timetoalign.core.provenance.belief_mask`.
    """
    if entries:
        rows = provenance_array(list(entries))
        states: tuple[BeliefState, ...] = ("false", "unlikely", "true")
        for state in states:
            if any(belief_mask(rows, believer, [state], context=context).to_pylist()):
                return state
    return "undecided"


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
        provenance: How the element was made and who believes it: one or
            more provenance entries, or None. A single entry given is held
            as the list of that entry. An element is never changed or
            deleted to repair flow control: it is rejected by a belief
            (:meth:`with_belief`), and another one added.
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
    provenance: tuple[Provenance, ...] | None = field(default=None, kw_only=True)

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
        ``label``, ``meta`` and ``provenance`` (a list of provenance
        entries) may be left out; keys the library does not know are kept
        and written back.

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
        object.__setattr__(
            self,
            "provenance",
            _provenance_entries(self.provenance, f"{self.role} {quote_id(self.id)}"),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id!r} @ {self.coordinate})"

    @property
    def is_rejected(self) -> bool:
        """Whether the element is rejected: some actor believes it ``"false"``.

        Any entry of its provenance holding such a belief rejects it. A
        rejected element is left out when flow control is played, and
        nothing may name it.
        """
        return any(
            belief.belief == "false"
            for entry in self.provenance or ()
            for belief in entry.beliefs
        )

    def with_belief(
        self, belief: Belief, *, context: Mapping[str, Any] | None = None
    ) -> Self:
        """A copy of the element whose provenance holds *belief*.

        Only the provenance changes. The belief replaces its actor's earlier
        one in every entry holding one
        (:meth:`Provenance.with_belief
        <timetoalign.core.provenance.Provenance.with_belief>`); where the
        actor holds none, the first entry records it, and an element
        without provenance gets one entry holding the belief alone. Every
        other actor's belief stands. A registry records a belief in an
        element it holds with :meth:`FlowControlRegistry.with_belief`.

        Args:
            belief: The belief; ``"false"`` rejects the element.
            context: The JSON-LD context under which actors compare as IRIs;
                None declares no prefix.

        Examples:
            >>> from timetoalign.core import Coordinate, TimeUnit
            >>> from timetoalign.core.provenance import Belief
            >>> barline = Barline(Coordinate(8, TimeUnit.quarters))
            >>> rejected = barline.with_belief(
            ...     Belief(actor="ana", belief="false", date="2026-11-25")
            ... )
            >>> barline.is_rejected, rejected.is_rejected
            (False, True)
            >>> rejected.to_dict()["provenance"]
            [{'beliefs': [{'actor': 'ana', 'belief': 'false', 'date': '2026-11-25'}]}]
        """
        entries = self.provenance
        if entries is None:
            return replace(self, provenance=(Provenance(beliefs=(belief,)),))
        holding = [
            entry.get_belief(belief.actor, context=context) is not None
            for entry in entries
        ]
        if not any(holding):
            holding[0] = True
        return replace(
            self,
            provenance=tuple(
                entry.with_belief(belief, context=context) if holds else entry
                for entry, holds in zip(entries, holding, strict=True)
            ),
        )

    def _passes_belief_filter(
        self,
        believer: str | None,
        beliefs: Collection[BeliefState],
        context: Mapping[str, Any] | None,
    ) -> bool:
        """Whether the element is read: not rejected, or kept for *believer*."""
        if believer is None:
            return not self.is_rejected
        state = _belief_state(self.provenance or (), believer, context)
        return state in check_belief_filter(beliefs)

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
            fields["provenance"] = [entry.to_dict() for entry in self.provenance]
        return fields

    def _renamed(self, renamed: Mapping[str, str]) -> Self:
        """Return the element naming the new ID of every element renamed."""
        return self


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
            without place. They carry no order and are held and written in
            the canonical one, whatever order they are given in: the
            destination without place, then those with ``by``, by value,
            then those with ``to``, by ID.
        name: A display name.
        label: A human-readable label.
        meta: Additional metadata.
        provenance: How the Jump was made and who believes it: provenance
            entries, or None.
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
        object.__setattr__(self, "targets", tuple(sorted(bound, key=_by_place)))

    def __repr__(self) -> str:
        count = len(self.targets)
        return (
            f"Jump({self.id!r} @ {self.coordinate}, "
            f"{count} destination{'' if count == 1 else 's'})"
        )

    @property
    def manifestations(self) -> tuple[FlowManifestation | str, ...]:
        """What the Jump looks like in the source: its destinations' manifestations."""
        return tuple(
            destination.manifestation
            for destination in self.targets
            if destination.manifestation is not None
        )

    def _own_fields(self) -> dict[str, Any]:
        return {
            "targets": [
                destination.to_dict(minimal=False) for destination in self.targets
            ]
        }

    def _renamed(self, renamed: Mapping[str, str]) -> Self:
        targets = tuple(destination._renamed(renamed) for destination in self.targets)
        if all(new is old for new, old in zip(targets, self.targets, strict=True)):
            return self
        return replace(self, targets=targets)


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

    def __repr__(self) -> str:
        shown = (
            "" if self.manifestation is None else f", {_written(self.manifestation)}"
        )
        return f"{type(self).__name__}({self.id!r}{shown} @ {self.coordinate})"

    def _own_fields(self) -> dict[str, Any]:
        if self.manifestation is None:
            return {}
        return {"manifestation": _written(self.manifestation)}


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
        provenance: How the element was made and who believes it:
            provenance entries, or None.
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
        provenance: How the element was made and who believes it:
            provenance entries, or None.
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
        provenance: How the element was made and who believes it:
            provenance entries, or None.
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
        provenance: How the element was made and who believes it:
            provenance entries, or None.
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


def _left_out(left_out: Mapping[str, str], name: str) -> str:
    """Words saying that *name* is an element held but not read, if it is."""
    return f": {left_out[name]}" if name in left_out else ""


def _canonical(
    elements: Iterable[FlowControlElement],
) -> tuple[FlowControlElement, ...]:
    """Sort elements by coordinate, then by ID."""
    return tuple(
        sorted(elements, key=lambda element: (element.coordinate.value, element.id))
    )


def _off_axis(value: Any, length: Coordinate | None) -> str | None:
    """Words saying where *value* lies off an axis of *length*, if it does.

    The axis runs from 0 to its length, both included; from 0 on where the
    length is unknown (None).
    """
    if value < 0:
        return "before 0"
    if length is not None and value > length.value:
        return f"beyond the length {length}"
    return None


class FlowControlView:
    """Flow-control elements on one axis, read only, in canonical order.

    A view holds :class:`FlowControlElement` objects, ordered by coordinate
    and then by ID, and reads them as flow control: the views by class
    (``jumps``, ``jump_targets``, ``breaks``, ``section_breaks``,
    ``barlines``), the Jump at a coordinate, the place a destination goes
    on at, the elements a reading keeps. Nothing changes what a view holds.
    :class:`FlowControlRegistry`, the flow control a timeline states, is the
    view that elements are added to; :attr:`Timeline.tree_flow_control
    <timetoalign.timelines.Timeline.tree_flow_control>` is the view of a
    timeline's flow control together with that of its descendants, which
    is what playing the timeline reads.

    A view refuses, as a registry does, what cannot be played among the
    elements not rejected: a second Jump at a coordinate, a ``to`` naming
    neither a :class:`JumpTarget` held nor a place the view knows, an
    ``after`` that does not name exactly one destination of another Jump
    held, an element or a place before 0. Two elements under one ID are
    refused too.

    Args:
        elements: The elements, held together.
        unit: The unit of the axis, if any.
        number_type: The number type of that axis, if any.

    Attributes:
        unit: The unit of the axis, if any.
        number_type: The number type of that axis, if any.

    Raises:
        ValueError: If the elements cannot be played together, or two of
            them share an ID.

    Examples:
        >>> from timetoalign.core import Coordinate
        >>> view = FlowControlView(
        ...     [Jump(Coordinate(8, "quarters"), [JumpDestination(repeat_count=1)])],
        ...     unit="quarters",
        ... )
        >>> view.jumps
        (Jump('jump_1' @ 8 quarters, 1 destination),)
        >>> view.place_of(view.jumps[0].targets[0], view.jumps[0])
        Coordinate(Fraction(0, 1), quarters)
    """

    def __init__(
        self,
        elements: Iterable[FlowControlElement] = (),
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
    ) -> None:
        self.unit: TimeUnit | None = None if unit is None else TimeUnit(unit)
        self.number_type: NumberType | None = (
            None if number_type is None else NumberType(number_type)
        )
        self._elements: tuple[FlowControlElement, ...] = ()
        # The places a destination may name beside the jump targets held (set
        # by the owning timeline, or by the tree a view gathers), and, by
        # element ID, the start on this axis of the timeline stating it.
        self._places: Callable[[], Mapping[str, Coordinate]] | None = None
        self._origins: Mapping[str, Coordinate] = {}
        self._hold(elements)

    @classmethod
    def _gathered(
        cls,
        elements: Iterable[FlowControlElement],
        *,
        unit: TimeUnit,
        number_type: NumberType,
        places: Mapping[str, Coordinate],
        origins: Mapping[str, Coordinate],
    ) -> Self:
        """Return the view of *elements* gathered from a tree onto one axis.

        Args:
            elements: The elements, each at its coordinate on the axis.
            unit: The axis's unit.
            number_type: The axis's number type.
            places: The places of the tree a destination may name, on the
                axis.
            origins: For each element's ID, the start on the axis of the
                timeline stating it.

        Raises:
            ValueError: As the constructor.
        """
        view = cls(unit=unit, number_type=number_type)
        table = dict(places)
        view._places = lambda: table
        view._origins = dict(origins)
        view._hold(elements)
        return view

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FlowControlView) or type(other) is not type(self):
            return NotImplemented
        return (self._elements, self.unit, self.number_type) == (
            other._elements,
            other.unit,
            other.number_type,
        )

    def __repr__(self) -> str:
        counts = ", ".join(
            f"{len(self._of(element_class))} {element_class.role}s"
            for element_class in _ROLES.values()
        )
        return f"{type(self).__name__}({counts})"

    # -- views --------------------------------------------------------------

    @property
    def elements(self) -> tuple[FlowControlElement, ...]:
        """Every element, by coordinate and then by ID."""
        return self._elements

    @property
    def jumps(self) -> tuple[Jump, ...]:
        """The Jumps, by coordinate and then by ID."""
        return self._of(Jump)

    @property
    def jump_targets(self) -> tuple[JumpTarget, ...]:
        """The JumpTargets, by coordinate and then by ID."""
        return self._of(JumpTarget)

    @property
    def breaks(self) -> tuple[Break, ...]:
        """The Breaks, by coordinate and then by ID."""
        return self._of(Break)

    @property
    def section_breaks(self) -> tuple[SectionBreak, ...]:
        """The SectionBreaks, by coordinate and then by ID."""
        return self._of(SectionBreak)

    @property
    def barlines(self) -> tuple[Barline, ...]:
        """The Barlines, by coordinate and then by ID."""
        return self._of(Barline)

    def _of(self, element_class: type[Any]) -> tuple[Any, ...]:
        return tuple(
            element for element in self._elements if type(element) is element_class
        )

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

    def _hold(self, elements: Iterable[FlowControlElement]) -> None:
        """Hold *elements*, refusing what cannot be played among them.

        Raises:
            ValueError: If two elements share an ID, or as
                :meth:`FlowControlRegistry.add_all` refuses elements, but for
                the length, which a view does not know.
        """
        staged = [self._on_axis(element) for element in elements]
        held: set[str] = set()
        for element in staged:
            if element.id in held:
                raise ValueError(
                    f"{element.role} {quote_id(element.id)} is held twice: an ID "
                    "names one flow-control element"
                )
            held.add(element.id)
        standing = [element for element in staged if not element.is_rejected]
        self._check(standing, standing, staged)
        self._check_on_axis(standing, standing, None)
        self._elements = _canonical(staged)

    def _check(
        self,
        new: Sequence[FlowControlElement],
        read: Sequence[FlowControlElement],
        held: Sequence[FlowControlElement] | None = None,
    ) -> None:
        """Refuse *new* elements that cannot be played among the elements *read*.

        *read* holds the elements the rules see (*new* among them), *held*
        every element of the registry (its own when None): a name found
        among *held* only is reported as naming an element that is rejected,
        or that the belief filter of the reading leaves out.

        Raises:
            ValueError: As :meth:`add_all`, but for the axis
                (:meth:`_check_on_axis`).
        """
        jumps = {element.id: element for element in read if isinstance(element, Jump)}
        targets = {element.id for element in read if isinstance(element, JumpTarget)}
        left_out = {
            element.id: f"{element.role} {quote_id(element.id)} "
            + (
                "is rejected"
                if element.is_rejected
                else "is left out by the belief filter"
            )
            for element in (self._elements if held is None else held)
        }
        for element in read:
            left_out.pop(element.id, None)
        places: Mapping[str, Coordinate] | None = None
        for element in new:
            what = f"{element.role} {quote_id(element.id)}"
            if isinstance(element, Jump):
                for other in jumps.values():
                    if (
                        other is not element
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
                                f"of its timeline{_left_out(left_out, destination.to)}. "
                                f"Available jump targets: {sorted(targets)}. "
                                f"Available places: {list(places)}"
                            )
                    if destination.after is not None:
                        self._destination_among(
                            jumps,
                            destination.after,
                            what,
                            holder=element.id,
                            left_out=left_out,
                        )
            elif isinstance(element, Break) and element.ends is not None:
                if element.ends.after is not None:
                    self._destination_among(
                        jumps, element.ends.after, what, left_out=left_out
                    )

    def _check_on_axis(
        self,
        new: Sequence[FlowControlElement],
        read: Sequence[FlowControlElement],
        length: Coordinate | None,
    ) -> None:
        """Refuse *new* elements that lie, or go on, outside an axis of *length*.

        The coordinate of an element and the place of every destination of
        a Jump lie from 0 to *length*, both included; from 0 on where the
        length is unknown (None). Places are found among the elements
        *read* (:meth:`place_of`), which :meth:`_check` has admitted.

        Raises:
            ValueError: Naming the element, and the place where it is one.
        """
        for element in new:
            what = f"{element.role} {quote_id(element.id)}"
            where = _off_axis(element.coordinate.value, length)
            if where is not None:
                raise ValueError(
                    f"{what} lies at {element.coordinate}, {where}: a flow-control "
                    "element lies from 0 to the length of its timeline"
                )
            if not isinstance(element, Jump):
                continue
            for destination in element.targets:
                if destination.place is None:
                    continue
                place = self.place_of(destination, element, elements=read)
                where = _off_axis(place.value, length)
                if where is None:
                    continue
                named = (
                    f"by {destination.by}, at {place}"
                    if destination.to is None
                    else f"at {quote_id(destination.to)}, which lies at {place}"
                )
                raise ValueError(
                    f"{what} at {element.coordinate} goes on {named}, {where}: a "
                    "place lies from 0 to the length of the timeline"
                )

    @staticmethod
    def _destination_among(
        jumps: Mapping[str, Jump],
        reference: DestinationReference,
        what: str,
        *,
        holder: str | None = None,
        left_out: Mapping[str, str] | None = None,
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
                f"no jump of this registry{_left_out(left_out or {}, reference.jump)}. "
                f"Available jumps: {list(jumps)}"
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
        rejected is returned; a rejected one where no other lies.

        Args:
            coordinate: A position on the registry's axis: a bare value, a
                :class:`~timetoalign.core.Coordinate` or an
                :class:`~timetoalign.core.IdCoordinate` in the registry's
                unit, read in the axis's number type.

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
        timeline gives (the start of a Region or of a child timeline, or the
        timeline's own start; in a view of a tree, of any timeline of the
        tree). A destination without place goes on at the nearest section
        break among *elements* that lies before the Jump, in a view of a
        tree stated by any timeline of it, but never before the start of
        the timeline stating the Jump, which counts as a section boundary
        for it; that start where no section break lies from it up to the
        Jump. The start is 0 in a registry and the timeline's place on the
        axis in a view of a tree: a child at offset 4 stating a repeat sign
        without place at 8 goes back to 4 when its root is played.

        Args:
            destination: The destination.
            jump: The Jump holding it.
            elements: The elements to find jump targets and section breaks
                among; the elements not rejected (:meth:`active_elements`)
                when None.

        Raises:
            KeyError: If ``to`` names nothing, or a rejected jump target.
        """
        among = self.active_elements() if elements is None else elements
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
        start = self._origins.get(jump.id)
        if start is None:
            start = Coordinate(0, axis.unit, number_type=axis.number_type)
        before = [
            element.coordinate
            for element in among
            if isinstance(element, SectionBreak)
            and start.value <= element.coordinate.value < axis.value
        ]
        if before:
            return max(before, key=lambda coordinate: coordinate.value)
        return start

    def active_elements(
        self,
        *,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
        context: Mapping[str, Any] | None = None,
    ) -> list[FlowControlElement]:
        """Return the elements that are read as flow control, in canonical order.

        Without a believer these are the elements not rejected: every
        element but those some actor believes ``"false"``
        (:attr:`FlowControlElement.is_rejected`). A believer narrows the
        reading to that actor's beliefs: the belief filter of
        :func:`~timetoalign.core.provenance.belief_mask`, read on each
        element's provenance, where an element without provenance, or
        without a belief of the believer's, is one the believer has not
        decided on, and one the believer holds two beliefs in (in two
        entries) is read as the more critical of them.

        Args:
            believer: The actor whose beliefs alone decide; None leaves out
                what any actor rejects.
            beliefs: With a believer, the belief states kept; every state
                but ``"false"`` by default.
            context: The JSON-LD context under which actors compare as IRIs.

        Raises:
            ValueError: If *beliefs* is no collection of belief states.

        Examples:
            >>> from timetoalign.core import Coordinate
            >>> registry = FlowControlRegistry(unit="quarters")
            >>> barline = registry.add(Barline(Coordinate(8, "quarters")))
            >>> _ = registry.with_belief(barline.id, "ana", "false", date="2026-11-25")
            >>> registry.elements, registry.active_elements()
            ((Barline('barline_1' @ 8 quarters),), [])
            >>> registry.active_elements(believer="bea")
            [Barline('barline_1' @ 8 quarters)]
        """
        check_belief_filter(beliefs)
        return [
            element
            for element in self._elements
            if element._passes_belief_filter(believer, beliefs, context)
        ]

    def has_flow_control(self) -> bool:
        """Whether the registry holds any element."""
        return bool(self._elements)

    def to_dict(self, *, minimal: bool = True) -> list[dict[str, Any]]:
        """Return the elements as the ``flow_control`` array.

        Each element is written as its own object, in canonical order
        whatever order the elements were added in: by coordinate, then by
        ID, values compared exactly and IDs by Unicode code point. Nothing
        is written that the registry does not hold.

        Args:
            minimal: Write each element in the minimal form (the default):
                every key at its default value left out, in the
                destinations of a Jump too. With False, the complete form:
                ``name``, ``label`` and ``meta`` of every element and
                ``determined``, ``reset`` and ``label`` of every
                destination are written whatever they hold. A key without
                a default (``manifestation``, ``ends``, ``provenance``,
                ``to``, ``by``, ``passes``, ``repeat_count``, ``after``) is
                written only when stated.
        """
        return [element.to_dict(minimal=minimal) for element in self._elements]


class FlowControlRegistry(FlowControlView):
    """The flow control of one timeline: its elements in canonical order.

    One collection holds every :class:`FlowControlElement`, ordered by
    coordinate and then by ID; ``jumps``, ``jump_targets``, ``breaks``,
    ``section_breaks`` and ``barlines`` are views of it by class. All of
    them are tuples: the collection changes through :meth:`add`,
    :meth:`add_all`, :meth:`with_belief` and :meth:`with_beliefs` alone,
    and no element ever leaves it. Every element carries an ID unique
    within the registry and, once the registry belongs to a timeline,
    within that timeline's whole manifest: an element whose minted ID is
    taken is stored under a freshly minted ID, and one whose given ID is
    taken is refused. A manifest writes the registry as the timeline's
    ``flow_control`` array.

    The registry refuses what cannot be played: a second Jump at a
    coordinate, or, once the registry belongs to a timeline, at a position
    of the root's axis that a Jump of the root's tree holds (the
    coordinate plus the offsets of the timeline and its ancestors, summed
    exactly), a ``to`` naming neither a :class:`JumpTarget` of the
    registry nor a place of its timeline, an ``after`` that does not name
    exactly one destination of another Jump of the registry, and an element
    or a destination's place lying outside the axis, which runs from 0 to
    the timeline's length (from 0 on while that length is unknown, and for
    a registry that belongs to no timeline).

    An element that some actor believes ``"false"`` is rejected
    (:attr:`FlowControlElement.is_rejected`, :meth:`with_belief`): it stays
    in the registry and is written, but it is out of every rule, as what is
    checked and as what is named. A rejected Jump and its replacement may
    therefore share a coordinate. An element that another element not
    rejected names (by ``to`` or ``after``) is not rejected alone: the
    elements naming it are rejected in the same step
    (:meth:`with_beliefs`).

    Args:
        elements: The elements the registry starts with, added together.
        unit: The unit of the axis the registry is bound to, if any.
        number_type: The number type of that axis, if any.

    Attributes:
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

    def __init__(
        self,
        elements: Iterable[FlowControlElement] = (),
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
    ) -> None:
        super().__init__(unit=unit, number_type=number_type)
        # Set by the owning timeline: admits IDs into its whole manifest,
        # gives the timeline's length (its places are the view's) and refuses
        # Jumps taking a position of its root's axis over the root's tree.
        self._admit: Callable[..., list[str]] | None = None
        self._length: Callable[[], Coordinate | None] | None = None
        self._refuse_jumps: Callable[..., None] | None = None
        self.add_all(elements)

    def _known_length(self) -> Coordinate | None:
        """The length of the registry's timeline, None when unknown or unowned."""
        return None if self._length is None else self._length()

    # -- adding -------------------------------------------------------------

    def add(self, element: FlowControlElement) -> FlowControlElement:
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
                is not the registry's, or an element that is not rejected
                cannot be played among those not rejected: a coordinate
                holds a second Jump, or a Jump lies at a position of the
                root's axis where a Jump of another timeline of the root's
                tree lies (:attr:`Timeline.tree_flow_control
                <timetoalign.timelines.Timeline.tree_flow_control>`), a
                ``to`` names neither a JumpTarget of the registry nor a
                place of its timeline, an ``after`` does
                not name exactly one destination of another Jump of the
                registry, or the element's coordinate or the place of one
                of its destinations lies before 0 or beyond the known
                length of the registry's timeline. Nothing has changed then.
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
        joined = [*self._elements, *stored]
        standing = [element for element in stored if not element.is_rejected]
        read = [element for element in joined if not element.is_rejected]
        self._check(standing, read, joined)
        self._check_on_axis(standing, read, self._known_length())
        if self._refuse_jumps is not None:
            self._refuse_jumps(standing, replaced=False)
        self._admitted_ids(entries, check_only=False)
        self._elements = _canonical(joined)
        return stored

    def _admitted_ids(self, entries: list[_Entry], *, check_only: bool) -> list[str]:
        """Return the ID each joining element is stored under.

        Raises:
            ValueError: If a given ID is already taken.
        """
        if self._admit is not None:
            return self._admit(entries, check_only=check_only)
        taken = {held.id: f"{held.role} {quote_id(held.id)}" for held in self._elements}
        plan = _plan_admissions(entries, taken)
        return [plan.get(index, entry[0]) for index, entry in enumerate(entries)]

    def _check_length(self, length: Coordinate) -> None:
        """Refuse a length of the timeline that leaves flow control outside it.

        Raises:
            ValueError: If an element that is not rejected, or the place of
                one of its destinations, lies beyond *length*.
        """
        read = [element for element in self._elements if not element.is_rejected]
        if read:
            self._check_on_axis(read, read, self._coordinate(length))

    # -- beliefs ------------------------------------------------------------

    def with_belief(
        self,
        element_id: str,
        actor: str,
        belief: BeliefValue,
        *,
        date: str | datetime.date | None = None,
        note: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> FlowControlElement:
        """Record *actor*'s belief in the element *element_id*.

        The registry holds the element's copy whose provenance records the
        belief (:meth:`FlowControlElement.with_belief`) in its place; the
        element's ID, coordinate and every other key stay. The belief
        ``"false"`` rejects the element, which is how flow control is
        repaired: the wrong element is rejected and another one added,
        never changed or removed.

        What names a rejected element would name nothing. An element that
        another element not rejected names, by a ``to`` or an ``after``, is
        therefore not rejected alone: :meth:`with_beliefs` rejects it
        together with the elements naming it, and their replacements are
        added afterwards.

        Args:
            element_id: The element's ID.
            actor: Who holds the belief, an IRI.
            belief: ``"true"``, ``"false"`` or ``"unlikely"``.
            date: The day, ``YYYY-MM-DD`` or a :class:`datetime.date`; today
                when None.
            note: A remark.
            context: The JSON-LD context under which actors compare as IRIs;
                None declares no prefix.

        Returns:
            The element as the registry holds it now.

        Raises:
            KeyError: If the registry holds no element *element_id*.
            ValueError: As :meth:`with_beliefs`. Nothing has changed then.

        Examples:
            >>> from timetoalign.core import Coordinate
            >>> registry = FlowControlRegistry(unit="quarters")
            >>> wrong = registry.add(
            ...     Jump(Coordinate(16, "quarters"), [JumpDestination(repeat_count=1)]))
            >>> registry.with_belief(wrong.id, "ana", "false").is_rejected
            True
            >>> registry.add(
            ...     Jump(Coordinate(16, "quarters"), [JumpDestination(by=-8, repeat_count=1)])
            ... ).id
            'jump_2'
            >>> registry.jump_at(16).id
            'jump_2'
        """
        return self.with_beliefs(
            [element_id], actor, belief, date=date, note=note, context=context
        )[0]

    def with_beliefs(
        self,
        element_ids: Iterable[str],
        actor: str,
        belief: BeliefValue,
        *,
        date: str | datetime.date | None = None,
        note: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> list[FlowControlElement]:
        """Record *actor*'s belief in several elements in one step, or in none.

        As :meth:`with_belief` for each element, the result checked as a
        whole: with the belief ``"false"``, an element and the elements
        naming it are rejected together, which neither could be alone.

        Args:
            element_ids: The elements' IDs.
            actor: Who holds the belief, an IRI.
            belief: ``"true"``, ``"false"`` or ``"unlikely"``.
            date: The day, ``YYYY-MM-DD`` or a :class:`datetime.date`; today
                when None.
            note: A remark.
            context: The JSON-LD context under which actors compare as IRIs;
                None declares no prefix.

        Returns:
            The elements as the registry holds them now, in the order given.

        Raises:
            KeyError: If the registry holds no element of one of the IDs.
            ValueError: If the belief or the date is invalid; an element
                rejected by the step is named, by a ``to`` or an ``after``,
                by an element that is not rejected after it; or an element
                no longer rejected by the step cannot be played among the
                elements not rejected (:meth:`add_all`). Nothing has changed
                then.

        Examples:
            >>> from timetoalign.core import Coordinate
            >>> registry = FlowControlRegistry(unit="quarters")
            >>> segno, leap = registry.add_all([
            ...     JumpTarget(Coordinate(4, "quarters"), id="segno"),
            ...     Jump(Coordinate(16, "quarters"), [JumpDestination(to="segno")], id="ds"),
            ... ])
            >>> registry.with_belief("segno", "ana", "false")
            Traceback (most recent call last):
            ...
            ValueError: jump_target 'segno' is not rejected alone: jump 'ds' names it. \
Reject the elements naming it in the same step (with_beliefs), then add their replacements
            >>> [e.is_rejected for e in registry.with_beliefs(["segno", "ds"], "ana", "false")]
            [True, True]
        """
        if isinstance(element_ids, str):
            raise TypeError(
                f"with_beliefs takes the IDs of several elements, not the string "
                f"{element_ids!r}: with_belief takes one"
            )
        wanted = list(element_ids)
        held = Belief(actor=actor, belief=belief, date=_day(date), note=note)
        index_of = {element.id: index for index, element in enumerate(self._elements)}
        for element_id in wanted:
            if element_id not in index_of:
                raise KeyError(
                    f"{element_id!r} names no flow-control element. Available "
                    f"elements: {list(index_of)}"
                )
        elements = list(self._elements)
        for element_id in dict.fromkeys(wanted):
            index = index_of[element_id]
            elements[index] = elements[index].with_belief(held, context=context)
        changed = [
            elements[index_of[element_id]] for element_id in dict.fromkeys(wanted)
        ]
        read = [element for element in elements if not element.is_rejected]
        self._refuse_named(
            [
                element
                for element in changed
                if element.is_rejected
                and not self._elements[index_of[element.id]].is_rejected
            ],
            read,
        )
        standing = [element for element in changed if not element.is_rejected]
        self._check(standing, read, elements)
        self._check_on_axis(standing, read, self._known_length())
        if self._refuse_jumps is not None:
            self._refuse_jumps(standing, replaced=True)
        self._elements = tuple(elements)
        return [elements[index_of[element_id]] for element_id in wanted]

    @staticmethod
    def _refuse_named(
        rejected: Sequence[FlowControlElement], read: Sequence[FlowControlElement]
    ) -> None:
        """Refuse rejecting an element that one of the elements *read* names.

        Raises:
            ValueError: Naming the first such element of *rejected* and
                every element naming it.
        """
        for element in rejected:
            naming: list[str] = []
            for other in read:
                names: list[str | None] = []
                if isinstance(other, Jump):
                    for destination in other.targets:
                        names.append(destination.to)
                        if destination.after is not None:
                            names.append(destination.after.jump)
                elif isinstance(other, Break) and other.ends is not None:
                    if other.ends.after is not None:
                        names.append(other.ends.after.jump)
                if element.id in names:
                    naming.append(f"{other.role} {quote_id(other.id)}")
            if naming:
                verb = "names" if len(naming) == 1 else "name"
                raise ValueError(
                    f"{element.role} {quote_id(element.id)} is not rejected alone: "
                    f"{', '.join(naming)} {verb} it. Reject the elements naming it "
                    "in the same step (with_beliefs), then add their replacements"
                )

    # -- maintenance --------------------------------------------------------

    def _reassign_id(self, old_id: str, new_id: str) -> None:
        """Store the element with ID *old_id* under the minted ID *new_id*.

        Every destination and condition naming it follows.
        """
        self._elements = tuple(
            (
                replace(element, id=new_id, _minted=True)
                if element.id == old_id
                else element
            )
            for element in self._elements
        )
        self._rename_references({old_id: new_id})

    def _rename_references(self, renamed: Mapping[str, str | None]) -> None:
        """Name the new ID of every renamed element or place; keep the others."""
        names = {old: new for old, new in renamed.items() if new is not None}
        if names:
            self._elements = _canonical(
                element._renamed(names) for element in self._elements
            )

    def _absorb(self, other: FlowControlRegistry) -> None:
        """Add every element of *other* together."""
        self.add_all(other.elements)

    def copy(self) -> FlowControlRegistry:
        """Return an independent registry with the same binding and content.

        The copy belongs to no timeline until one adopts it.
        """
        owner = (self._admit, self._places, self._length, self._refuse_jumps)
        self._admit = self._places = self._length = self._refuse_jumps = None
        try:
            return deepcopy(self)
        finally:
            self._admit, self._places, self._length, self._refuse_jumps = owner


# endregion
