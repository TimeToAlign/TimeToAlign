"""Reading and writing TimeSkeleton manifests.

A TimeSkeleton manifest is one JSON object: the version marker
``timeskeleton``, an array of root ``timelines`` and, optionally, the
manifest's ``id``, ``name`` and ``meta``, its ``groups``, its
``measure_maps`` and its ``graphical``. :class:`~timetoalign.timelines.Timeline`,
:class:`~timetoalign.timelines.TimelineGroup` and
:class:`~timetoalign.alignment.AlignmentBundle` each write one with
``to_dict`` and read one with ``from_dict``. This module holds what they
share: the version marker, the reading of the manifest around its timelines
and groups, the writing of that envelope and of the measure maps its
timelines name, and the codec of a bare value on a timeline's own axis.

A value on a timeline's own axis (``length``, ``offset``, region ``start``
and ``end``, break and marker ``coordinate``, jump ``from_coordinate`` and
``to_coordinate``) is written bare, represented as the axis's number type
selects: a JSON integer for ``int``, a JSON number for ``float``, a
``[numerator, denominator]`` pair for ``fraction``. Reading follows the
number type, never the lexical form: an integral JSON number on an ``int``
axis is that integer, an integer on a ``float`` axis that float, and a pair
with integral float members (``[1, 8.0]``) that Fraction. A value no reading
of the number type admits is refused.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any, ClassVar

from timetoalign.core import Coordinate, NumberType, TimeUnit
from timetoalign.core.ids import _OpenObject, mint_id, validate_id
from timetoalign.core.time import is_fraction_json, number_from_json, number_to_json
from timetoalign.maps.base import MEASURE_MAP_ID_PREFIX

if TYPE_CHECKING:
    from .base import Timeline
    from .structure import MeasureMap

TIMESKELETON_VERSION = "0.7.0"
"""The TimeSkeleton version the library reads and writes."""

_STYLING_DEFAULTS: dict[str, Any] = {"styles": {}, "statements": []}
"""The members of ``graphical`` other than ``order``, with their defaults."""


# region Bare values on a timeline's own axis


def axis_value_to_json(coordinate: Coordinate) -> int | float | list[int]:
    """Write a coordinate on its timeline's own axis as a bare value."""
    return number_to_json(coordinate.value)


def _json_kind(data: Any) -> str | None:
    """Name the number representation of *data*, or None if it is none."""
    if isinstance(data, bool):
        return None
    if isinstance(data, int):
        return "an int"
    if isinstance(data, float):
        return "a float"
    if is_fraction_json(data):
        return "a fraction"
    return None


_AXIS_KIND = {
    NumberType.int: "an int",
    NumberType.float: "a float",
    NumberType.fraction: "a fraction",
}


def _suggestion(data: Any, number_type: NumberType) -> str:
    """What to write instead of *data* on an axis of *number_type*.

    The rewritten value when the rewriting is exact, otherwise the axis's
    number type with its article.
    """
    value = Fraction(number_from_json(data))
    if number_type is NumberType.fraction:
        return json.dumps(number_to_json(value))
    if number_type is NumberType.float:
        written = float(value)
        return repr(written) if Fraction(written) == value else _AXIS_KIND[number_type]
    if value.denominator == 1:
        return str(int(value))
    return _AXIS_KIND[number_type]


def axis_value_from_json(
    data: Any,
    *,
    key: str,
    owner: str,
    unit: TimeUnit,
    number_type: NumberType,
) -> Coordinate:
    """Read a bare value on a timeline's own axis.

    The axis's number type decides how the value is read
    (:func:`~timetoalign.core.time.number_from_json`): an integral JSON
    number is read on an ``int`` axis, an integer on a ``float`` axis, a pair
    with integral float members on a ``fraction`` axis. A non-integral value
    on an ``int`` axis, a pair on an ``int`` or ``float`` axis and a bare
    number on a ``fraction`` axis are refused.

    Args:
        data: The JSON value.
        key: The key the value was found under, for the error.
        owner: The id of the object holding the key, for the error.
        unit: The axis's unit.
        number_type: The axis's number type.

    Returns:
        The coordinate on the axis.

    Raises:
        ValueError: If *data* is no number, or no reading of the axis's
            number type admits it, for example ``length 32 of
            'score' is an int on a fraction axis: write [32, 1]``; the
            rewritten value is named only when the rewriting is exact,
            otherwise the axis's number type (``write a float``).
    """
    kind = _json_kind(data)
    if kind is None:
        raise ValueError(f"{key} {json.dumps(data)} of '{owner}' is not a number")
    if isinstance(data, float) and not math.isfinite(data):
        raise ValueError(f"{key} {data!r} of '{owner}' is not finite")
    try:
        value = number_from_json(data, number_type)
    except ValueError as error:
        if "int64" in str(error):
            raise ValueError(f"{key} of '{owner}': {error}") from error
        raise ValueError(
            f"{key} {json.dumps(data)} of '{owner}' is {kind} on "
            f"{_AXIS_KIND[number_type]} axis: write {_suggestion(data, number_type)}"
        ) from error
    return Coordinate(value, unit, number_type=number_type)


def spell(value: Any) -> str:
    """Spell a JSON value as a message quotes it.

    A number as JSON writes it, an array as its items separated by ``, ``
    within brackets (``[32, 1]``), anything else as compact JSON: the
    spelling of the TimeSkeleton validators.
    """
    if isinstance(value, list):
        return "[" + ", ".join(spell(item) for item in value) + "]"
    if isinstance(value, bool) or not isinstance(value, int | float):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return repr(value)


def _spelled_on(value: Fraction, number_type: NumberType) -> str:
    """Spell *value* as an axis of *number_type* writes it, as a pair if inexact."""
    if number_type is NumberType.int and value.denominator == 1:
        return spell(value.numerator)
    if number_type is NumberType.float and Fraction(float(value)) == value:
        return spell(float(value))
    return spell([value.numerator, value.denominator])


def check_children_fit(data: Mapping[str, Any], number_type: NumberType) -> None:
    """Refuse a child beyond the known ``length`` of the timeline object *data*.

    Child by child in array order: an ``offset`` beyond the length, or else an
    end beyond it where the end is known (offset plus length, or, in a
    segment line, a segment of unknown length's successor's offset). A child
    of another unit is left to the reader, which refuses it. The messages
    are the TimeSkeleton validators', values spelled as written.

    Raises:
        ValueError: ``offset 75.0 of 'take' is beyond the length 60.0 of its
            parent 'recording'``, or ``child 'coda' ends at [12, 1], beyond
            the length [10, 1] of its parent 'score'``.
    """
    length = data["length"]
    if length is None:
        return
    bound = Fraction(number_from_json(length))
    children = [
        child for child in data.get("children", []) if child["unit"] == data["unit"]
    ]
    offsets = [Fraction(number_from_json(child["offset"])) for child in children]
    ends: list[Fraction | None] = [
        (
            None
            if child["length"] is None
            else offset + Fraction(number_from_json(child["length"]))
        )
        for child, offset in zip(children, offsets, strict=True)
    ]
    if data.get("is_segment_line") is True:
        order = sorted(range(len(children)), key=lambda index: offsets[index])
        for index, successor in zip(order, order[1:]):
            if ends[index] is None:
                ends[index] = offsets[successor]
    parent = data["id"]
    for child, offset, end in zip(children, offsets, ends, strict=True):
        if offset > bound:
            raise ValueError(
                f"offset {spell(child['offset'])} of '{child['id']}' is beyond the "
                f"length {spell(length)} of its parent '{parent}'"
            )
        if end is not None and end > bound:
            raise ValueError(
                f"child '{child['id']}' ends at {_spelled_on(end, number_type)}, "
                f"beyond the length {spell(length)} of its parent '{parent}'"
            )


# endregion


# region Objects


def require_object(
    data: Any, what: str, required: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Return *data* if it is a JSON object holding every *required* key.

    Raises:
        ValueError: If *data* is not an object or lacks a required key.
    """
    if not isinstance(data, dict):
        raise ValueError(f"{what} must be an object, got {type(data).__name__}")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{what} lacks the key(s) {missing}")
    return data


def read_name(data: Mapping[str, Any], what: str) -> str | None:
    """Read an optional ``name``: a string, or null (the default).

    Raises:
        ValueError: If the name is neither.
    """
    name = data.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError(f"{what} has a non-string name {name!r}")
    return name


def read_meta(data: Mapping[str, Any], what: str) -> dict[str, Any]:
    """Read an optional ``meta``: an object, ``{}`` by default.

    Raises:
        ValueError: If the meta is not an object.
    """
    meta = data.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError(f"{what} has a non-object meta")
    return copy.deepcopy(meta)


def read_id(value: Any, what: str) -> str:
    """Read a given id.

    Raises:
        ValueError: If the id is not a string or not a valid id.
    """
    if not isinstance(value, str):
        raise ValueError(f"{what} id must be a string, got {value!r}")
    return validate_id(value)


def read_list(data: Mapping[str, Any], key: str, what: str) -> list[Any]:
    """Read an optional array, ``[]`` by default.

    Raises:
        ValueError: If the value is not an array.
    """
    value = data.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{what}'s {key!r} must be an array")
    return value


# endregion


# region The manifest


@dataclass(eq=False)
class Manifest(_OpenObject):
    """What a manifest holds beside its timelines and groups.

    The manifest's ``id`` (absent when it has none), ``name`` and ``meta``, the
    members of its ``graphical`` other than ``order`` (``styling``) and the
    band orders of its timelines (``band_orders``), both kept as given and
    never interpreted, whether it holds ``graphical`` at all, the measure
    maps no timeline names, and every key the library does not read.

    Attributes:
        id: The manifest id, or None when it has none (an anonymous bundle,
            or one named by ``name`` alone).
        name: The manifest name, or None.
        meta: The manifest's free metadata.
        styling: ``graphical`` without ``order``: ``styles``,
            ``statements`` and any other key, kept as given.
        band_orders: The entries of ``graphical.order`` keyed by a
            timeline's id, each that timeline's band order, kept as given.
        has_graphical: Whether the manifest holds a ``graphical`` object,
            which is then written back even with every member at its
            default.
        measure_maps: The entries of ``measure_maps`` no timeline names,
            written back with the measure maps the timelines name.
    """

    _dict_keys: ClassVar[tuple[str, ...]] = (
        "timeskeleton",
        "id",
        "name",
        "meta",
        "timelines",
        "groups",
        "measure_maps",
        "graphical",
    )
    _dict_defaults: ClassVar[Mapping[str, Any]] = {
        "name": None,
        "meta": {},
        "groups": [],
        "measure_maps": [],
    }

    id: str | None = None
    name: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    styling: dict[str, Any] = field(default_factory=dict)
    band_orders: dict[str, Any] = field(default_factory=dict)
    has_graphical: bool = False
    measure_maps: list[MeasureMap] = field(default_factory=list)

    @classmethod
    def read(cls, data: Any) -> tuple[Manifest, ManifestContent]:
        """Read a manifest: its envelope, its timelines and its raw groups.

        The measure maps are read first, then every root timeline is
        rebuilt, its conversion maps reading the measure maps they name, and
        every timeline of the roots' trees is indexed by id. Groups are left
        as JSON for the class reading them. ``graphical.order`` is split by
        its keys: a group's display order goes to the content, a timeline's
        band order to the envelope, unchecked. The measure maps no timeline
        names are kept on the envelope.

        Returns:
            The envelope, and the manifest's content.

        Raises:
            ValueError: If *data* is no TimeSkeleton manifest (``"not a
                TimeSkeleton manifest"``), names another version, or is
                malformed: a timeline or measure-map id occurring twice, the
                manifest id equal to a timeline or group id, a measure-map id
                equal to a timeline, group or manifest id, a
                ``graphical.order`` that is no object or names no group or
                timeline.
        """
        if not isinstance(data, dict) or "timeskeleton" not in data:
            raise ValueError("not a TimeSkeleton manifest")
        version = data["timeskeleton"]
        if version != TIMESKELETON_VERSION:
            raise ValueError(
                f"unsupported TimeSkeleton version {version!r}: this library reads "
                f"TimeSkeleton {TIMESKELETON_VERSION} only"
            )
        require_object(data, "a TimeSkeleton manifest", ("timelines",))
        manifest_id = read_id(data["id"], "the manifest") if "id" in data else None
        manifest = cls(
            id=manifest_id,
            name=read_name(data, "the manifest"),
            meta=read_meta(data, "the manifest"),
        )
        manifest._keep_unknown_keys(data)

        measure_maps = _read_measure_maps(
            read_list(data, "measure_maps", "the manifest")
        )
        roots, timelines = _read_roots(data["timelines"], measure_maps)
        groups = read_list(data, "groups", "the manifest")
        group_ids = [
            group.get("id") if isinstance(group, dict) else None for group in groups
        ]
        for measure_map_id in measure_maps:
            if (
                measure_map_id in timelines
                or measure_map_id in group_ids
                or measure_map_id == manifest_id
            ):
                raise ValueError(
                    f"measure map id {measure_map_id!r} is also a timeline, group or "
                    "manifest id"
                )
        named = {
            id(measure_map)
            for timeline in timelines.values()
            for cmap in timeline._all_conversion_maps()
            for measure_map in cmap._measure_maps()
        }
        manifest.measure_maps = [
            measure_map
            for measure_map in measure_maps.values()
            if id(measure_map) not in named
        ]

        order: dict[str, Any] = {}
        if "graphical" in data:
            graphical = require_object(data["graphical"], "'graphical'")
            manifest.has_graphical = True
            order = graphical.get("order", {})
            if not isinstance(order, dict):
                raise ValueError(
                    "'graphical.order' must map group or timeline ids to orders"
                )
            unknown = sorted(
                str(key)
                for key in order
                if key not in group_ids and key not in timelines
            )
            if unknown:
                raise ValueError(
                    f"'graphical.order' names no group or timeline: {unknown}"
                )
            manifest.band_orders = {
                key: copy.deepcopy(entry)
                for key, entry in order.items()
                if key not in group_ids
            }
            order = {key: entry for key, entry in order.items() if key in group_ids}
            # A styling member at its default is left out, so that the complete
            # and the minimal form read to the same styling.
            manifest.styling = {
                key: copy.deepcopy(value)
                for key, value in graphical.items()
                if key != "order" and _STYLING_DEFAULTS.get(key, object()) != value
            }

        if manifest_id is not None and (
            manifest_id in timelines or manifest_id in group_ids
        ):
            raise ValueError(
                f"manifest id {manifest_id!r} is also a timeline or group id"
            )
        return manifest, ManifestContent(roots, timelines, groups, order)

    def write(
        self,
        roots: list[Timeline],
        *,
        groups: list[dict[str, Any]] = (),  # type: ignore[assignment]
        order: dict[str, list[str]] | None = None,
        minimal: bool = False,
        events: bool = False,
        external_references: bool = False,
    ) -> dict[str, Any]:
        """Write the manifest around root timelines and written groups.

        The roots are written sorted by id, each with its
        :attr:`~timetoalign.timelines.Timeline.root_offset` as its
        ``offset`` when it has one. ``measure_maps`` holds every measure map
        the timelines name and every one the manifest kept, sorted by id
        (see :class:`MeasureMapEntries`). ``graphical.order`` holds the
        groups' member orders and, unchanged, the band order of every
        written timeline that has one; the band order of a timeline not
        written, removed or given another id, is left out, so that every
        key names a timeline or group of the manifest. ``graphical`` is
        written when there is an order to keep, styling, or a ``graphical``
        read with the manifest; ``id`` only when the manifest has one.

        Args:
            roots: The root timelines.
            groups: The written groups.
            order: Every group's member order, keyed by group id.
            minimal: Leave out every key at its default.
            events: Write each timeline's event rows.
            external_references: Write each timeline's external reference
                table.

        Raises:
            ValueError: If a measure map cannot be written: two different
                ones carry one given id, or one's given id is a timeline's,
                a group's or the manifest's.
        """
        trees = [
            timeline
            for root in roots
            for timeline in (root, *(child for _, child in root._iter_descendants()))
        ]
        reserved = {timeline.id for timeline in trees}
        reserved.update(group["id"] for group in groups)
        if self.id is not None:
            reserved.add(self.id)
        entries = MeasureMapEntries(
            reserved,
            [
                *self.measure_maps,
                *(
                    measure_map
                    for timeline in trees
                    for cmap in timeline._all_conversion_maps()
                    for measure_map in cmap._measure_maps()
                ),
            ],
        )
        timelines = [
            root._to_timeline_object(
                minimal=minimal,
                events=events,
                external_references=external_references,
                offset=root._root_offset,
                entries=entries,
            )
            for root in sorted(roots, key=lambda root: root.id)
        ]
        for measure_map in self.measure_maps:
            entries.id_of(measure_map)
        fields: dict[str, Any] = {"timeskeleton": TIMESKELETON_VERSION}
        if self.id is not None:
            fields["id"] = self.id
        fields.update(
            name=self.name,
            meta=copy.deepcopy(self.meta),
            timelines=timelines,
            groups=list(groups),
            measure_maps=entries.write(minimal=minimal),
        )
        written = _timeline_ids(timelines)
        band_orders = {
            key: copy.deepcopy(entry)
            for key, entry in self.band_orders.items()
            if key in written
        }
        order = dict(sorted({**band_orders, **(order or {})}.items()))
        if order or self.styling or self.has_graphical:
            graphical: dict[str, Any] = {"order": order}
            for key, default in _STYLING_DEFAULTS.items():
                value = self.styling.get(key, default)
                if not (minimal and value == default):
                    graphical[key] = copy.deepcopy(value)
            if minimal and not order:
                del graphical["order"]
            graphical.update(
                {
                    key: copy.deepcopy(value)
                    for key, value in self.styling.items()
                    if key not in _STYLING_DEFAULTS
                }
            )
            fields["graphical"] = graphical
        return self._dict_form(fields, minimal=minimal)


class MeasureMapEntries:
    """The measure maps a manifest being written holds, each under one id.

    A measure map's id is unique together with the manifest's timeline,
    group and manifest ids. A given id is kept: two different measure maps
    carrying one, or one carrying a timeline's, group's or manifest's id,
    are refused. A minted id is minted again around every id the manifest
    holds, the given ones taken first. Equal measure maps of equal name and
    meta share one entry, so two timelines holding the same measure map, or
    equal ones, name one entry.

    While a timeline's conversion maps are written, :attr:`own` is that
    timeline's measure map, which a measure-derived map names when it
    derives from it.

    Args:
        reserved: The timeline, group and manifest ids of the manifest.
        measure_maps: The measure maps the manifest names, whose given ids
            are taken before any is minted.

    Raises:
        ValueError: If a given id is refused.
    """

    def __init__(
        self, reserved: set[str], measure_maps: list[MeasureMap] = ()  # type: ignore[assignment]
    ) -> None:
        self.own: MeasureMap | None = None
        self._reserved = reserved
        self._entries: dict[str, MeasureMap] = {}
        # By object identity: the id each measure map met is written under,
        # with the map kept alive so that its identity is not reused.
        self._written: dict[int, tuple[MeasureMap, str]] = {}
        for measure_map in measure_maps:
            if not measure_map._id_minted:
                self._admit_given(measure_map)

    def id_of(self, measure_map: MeasureMap) -> str:
        """Return the id under which *measure_map* is written, adding it.

        Raises:
            ValueError: If its given id is refused.
        """
        known = self._written.get(id(measure_map))
        if known is not None:
            return known[1]
        if not measure_map._id_minted:
            return self._admit_given(measure_map)
        for entry_id, entry in self._entries.items():
            if _same_entry(entry, measure_map):
                self._written[id(measure_map)] = (measure_map, entry_id)
                return entry_id
        entry_id = mint_id(MEASURE_MAP_ID_PREFIX, {*self._entries, *self._reserved})
        self._entries[entry_id] = measure_map
        self._written[id(measure_map)] = (measure_map, entry_id)
        return entry_id

    def id_of_rows(self, rows: list[dict[str, Any]]) -> str:
        """Return the id of the measure map compressed *rows* describe, adding it."""
        from .structure import MeasureMap

        return self.id_of(MeasureMap.from_rows(rows))

    def write(self, *, minimal: bool) -> list[dict[str, Any]]:
        """Return the ``measure_maps`` entries, sorted by id."""
        return [
            {**self._entries[entry_id].to_dict(minimal=minimal), "id": entry_id}
            for entry_id in sorted(self._entries)
        ]

    def _admit_given(self, measure_map: MeasureMap) -> str:
        """Add a measure map under its given id, and return the id.

        Raises:
            ValueError: If the id is a timeline's, group's or manifest's, or
                another measure map carries it.
        """
        entry_id = measure_map.id
        if entry_id in self._reserved:
            raise ValueError(
                f"measure map id {entry_id!r} is also a timeline, group or manifest id"
            )
        held = self._entries.get(entry_id)
        if held is not None and not _same_entry(held, measure_map):
            raise ValueError(f"two different measure maps carry the id {entry_id!r}")
        self._entries.setdefault(entry_id, measure_map)
        self._written[id(measure_map)] = (measure_map, entry_id)
        return entry_id


def _same_entry(left: MeasureMap, right: MeasureMap) -> bool:
    """Whether two measure maps write the same entry but for its id."""
    return (
        left == right
        and left.name == right.name
        and left.meta == right.meta
        and dict(left._unknown_keys) == dict(right._unknown_keys)
        and dict(left._unknown_row_keys) == dict(right._unknown_row_keys)
    )


@dataclass
class ManifestContent:
    """The timelines and groups a manifest holds.

    Attributes:
        roots: The root timelines, rebuilt, in the manifest's order.
        timelines: Every timeline of the roots' trees, by id.
        groups: The manifest's groups, as JSON.
        order: The display orders of ``graphical.order``: member orders
            keyed by group id.
    """

    roots: list[Timeline]
    timelines: dict[str, Timeline]
    groups: list[Any]
    order: dict[str, Any]

    def candidates(self) -> list[tuple[str, str | None]]:
        """Every timeline of the manifest as ``(id, name)``, for an error."""
        return [(timeline.id, timeline._name) for timeline in self.timelines.values()]


def _timeline_ids(timelines: list[dict[str, Any]]) -> set[str]:
    """The ids of written timeline objects and of all their children."""
    ids: set[str] = set()
    pending = list(timelines)
    while pending:
        timeline = pending.pop()
        ids.add(timeline["id"])
        pending.extend(timeline.get("children", []))
    return ids


def _read_measure_maps(items: list[Any]) -> dict[str, MeasureMap]:
    """Read the ``measure_maps`` entries, by id.

    Raises:
        ValueError: If an entry is malformed or an id occurs twice.
    """
    from .structure import MeasureMap

    measure_maps: dict[str, MeasureMap] = {}
    for item in items:
        measure_map = MeasureMap.from_dict(item)
        if measure_map.id in measure_maps:
            raise ValueError(
                f"measure map id {measure_map.id!r} occurs twice in 'measure_maps'"
            )
        measure_maps[measure_map.id] = measure_map
    return measure_maps


def _read_roots(
    items: Any, measure_maps: Mapping[str, MeasureMap]
) -> tuple[list[Timeline], dict[str, Timeline]]:
    """Rebuild the root timelines and index every timeline of their trees by id.

    Raises:
        ValueError: If *items* is not an array of timeline objects, or a
            timeline id occurs twice across the trees.
    """
    from .base import Timeline

    if not isinstance(items, list):
        raise ValueError(f"'timelines' must be an array, got {type(items).__name__}")
    roots: list[Timeline] = []
    index: dict[str, Timeline] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(
                f"each entry of 'timelines' must be a timeline object, got "
                f"{type(item).__name__}"
            )
        root = Timeline._from_timeline_object(item, measure_maps=measure_maps)
        roots.append(root)
        for timeline in [
            root,
            *(descendant for _, descendant in root._iter_descendants()),
        ]:
            if timeline.id in index:
                raise ValueError(
                    f"timeline id {timeline.id!r} occurs twice in 'timelines'"
                )
            index[timeline.id] = timeline
    return roots, index


def select(
    candidates: list[Any],
    *,
    id: str | None,
    name: str | None,
    what: str,
    names: list[tuple[str, str | None]],
) -> Any:
    """Return the one candidate whose id and name match the given ones.

    Args:
        candidates: Objects with ``id`` and ``_name``-or-``name``.
        id: The id to match, or None to match any.
        name: The name to match, or None to match any.
        what: What is selected (``"timeline"``, ``"group"``), for the error.
        names: Every candidate as ``(id, name)``, for the error.

    Raises:
        ValueError: If no candidate or several match; the message names the
            manifest's candidates.
    """
    matches = [
        candidate
        for candidate in candidates
        if (id is None or candidate.id == id)
        and (name is None or _name_of(candidate) == name)
    ]
    if len(matches) == 1:
        return matches[0]
    wanted = ", ".join(
        f"{key} {value!r}"
        for key, value in (("id", id), ("name", name))
        if value is not None
    )
    found = "no" if not matches else "several"
    raise ValueError(
        f"the manifest holds {found} {what} with {wanted}; its {what}s: {listed(names)}"
    )


def matches_any(candidates: list[Any], *, id: str | None, name: str | None) -> bool:
    """Whether the given id or the given name names one of the candidates.

    Each given argument is matched on its own, so an argument naming
    something the manifest holds asks for a selection even when the other
    names nothing.
    """
    return any(
        (id is not None and candidate.id == id)
        or (name is not None and _name_of(candidate) == name)
        for candidate in candidates
    )


def listed(names: list[tuple[str, str | None]]) -> str:
    """Spell candidates given as ``(id, name)`` for an error message."""
    return ", ".join(f"{cid!r} (name {cname!r})" for cid, cname in names) or "none"


def _name_of(candidate: Any) -> str | None:
    """Return a candidate's own name, never a derived display name."""
    return getattr(candidate, "_name", candidate.name)


# endregion
