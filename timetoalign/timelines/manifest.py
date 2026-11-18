"""Reading and writing TimeToShare manifests.

A TimeToShare manifest is one JSON object: the version marker
``timetoshare``, an array of root ``timelines`` and, optionally, the
manifest's ``id``, ``name`` and ``meta``, its ``groups``, its
``measure_maps``, its ``schemes`` and ``links`` (the description layer, see
:mod:`timetoalign.timelines.description`) and its ``graphical``.
:class:`~timetoalign.timelines.Timeline`,
:class:`~timetoalign.timelines.TimelineGroup` and
:class:`~timetoalign.alignment.AlignmentBundle` each write one with
``to_dict`` and read one with ``from_dict``. This module holds what they
share: the version marker, the reading of the manifest around its timelines
and groups, the writing of that envelope and of the measure maps its
timelines name, the reading of links and schemes, and the codec of a bare
value on a timeline's own axis.

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
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

import timetoshare
from timetoshare import Link, Scheme, child_ends, description_problems, loads
from timetoshare.serialization import Serialization

from timetoalign.core import Coordinate, NumberType, TimeUnit
from timetoalign.core.ids import _OpenObject, mint_id, quote_id, validate_id
from timetoalign.core.time import is_fraction_json, number_from_json, number_to_json
from timetoalign.maps.base import MEASURE_MAP_ID_PREFIX

from .description import _named_schemes, _renamed_parts

if TYPE_CHECKING:
    from .base import Timeline
    from .structure import MeasureMap

TIMETOSHARE_VERSION = "0.2.0"
"""The TimeToShare version the library reads and writes."""

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
    spelling of the TimeToShare validators.
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
    segment line, a segment of unknown length's successor's offset). Where
    the parent and the child are both ``float``, the end is the double sum a
    reader computes, exact otherwise; the ends are TimeToShare's
    (``child_ends``). A child of another unit is left to the reader, which
    refuses it. The messages are the TimeToShare validators', values spelled
    as written.

    Raises:
        ValueError: ``offset 75.0 of 'take' is beyond the length 60.0 of its
            parent 'recording'``, or ``child 'coda' ends at [12, 1], beyond
            the length [10, 1] of its parent 'score'``.
    """
    length = data["length"]
    if length is None:
        return
    bound = Fraction(number_from_json(length))
    parent = data["id"]
    for child, end in zip(data.get("children", []), child_ends(data), strict=True):
        if child["unit"] != data["unit"]:
            continue
        if Fraction(number_from_json(child["offset"])) > bound:
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


def read_links(data: Mapping[str, Any], what: str) -> list[Link]:
    """Read an optional ``links`` array of TimeToShare links, ``[]`` by default.

    Each link is read by ``Link.from_dict``, which keeps the keys the format
    does not declare. The rules of a link are TimeToShare's, which
    :func:`check_description` applies before the manifest is read.

    Raises:
        ValueError: If ``links`` is not an array or a link is no object
            holding ``relation`` and ``level``.
    """
    return [
        Link.from_dict(require_object(item, f"a link of {what}", ("relation", "level")))
        for item in read_list(data, "links", what)
    ]


def read_schemes(data: Mapping[str, Any]) -> list[Scheme]:
    """Read the manifest's optional ``schemes`` array, ``[]`` by default.

    Raises:
        ValueError: If ``schemes`` is not an array, or a scheme is no object
            or has no valid id.
    """
    schemes = []
    for item in read_list(data, "schemes", "the manifest"):
        require_object(item, "a scheme", ("id",))
        read_id(item["id"], "a scheme")
        schemes.append(Scheme.from_dict(item))
    return schemes


def check_description(data: Mapping[str, Any]) -> None:
    """Refuse a manifest whose description layer breaks a rule of the format.

    The rules of the description layer — the shape of every link and
    scheme, scheme ids unique in the manifest, the dates of assertions, the
    schemes identifiers name and the elements part slots name — are
    TimeToShare's: ``timetoshare.description_problems`` checks them on the
    whole manifest and nothing else, and the first problem it finds is
    raised. A manifest is checked as it is read, before anything else is
    read, and as it is written, so that an invalid one is never emitted.

    Raises:
        ValueError: The first problem, its path in *data* first:
            ``#/links/0/parts/0/represented_by/0: 'ghost' names no timeline
            or region of this manifest``.
    """
    problems = description_problems(data)
    if problems:
        raise problems[0]


def schemes_named(
    declared: Iterable[Scheme], timelines: Iterable[Timeline]
) -> list[Scheme]:
    """Return the schemes the links of *timelines* name that *declared* lacks.

    A link names a scheme by id in an identifier of its own, of a part slot
    or of a ``part_of`` entry. Each id the links of a timeline or of its
    regions name is looked up among the schemes *declared* and the schemes
    the timeline knows (``Timeline._scheme_sources``): those of the bundles
    and groups its tree belongs to and of the manifest it was read from. An
    id no scheme carries is left to the check of the manifest written.

    Args:
        declared: The schemes the manifest declares already.
        timelines: The timelines whose links are written.

    Returns:
        The schemes to declare beside *declared*, in the order found.

    Raises:
        ValueError: If two different schemes carry an id the links name
            (``two different schemes carry the id 'opus'``).
    """
    held = {scheme.id: scheme for scheme in declared}
    found: list[Scheme] = []
    for timeline in timelines:
        named = _named_schemes(timeline._held_links())
        if not named:
            continue
        for scheme in timeline._scheme_sources():
            if scheme.id not in named:
                continue
            known = held.get(scheme.id)
            if known is None:
                held[scheme.id] = scheme
                found.append(scheme)
            elif known != scheme:
                raise ValueError(
                    f"two different schemes carry the id {quote_id(scheme.id)}"
                )
    return found


def _written_links(fields: Mapping[str, Any]) -> Iterator[list[Any]]:
    """Yield every ``links`` array of a manifest written, to change in place.

    The manifest's, every timeline's and every region's.
    """
    if isinstance(fields.get("links"), list):
        yield fields["links"]
    pending = list(fields["timelines"])
    while pending:
        timeline = pending.pop()
        if isinstance(timeline.get("links"), list):
            yield timeline["links"]
        for region in timeline.get("regions", []):
            if isinstance(region.get("links"), list):
                yield region["links"]
        pending.extend(timeline.get("children", []))


def _part_targets(timelines: list[dict[str, Any]]) -> set[str]:
    """The ids a part slot may name in a manifest written: timelines and regions."""
    found: set[str] = set()
    pending = list(timelines)
    while pending:
        timeline = pending.pop()
        found.add(timeline["id"])
        found.update(region["id"] for region in timeline.get("regions", []))
        pending.extend(timeline.get("children", []))
    return found


def _leave_out_unwritten(fields: dict[str, Any], roots: list[Timeline]) -> None:
    """Drop from the part slots written the elements the write leaves out.

    A timeline or a group written on its own writes part of the manifests
    it belongs to: a part slot of a link written may name a timeline or
    region of those manifests (``Timeline._manifest_ids`` of the roots
    written) that this one does not hold. The id is left out of the slot,
    and a slot left with none is ``absent`` as ``not_represented``: the
    part exists, and this manifest leaves it out. An id no manifest of the
    roots holds is left as it is, for the check of the manifest written to
    refuse.
    """
    targets = _part_targets(fields["timelines"])
    outside = {
        element
        for links in _written_links(fields)
        for link in links
        for slot in link.get("parts", [])
        for element in slot.get("represented_by", [])
        if element not in targets
    }
    if not outside:
        return
    known: set[str] = set()
    for root in roots:
        known.update(root._manifest_ids())
    renamed: dict[str, str | None] = dict.fromkeys(outside & known)
    if not renamed:
        return
    for links in _written_links(fields):
        for index, data in enumerate(links):
            link = Link.from_dict(data)
            pruned = _renamed_parts(link, renamed)
            if pruned is not link:
                links[index] = pruned.to_dict(minimal=False)


# endregion


# region The manifest


@dataclass(eq=False)
class Manifest(_OpenObject):
    """What a manifest holds beside its timelines and groups.

    The manifest's ``id`` (absent when it has none), ``name`` and ``meta``, its
    schemes and links, the members of its ``graphical`` other than ``order``
    (``styling``) and the band orders of its timelines (``band_orders``),
    both kept as given and never interpreted, whether it holds ``graphical``
    at all, the measure maps no timeline names, and every key the library
    does not read.

    Attributes:
        id: The manifest id, or None when it has none (an anonymous bundle,
            or one named by ``name`` alone).
        name: The manifest name, or None.
        meta: The manifest's free metadata.
        schemes: The identifier schemes the manifest declares, written
            sorted by id.
        links: The manifest's links, in their order.
        styling: ``graphical`` without ``order``: ``styles``,
            ``statements``, ``timestamps`` and any other key, kept as given;
            a named timestamp's ``display`` is written in canonical order,
            its default stated in the complete form only.
        band_orders: The entries of ``graphical.order`` keyed by a
            timeline's id, each that timeline's band order, kept as given.
        has_graphical: Whether the manifest holds a ``graphical`` object,
            which is then written back even with every member at its
            default.
        measure_maps: The entries of ``measure_maps`` no timeline names,
            written back with the measure maps the timelines name.
    """

    _dict_keys: ClassVar[tuple[str, ...]] = (
        "timetoshare",
        "id",
        "name",
        "meta",
        "timelines",
        "groups",
        "measure_maps",
        "schemes",
        "links",
        "graphical",
    )

    id: str | None = None
    name: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    schemes: list[Scheme] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    styling: dict[str, Any] = field(default_factory=dict)
    band_orders: dict[str, Any] = field(default_factory=dict)
    has_graphical: bool = False
    measure_maps: list[MeasureMap] = field(default_factory=list)

    @classmethod
    def read(cls, data: Any) -> tuple[Manifest, ManifestContent]:
        """Read a manifest: its envelope, its timelines and its raw groups.

        The description layer is checked first, by TimeToShare's rules
        (:func:`check_description`). The measure maps are read next, then
        every root timeline is rebuilt, its conversion maps reading the
        measure maps they name, and every timeline of the roots' trees is
        indexed by id. Every id is unique across the manifest: the
        manifest's own, every timeline, root or descendant, group, measure
        map, conversion map at any depth, region, flow-control element and
        scheme. Groups are left
        as JSON for the class reading them. ``graphical.order`` is split by
        its keys: a group's display order goes to the content, a timeline's
        band order to the envelope, unchecked. The measure maps no timeline
        names, the schemes and the manifest's links are kept on the envelope;
        every timeline keeps the schemes too, which its links may name
        wherever it is written.

        Returns:
            The envelope, and the manifest's content.

        Raises:
            ValueError: If *data* is no TimeToShare manifest (``"not a
                TimeToShare manifest"``), names another version, or is
                malformed: a link or scheme breaking a rule of the format,
                an id held twice anywhere in the manifest (see
                :func:`unique_ids`), a ``graphical.order`` that is no object
                or names no group or timeline.
        """
        if not isinstance(data, dict) or "timetoshare" not in data:
            raise ValueError("not a TimeToShare manifest")
        version = data["timetoshare"]
        if version != TIMETOSHARE_VERSION:
            raise ValueError(
                f"unsupported TimeToShare version {version!r}: this library reads "
                f"TimeToShare {TIMETOSHARE_VERSION} only"
            )
        require_object(data, "a TimeToShare manifest", ("timelines",))
        check_description(data)
        manifest_id = read_id(data["id"], "the manifest") if "id" in data else None
        manifest = cls(
            id=manifest_id,
            name=read_name(data, "the manifest"),
            meta=read_meta(data, "the manifest"),
        )
        manifest._keep_unknown_keys(data)

        holders: list[tuple[str, str]] = []
        if manifest_id is not None:
            holders.append((manifest_id, "the manifest"))
        read_maps = _read_measure_maps(read_list(data, "measure_maps", "the manifest"))
        holders.extend(
            (measure_map.id, f"measure map {quote_id(measure_map.id)}")
            for measure_map in read_maps
        )
        unique_ids(holders)
        measure_maps = {measure_map.id: measure_map for measure_map in read_maps}
        roots, timelines = _read_roots(data["timelines"], measure_maps)
        holders.extend(
            (element_id, description)
            for root in roots
            for element_id, description in root._tree_ids().items()
        )
        groups = read_list(data, "groups", "the manifest")
        group_ids = [
            group.get("id") if isinstance(group, dict) else None for group in groups
        ]
        holders.extend(
            (group_id, f"group {quote_id(group_id)}")
            for group_id in group_ids
            if isinstance(group_id, str)
        )
        manifest.schemes = read_schemes(data)
        holders.extend(
            (scheme.id, f"scheme {quote_id(scheme.id)}") for scheme in manifest.schemes
        )
        ids = unique_ids(holders)
        manifest.links = read_links(data, "the manifest")
        if manifest.schemes:
            # Each timeline keeps the schemes its links may name, so that it
            # writes them wherever it is written.
            schemes = tuple(manifest.schemes)
            for timeline in timelines.values():
                timeline._source_schemes = schemes
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
            # The styling is kept in TimeToShare's minimal form, so that the
            # complete and the minimal form read to the same styling.
            styling = {key: value for key, value in graphical.items() if key != "order"}
            manifest.styling = timetoshare.minimal({"graphical": styling})["graphical"]

        return manifest, ManifestContent(roots, timelines, groups, order, ids)

    def write(
        self,
        roots: list[Timeline],
        *,
        groups: list[dict[str, Any]] = (),  # type: ignore[assignment]
        order: dict[str, list[str]] | None = None,
        minimal: bool = True,
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
        ``links`` are written in their order and ``schemes`` sorted by id:
        the manifest's own, and every scheme a link of a timeline written
        names that the timeline knows (:func:`schemes_named`). A part slot
        naming an element the manifest leaves out leaves it out too
        (:func:`_leave_out_unwritten`). The form is TimeToShare's: the
        manifest is built whole and put in its form by
        :func:`timetoshare.minimal` or :func:`timetoshare.complete`. The
        manifest written is checked as a manifest read is
        (:func:`check_description`).

        Args:
            roots: The root timelines.
            groups: The written groups, each in the complete form.
            order: Every group's member order, keyed by group id.
            minimal: Write the minimal form (the default): every key at its
                default value left out. With False, the complete form.
            events: Write each timeline's event rows.
            external_references: Write each timeline's external reference
                table.

        Raises:
            ValueError: If an id is held twice among the manifest, its
                timelines' trees, its groups and its schemes
                (:func:`unique_ids`), two different schemes carry an id a
                link names (:func:`schemes_named`), a measure map cannot be
                written (two different ones carry one given id, or one's
                given id is held by another element), or the description
                layer written breaks a rule of the format
                (:func:`check_description`).
        """
        trees = [
            timeline
            for root in roots
            for timeline in (root, *(child for _, child in root._iter_descendants()))
        ]
        schemes = [*self.schemes, *schemes_named(self.schemes, trees)]
        holders: list[tuple[str, str]] = []
        if self.id is not None:
            holders.append((self.id, "the manifest"))
        holders.extend(
            (element_id, description)
            for root in roots
            for element_id, description in root._tree_ids().items()
        )
        holders.extend(
            (group["id"], f"group {quote_id(group['id'])}") for group in groups
        )
        holders.extend(
            (scheme.id, f"scheme {quote_id(scheme.id)}") for scheme in schemes
        )
        entries = MeasureMapEntries(
            unique_ids(holders),
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
                events=events,
                external_references=external_references,
                offset=root._root_offset,
                entries=entries,
            )
            for root in sorted(roots, key=lambda root: root.id)
        ]
        for measure_map in self.measure_maps:
            entries.id_of(measure_map)
        fields: dict[str, Any] = {"timetoshare": TIMETOSHARE_VERSION}
        if self.id is not None:
            fields["id"] = self.id
        fields.update(
            name=self.name,
            meta=copy.deepcopy(self.meta),
            timelines=timelines,
            groups=list(groups),
            measure_maps=entries.write(),
            schemes=[
                scheme.to_dict(minimal=False)
                for scheme in sorted(schemes, key=lambda scheme: scheme.id)
            ],
            links=[link.to_dict(minimal=False) for link in self.links],
        )
        _leave_out_unwritten(fields, roots)
        written = _timeline_ids(timelines)
        band_orders = {
            key: copy.deepcopy(entry)
            for key, entry in self.band_orders.items()
            if key in written
        }
        order = dict(sorted({**band_orders, **(order or {})}.items()))
        if order or self.styling or self.has_graphical:
            fields["graphical"] = {"order": order, **copy.deepcopy(self.styling)}
        built = self._dict_form(fields, minimal=False)
        result = timetoshare.minimal(built) if minimal else timetoshare.complete(built)
        check_description(result)
        return result

    def _rename_parts(self, renamed: Mapping[str, str | None]) -> None:
        """Rename the elements the part slots of the manifest's links name.

        See :func:`~timetoalign.timelines.description._renamed_parts`.
        """
        self.links = [_renamed_parts(link, renamed) for link in self.links]


class MeasureMapEntries:
    """The measure maps a manifest being written holds, each under one id.

    Every id of the manifest is unique: :attr:`taken` holds every id the
    manifest written holds besides its measure maps, mapped to its holder,
    and grows with every id minted while it is written (a ``jump_to`` a
    jump needs, see
    :meth:`~timetoalign.timelines.flowcontrol.FlowControlRegistry.to_dict`).
    A given measure map id is kept: two different measure maps carrying
    one, or one carrying an id another element holds, are refused. A minted
    id is minted again around every id the manifest holds, the given ones
    taken first. Equal measure maps of equal name and meta share one entry,
    so two timelines holding the same measure map, or equal ones, name one
    entry.

    While a timeline's conversion maps are written, :attr:`own` is that
    timeline's measure map, which a measure-derived map names when it
    derives from it.

    Args:
        taken: Every id of the manifest but its measure maps', mapped to its
            holder.
        measure_maps: The measure maps the manifest names, whose given ids
            are taken before any is minted.

    Raises:
        ValueError: If a given id is refused.
    """

    def __init__(
        self, taken: dict[str, str], measure_maps: list[MeasureMap] = ()  # type: ignore[assignment]
    ) -> None:
        self.own: MeasureMap | None = None
        self.taken = taken
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
        entry_id = mint_id(MEASURE_MAP_ID_PREFIX, {*self._entries, *self.taken})
        self._entries[entry_id] = measure_map
        self._written[id(measure_map)] = (measure_map, entry_id)
        return entry_id

    def id_of_rows(self, rows: list[dict[str, Any]]) -> str:
        """Return the id of the measure map compressed *rows* describe, adding it."""
        from .structure import MeasureMap

        return self.id_of(MeasureMap.from_rows(rows))

    def write(self) -> list[dict[str, Any]]:
        """Return the ``measure_maps`` entries, sorted by id, in the complete form."""
        return [
            {**self._entries[entry_id].to_dict(minimal=False), "id": entry_id}
            for entry_id in sorted(self._entries)
        ]

    def _admit_given(self, measure_map: MeasureMap) -> str:
        """Add a measure map under its given id, and return the id.

        Raises:
            ValueError: If another element of the manifest, or another
                measure map, carries the id.
        """
        entry_id = measure_map.id
        if entry_id in self.taken:
            raise ValueError(
                f"measure map {quote_id(entry_id)} cannot be written: its id is already "
                f"held by {self.taken[entry_id]}"
            )
        held = self._entries.get(entry_id)
        if held is not None and not _same_entry(held, measure_map):
            raise ValueError(
                f"two different measure maps carry the id {quote_id(entry_id)}"
            )
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
        ids: Every id of the manifest, mapped to a description of its
            holder.
    """

    roots: list[Timeline]
    timelines: dict[str, Timeline]
    groups: list[Any]
    order: dict[str, Any]
    ids: dict[str, str]

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


def _read_measure_maps(items: list[Any]) -> list[MeasureMap]:
    """Read the ``measure_maps`` entries, in order.

    Raises:
        ValueError: If an entry is malformed.
    """
    from .structure import MeasureMap

    return [MeasureMap.from_dict(item) for item in items]


def _read_roots(
    items: Any, measure_maps: Mapping[str, MeasureMap]
) -> tuple[list[Timeline], dict[str, Timeline]]:
    """Rebuild the root timelines and index every timeline of their trees by id.

    Raises:
        ValueError: If *items* is not an array of timeline objects.
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
            index.setdefault(timeline.id, timeline)
    return roots, index


def unique_ids(holders: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Map each id to its holder, refusing an id held twice.

    Args:
        holders: ``(id, description)`` of every element of a manifest.

    Raises:
        ValueError: ``the id 'intro' is held twice in the manifest: by
            region 'intro' of timeline 'score' and by timeline 'intro'``.
    """
    ids: dict[str, str] = {}
    for element_id, holder in holders:
        if element_id in ids:
            raise ValueError(
                f"the id {quote_id(element_id)} is held twice in the manifest: by "
                f"{ids[element_id]} and by {holder}"
            )
        ids[element_id] = holder
    return ids


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
        f"{key} {quote_id(value) if key == 'id' else repr(value)}"
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
    return (
        ", ".join(f"{quote_id(cid)} (name {cname!r})" for cid, cname in names) or "none"
    )


def _name_of(candidate: Any) -> str | None:
    """Return a candidate's own name, never a derived display name."""
    return getattr(candidate, "_name", candidate.name)


# endregion


# region Manifest files


@dataclass(frozen=True)
class FileForm(timetoshare.FileForm):
    """How a manifest file is written: TimeToShare's file form, and the timeline tables.

    The form, the styling and the serialization are ``timetoshare.FileForm``'s;
    the two tables are the library's own.

    Attributes:
        events: Whether any timeline object holds its event rows
            (``events``).
        external_references: Whether any timeline object holds its
            external reference table (``external_references``).
    """

    events: bool = False
    external_references: bool = False

    @classmethod
    def of(cls, text: str, data: Any) -> FileForm:
        """Infer the form, styling, serialization and table keys of a manifest file.

        The form, the styling and the serialization are the ones
        ``timetoshare.file_form`` reads off *text*; ``events`` and
        ``external_references`` are each written when a timeline object of
        the manifest, at any depth, holds the key.

        Args:
            text: The file's text.
            data: The manifest *text* holds.
        """
        keys = {key for timeline in _timeline_objects(data) for key in timeline}
        return cls(
            **asdict(timetoshare.file_form(text)),
            events="events" in keys,
            external_references="external_references" in keys,
        )


_NEW_FILE = FileForm(minimal=True, styling=True, serialization="yaml")
"""The form of an object not read from a file: minimal, styled, canonical YAML."""


def _timeline_objects(data: Any) -> Iterator[dict[str, Any]]:
    """Yield every timeline object of a manifest, the children included."""
    pending = list(data.get("timelines") or ())
    while pending:
        timeline = pending.pop()
        if isinstance(timeline, dict):
            yield timeline
            pending.extend(timeline.get("children") or ())


class ManifestFileMixin:
    """Read a manifest file and write it back in the form it came in.

    Shared by :class:`~timetoalign.timelines.Timeline`,
    :class:`~timetoalign.timelines.TimelineGroup` and
    :class:`~timetoalign.alignment.AlignmentBundle`, each of which reads a
    manifest with ``from_dict`` and writes one with ``to_dict``. An object
    read from a file keeps the file's :class:`FileForm`, which
    :meth:`to_file` follows unless told otherwise: a file in canonical YAML,
    pretty JSON or RFC 8785 JSON keeps its bytes; any other text is written
    back in the canonical form of the serialization it was read as. A file
    read, changed and written back changes only where the object changed.
    """

    _file_form: FileForm | None = None
    """The form of the file the object was read from, or None."""

    @classmethod
    def from_dict(cls, data: dict[str, Any], **selection: str | None) -> Self:
        """Read the object from a manifest; each class using the mixin defines it.

        Args:
            data: The manifest.
            **selection: The keyword arguments that select one timeline or
                group of the manifest.
        """
        raise NotImplementedError

    def to_dict(
        self,
        *,
        minimal: bool = True,
        events: bool = False,
        external_references: bool = False,
    ) -> dict[str, Any]:
        """Write the object's manifest; each class using the mixin defines it.

        Args:
            minimal: Write the minimal form (True) or the complete form
                (False).
            events: Write every timeline's event rows.
            external_references: Write every timeline's external references.
        """
        raise NotImplementedError

    @classmethod
    def from_file(cls, path: str | PathLike[str], **selection: str | None) -> Self:
        """Read a manifest file in any serialization, as ``from_dict`` reads a manifest.

        The text is read with ``timetoshare.loads``: canonical YAML, pretty
        JSON, RFC 8785 JSON or any other JSON or YAML text. The object read
        records the file's form, styling and serialization, as
        ``timetoshare.file_form`` reads them, and whether its timeline
        objects hold ``events`` and ``external_references``, which
        :meth:`to_file` writes back.

        Args:
            path: The manifest file, UTF-8.
            **selection: The keyword arguments of ``from_dict`` that select
                one timeline or group of the manifest (``timeline_id`` and
                ``name`` for a timeline, ``group_id`` and ``name`` for a
                group).

        Returns:
            The object read.

        Raises:
            ValueError: If the text cannot be read (with the line and column
                of the fault), or ``from_dict`` refuses the manifest.
        """
        with Path(path).open(encoding="utf-8", newline="") as file:
            text = file.read()
        data = loads(text)
        result = cls.from_dict(data, **selection)
        result._file_form = FileForm.of(text, data)
        return result

    def to_file(
        self,
        path: str | PathLike[str],
        *,
        minimal: bool | None = None,
        styling: bool | None = None,
        serialization: Serialization | None = None,
        events: bool | None = None,
        external_references: bool | None = None,
    ) -> Path:
        """Write the object's manifest to a file.

        The manifest is ``to_dict``'s complete form, written with
        ``timetoshare.dumps_file`` in the form, styling and serialization
        chosen: every object's keys sorted, its arrays in canonical order, so
        that equal manifests give equal text and a change shows in a diff as the
        lines it touches. Each argument left None takes the value the file
        the object was read from had (see :meth:`from_file`); for an object
        not read from a file, the minimal form, the styling included, in
        canonical YAML, without event rows or external references.

        Args:
            path: The file to write, UTF-8 with line feeds; it is
                overwritten.
            minimal: Write the minimal form (True) or the complete form
                (False).
            styling: Write the ``graphical`` object, when the manifest has
                one (True), or leave it out (False).
            serialization: ``"yaml"`` (canonical YAML), ``"json"`` (pretty
                JSON) or ``"jcs"`` (RFC 8785 JSON).
            events: Write every timeline's event rows (``events``).
            external_references: Write every timeline's external reference
                table (``external_references``).

        Returns:
            The path written.

        Raises:
            ValueError: If ``to_dict`` cannot write the manifest, or the
                serialization refuses a value of it.

        Examples:
            >>> import tempfile
            >>> from fractions import Fraction
            >>> from pathlib import Path
            >>> from timetoalign.timelines import ContinuousLogicalTimeline, Timeline
            >>> score = ContinuousLogicalTimeline(length=Fraction(32), uid="piece")
            >>> path = score.to_file(Path(tempfile.mkdtemp()) / "piece.yaml")
            >>> print(path.read_text(encoding="utf-8"), end="")
            timelines:
              -
                id: piece
                length: [32, 1]
                unit: quarters
            timetoshare: "0.2.0"
            >>> Timeline.from_file(path).length
            Coordinate(Fraction(32, 1), quarters)
        """
        given = {
            "minimal": minimal,
            "styling": styling,
            "serialization": serialization,
            "events": events,
            "external_references": external_references,
        }
        form = replace(
            self._file_form or _NEW_FILE,
            **{key: value for key, value in given.items() if value is not None},
        )
        manifest = self.to_dict(
            minimal=False,
            events=form.events,
            external_references=form.external_references,
        )
        text = timetoshare.dumps_file(manifest, form)
        path = Path(path)
        path.write_text(text, encoding="utf-8", newline="")
        return path


# endregion
