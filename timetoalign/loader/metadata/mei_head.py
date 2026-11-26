"""Import an MEI header as statements on a bundle and its timelines.

An MEI file's ``meiHead`` describes the work encoded, its expressions, the
manifestations that embody them and the items that exemplify those.
:class:`MeiHeadLoader` reads that description and states it as
property–value pairs with provenance (see
:mod:`timetoalign.timelines.statements`), in LRMoo and MeLODy's vocabulary:

* the bundle is about the work, a value node under ``schema:about`` whose IRI
  is the MEI file's address followed by ``#`` and the work's ``xml:id``; the
  bundle points to the MEI file with ``rdfs:seeAlso``;
* an expression is a value node as well, written under ``lrmoo:R4_embodies``
  of every manifestation embodying it (a manifestation's ``isEmbodimentOf``
  or the expression's ``hasEmbodiment``), else under the work's
  ``lrmoo:R3_is_realised_in``;
* each manifestation becomes a graphical timeline in millimetres of unknown
  length (or lands on an element of an existing bundle);
* each item is a value node of its manifestation, with the IRI built as a
  work's when it has an ``xml:id`` (or lands on an element).

What is linked-data-ready as encoded (relations, ``@sameas``, authority
IRIs, identifiers, titles, dates, keys, meters, media of performance) is
stated; a name without an IRI is offered as a candidate, for reconciliation.
Every pair's provenance entry names the MEI element it comes from as its
basis; the import records how a pair was made and holds no belief. History,
notes, bibliography and incipits stay in the MEI file.

Examples:
    >>> import tempfile, pathlib
    >>> from timetoalign.loader.metadata import MeiHeadLoader
    >>> header = '''<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.1">
    ...   <meiHead><workList><work xml:id="work_1">
    ...     <title xml:id="title_1">Hr. Olufs Sang</title>
    ...     <composer><persName xml:id="persName_1">Carl Nielsen</persName></composer>
    ...   </work></workList></meiHead></mei>'''
    >>> folder = pathlib.Path(tempfile.mkdtemp())
    >>> _ = (folder / "song.mei").write_text(header, encoding="utf-8")
    >>> bundle = MeiHeadLoader.from_file(folder / "song.mei").create_bundle(date="2026-11-19")
    >>> bundle.id, bundle.timeline_ids
    ('song', [])
    >>> work = bundle.statements["schema:about"][0]
    >>> work["id"], work["type"], work["melod:hasTitle"]
    ('song.mei#work_1', ['melod:Work'], [{'type': ['melod:Title'], 'rdfs:label': ['Hr. Olufs Sang']}])
    >>> [entry["property"] for entry in work["provenance"]]
    ['rdf:type', 'melod:hasTitle', 'rel:cmp']
    >>> candidates = bundle.get_candidates(node=("schema:about", 0))
    >>> [(entry["value"], entry["basis"], entry["note"]) for entry in candidates]
    [({'label': 'Carl Nielsen'}, ['song.mei#persName_1'], 'unreconciled')]
    >>> "beliefs" in candidates[0]
    False
"""

from __future__ import annotations

import datetime
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal
from urllib.parse import quote, urljoin

import timetoshare
from timetoshare import load_catalogue, to_file_id

from timetoalign.alignment import AlignmentBundle
from timetoalign.core import TimeUnit
from timetoalign.loader._sources import Source, describe_source, readable
from timetoalign.loader.base import Loader
from timetoalign.loader.metadata._statements import (
    Plan,
    Target,
    day,
    refuse_arguments,
    refuse_value_entity,
)
from timetoalign.timelines import ContinuousGraphicalTimeline
from timetoalign.timelines.statements import LIBRARY_AGENT, NodePath

__all__ = ["MeiHeadLoader"]

MEI_NAMESPACE = "http://www.music-encoding.org/ns/mei"
_XML = "{http://www.w3.org/XML/1998/namespace}"

Kind = Literal["work", "expression", "manifestation", "item"]

_TYPES: dict[str, str] = {
    "work": "melod:Work",
    "expression": "melod:Expression",
    "manifestation": "melod:Manifestation",
    "item": "melod:Item",
}

_TITLE_TYPES: dict[str, str] = {
    "main": "MainTitle",
    "alternative": "AlternativeTitle",
    "subordinate": "SubordinateTitle",
    "abbreviated": "AbbreviatedTitle",
    "uniform": "UniformTitle",
    "original": "OriginalTitle",
    "translated": "TranslatedTitle",
    "desc": "DescTitle",
    "text_source": "TextSourceTitle",
}

_ROLE_ELEMENTS = frozenset(
    (
        "composer",
        "lyricist",
        "librettist",
        "arranger",
        "author",
        "contributor",
        "editor",
    )
)

_ROLE_CODES: dict[str, str] = {
    "composer": "cmp",
    "lyricist": "lyr",
    "librettist": "lbt",
    "arranger": "arr",
    "author": "aut",
    "editor": "edt",
    "contributor": "ctb",
    "performer": "prf",
    "conductor": "cnd",
    "publisher": "pbl",
    "dedicatee": "dte",
    "translator": "trl",
    "transcriber": "trc",
    "copyist": "scr",
    "engraver": "egr",
    "printer": "prt",
    "illustrator": "ill",
}

_ACCIDENTALS: dict[str, str] = {
    "s": "Sharp",
    "f": "Flat",
    "n": "Natural",
    "ss": "DoubleSharp",
    "x": "DoubleSharp",
    "ff": "DoubleFlat",
}

_MODES = frozenset(
    (
        "major",
        "minor",
        "dorian",
        "phrygian",
        "lydian",
        "mixolydian",
        "aeolian",
        "locrian",
    )
)

# FRBR relations of MEI's data.FRBRRELATIONSHIP: LRMoo's properties for the
# realisation, embodiment and exemplification relations; MeLODy's term,
# followed by a Dublin Core fallback, for parts and derivations; MeLODy's
# term alone for the rest.
_LRMOO_RELATIONS: dict[str, str] = {
    "hasRealization": "lrmoo:R3_is_realised_in",
    "isRealizationOf": "lrmoo:R3i_realises",
    "hasEmbodiment": "lrmoo:R4i_is_embodied_in",
    "isEmbodimentOf": "lrmoo:R4_embodies",
    "hasExemplar": "lrmoo:R7i_is_exemplified_by",
    "isExemplarOf": "lrmoo:R7_exemplifies",
}
_DERIVATIONS = frozenset(
    (
        "isAbridgementOf",
        "isAdaptationOf",
        "isArrangementOf",
        "isImitationOf",
        "isReconfigurationOf",
        "isReproductionOf",
        "isRevisionOf",
        "isSummarizationOf",
        "isTransformationOf",
        "isTranslationOf",
    )
)
_MELOD_RELATIONS = frozenset(
    (
        "hasAbridgement",
        "hasAdaptation",
        "hasAlternate",
        "isAlternateOf",
        "hasArrangement",
        "hasComplement",
        "isComplementOf",
        "hasImitation",
        "hasReconfiguration",
        "hasReproduction",
        "hasRevision",
        "hasSuccessor",
        "isSuccessorOf",
        "hasSummarization",
        "hasSupplement",
        "isSupplementOf",
        "hasTransformation",
        "hasTranslation",
    )
)

_ABSOLUTE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_DATE_TYPES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\d{4}"), "xsd:gYear"),
    (re.compile(r"\d{4}-\d{2}"), "xsd:gYearMonth"),
    (re.compile(r"\d{4}-\d{2}-\d{2}"), "xsd:date"),
)


# region Reading MEI


def _tag(name: str) -> str:
    return f"{{{MEI_NAMESPACE}}}{name}"


def _local(element: ET.Element) -> str:
    return element.tag.rpartition("}")[2] if isinstance(element.tag, str) else ""


def _children(element: ET.Element, *path: str) -> list[ET.Element]:
    """The elements at *path* below *element*, MEI names, document order."""
    return element.findall("/".join(_tag(name) for name in path))


def _xml_id(element: ET.Element) -> str | None:
    return element.get(f"{_XML}id")


def _text(element: ET.Element, *, skip: str | None = None) -> str:
    """The element's whitespace-normalised text, leaving out *skip* children."""
    parts = [element.text or ""]
    for child in element:
        if _local(child) != skip:
            parts.extend(child.itertext())
        parts.append(child.tail or "")
    return " ".join("".join(parts).split())


@dataclass(frozen=True)
class MeiHead:
    """A parsed ``meiHead`` and the file it was read from."""

    element: ET.Element
    path: Source


@dataclass(frozen=True)
class _Pair:
    """One pair to state or offer as a candidate, and the MEI element it comes from.

    *inner* are the pairs of the value node this pair states, applied inside
    it once it is stated.
    """

    property: str
    value: Any
    source: ET.Element | None
    candidate: bool = False
    note: str | None = None
    inner: tuple[_Pair, ...] = ()


# endregion


class MeiHeadLoader(Loader[MeiHead]):
    """Read an MEI header and state what it says as statements.

    MEI 5.1 headers are read, and MEI 4.0.1 headers using ``workList`` and
    ``manifestationList``. ``load`` takes one MEI file whose root is ``mei``
    or ``meiHead``; :meth:`create_bundle` states its description on a new
    bundle or on one given.

    Examples:
        >>> from timetoalign.loader.metadata import MeiHeadLoader
        >>> MeiHeadLoader().create_bundle()
        Traceback (most recent call last):
        ...
        ValueError: MeiHeadLoader holds no source: load an MEI file first.
    """

    _single_source: ClassVar[bool] = True

    def __init__(self, *, seed: int | None = None) -> None:
        """Create a loader holding no header.

        Args:
            seed: The seed of the loader's ID minter (see ``Loader.load``).
        """
        super().__init__(seed=seed)
        self._head: MeiHead | None = None

    def _load_source(self, source: Source) -> MeiHead:
        """Parse *source* and return its ``meiHead``.

        Raises:
            ValueError: If the file's root is neither an ``mei`` element
                holding a ``meiHead`` nor a ``meiHead``.
        """
        with readable(source) as document:
            root = ET.parse(document).getroot()
        if root.tag == _tag("meiHead"):
            head: ET.Element | None = root
        elif root.tag == _tag("mei"):
            head = root.find(_tag("meiHead"))
        else:
            head = None
        if head is None:
            raise ValueError(f"{describe_source(source)} holds no meiHead")
        return MeiHead(head, source)

    def _accept_source(
        self, path: Source, source_meta: dict[str, Any], payload: MeiHead
    ) -> None:
        super()._accept_source(path, source_meta, payload)
        self._head = payload

    def clear(self) -> None:
        """Release the header and the source."""
        super().clear()
        self._head = None

    def create_bundle(
        self,
        *,
        into: AlignmentBundle | None = None,
        subjects: Mapping[str, str] | None = None,
        id: str | None = None,
        base: str | None = None,
        stated_by: str | Sequence[str] = LIBRARY_AGENT,
        date: str | datetime.date | None = None,
    ) -> AlignmentBundle:
        """State the header's description on a bundle and its timelines.

        The work becomes a value node under the bundle's ``schema:about``,
        each manifestation a new graphical timeline in millimetres of
        unknown length, named by its first title; expressions and items are
        value nodes. A name without an IRI is offered as a candidate. A
        pair a subject already states or holds an entry of (a candidate, a
        pair someone believes) is passed over, and the pairs of a value node
        go into it only while the subject states it, so importing a header
        twice into one bundle adds nothing and every belief recorded
        stands. The import holds no belief. Every pair
        is checked before the bundle changes: a refused one leaves it as it
        was.

        Args:
            into: The bundle to state the description on, returned; a new
                bundle when None.
            subjects: Maps the ``xml:id`` of a manifestation or an item to
                the ID of a timeline (at any depth) or region of *into*,
                which then takes that entity's statements in place of a new
                timeline or value node.
            id: The ID of the new bundle; the file's stem when None, with
                every character outside ``[A-Za-z0-9_.-]`` replaced by
                ``_``.
            base: The MEI file's address: relative targets resolve against
                it, and every basis and IRI built from an ``xml:id`` begins
                with it. The file name, percent-encoded, when None.
            stated_by: Who states the pairs: an IRI, or several.
            date: The day of the provenance entries, ``YYYY-MM-DD`` or a
                :class:`datetime.date`; today when None.

        Returns:
            The bundle holding the statements.

        Raises:
            ValueError: Before anything changes, if *id* is given with
                *into*, *subjects* without *into*, a key of *subjects* names
                no work, expression, manifestation or item, or names a work
                or an expression, a value names no timeline or region of
                *into*, the header holds more than one work, a work or an
                expression has no ``xml:id``, or a pair is refused (``the
                prefix 'doi' is declared neither by the catalogue nor on the
                new timeline of manifestation 'source_2'``).
        """
        if self._head is None:
            raise ValueError(
                f"{type(self).__name__} holds no source: load an MEI file first."
            )
        refuse_arguments(into, id, subjects)
        path = self._head.path
        run = _Import(
            self._head.element,
            base=quote(path.name) if base is None else base,
            stated_by=[stated_by] if isinstance(stated_by, str) else list(stated_by),
            day=day(date),
        )
        bundle = into if into is not None else AlignmentBundle(id=id or _stem_id(path))
        plan = Plan(bundle, self._id_minter)
        mapped = run.check(plan, dict(subjects or {}))
        run.plan(plan, mapped)
        plan.commit()
        return bundle


def _stem_id(path: Source) -> str:
    """The file's stem with every character outside ``[A-Za-z0-9_.-]`` replaced."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", path.stem)


class _Import:
    """One import of a header into a bundle."""

    def __init__(
        self, head: ET.Element, *, base: str, stated_by: list[str], day: str
    ) -> None:
        self.base = base
        self.stated_by = stated_by
        self.day = day
        self.catalogue = load_catalogue()
        self.works = _children(head, "workList", "work")
        self.manifestations = _children(head, "manifestationList", "manifestation")
        self.expressions = [
            expression
            for work in self.works[:1]
            for expression in _children(work, "expressionList", "expression")
        ]
        items = [
            item
            for manifestation in self.manifestations
            for item in _children(manifestation, "itemList", "item")
        ]
        entities: list[tuple[Kind, list[ET.Element]]] = [
            ("work", self.works),
            ("expression", self.expressions),
            ("manifestation", self.manifestations),
            ("item", items),
        ]
        self.kinds: dict[str, Kind] = {}
        for kind, elements in entities:
            for element in elements:
                if (xml_id := _xml_id(element)) is not None:
                    self.kinds.setdefault(xml_id, kind)
        self.expression_by_id = {_xml_id(e): e for e in self.expressions}
        self.manifestation_by_id = {_xml_id(m): m for m in self.manifestations}
        self.embodied, self.embodied_by_expression = self._embodiments()
        # The ID of the element an entity's statements land on, by xml:id.
        self.elements: dict[str, str] = {}

    # region Checks and the order of writing

    def check(self, plan: Plan, subjects: dict[str, str]) -> dict[str, Target]:
        """Refuse what the import cannot do, before anything changes.

        Returns:
            The element each key of *subjects* maps to.
        """
        for key in subjects:
            kind = self.kinds.get(key)
            if kind is None:
                raise ValueError(
                    f"subjects names '{key}', which is no work, expression, "
                    "manifestation or item of the meiHead"
                )
            if kind in ("work", "expression"):
                refuse_value_entity(key, kind)
        mapped = {key: plan.mapped(key, value) for key, value in subjects.items()}
        if len(self.works) > 1:
            raise ValueError(
                f"the meiHead holds {len(self.works)} works; a bundle is about one work"
            )
        for kind, elements_of_kind in (
            ("work", self.works),
            ("expression", self.expressions),
        ):
            for number, element in enumerate(elements_of_kind, start=1):
                if _xml_id(element) is None:
                    raise ValueError(
                        f"{kind} {number} of the meiHead has no xml:id, from which "
                        "its IRI is built"
                    )
        return mapped

    def plan(self, plan: Plan, mapped: dict[str, Target]) -> None:
        """Plan the new timelines, then every pair."""
        targets: list[tuple[ET.Element, Target]] = []
        for number, manifestation in enumerate(self.manifestations, start=1):
            xml_id = _xml_id(manifestation)
            if xml_id is not None and xml_id in mapped:
                target = mapped[xml_id]
            else:
                titles = self._titles("manifestation", manifestation)
                timeline = ContinuousGraphicalTimeline(
                    length=None,
                    unit=TimeUnit.millimeters,
                    uid=plan.timeline_id(TimeUnit.millimeters),
                    name=_text(titles[0]) if titles else None,
                )
                entity = f"'{xml_id}'" if xml_id is not None else str(number)
                target = plan.new_timeline(timeline, f"manifestation {entity}")
            if xml_id is not None:
                self.elements[xml_id] = target.id
            targets.append((manifestation, target))
        for key, target in mapped.items():
            if self.kinds[key] == "item":
                self.elements[key] = target.id

        about = plan.bundle_target
        for work in self.works:
            self._apply(
                plan,
                about,
                (),
                _Pair(
                    "schema:about",
                    {"id": self._iri(work)},
                    work,
                    inner=tuple(self._pairs("work", work)),
                ),
            )
        self._apply(
            plan,
            about,
            (),
            _Pair("rdfs:seeAlso", {"id": self._compact(self.base)}, None),
        )
        for manifestation, target in targets:
            for pair in self._pairs("manifestation", manifestation):
                self._apply(plan, target, (), pair)
            for item in _children(manifestation, "itemList", "item"):
                xml_id = _xml_id(item)
                if xml_id is not None and xml_id in mapped:
                    for pair in self._pairs("item", item):
                        self._apply(plan, mapped[xml_id], (), pair)

    # endregion

    # region Writing one pair

    def _basis(self, element: ET.Element | None) -> str:
        xml_id = None if element is None else _xml_id(element)
        return self.base if xml_id is None else f"{self.base}#{xml_id}"

    def _apply(self, plan: Plan, target: Target, node: NodePath, pair: _Pair) -> None:
        """State *pair* at *node* of *target*, then its inner pairs; or offer it.

        A candidate pair is offered, never stated. The inner pairs go into
        the value as the target holds it now; a value the target does not
        state (a candidate, or one withdrawn) takes none.
        """
        basis = [self._basis(pair.source)]
        if pair.candidate:
            plan.offer(
                target,
                pair.property,
                pair.value,
                stated_by=self.stated_by,
                date=self.day,
                basis=basis,
                note=pair.note,
                node=node,
            )
            return
        provenance: dict[str, Any] = {
            "stated_by": self.stated_by,
            "date": self.day,
            "method": "imported",
            "basis": basis,
        }
        if pair.note is not None:
            provenance["note"] = pair.note
        plan.state(target, pair.property, pair.value, provenance, node)
        if not pair.inner:
            return
        index = plan.held_index(target, pair.property, pair.value, node)
        if index is not None:
            for inner in pair.inner:
                self._apply(plan, target, (*node, pair.property, index), inner)

    # endregion

    # region IRIs

    def _compact(self, iri: str) -> str:
        return timetoshare.compact_iri(
            self.catalogue.prefixes, iri, aliases=self.catalogue.aliases
        )

    def _ref(self, token: str) -> dict[str, str]:
        """A target or same-as token as a node naming what it names."""
        if token.startswith("#"):
            element_id = self.elements.get(token[1:])
            if element_id is not None:
                return {"id": to_file_id(element_id)}
            return {"id": self._compact(self.base + token)}
        if _ABSOLUTE.match(token):
            return {"id": self._compact(token)}
        return {"id": self._compact(urljoin(self.base, token))}

    def _iri(self, element: ET.Element) -> str:
        """The IRI of a work, an expression or an item: the file's address and its xml:id."""
        return self._compact(f"{self.base}#{_xml_id(element)}")

    def _authority_iris(self, element: ET.Element) -> list[str]:
        """``@auth.uri`` completed by each token of ``@codedval``."""
        authority, coded = element.get("auth.uri"), element.get("codedval")
        if not authority or not coded:
            return []
        return [self._ref(authority + token)["id"] for token in coded.split()]

    def _iris(self, element: ET.Element) -> list[str]:
        """Each token of ``@sameas``, then the authority IRIs."""
        same = [
            self._ref(token)["id"] for token in (element.get("sameas") or "").split()
        ]
        return same + self._authority_iris(element)

    # endregion

    # region The pairs of an entity

    def _pairs(self, kind: Kind, element: ET.Element) -> list[_Pair]:
        """The pairs of an entity, in the order they are written."""
        pairs = [_Pair("type", _TYPES[kind], element)]
        pairs += self._title_pairs(kind, element)
        pairs += [
            self._identifier(identifier)
            for identifier in _children(element, "identifier")
        ]
        pairs += self._names(element)
        if kind == "item":
            pairs += self._location(_children(element, "physLoc"))
        if kind in ("work", "expression"):
            pairs += [
                self._date("schema:dateCreated", d)
                for d in _children(element, "creation", "date")
            ]
        elif kind == "manifestation":
            pairs += [
                self._date("schema:datePublished", d)
                for d in _children(element, "pubStmt", "date")
            ]
        if kind in ("work", "expression"):
            pairs += [self._key(key) for key in _children(element, "key")]
            pairs += [self._meter(meter) for meter in _children(element, "meter")]
            pairs += [self._medium(medium) for medium in self._media(element)]
        pairs += [
            _Pair("owl:sameAs", {"id": iri}, element) for iri in self._iris(element)
        ]
        pairs += self._nesting(kind, element)
        pairs += self._relations(kind, element)
        return pairs

    def _titles(self, kind: Kind, element: ET.Element) -> list[ET.Element]:
        if kind == "manifestation":
            found = _children(element, "titleStmt", "title")
        else:
            found = _children(element, "title") + _children(
                element, "titleStmt", "title"
            )
        return [title for title in found if _text(title)]

    def _title_pairs(self, kind: Kind, element: ET.Element) -> list[_Pair]:
        if kind == "item":
            return []
        pairs = []
        for title in self._titles(kind, element):
            text = _text(title)
            language = title.get(f"{_XML}lang")
            label: Any = {"@value": text, "@language": language} if language else text
            if kind == "expression":
                pairs.append(_Pair("rdfs:label", label, title))
                continue
            value: dict[str, Any] = {"type": ["melod:Title"], "rdfs:label": [label]}
            if (title_type := title.get("type")) is not None:
                term = _TITLE_TYPES.get(title_type, "OtherTitle")
                value["melod:hasTitleType"] = [{"id": f"melodTerms:{term}"}]
            pairs.append(_Pair("melod:hasTitle", value, title))
        return pairs

    def _identifier(self, identifier: ET.Element) -> _Pair:
        value: dict[str, Any] = {"type": ["melod:Identifier"]}
        label = identifier.get("label") or identifier.get("type")
        if label:
            value["rdfs:label"] = [label]
        value["owl:hasValue"] = [_text(identifier)]
        return _Pair("melod:hasIdentifier", value, identifier)

    def _names(self, element: ET.Element) -> list[_Pair]:
        """Row by row, the persons and bodies named in a role, document order."""
        title_statements = _children(element, "titleStmt")
        containers = [
            (element, False),
            *((statement, False) for statement in title_statements),
            *(
                (responsibility, True)
                for statement in title_statements
                for responsibility in _children(statement, "respStmt")
            ),
        ]
        found: list[tuple[ET.Element, ET.Element | None]] = []
        for container, responsibility in containers:
            for child in container:
                if _local(child) in _ROLE_ELEMENTS:
                    names = [n for n in child if _local(n) in ("persName", "corpName")]
                    if names:
                        found.extend((name, child) for name in names)
                    elif _text(child):
                        found.append((child, None))
                elif (
                    responsibility
                    and _local(child) in ("persName", "corpName")
                    and child.get("role")
                ):
                    found.append((child, None))
        order = {id(node): position for position, node in enumerate(element.iter())}
        found.sort(key=lambda pair: order[id(pair[0])])
        return [self._name(name, role_element) for name, role_element in found]

    def _name(self, name: ET.Element, role_element: ET.Element | None) -> _Pair:
        role = (
            name.get("role")
            or (role_element.get("role") if role_element is not None else None)
            or _local(role_element if role_element is not None else name)
        )
        code = _ROLE_CODES.get(role)
        note = None
        if code is None:
            if re.fullmatch(r"[a-z]{3}", role):
                code = role
            else:
                code, note = "ctb", f"role '{role}'"
        iris = self._iris(name)
        if not iris and role_element is not None:
            iris = self._iris(role_element)
        text = _text(name)
        if iris:
            return _Pair(f"rel:{code}", {"id": iris[0], "label": text}, name, note=note)
        unreconciled = "unreconciled" if note is None else f"unreconciled; {note}"
        return _Pair(
            f"rel:{code}", {"label": text}, name, candidate=True, note=unreconciled
        )

    def _date(self, property: str, date: ET.Element) -> _Pair:
        isodate = date.get("isodate")
        value: dict[str, Any]
        if isodate:
            value = {
                "type": ["melod:ExactDate"],
                "melod:isodate": [_date_value(isodate)],
            }
        else:
            value = {"type": ["melod:Date"]}
            for attribute, key in (
                ("notbefore", "melod:notBefore"),
                ("notafter", "melod:notAfter"),
            ):
                if bound := date.get(attribute):
                    value[key] = [_date_value(bound)]
        text = _text(date)
        if text and text != isodate:
            value["schema:description"] = [text]
        return _Pair(property, value, date)

    def _key(self, key: ET.Element) -> _Pair:
        value: dict[str, Any] = {"type": ["melod:Key"]}
        if pitch := key.get("pname"):
            value["melod:hasPitch"] = [{"id": f"melodTerms:{pitch.upper()}"}]
        if (accidental := _ACCIDENTALS.get(key.get("accid") or "")) is not None:
            value["melod:hasAccidental"] = [{"id": f"melodTerms:{accidental}"}]
        if (mode := key.get("mode")) is not None and mode.lower() in _MODES:
            value["melod:isInMode"] = [{"id": f"melodTerms:{mode.capitalize()}"}]
        return _Pair("mus:U11_has_key", value, key)

    def _meter(self, meter: ET.Element) -> _Pair:
        value: dict[str, Any] = {"type": ["melod:Meter"]}
        if (count := meter.get("count")) is not None:
            value["melod:hasMeterCount"] = [int(count) if count.isdigit() else count]
        if (unit := meter.get("unit")) is not None:
            value["melod:inMeterUnit"] = [int(unit)]
        if (symbol := meter.get("sym")) is not None:
            value["melod:usesMeterSymbol"] = [symbol]
        return _Pair("melod:hasMeter", value, meter)

    def _media(self, element: ET.Element) -> Iterator[ET.Element]:
        """Every ``perfRes`` of the entity's ``perfResList``s, nested lists included."""

        def walk(listing: ET.Element) -> Iterator[ET.Element]:
            """Yield the ``perfRes`` children of *listing*, descending into nested lists."""
            for child in listing:
                if _local(child) == "perfRes":
                    yield child
                elif _local(child) == "perfResList":
                    yield from walk(child)

        for listing in _children(element, "perfMedium", "perfResList"):
            yield from walk(listing)

    def _medium(self, medium: ET.Element) -> _Pair:
        value: dict[str, Any] = {}
        if text := _text(medium, skip="perfRes"):
            value["rdfs:label"] = [text]
        if coded := medium.get("codedval"):
            value["skos:notation"] = [coded]
        if (count := medium.get("count")) is not None:
            value["mus:U30_foresees_quantity_of_medium_of_performance"] = [int(count)]
        if iris := self._authority_iris(medium):
            value["owl:sameAs"] = [{"id": iri} for iri in iris]
        return _Pair("mus:U2_foresees_use_of_medium_of_performance", value, medium)

    def _location(self, locations: list[ET.Element]) -> list[_Pair]:
        """The shelfmarks and keepers of an item's ``physLoc``s."""
        pairs = [
            _Pair("melod:hasShelfmark", _text(identifier), identifier)
            for location in locations
            for identifier in _children(location, "identifier")
            if _text(identifier)
        ]
        for location in locations:
            for repository in _children(location, "repository"):
                names = [n for n in repository if _local(n) in ("corpName", "persName")]
                keeper = names[0] if names else repository
                if not (text := _text(keeper)):
                    continue
                iris = self._iris(keeper) or self._iris(repository)
                property = "cidoc:P50_has_current_keeper"
                if iris:
                    pairs.append(
                        _Pair(property, {"id": iris[0], "label": text}, keeper)
                    )
                else:
                    pairs.append(
                        _Pair(
                            property,
                            {"label": text},
                            keeper,
                            candidate=True,
                            note="unreconciled",
                        )
                    )
        return pairs

    def _nesting(self, kind: Kind, element: ET.Element) -> list[_Pair]:
        """A work's expressions; a manifestation's items."""
        if kind == "work":
            return [
                _Pair(
                    "lrmoo:R3_is_realised_in",
                    {"id": self._iri(expression)},
                    expression,
                    inner=(
                        ()
                        if expression in self.embodied
                        else tuple(self._pairs("expression", expression))
                    ),
                )
                for expression in _children(element, "expressionList", "expression")
            ]
        if kind != "manifestation":
            return []
        pairs = []
        for item in _children(element, "itemList", "item"):
            xml_id = _xml_id(item)
            if xml_id is not None and xml_id in self.elements:
                pairs.append(
                    _Pair(
                        "lrmoo:R7i_is_exemplified_by",
                        {"id": to_file_id(self.elements[xml_id])},
                        item,
                    )
                )
            else:
                pairs.append(self._item_node(self._pairs("item", item), item))
        if element.get("singleton") == "true" and (
            locations := _children(element, "physLoc")
        ):
            pairs.append(
                self._item_node(
                    [
                        _Pair("type", _TYPES["item"], element),
                        *self._location(locations),
                    ],
                    element,
                )
            )
        return pairs

    def _item_node(self, pairs: list[_Pair], source: ET.Element) -> _Pair:
        """An item as one value node: its stated pairs inside, its candidates after.

        An item with an ``xml:id`` is the node of the IRI ``<base>#<xml:id>``,
        as works and expressions are; any other item (and the item a
        singleton manifestation holds) is a blank node.
        """
        node: dict[str, Any] = {}
        if _local(source) == "item" and _xml_id(source) is not None:
            node["id"] = self._iri(source)
        for pair in pairs:
            if not pair.candidate:
                node.setdefault(pair.property, []).append(pair.value)
        return _Pair(
            "lrmoo:R7i_is_exemplified_by",
            node,
            source,
            inner=tuple(pair for pair in pairs if pair.candidate),
        )

    def _targets(
        self, element: ET.Element, rel: str
    ) -> Iterator[tuple[ET.Element, str]]:
        """Each relation *rel* of *element* with the ``xml:id`` of each ``#`` target."""
        for relation in _children(element, "relationList", "relation"):
            if relation.get("rel") == rel:
                for token in (relation.get("target") or "").split():
                    if token.startswith("#"):
                        yield relation, token[1:]

    def _embodiments(
        self,
    ) -> tuple[list[ET.Element], dict[ET.Element, list[tuple[ET.Element, ET.Element]]]]:
        """The expressions manifestations embody, by either side of the relation.

        A manifestation's ``isEmbodimentOf`` and an expression's
        ``hasEmbodiment`` state one fact: the expression is written under
        that manifestation's ``lrmoo:R4_embodies``.

        Returns:
            The expressions some manifestation embodies; and per
            manifestation, the expressions only a ``hasEmbodiment`` of the
            expression places under it, each with that relation.
        """
        embodied: list[ET.Element] = []
        by_expression: dict[ET.Element, list[tuple[ET.Element, ET.Element]]] = {}
        stated: set[tuple[int, int]] = set()
        for manifestation in self.manifestations:
            for _, target in self._targets(manifestation, "isEmbodimentOf"):
                if (expression := self.expression_by_id.get(target)) is not None:
                    stated.add((id(manifestation), id(expression)))
                    embodied.append(expression)
        for expression in self.expressions:
            for relation, target in self._targets(expression, "hasEmbodiment"):
                manifestation = self.manifestation_by_id.get(target)
                if manifestation is None:
                    continue
                embodied.append(expression)
                if (id(manifestation), id(expression)) not in stated:
                    stated.add((id(manifestation), id(expression)))
                    by_expression.setdefault(manifestation, []).append(
                        (expression, relation)
                    )
        return embodied, by_expression

    def _relations(self, kind: Kind, element: ET.Element) -> list[_Pair]:
        pairs = []
        word = kind.capitalize()
        for relation in _children(element, "relationList", "relation"):
            rel = relation.get("rel") or ""
            if rel in _LRMOO_RELATIONS:
                properties = [_LRMOO_RELATIONS[rel]]
            elif rel == "hasPart":
                properties = [f"melod:has{word}Part", "dcterms:hasPart"]
            elif rel == "isPartOf":
                properties = [f"melod:is{word}PartOf", "dcterms:isPartOf"]
            elif rel in _DERIVATIONS:
                properties = [f"melod:{rel}", "dcterms:source"]
            elif rel in _MELOD_RELATIONS:
                properties = [f"melod:{rel}"]
            else:
                continue
            for token in (relation.get("target") or "").split():
                local = token[1:] if token.startswith("#") else None
                if (
                    kind == "expression"
                    and rel == "hasEmbodiment"
                    and local in self.manifestation_by_id
                ):
                    continue  # the manifestation states it, by lrmoo:R4_embodies
                expression = self.expression_by_id.get(local) if local else None
                embodies = kind == "manifestation" and rel == "isEmbodimentOf"
                for property in properties:
                    if embodies and expression is not None:
                        pairs.append(self._embodiment(expression, relation))
                    else:
                        pairs.append(_Pair(property, self._ref(token), relation))
        if kind == "manifestation":
            pairs += [
                self._embodiment(expression, relation)
                for expression, relation in self.embodied_by_expression.get(element, [])
            ]
        return pairs

    def _embodiment(self, expression: ET.Element, relation: ET.Element) -> _Pair:
        """The expression in full under ``lrmoo:R4_embodies``, from *relation*."""
        return _Pair(
            "lrmoo:R4_embodies",
            {"id": self._iri(expression)},
            relation,
            inner=tuple(self._pairs("expression", expression)),
        )

    # endregion


def _date_value(text: str) -> Any:
    """An ISO date as a typed literal by its form; any other form as the string."""
    for pattern, datatype in _DATE_TYPES:
        if pattern.fullmatch(text):
            return {"@value": text, "@type": datatype}
    return text
