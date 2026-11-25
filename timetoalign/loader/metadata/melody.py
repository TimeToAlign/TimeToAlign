"""The MeLODy import: a MeLODy data repository's Turtle as statements.

MeLODy keeps its metadata as one Turtle file per entity (a work, an
expression, a manifestation, an item, a person, a place, ...), written in
LRMoo, its own ``melod:`` terms, schema.org, DOREMUS and CIDOC CRM. Its
statements are already property–value pairs and small nodes, so
:class:`MelodyLoader` carries them as they are:

* the work becomes a value node under the bundle's ``schema:about``;
* each expression of the work becomes a value node under
  ``lrmoo:R4_embodies`` of every manifestation embodying it (under the work
  node's ``lrmoo:R3_is_realised_in`` where no manifestation embodies it);
* each manifestation embodying one of them becomes a timeline in
  millimetres of unknown length (or the element ``subjects`` maps it to);
* each item exemplifying such a manifestation becomes a value node under
  the manifestation's ``lrmoo:R7i_is_exemplified_by`` (or the element
  ``subjects`` maps it to).

Every pair stated carries a provenance entry naming the file it was read
from as its ``basis``; the import records how a pair was made and holds no
belief. IRIs are compacted with the prefixes of
TimeToShare's catalogue, which reads MeLODy's ``https://schema.org/`` as
``schema:``; an IRI of a namespace the catalogue lacks stays a full IRI.

The import needs rdflib (``pip install "timetoalign[rdf]"``).
"""

from __future__ import annotations

import datetime
import json
import re
import urllib.parse
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import timetoshare
from timetoshare import load_catalogue

from timetoalign.alignment.bundle import AlignmentBundle
from timetoalign.core import TimeUnit
from timetoalign.loader.base import Loader
from timetoalign.loader.metadata._statements import (
    Plan,
    Target,
    day,
    refuse_arguments,
    refuse_value_entity,
)
from timetoalign.timelines.statements import LIBRARY_AGENT
from timetoalign.timelines.types import ContinuousGraphicalTimeline

__all__ = ["MelodyLoader", "MelodyRecord"]

_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_XSD = "http://www.w3.org/2001/XMLSchema#"
_OWL = "http://www.w3.org/2002/07/owl#"
_SKOS = "http://www.w3.org/2004/02/skos/core#"
_MELOD = "https://lod.academy/melod/vocab/ontology#"
_LRMOO = "http://iflastandards.info/ns/lrm/lrmoo/"
_SCHEMAS = ("http://schema.org/", "https://schema.org/")

_TYPE = _RDF + "type"
_SAME_AS = _OWL + "sameAs"
_WORK = _MELOD + "Work"
_EXPRESSION = _MELOD + "Expression"
_REALISED_IN = _LRMOO + "R3_is_realised_in"
_EMBODIES = _LRMOO + "R4_embodies"
_EXEMPLIFIES = _LRMOO + "R7_exemplifies"
_EXEMPLIFIED_BY = _LRMOO + "R7i_is_exemplified_by"

#: The top-level directories of a MeLODy repository that hold no entity: its
#: indexes, its configuration and the datasets generated from the entities.
_NOT_ENTITIES = ("indexes", "configuration", "datasets")

_INTEGER = re.compile("[+-]?[0-9]+")
_BOOLEANS = {"true": True, "1": True, "false": False, "0": False}

_RDFLIB_MISSING = 'MelodyLoader needs rdflib: pip install "timetoalign[rdf]"'


def _rdflib() -> Any:
    """Import rdflib, or say how to install it."""
    try:
        import rdflib
    except ImportError as error:
        raise ImportError(_RDFLIB_MISSING) from error
    return rdflib


@dataclass(frozen=True)
class MelodyRecord:
    """One source read: a Turtle file or a MeLODy repository directory.

    Attributes:
        source: The path given to :meth:`MelodyLoader.load`.
        files: Per Turtle file read, its path relative to the directory (a
            file given directly: its name) and its ``rdflib.Graph``, in
            sorted relative-path order. Two sources may hold files of one
            relative path; each is read.
    """

    source: Path
    files: tuple[tuple[str, Any], ...]


class MelodyLoader(Loader[MelodyRecord]):
    """Read MeLODy's Turtle and carry its triples as statements.

    ``load(*sources)`` takes Turtle files and MeLODy data repositories: a
    directory is read as one, every ``*.ttl`` at any depth except under
    ``indexes/``, ``configuration/`` and ``datasets/``. Each file is parsed
    into its own graph. :meth:`create_bundle` then writes one work, its
    manifestations and their items as statements on a bundle, as the module
    describes.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from timetoalign.loader.metadata.melody import MelodyLoader
        >>> folder = Path(tempfile.mkdtemp())
        >>> _ = (folder / "work.ttl").write_text(
        ...     "@prefix melod: <https://lod.academy/melod/vocab/ontology#> .\\n"
        ...     "@prefix schema: <https://schema.org/> .\\n"
        ...     "<urn:uuid:works/1> a melod:Work ; schema:creativeWorkStatus 'authentic' .\\n",
        ...     encoding="utf-8",
        ... )
        >>> bundle = MelodyLoader.from_file(folder / "work.ttl").create_bundle(
        ...     id="clair_de_lune", date="2026-11-19"
        ... )
        >>> list(bundle.statements)
        ['schema:about']
        >>> work = bundle.statements["schema:about"][0]
        >>> work["id"], work["type"], work["schema:creativeWorkStatus"]
        ('urn:uuid:works/1', ['melod:Work'], ['authentic'])
        >>> bundle.provenance[0]["basis"]
        ['work.ttl']
    """

    def __init__(self) -> None:
        """Initialise a loader holding no source."""
        super().__init__()
        self._records: list[MelodyRecord] = []

    def _load_source(self, source: Path) -> MelodyRecord:
        """Parse a Turtle file, or every entity file of a repository directory.

        Raises:
            ValueError: If a file is no Turtle (``<path>: <rdflib's error>``).
            ImportError: If rdflib is not installed.
        """
        rdflib = _rdflib()
        if source.is_dir():
            paths = sorted(
                (path.relative_to(source).as_posix(), path)
                for path in source.rglob("*.ttl")
                if path.is_file()
                and path.relative_to(source).parts[0] not in _NOT_ENTITIES
            )
        else:
            paths = [(source.name, source)]
        files = []
        for relative, path in paths:
            graph = rdflib.Graph()
            try:
                graph.parse(path, format="turtle")
            except Exception as error:  # rdflib raises several error classes
                raise ValueError(f"{path}: {error}") from error
            files.append((relative, graph))
        return MelodyRecord(source=source, files=tuple(files))

    def _accept_source(
        self, path: Path, source_meta: dict[str, Any], payload: MelodyRecord
    ) -> None:
        super()._accept_source(path, source_meta, payload)
        self._records.append(payload)

    @property
    def records(self) -> list[MelodyRecord]:
        """The sources read, each with its files and graphs, in loading order."""
        return list(self._records)

    def clear(self) -> None:
        """Forget every source read."""
        super().clear()
        self._records.clear()

    def create_bundle(
        self,
        *,
        work: str | None = None,
        into: AlignmentBundle | None = None,
        subjects: Mapping[str, str] | None = None,
        id: str | None = None,
        base: str = "",
        stated_by: str | Sequence[str] = LIBRARY_AGENT,
        date: str | datetime.date | None = None,
    ) -> AlignmentBundle:
        """State one work, its manifestations and their items on a bundle.

        The bundle states ``schema:about`` the work, a value node carrying
        the work's triples. Each expression of the work is a value node
        under ``lrmoo:R4_embodies`` of every manifestation embodying it, or
        under the work node's ``lrmoo:R3_is_realised_in`` when none does.
        Each manifestation embodying an expression of the work is a new
        :class:`~timetoalign.timelines.ContinuousGraphicalTimeline` in
        millimetres, of unknown length, named by its ``rdfs:label`` or else
        by the label of its first title; the timelines join the bundle
        ungrouped before any statement is added. Each item exemplifying such
        a manifestation is a value node under the manifestation's
        ``lrmoo:R7i_is_exemplified_by``.

        A subject (the manifestation's timeline, a mapped element) states
        its ``type``, its own IRI as ``owl:sameAs``, then its other triples
        as they are: in the code-point order of the compact property,
        values sorted by their canonical JSON. An IRI naming an entity with
        an element is ``{"id": "#<element>"}``; another IRI ``{"id": ...}``
        with the ``label`` and ``owl:sameAs`` the graph states of it; a
        blank node a node of its own triples, an ``rdf:List`` ``{"list":
        [...]}``. Each pair stated carries an entry holding no belief:
        method ``imported``, basis the subject's file (and the file of the
        entity whose node the value is, where it differs). A pair the
        subject already states or holds an entry of (a candidate, a pair
        someone believes) is skipped, so importing twice into one bundle
        adds nothing and every belief recorded stands. Every pair is checked
        before the bundle changes: a refused one leaves it as it was.

        Args:
            work: The IRI of the ``melod:Work`` the bundle is about; the
                only work of the graph when None.
            into: A bundle to import into instead of a new one; it is
                returned.
            subjects: Maps the IRI of a manifestation or an item of the work
                to the ID of a timeline (at any depth) or region of
                ``into``, which then receives that entity's statements
                instead of a new timeline or value node.
            id: The ID of a new bundle; minted (and never written) when
                None. Pass one so that the bundle has an IRI of its own.
            base: Written before each file's relative path in a ``basis``.
            stated_by: Who the entries name as having stated the pairs.
            date: The day of the entries; today when None.

        Returns:
            The bundle: ``into``, or a new one.

        Raises:
            ValueError: Before anything changes, if ``id`` is given with
                ``into``; ``subjects`` is given without ``into``; the graph
                holds no work, or several and ``work`` is None; ``work`` is
                no ``melod:Work`` of the graph; ``subjects`` names no entity
                of the graph, a work, an expression, or an entity that is no
                manifestation or item of the work, or maps one to an ID
                naming no timeline or region of ``into``; a blank node
                contains itself; a pair is refused (``the prefix 'doi' is
                declared neither by the catalogue nor on the new timeline of
                manifestation 'urn:uuid:manifestations/4'``).
        """
        refuse_arguments(into, id, subjects)
        graph = _Repository(self._records)
        work_iri = graph.work(work)
        expressions = graph.objects(work_iri, _REALISED_IN)
        manifestations = sorted(
            {m for e in expressions for m in graph.subjects(_EMBODIES, e)}
        )
        items = {m: graph.subjects(_EXEMPLIFIES, m) for m in manifestations}
        bundle = into if into is not None else AlignmentBundle(id=id or "")
        plan = Plan(bundle)
        targets = self._mapped_elements(
            graph, work_iri, manifestations, items, subjects or {}, plan
        )
        importer = _Importer(
            graph,
            work_iri,
            expressions,
            manifestations,
            items,
            base=base,
            provenance={
                "stated_by": (
                    [stated_by] if isinstance(stated_by, str) else list(stated_by)
                ),
                "date": day(date),
                "method": "imported",
            },
        )
        for manifestation in manifestations:
            if manifestation not in targets:
                timeline = ContinuousGraphicalTimeline(
                    length=None,
                    unit=TimeUnit.millimeters,
                    name=importer.timeline_name(manifestation),
                )
                targets[manifestation] = plan.new_timeline(
                    timeline, f"manifestation '{manifestation}'"
                )
        element_ids = {iri: target.id for iri, target in targets.items()}
        for subject, property, value, basis in importer.plan(element_ids):
            target = plan.bundle_target if subject is None else targets[subject]
            plan.state(target, property, value, {**importer.provenance, "basis": basis})
        plan.commit()
        return bundle

    @staticmethod
    def _mapped_elements(
        graph: _Repository,
        work: str,
        manifestations: Sequence[str],
        items: Mapping[str, Sequence[str]],
        subjects: Mapping[str, str],
        plan: Plan,
    ) -> dict[str, Target]:
        """Resolve *subjects* to the elements of the bundle, refusing what cannot map."""
        allowed = {*manifestations, *(i for found in items.values() for i in found)}
        mapped: dict[str, Target] = {}
        for key, element_id in subjects.items():
            types = graph.types(key)
            if not types:
                raise ValueError(
                    f"subjects names '{key}', which is no entity of the graph"
                )
            for kind, word in ((_WORK, "work"), (_EXPRESSION, "expression")):
                if kind in types:
                    refuse_value_entity(key, word)
            if key not in allowed:
                raise ValueError(
                    f"subjects maps '{key}', which is no manifestation or item of the "
                    f"work '{work}'"
                )
            mapped[key] = plan.mapped(key, element_id)
        return mapped


# region The graph


class _Repository:
    """The graphs of every file read, and what the import asks of them.

    A file is keyed by its relative path and the position of its source in
    loading order, so that two sources holding files of one relative path
    (``a/x.ttl`` and ``b/x.ttl`` given directly) are both read; a basis
    names a file by its relative path.
    """

    def __init__(self, records: Sequence[MelodyRecord]) -> None:
        rdflib = _rdflib()
        self._rdflib = rdflib
        self.files: dict[tuple[str, int], Any] = {}
        self.union = rdflib.Graph()
        for position, record in enumerate(records):
            for relative, graph in record.files:
                self.files[relative, position] = graph
                self.union += graph
        # An entity's file is the first file, in relative-path order (then
        # loading order), whose graph types it.
        self.entity_files: dict[str, tuple[str, int]] = {}
        for file in sorted(self.files):
            for subject in self.files[file].subjects(rdflib.RDF.type, None):
                if isinstance(subject, rdflib.URIRef):
                    self.entity_files.setdefault(str(subject), file)

    def uri(self, iri: str) -> Any:
        """The RDF node of *iri*."""
        return self._rdflib.URIRef(iri)

    def types(self, iri: str) -> set[str]:
        """The classes any file types *iri* with."""
        return {
            str(o) for o in self.union.objects(self.uri(iri), self._rdflib.RDF.type)
        }

    def objects(self, iri: str, predicate: str) -> list[str]:
        """The entities *iri* has as objects of *predicate*, sorted."""
        found = self.union.objects(self.uri(iri), self.uri(predicate))
        return sorted({str(o) for o in found if str(o) in self.entity_files})

    def subjects(self, predicate: str, iri: str) -> list[str]:
        """The entities that have *iri* as object of *predicate*, sorted."""
        found = self.union.subjects(self.uri(predicate), self.uri(iri))
        return sorted({str(s) for s in found if str(s) in self.entity_files})

    def work(self, work: str | None) -> str:
        """The work the bundle is about.

        Raises:
            ValueError: If *work* is no work of the graph, or it is None and
                the graph holds no work or several.
        """
        works = sorted(
            str(s)
            for s in self.union.subjects(self._rdflib.RDF.type, self.uri(_WORK))
            if isinstance(s, self._rdflib.URIRef)
        )
        if work is not None:
            if work not in works:
                raise ValueError(f"'{work}' is no melod:Work of the graph")
            return work
        if not works:
            raise ValueError("the graph holds no melod:Work")
        if len(works) > 1:
            listed = ", ".join(f"'{iri}'" for iri in works)
            raise ValueError(
                f"the graph holds {len(works)} works: give work, one of {listed}"
            )
        return works[0]

    def file_of(self, iri: str) -> str:
        """The relative path of the entity's file."""
        return self.entity_files[iri][0]

    def graph_of(self, iri: str) -> Any:
        """The graph of the entity's file."""
        return self.files[self.entity_files[iri]]

    def describes(self, node: Any) -> bool:
        """Whether any file holds a triple about *node*."""
        return (node, None, None) in self.union


# endregion


# region Values


class _Importer:
    """Turns the triples of the subjects into the pairs a bundle states."""

    def __init__(
        self,
        graph: _Repository,
        work: str,
        expressions: Sequence[str],
        manifestations: Sequence[str],
        items: Mapping[str, Sequence[str]],
        *,
        base: str,
        provenance: dict[str, Any],
    ) -> None:
        self.graph = graph
        self.rdflib = graph._rdflib
        self.work = work
        self.expressions = list(expressions)
        self.manifestations = list(manifestations)
        self.items = items
        self.base = base
        self.provenance = provenance
        catalogue = load_catalogue()
        self._prefixes = catalogue.prefixes
        self._aliases = catalogue.aliases
        # The expressions no manifestation embodies: written in full under
        # the work node, the others under the manifestations embodying them.
        self._unembodied = [
            e for e in self.expressions if not graph.subjects(_EMBODIES, e)
        ]
        self._elements: Mapping[str, str] = {}

    # region Plan

    def plan(
        self, elements: Mapping[str, str]
    ) -> list[tuple[str | None, str, Any, list[str]]]:
        """Yield ``(subject, property, value, basis)`` for every pair, in order.

        *subject* is None for the bundle, else the IRI of a manifestation or
        mapped item; *elements* maps the IRIs that have an element to its ID.
        """
        self._elements = elements
        pairs: list[tuple[str | None, str, Any, list[str]]] = []
        work_file = self.graph.file_of(self.work)
        work_node = self._entity_node(
            self.work, expand={_REALISED_IN: self._unembodied}
        )
        expression_files = (self.graph.file_of(e) for e in self._unembodied)
        pairs.append(
            (None, "schema:about", work_node, self._basis(work_file, *expression_files))
        )
        for manifestation in self.manifestations:
            items = self.items[manifestation]
            pairs.extend(
                (manifestation, property, value, basis)
                for property, value, basis in self._subject_pairs(
                    manifestation,
                    expand={_EMBODIES: self.expressions},
                    leave_out={_EXEMPLIFIED_BY: items},
                )
            )
            for item in items:
                value = (
                    self._element(item)
                    if item in elements
                    else self._entity_node(item, expand={})
                )
                basis = self._basis(self.graph.file_of(item))
                pairs.append(
                    (manifestation, "lrmoo:R7i_is_exemplified_by", value, basis)
                )
            for item in items:
                if item in elements:
                    pairs.extend(
                        (item, property, value, basis)
                        for property, value, basis in self._subject_pairs(item, {}, {})
                    )
        return pairs

    def timeline_name(self, iri: str) -> str | None:
        """The name of a manifestation's new timeline: its label, or its first title's."""
        graph = self.graph.graph_of(iri)
        subject = self.graph.uri(iri)
        label = self._first_text(graph.objects(subject, self.rdflib.RDFS.label))
        if label is not None:
            return label
        titles = sorted(
            (
                self._value(title, graph, self.graph.file_of(iri), ())
                for title in graph.objects(subject, self.graph.uri(_MELOD + "hasTitle"))
            ),
            key=_canonical,
        )
        if not titles or not isinstance(titles[0], dict):
            return None
        labels = titles[0].get("rdfs:label", [])
        if not labels:
            return None
        first = labels[0]
        return first["@value"] if isinstance(first, dict) else str(first)

    def _basis(self, *files: str) -> list[str]:
        """The basis of a pair: its files, each once, after the base."""
        seen: list[str] = []
        for file in files:
            if file not in seen:
                seen.append(file)
        return [self.base + _quoted(file) for file in seen]

    def _subject_pairs(
        self,
        iri: str,
        expand: Mapping[str, Sequence[str]],
        leave_out: Mapping[str, Sequence[str]],
    ) -> Iterator[tuple[str, Any, list[str]]]:
        """The pairs an element standing for the entity *iri* states, in order."""
        file = self.graph.file_of(iri)
        graph = self.graph.graph_of(iri)
        subject = self.graph.uri(iri)
        for type_iri in sorted(
            self.compact(str(o)) for o in graph.objects(subject, self.rdflib.RDF.type)
        ):
            yield "type", type_iri, self._basis(file)
        yield "owl:sameAs", {"id": self.compact(iri)}, self._basis(file)
        for property, values in self._properties(
            subject, graph, file, (), expand, leave_out
        ):
            for value, entity in values:
                if entity is None:
                    yield property, value, self._basis(file)
                else:
                    yield property, value, self._basis(file, self.graph.file_of(entity))

    def _entity_node(
        self, iri: str, expand: Mapping[str, Sequence[str]]
    ) -> dict[str, Any]:
        """The value node of an entity: its compact IRI as ``id``, then its triples."""
        graph = self.graph.graph_of(iri)
        subject = self.graph.uri(iri)
        node: dict[str, Any] = {"id": self.compact(iri)}
        node.update(self._node(subject, graph, self.graph.file_of(iri), (), expand))
        return node

    def _node(
        self,
        subject: Any,
        graph: Any,
        file: str,
        within: tuple[Any, ...],
        expand: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, Any]:
        """The ``type`` and pairs of a node built from the triples of *subject*."""
        node: dict[str, Any] = {}
        types = sorted(
            self.compact(str(o)) for o in graph.objects(subject, self.rdflib.RDF.type)
        )
        if types:
            node["type"] = types
        for property, values in self._properties(
            subject, graph, file, within, expand or {}, {}
        ):
            node[property] = [value for value, _ in values]
        return node

    def _properties(
        self,
        subject: Any,
        graph: Any,
        file: str,
        within: tuple[Any, ...],
        expand: Mapping[str, Sequence[str]],
        leave_out: Mapping[str, Sequence[str]],
    ) -> list[tuple[str, list[tuple[Any, str | None]]]]:
        """The triples of *subject* but its types, by compact property.

        Properties in code-point order, each with its values sorted by
        their canonical JSON; a value that is an expanded entity's node is
        given with that entity's IRI, any other with None.
        """
        grouped: dict[str, list[tuple[Any, str | None]]] = {}
        for predicate, obj in graph.predicate_objects(subject):
            predicate_iri = str(predicate)
            if predicate_iri == _TYPE:
                continue
            if str(obj) in leave_out.get(predicate_iri, ()):
                continue
            if isinstance(obj, self.rdflib.URIRef) and str(obj) in expand.get(
                predicate_iri, ()
            ):
                entry = (self._entity_node(str(obj), expand={}), str(obj))
            else:
                entry = (self._value(obj, graph, file, (*within, subject)), None)
            grouped.setdefault(self.compact(predicate_iri), []).append(entry)
        return [
            (
                property,
                sorted(grouped[property], key=lambda entry: _canonical(entry[0])),
            )
            for property in sorted(grouped)
        ]

    def _value(self, obj: Any, graph: Any, file: str, within: tuple[Any, ...]) -> Any:
        """The value an object of a triple read from *file* is written as."""
        rdflib = self.rdflib
        if isinstance(obj, rdflib.Literal):
            return self._literal(obj)
        if isinstance(obj, rdflib.BNode):
            if obj in within:
                raise ValueError(f"a blank node of {file} contains itself")
            if (obj, rdflib.RDF.first, None) in graph:
                from rdflib.collection import Collection

                return {
                    "list": [
                        self._value(item, graph, file, (*within, obj))
                        for item in Collection(graph, obj)
                    ]
                }
            return self._node(obj, graph, file, within)
        iri = str(obj)
        if obj == rdflib.RDF.nil:
            return {"list": []}
        if iri in self._elements:
            return self._element(iri)
        return self._reference(iri)

    def _element(self, iri: str) -> dict[str, str]:
        """``{"id": "#<element>"}`` for an entity that has an element."""
        return {"id": timetoshare.to_file_id(self._elements[iri])}

    def _reference(self, iri: str) -> dict[str, Any]:
        """An IRI as a thing, with the label and sameness the graph states of it."""
        value: dict[str, Any] = {"id": self.compact(iri)}
        subject = self.graph.uri(iri)
        if not self.graph.describes(subject):
            return value
        union = self.graph.union
        rdfs, skos = self.rdflib.RDFS, self.rdflib.Namespace(_SKOS)
        label = self._first_text(union.objects(subject, rdfs.label))
        if label is None:
            label = self._first_text(union.objects(subject, skos.prefLabel))
        if label is None:
            label = self._first_text(self._schema_objects(subject, "name"))
        if label is None:
            given = self._first_text(self._schema_objects(subject, "givenName"))
            family = self._first_text(self._schema_objects(subject, "familyName"))
            if given is not None and family is not None:
                label = f"{given} {family}"
        if label is not None:
            value["label"] = label
        same = sorted(
            self.compact(str(o))
            for o in union.objects(subject, self.graph.uri(_SAME_AS))
            if isinstance(o, self.rdflib.URIRef)
        )
        if same:
            value["owl:sameAs"] = [{"id": other} for other in same]
        return value

    def _first_text(self, objects: Iterable[Any]) -> str | None:
        """The first, in sorted order, of the texts of the literals among *objects*."""
        texts = sorted(str(o) for o in objects if isinstance(o, self.rdflib.Literal))
        return texts[0] if texts else None

    def _schema_objects(self, subject: Any, name: str) -> Iterator[Any]:
        for namespace in _SCHEMAS:
            yield from self.graph.union.objects(
                subject, self.graph.uri(namespace + name)
            )

    def _literal(self, literal: Any) -> Any:
        """A literal as the file writes it; no number becomes a float."""
        lexical = str(literal)
        if literal.language:
            return {"@value": lexical, "@language": literal.language}
        datatype = None if literal.datatype is None else str(literal.datatype)
        if datatype is None or datatype == _XSD + "string":
            return lexical
        if datatype == _XSD + "integer" and _INTEGER.fullmatch(lexical):
            return int(lexical)
        if datatype == _XSD + "boolean" and lexical in _BOOLEANS:
            return _BOOLEANS[lexical]
        return {"@value": lexical, "@type": self.compact(datatype)}

    def compact(self, iri: str) -> str:
        """An IRI with a prefix of the catalogue, its aliases applied, else as given."""
        return timetoshare.compact_iri(self._prefixes, iri, aliases=self._aliases)

    # endregion


def _canonical(value: Any) -> str:
    """The canonical JSON a value is sorted by."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _quoted(relative: str) -> str:
    """A relative path percent-encoded segment by segment."""
    return "/".join(
        urllib.parse.quote(segment, safe="") for segment in relative.split("/")
    )


# endregion
