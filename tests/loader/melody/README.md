# MeLODy Import Tests - Validation Strategy

`MelodyLoader` reads a MeLODy data repository (one Turtle file per entity, in
LRMoo, MeLODy's own `melod:` terms, schema.org and DOREMUS) and carries its
triples as statements on a bundle, its manifestation timelines and value
nodes. These tests prove the mapping, not the RDF parser: rdflib is trusted to
read Turtle, and every expectation is written from the mapping below, never
copied from a run.

## Why a hand-written repository proves the mapping

The fixture repository is written by the test module into `tmp_path` (nothing
is vendored). Its shapes follow MeLODy's published data: the work states
`lrmoo:R3_is_realised_in` its expressions, a manifestation
`lrmoo:R4_embodies` an expression, an item `lrmoo:R7_exemplifies` its
manifestation; titles, identifiers, contributions, dates and meters are blank
nodes typed `melod:Title`, `melod:Identifier`, `melod:Contribution`,
`melod:Date`, `melod:Meter`; MeLODy writes schema.org as `https://schema.org/`.
A hand-written repository can prove the mapping because each rule of the
mapping is exercised at least once by a triple whose expected rendering
follows from that rule alone:

| Rule | Where it occurs |
|---|---|
| the work is a value node under the bundle's `schema:about`, the bundle untyped | the main repository |
| an expression embodied by a manifestation is a full node under that manifestation's `lrmoo:R4_embodies`, and a reference under the work's `lrmoo:R3_is_realised_in` | the main repository |
| an expression no manifestation embodies is a full node under the work's `lrmoo:R3_is_realised_in` | the kinds repository |
| a manifestation is a new millimetre timeline of unknown length, named by its title | the main repository |
| an item is a value node under its manifestation's `lrmoo:R7i_is_exemplified_by` | the main repository |
| a manifestation or an item mapped by `subjects` receives its pairs; the others name it `#<element>` | the mapped case |
| a subject states `type` (sorted), its own IRI as `owl:sameAs`, then its other triples in the code-point order of the compact property, values sorted by their canonical JSON | every subject |
| an IRI of an entity with an element is `{"id": "#<element>"}` | `lrmoo:R7_exemplifies` in the item node, the mapped case |
| another IRI is `{"id": compact IRI}`, with `label` and `owl:sameAs` when the graph holds triples about it | the Debussy agent (`schema:givenName` + `schema:familyName`, `owl:sameAs`); the kinds repository (`rdfs:label`, `skos:prefLabel`, `schema:name` in both schema.org namespaces, an IRI without triples) |
| a namespace the catalogue lacks stays a full IRI | the kinds repository |
| a blank node is a node; an `rdf:List` is `{"list": [...]}` in its order; a blank node containing itself is refused | titles, meters, dates; the kinds repository; the cycle case |
| literals: plain and `xsd:string` a string, `xsd:integer` an integer, `xsd:boolean` a boolean, a language tag a value object with `@language`, any other datatype a value object with `@type` (so `xsd:decimal` never becomes a float) | the meter count, the duration, the French title; the kinds repository |
| `https://schema.org/` compacts to `schema:` through the catalogue's alias | `schema:creativeWorkStatus`, `schema:dateCreated` |
| every pair stated carries a `confirmed` entry: `stated_by` the library, the date given, `method: imported`, `basis` the base followed by the subject's file (percent-encoded per segment); a pair whose value is an entity's node adds that entity's file when it differs | every subject; the base and file-name cases |
| files under `indexes/`, `configuration/`, `datasets/` are not read | the main repository's `indexes/works.ttl`, which types the work `melod:WorkEntity`: read, it would type the work twice and become the work's file |
| a pair already stated, proposed or rejected is skipped | importing twice; a rejected pair |

Every value written is checked by `timetoshare.validate` through
`to_dict()`, with no warning, and the manifest read back writes itself
unchanged: the import produces a manifest the format accepts, and the
library's reader and writer keep it.

## The mapping, as the expectations apply it

* The bundle states one pair, `schema:about` the work node `{"id": W,
  "type": [...], ...W's triples}`. Works and expressions are values, never
  elements: neither becomes a timeline and `subjects` refuses to map one.
* The expressions are the objects of `W lrmoo:R3_is_realised_in`; the
  manifestations every `M` with `M lrmoo:R4_embodies E` for such an `E`; the
  items of `M` every `I` with `I lrmoo:R7_exemplifies M`.
* An entity's file is the first file, in sorted relative-path order, whose
  graph types it; its pairs and blank nodes are read from that file's graph.
  Labels and `owl:sameAs` of a referenced IRI come from any file.
* A node's `type` and keys follow the subject's rules; an expanded entity
  node (work, expression, item) has its compact IRI as `id` and no
  `owl:sameAs` of itself.

## Cases

1. **Main repository** (`works/1.ttl`, `expressions/2.ttl`,
   `persons/3.ttl`, `manifestations/4.ttl`, `items/5.ttl`,
   `indexes/works.ttl`), `create_bundle(id="clair_de_lune",
   date="2026-11-19")`: timelines `["cgt_1"]`; bundle and `cgt_1` statements
   and provenance exact; `cgt_1` in millimetres, length unknown, named
   "Suite bergamasque, first edition"; the written manifest validates
   without warnings, its `@context` prefixes are exactly `dcterms lrmoo melod
   melodTerms owl rdf rdfs schema wd xsd`, and it round-trips.
2. **Re-import** into the bundle of case 1 with the manifestation mapped to
   `cgt_1`: `to_dict()` unchanged.
3. **A rejected pair** stays rejected: the manifestation's title rejected on
   `cgt_1`, then a re-import leaves the title unstated and the entry
   rejected.
4. **Mapped elements**: the repository plus a second work; `into` holds
   `clt_6` (quarters, 27) with `region_1` (0 to 12); the manifestation maps
   to `clt_6` and the item to `region_1`: no timeline is added, `clt_6`
   states the manifestation's pairs with the item as `{"id": "#region_1"}`,
   `region_1` the item's pairs with `lrmoo:R7_exemplifies` `{"id":
   "#clt_6"}`.
5. **Errors**, messages exact, `into` unchanged after each: two works
   without `work`; a `work` that is no work; no work at all; `id` with
   `into`; `subjects` without `into`; a key that is no entity; a work, an
   expression, and a person as keys; a value naming no element; a parse
   error (the expected text is rdflib's own error for the same file); a
   blank node containing itself.
6. **Kinds repository**, one file named with a space and given directly:
   every literal and object kind of the mapping not in the main repository;
   the basis is the percent-encoded file name, and with `base` given the
   base followed by it.
7. **rdflib missing** (its import made to fail): `load` raises
   `ImportError('MelodyLoader needs rdflib: pip install "timetoalign[rdf]"')`.
