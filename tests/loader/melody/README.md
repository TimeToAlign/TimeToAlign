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
| every pair stated carries one entry, holding no `beliefs` (a loader believes nothing): `stated_by` the library, the date given, `method: imported`, `basis` the base followed by the subject's file (percent-encoded per segment); a pair whose value is an entity's node adds that entity's file when it differs | every subject; the base and file-name cases |
| files under `indexes/`, `configuration/`, `datasets/` are not read | the main repository's `indexes/works.ttl`, which types the work `melod:WorkEntity`: read, it would type the work twice and become the work's file |
| a pair the subject states or holds an entry of is skipped, so a recorded belief stands | importing twice; a pair an editor believes `false` |
| a file is read per source and relative path: two sources holding files of one relative path are both read, each named by that path in a `basis` | two files `x.ttl` in two folders, given directly |
| every pair is checked before the bundle changes | a manifestation stating `owl:sameAs <doi:10.1000/x>`, a prefix no context declares |

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

The loader mints the ID of every timeline it adds from its minter
(`cgt_…`); the tests load with `seed=0`, so the first new timeline is the
first `cgt` ID of a fresh `IdMinter(0)`, written `cgt_…` below.

1. **Main repository** (`works/1.ttl`, `expressions/2.ttl`,
   `persons/3.ttl`, `manifestations/4.ttl`, `items/5.ttl`,
   `indexes/works.ttl`), `create_bundle(id="clair_de_lune",
   date="2026-11-19")`: timelines `["cgt_…"]`; bundle and `cgt_…` statements
   and provenance exact; `cgt_…` in millimetres, length unknown, named
   "Suite bergamasque, first edition"; the written manifest validates
   without warnings, its `@context` prefixes are exactly `dcterms lrmoo melod
   melodTerms owl rdf rdfs schema wd xsd`, and it round-trips.
2. **Re-import** into the bundle of case 1 with the manifestation mapped to
   `cgt_…`: `to_dict()` unchanged.
3. **A pair believed `false`** keeps its belief: an editor believes the
   manifestation's title on `cgt_…` `false`; a re-import leaves `to_dict()`
   unchanged, the title stated (a belief never removes a pair), its entry
   the import's members plus the editor's belief, and the editor's filtered
   provenance (`get_provenance(believer=)`) without the title's entry.
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
   blank node containing itself; a pair refused while it is stated (the
   manifestation's `owl:sameAs <doi:10.1000/x>`): no timeline is added and
   the message names the timeline the import would have added by its
   manifestation, as it holds no ID yet.
6. **Kinds repository**, a folder holding a work file whose name has a
   space and an expression file: every literal and object kind of the
   mapping not in the main repository; the basis is the work file's path,
   percent-encoded, and with `base` given the base followed by it.
7. **Files given directly**: the main repository's work and expression
   files given as two sources: no manifestation, so the expression is a
   full node under the work's `lrmoo:R3_is_realised_in` and the basis names
   both files by their names. Two files of one name, `a/x.ttl` (the work)
   and `b/x.ttl` (the expression, the manifestation and the person), given
   directly: both are read, the result is the main repository's without the
   item, and every basis is `x.ttl`.
8. **rdflib missing** (its import made to fail): `load` raises
   `ImportError('MelodyLoader needs rdflib: pip install "timetoalign[rdf]"')`.
