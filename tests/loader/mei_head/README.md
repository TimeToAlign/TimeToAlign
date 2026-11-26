# The MEI header import

`MeiHeadLoader` reads the `meiHead` of an MEI file and states what it says
about the work, its expressions, manifestations and items as statements with
provenance on a bundle and its timelines. The statement API itself is tested
in `tests/timelines/`; these tests prove the mapping from MEI elements to
pairs, where each pair lands, and what an import into an existing bundle does.

## Why hand-written headers prove the mapping

No MEI file is vendored: the specimens the mapping was designed from carry no
licence that allows it. Two headers are written by hand instead, as module
constants written into `tmp_path`, and together they hold every case of the
mapping at least once:

* `cnw0133.xml` (MEI 5.1), after a song of the Carl Nielsen catalogue: a work
  with two typed titles in two languages, a catalogue number, a composer
  given only by name, a lyricist with an authority IRI (`@auth.uri` completed
  by `@codedval`), an exact creation date, one expression (title, key, meter,
  two media of performance), a relation with a fallback property; two
  manifestations, one embodying the expression through a relation and one
  with `@sameas`, a main title and an item (an identifier, a shelfmark, a
  repository given only by name).
* `andante.xml` (MEI 4.0.1): a work with an authority IRI of its own
  (`@auth.uri` and `@codedval`), an identifier labelled by its `@type`, a
  title and a title of a type MeLODy does not list (`OtherTitle`) in its
  `titleStmt`; in `titleStmt/respStmt` a dedicatee, a name whose role is a
  three-letter code and one of an unknown role, none with an IRI; a composer
  element holding only text, a date given by its bounds, a meter whose
  count is no number; one expression no manifestation embodies (with a
  language-tagged title, a contributor of an unknown role that has an IRI,
  a meter of count and unit, a medium with a count and an authority IRI and
  one in a nested list); a singleton manifestation with a publication date,
  whose `physLoc` names a repository with an IRI, and a `hasPart` relation.

Three variants of `cnw0133.xml` cover what one header cannot hold at once:
the expression placed by its own `hasEmbodiment` instead of the
manifestation's `isEmbodimentOf`; two items of equal content; a `@sameas`
whose prefix no context declares.

Every expected statement and provenance entry is written from the mapping,
never copied from a run: the tests compare whole dictionaries for equality,
so a pair too many, a pair missing, a value in another form or an entry with
another basis fails.

## What lands where

* The **bundle** is never typed. The work is a value node under its
  `schema:about`, with the IRI `<base>#<xml:id>`; the bundle also points to
  the MEI file with `rdfs:seeAlso`.
* An **expression** is a value node with the IRI `<base>#<xml:id>`, written
  in full under `lrmoo:R4_embodies` of each manifestation embodying it; the
  work then names it by its IRI alone under `lrmoo:R3_is_realised_in`. Either
  side of the relation places it: the manifestation's `isEmbodimentOf` or
  the expression's `hasEmbodiment`. The variant stating only the latter
  gives the first header's statements exactly, the `lrmoo:R4_embodies` entry
  based on the expression's relation, and no `lrmoo:R4i_is_embodied_in` in
  the expression: one fact is stated once. An expression no manifestation
  embodies is written in full under the work's `lrmoo:R3_is_realised_in`
  (the second header proves this branch).
* Each **manifestation** becomes a new graphical timeline in millimetres of
  unknown length, named by its first title, its ID minted from the loader's
  minter (`cgt_…`; the tests load with `seed=0` and name the IDs a fresh
  `IdMinter(0)` mints), so a value naming it (`{"id": "#cgt_…"}`) carries its
  final ID.
* An **item** is a value node under its manifestation's
  `lrmoo:R7i_is_exemplified_by`, its pairs stated inside the node; a
  repository without an IRI is a candidate inside the node. An item with an
  `xml:id` has the IRI `<base>#<xml:id>` as its `id`, so two items of equal
  content stay two nodes (the variant with two such items); the item of a
  singleton manifestation, which has no element of its own, is a blank node.

The tests check the timelines' IDs, units, lengths and names, each holder's
`statements` and `provenance`, and the manifest written: `to_dict()` passes
`timetoshare.validate`, has no warnings, its `@context` declares exactly the
prefixes used, and it reads back to an equal manifest.

## Provenance

Every pair the import writes has one entry saying how it was made:
`stated_by` the library (or the agent given), the date given, `method:
imported`, and as `basis` the MEI file followed by `#<xml:id>` of the element
the pair comes from (the file alone when that element has no `xml:id`). The
entry holds no `beliefs`: a loader records how a pair was made and believes
nothing, so the expected entries are written without the key and the whole
dictionaries compared. A name without an IRI is a candidate (an entry with
no stated pair), never stated, with the note `unreconciled`. The default
date is not tested here: it is the statement API's.

## Importing into a bundle

* `into` with `subjects` lands a manifestation's statements on an existing
  timeline and an item's on an existing region (which the manifestation then
  names by `{"id": "#region_1"}`); no timeline is added for a mapped entity.
* Importing the same header again, with every manifestation mapped to the
  element the first import gave it, leaves the manifest written unchanged:
  a pair the subject states or holds an entry of is skipped. A candidate an
  editor believes `false` is not offered again: the manifest is unchanged,
  the entry keeps the editor's belief, and `get_candidates(believer=)` of
  that editor leaves it out while the unfiltered call keeps it.
* A decision taken on a value node survives a second import, which then
  changes nothing: the work an editor believes `false` under `schema:about`
  stays stated with that belief, and the editor's filtered provenance leaves
  it out; an expression an editor withdrew from a manifestation's
  `lrmoo:R4_embodies` (removed, then believed `false`) is not stated again
  and its pairs are not put back, since the manifestation no longer holds
  the node; the keeper candidate inside an item node stated there by an
  editor (the item is found by its IRI though its content changed; the
  entry keeps the import's members and gains the editor's belief `true`).
* `base` sets the address against which relative targets resolve and from
  which every basis and IRI is built; the default is the file name,
  percent-encoded, and the bundle's default ID is the file's stem with every
  character outside `[A-Za-z0-9_.-]` replaced by `_`.

## Refusals

Each refusal is checked by its exact message, and after each refusal of an
import `into` a bundle, the bundle's manifest is unchanged: `id` together
with `into`, `subjects` without `into`, a key naming no work, expression,
manifestation or item, a key naming a work or an expression (they are values,
never elements), a value naming no timeline or region of the bundle, a header
with two works, and a file holding no `meiHead`. A pair refused while the
import states it refuses the whole import before the bundle changes: the
variant whose manifestation has `@sameas="doi:10.1000/x"` (a prefix neither
the catalogue nor the holder declares) leaves the bundle's manifest
unchanged and adds no timeline; the message names the timeline the import
would have added by its manifestation, as it holds no ID yet.
