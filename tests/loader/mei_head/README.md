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
  with `@sameas`, a main title and an item (shelfmark, a repository given
  only by name).
* `andante.xml` (MEI 4.0.1): a work titled in its `titleStmt`, a dedicatee
  in `titleStmt/respStmt`, a composer element holding only text, a date
  given by its bounds, one expression no manifestation embodies (with a
  language-tagged title and a contributor of an unknown role that has an
  IRI), a singleton manifestation whose `physLoc` names a repository with an
  IRI, and a `hasPart` relation.

Every expected statement and provenance entry is written from the mapping,
never copied from a run: the tests compare whole dictionaries for equality,
so a pair too many, a pair missing, a value in another form or an entry with
another basis fails.

## What lands where

* The **bundle** is never typed. The work is a value node under its
  `schema:about`, with the IRI `<base>#<xml:id>`; the bundle also points to
  the MEI file with `rdfs:seeAlso`.
* An **expression** is a value node with the IRI `<base>#<xml:id>`, written
  in full under `lrmoo:R4_embodies` of each manifestation that names it in a
  relation `isEmbodimentOf`; the work then names it by its IRI alone under
  `lrmoo:R3_is_realised_in`. An expression no manifestation embodies is
  written in full under the work's `lrmoo:R3_is_realised_in` (the second
  header proves this branch).
* Each **manifestation** becomes a new graphical timeline in millimetres of
  unknown length, named by its first title, added to the bundle before any
  statement, so a value naming it (`{"id": "#cgt_1"}`) carries its final ID.
* An **item** is a value node under its manifestation's
  `lrmoo:R7i_is_exemplified_by`, its pairs stated inside the node; a
  repository without an IRI is proposed inside the node.

The tests check the timelines' IDs, units, lengths and names, each holder's
`statements` and `provenance`, and the manifest written: `to_dict()` passes
`timetoshare.validate`, has no warnings, its `@context` declares exactly the
prefixes used, and it reads back to an equal manifest.

## Provenance

Every stated pair has one `confirmed` entry: `stated_by` the library (or the
agent given), the date given, `method: imported`, and as `basis` the MEI file
followed by `#<xml:id>` of the element the pair comes from (the file alone
when that element has no `xml:id`). A name without an IRI is proposed, never
stated, with the note `unreconciled`. The default date is not tested here: it
is the statement API's.

## Importing into a bundle

* `into` with `subjects` lands a manifestation's statements on an existing
  timeline and an item's on an existing region (which the manifestation then
  names by `{"id": "#region_1"}`); no timeline is added for a mapped entity.
* Importing the same header again, with every manifestation mapped to the
  element the first import gave it, leaves the manifest written unchanged:
  a pair the subject states, proposes or rejects is skipped. A rejected
  proposal is not proposed again.
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
with two works, and a file holding no `meiHead`.
