# Graphical loader tests: areas of page images

This README documents the validation logic of the three readers of page
images and of the two modules they share:

- `DmosLoader` (`test_dmos_loader.py`): pages, systems and measures of a
  DMOS JSON file;
- `MeiFacsimileLoader` (`test_mei_facsimile_loader.py`): surfaces and zones
  of an MEI facsimile;
- `IIIFManifestLoader` (`test_iiif_loader.py`): the canvases of a IIIF
  manifest, along a time path that is given;
- time paths (`test_paths.py`) and the builder of pages and volumes
  (`test_pages.py`), in "Time paths and pages".

Nothing in a page image is a time axis until a path through it is stated: a
line of music is one horizontal piece, a page the pieces one after the
other. Every reader builds the same tree from such paths, all in `pixels`:
an **accolade** per piece, as long as the piece; a **page**, the segment
line of its accolades; a **volume**, the segment line of the pages. Each
area drawn on the image (a DMOS measure, an MEI zone) is a `Box` event of
the accolade it lies on: its shape (a `Quadrilateral` or a `BoundingBox`)
in the image's pixel space, its interval on the accolade.

The fixtures are small excerpts of real files, written by the tests to
`tmp_path`; no corpus is fetched. All expected values are exact integers
(fractions where a line lies between two pixels) read from the source files
by hand (named below), never computed by the code under test.

## What proves a reader right

A reader is right when four things hold, and each test checks them on what
the loader returns, not on its internals:

1. **The tree mirrors the time path.** Every piece of a path is an
   accolade, every page the segment line of its accolades in the path's
   order, the volume the segment line of the pages in document order, with
   the names the file gives. No length is unknown and none is an image
   size: an accolade is as long as its piece, a page and a volume as long
   as the sum of their segments. An accolade states its area in
   `meta["box"]`; a page states no `meta["box"]`, since its axis is a
   concatenation and no area. What has no path (a page without systems, a
   surface without zones) is no timeline.
2. **Every shape is the source's, corner for corner.** Each Box equals a Box
   built by hand from the file's corners, written in the canonical order
   (top left, top right, bottom right, bottom left). DMOS writes its corners
   in two other orders; the expected values are the source's numbers moved
   into the canonical order by hand, so a reader that kept DMOS's order, or
   read the wrong order for measures or systems, gives a different
   quadrilateral (or a crossed one, which the shape refuses).
3. **Every Box keeps the image-space invariant.** For a Box on timeline T,
   `start = x_min - o(T)` and `duration = x_max - x_min`, where `o(T)` is the
   x of T's `meta["box"]` (0 where T states none). The tests check it with a
   helper of their own over every Box of every returned timeline, so a reader
   that placed a measure on its page's axis instead of its accolade's fails
   even where the single expected values were not spelled out.
4. **The Boxes survive a package.** A manifest file holds no event, so a
   volume written alone comes back with empty accolades; the Boxes live in
   the package's `boxes.parquet`. For each reader of Boxes one volume is put
   in an `AlignmentBundle`, written to `tmp_path` and read back, and the
   tree read is compared with literals, never with the volume written: a
   writer and a reader that dropped or moved the same thing would agree
   with each other and still be wrong.

Refusals the library composes are asserted whole.

## `DmosLoader`

**Source:** the DMOS optical music recognition output for the Gallica score
`ark:/12148/bpt6k11620688` (4 pages, 17 systems, 67 measures; `no_page` 1 to
4, `page_url` `https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688/f2` to
`/f5`, `score_image_url` the same without `/fN`). Two excerpts are written by
the tests, every corner copied from the file:

- *The page*: page 1 with its systems 1 and 2 and measures 1 and 2 of
  system 1. System 2 keeps no measures: DMOS output holds systems with no
  measure, and the reader must take one.
- *The scan*: all four pages with the regions of all 17 systems, and four
  measures: measures 1 and 2 of page 1's system 1, measure 1 of its system 2
  and measure 1 of page 2's system 1. It carries the lengths of the whole
  file, since a length depends on the system regions only.

**What the numbers are, by hand.**

- **System corners** are written top left, bottom left, top right, bottom
  right. System 1 of page 1 `[[1221,1589],[1221,2266],[4097,1589],
  [4101,2266]]` is the quadrilateral (1221,1589) (4097,1589) (4101,2266)
  (1221,2266): x 1221 to 4101, y 1589 to 2266. System 2
  `[[971,2493],[971,3192],[4104,2493],[4112,3192]]`: x 971 to 4112, y 2493 to
  3192.
- **A page's time path is its systems, one horizontal piece each**: from the
  region's least x to its greatest, at the middle of its y extent, with half
  that extent as half-breadth. Page 1: (x0 1221, x1 4101, y 3855/2, 677/2),
  (971, 4112, 5685/2, 699/2), (983, 4125, 7549/2, 721/2), (994, 4142, 4729,
  362): the last system spans y 4367 to 5091, an even sum, so its `y` and
  `tolerance` are integers, the others fractions, never floats. Nothing is
  grouped: system 2 starts 250 pixels left of system 1 and is a piece of its
  own because the file states it, and the paths are keyed by `no_page`.
- **Lengths** (x extents of the 17 regions). Page 1: 2880 + 3141 + 3142 +
  3148 = 12311, the accolades at offsets 0, 2880, 6021, 9163. Page 2 (x
  519–3661, 522–3659, 522–3659, 523–3658, 522–3661): 3142 + 3137 + 3137 +
  3135 + 3139 = 15690. Page 3 (913–4060, 918–4054, 913–4051, 907–4048): 3147
  + 3136 + 3138 + 3141 = 12562. Page 4 (508–3650, 504–3643, 494–3634,
  489–3628): 3142 + 3139 + 3140 + 3139 = 12560. Volume: 53123, the pages at
  0, 12311, 28001, 40563.
- **Measure corners** are written top left, top right, bottom left, bottom
  right. Measure 1 `[[1221,1589],[2151,1589],[1221,2266],[2161,2266]]` is
  (1221,1589) (2151,1589) (2161,2266) (1221,2266): on system 1's accolade
  start 0, duration 940. Measure 2 is (2151,1589) (2766,1589) (2775,2266)
  (2161,2266): start 930 (2151 − 1221), duration 624. The two overlap by 10
  pixels at the bottom (the right edge of measure 1 leans), so the intervals
  are [0, 940) and [930, 1554). Measure 1 of system 2 spans x 971 to 1945:
  start 0, duration 974. Measure 1 of page 2's system 1 spans x 519 to 1522:
  start 0, duration 1003.

**What proves the loader right.** The tests read only what the loader
returns and compare it with the numbers above:

1. **The tree mirrors the path.** `create_timeline()` is a locked
   `SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]` of length 53123
   whose segments are the pages at the offsets above, each a locked segment
   line of its systems' accolades (unlocked plain timelines) at theirs; the
   length of every page is the sum of its accolades. `create_timelines()`
   gives the same pages as locked roots. A loader that still laid a page
   along its image's width, or put the systems on it as overlapping children
   at their least x (offsets 1221 and 971), fails on class, offsets and
   lengths at once.
2. **Who states what.** An accolade states `meta["box"]` (its system's
   quadrilateral in canonical order, bottom right (4101, 2266) for system 1)
   and `meta["image"]` (the `page_url`); a page states `meta["image"]` only;
   the volume states the file's `score_image_url` as `meta["image"]`, and
   nothing where the file has none. No timeline has an unknown length, and no
   argument takes an image size.
3. **Every shape is the source's and every Box keeps the image-space
   invariant** (see above), checked by the helper over the whole tree, so a
   measure placed on its page's or the volume's axis fails.
4. **Names and minted ids.** Pages are named by `no_page`, accolades by
   DMOS's system id as a string. DMOS output repeats a system id on some
   pages, so two systems numbered 1 give two accolades named `"1"` with
   distinct ids. Minted ids come from the loader's minter in the page
   builder's order (pinned in "Time paths and pages"); here each is checked
   by its shape (`dgt_…`, Boxes `box_…`: prefix and 13 base-36 characters).
5. **Identity given at creation.** `uid="scan"` names the volume; the pages
   are minted. `box_ids="m{n}"` numbers the scan's four measures
   through the pages: `m1`, `m2` on page 1's first accolade, `m3` on its
   second, `m4` on page 2's first. `accolade_ids="p{page}_s{accolade}"`
   gives `p1_s1` … `p1_s4`, `p2_s1` … `p4_s4`; `page_ids` as a sequence
   names the four pages in document order; a `box_ids` sequence of four
   names the measures in document order. Refusals come before anything is
   built, which the tests prove by creating again from the same loader and
   finding the id a fresh loader of the file mints first, not drawn: three ids for four Boxes, a
   template field that does not exist (`{system}`), an id holding a control
   character, and `uid` equal to a page id.
6. **A page without systems is left out.** A file whose page 2 of three has
   `"systems": []` gives pages named `"1"` and `"3"`, paths keyed 1 and 3,
   and a volume of two segments; a file with no system at all gives no page,
   no path, and a refused volume (`A volume needs at least one page.`).
7. **What is not read.** Measure staff headers and voices are left in the
   excerpts and must leave no trace: each accolade holds exactly its Boxes,
   pages and volume hold no event of their own.

**Refusals at loading.** A measure whose corners, read in DMOS's measure
order, do not run around a convex outline is refused with the shape's own
message, prefixed by where it stands (`DMOS page 1, system 1, measure 1:
...`), since DMOS measures have no ids; so is a corner that is no whole
number of pixels (`1221.5`: geometry is never rounded). Creating before
`load()` and a second source are refused (the loader reads one file).

**Known issue: a measure leaving its system.** A measure is a Box of its
system's accolade, which spans exactly the system region's x extent, so a
measure reaching left or right of that extent has no place there; optical
music recognition can produce one (measure edges lean: measure 1's right
edge runs from x 2151 to 2161). One rule for both edges: the file is refused
at loading with one message naming page, system and measure and both
extents; no negative start is produced and the system is not widened.
Separate cases, each with the excerpt's system 1 (x 1221 to 4101): measure 1
at `[[1200,1589],[2151,1589],[1200,2266],[2161,2266]]` (x 1200 to 2161, left
of the system) gives `DMOS page 1, system 1, measure 1 spans x 1200 to 2161,
which leaves its system's region, spanning x 1221 to 4101.`; measure 2 at
`[[3500,1589],[4110,1589],[3500,2266],[4120,2266]]` after measure 1 (x 3500
to 4120, right of it) gives the same sentence for `measure 2` and `3500 to
4120`. A measure spanning the system exactly (x 1221 to 4101, corners those
of system 1 in measure order) loads as `box_…`, start 0, duration 2880: both
edges are inside.

**A system without width.** DMOS output holds, on some pages, a system
whose region is a vertical line and which holds no measure (seen in the
output for Debussy's *Pelléas et Mélisande*, act 2: page 12, second system,
`[[22,929],[22,1185],[22,929],[22,1185]]`, x 22 to 22). A horizontal piece
runs from one x to a greater one, so such a system lies along no time axis.
The rule: it is **no part of the time path**. The file loads with the system
as written (`repr` counts it), loading says once per file, in one warning of
the loader's logger, which systems are left out, and `create_paths()`,
`create_timelines()` and `create_timeline()` all leave them out, as they
leave out a page without systems; a page all of whose systems are left out
is left out too. What proves it is that the three methods agree and that
nothing else moves: the remaining systems keep their extents, the offsets
and minted ids close up over the gap, and positions in id templates count
the accolades made.

The case, with the regions of the page excerpt: page 1 holds system 1 with
measure 1, the line (DMOS id 1, no measure), and system 2 with its measure
1; page 2 holds the line alone; page 3 holds system 2's region and the
line. By hand:

- the loader's `repr` is `DmosLoader(pages=3, systems=6, measures=2)`, and
  exactly one warning is logged by `timetoalign.loader.graphical.dmos.
  DmosLoader`, also after all three `create_*` calls: `<file>: 3 system(s)
  without x extent and without measures are no part of the time path and
  are left out: page 1, system 2; page 2, system 1; page 3, system 2.`
  (positions on the page, from 1). The scan, which holds no such system,
  logs nothing;
- paths: keys 1 and 3; page 1 the two pieces (1221, 4101, 3855/2, 677/2)
  and (971, 4112, 5685/2, 699/2) at offsets 0 and 2880; page 3 the one
  piece of system 2;
- pages: `dgt_…` named `"1"`, length 2880 + 3141 = 6021, its accolades
  `dgt_…` (name `"1"`, 2880) and `dgt_…` (name `"2"`, 3141) at 0 and 2880,
  each with its `box_…`; then `dgt_…` named `"3"`, length 3141, its
  accolade `dgt_…`. A reader that kept the line as a zero-length accolade
  mints `dgt_…` for it and names two accolades `"1"`;
- volume: length 6021 + 3141 = 9162, pages `"1"` and `"3"` at 0 and 6021;
  with `accolade_ids="p{page}_s{accolade}"` and `box_ids="m{n}"` the
  accolades are `p1_s1` (`m1`), `p1_s2` (`m2`) and `p2_s1` (no Box): page 3
  is the second page created, system 2 the second accolade of its page;
- a file whose only system is the line gives no path, no page, and the
  refused volume of a file without systems (`A volume needs at least one
  page.`).

**Known issue: a system without width that holds measures.** Leaving such a
system out would drop its measures, and no accolade of length 0 can hold a
Box, so it stays refused: the file loads (a measure on the line, `[[22,929],
[22,929],[22,1185],[22,1185]]`, does not leave its system's x extent), and
each of the three `create_*` methods refuses with `DMOS page 1, system 3
has no x extent (its region spans x 22 to 22) and holds 1 measure(s): it
cannot be a piece of a time path.` Separate case: page 1 with system 1, the
empty line and a line holding that measure; the position named is the
system's on its page (3), counting the system left out before it, and the
warning names that one (`page 1, system 2`) alone.

**The package round trip** (principle 4). The scan's volume, created with
`uid="scan"` and `box_ids="m{n}"`, is written in a bundle to
`package/package.yaml`; the directory then holds exactly `boxes.parquet`
and `package.yaml`. Read back, every value is one derived above: the volume
`scan` is a locked segment line of length 53123 stating the
`score_image_url`; its pages `dgt_…`, `dgt_…`, `dgt_…`, `dgt_…` (each
page's number follows its predecessor's accolades: 4, 5 and 4 of them),
named `"1"` to `"4"`, of lengths 12311, 15690, 12562, 12560 at offsets 0,
12311, 28001, 40563, each stating its `page_url` only; page 1's accolades
`dgt_…` to `dgt_…` of lengths 2880, 3141, 3142, 3148 at 0, 2880, 6021,
9163; the `meta["box"]` of the first two the regions of systems 1 and 2 in
canonical order, corner for corner (bottom right (4101, 2266) and (4112,
3192)). The four Boxes are compared whole (id, start, duration and the four
corners): `m1` [0, 940) and `m2` [930, 1554) on `dgt_…`, `m3` [0, 974) on
`dgt_…`, `m4` [0, 1003) on `dgt_…`, page 2's first accolade. The 17
accolades hold four Boxes together, volume and pages none, and the
image-space invariant holds over the tree read.

## `MeiFacsimileLoader`

**Source:** `001_Ach bleib bei uns, Herr Jesu Christ_facsimile.mei` of the
Anhaltisches Choralbuch digital edition (CC BY 4.0): its one `surface`
(`xml:id` `surface_01M7DF2D8Q8FYGETPJXXMG03QP`, `n` 15, `ulx` 0 `uly` 0
`lrx` 3999 `lry` 2999) with its `graphic` (`target`
`facsimile/20251230_180023.jpg`, `width` 4000, `height` 3000) and its nine
`measure` zones, and the first two measures pointing to them by `@facs`. The
zones, as (`ulx`, `uly`, `lrx`, `lry`), called zone 1 to 9 below:

| | zone | | zone |
|---|---|---|---|
| 1 | 490, 344, 973, 877 | 6 | 498, 858, 1496, 1337 |
| 2 | 944, 368, 1523, 864 | 7 | 1465, 885, 2146, 1353 |
| 3 | 1481, 388, 2082, 912 | 8 | 2110, 886, 2748, 1357 |
| 4 | 2034, 402, 2694, 884 | 9 | 2704, 886, 3302, 1387 |
| 5 | 2669, 389, 3296, 916 | | |

**What has to be proved.** MEI states pages and zones but no systems, and a
page's width is no time axis. The loader is right when the timeline follows
the page's time path: one accolade per line of music, each as long as its
zones reach in x, the page their concatenation, and each zone a Box of the
accolade it lies on, placed from that accolade's left edge. Every number
below is derived by hand from the table, so a loader that kept a page-wide
timeline, grouped wrongly, or placed a zone from the image's left edge gives
other numbers.

- **The grouping, by hand.** Left edges in document order: 490, 944, 1481,
  2034, 2669, 498, 1465, 2110, 2704; their span is 2704 - 490 = 2214. A zone
  begins a new accolade where its left edge falls back from the previous
  one's by more than half the span: the only fall is 2669 to 498 = 2171, and
  2 x 2171 > 2214. So zones 1 to 5 are accolade 1, zones 6 to 9 accolade 2.
- **The paths** (`create_paths()`, keyed by the surface's `xml:id`):
  accolade 1 runs x 490 to 3296 (least `ulx`, greatest `lrx`), y 344 to 916,
  so its line is at y (344 + 916) / 2 = 630 with tolerance 286; accolade 2
  runs x 498 to 3302, y 858 to 1387, line at y 2245/2, tolerance 529/2 (kept
  as fractions, never rounded). A surface without `xml:id` is keyed by its
  position in the file from 0; a surface without a zone taking part has no
  path.
- **The page** (`create_timelines()`) is a locked root
  `SegmentLine[DiscreteGraphicalTimeline]` with the surface's `xml:id`, named
  `"15"`, length 2806 + 2804 = 5610 (not the graphic's 4000), `meta ==
  {"image": "facsimile/20251230_180023.jpg"}`: a page states no
  `meta["box"]`, the graphic's size is on no timeline.
- **The accolades** are its segments: `accolade_1` (`dgt_…`) at offset 0,
  length 2806, `meta["box"]` (490, 344, 2806, 572); `accolade_2` (`dgt_…`)
  at offset 2806, length 2804, `meta["box"]` (498, 858, 2804, 529); both
  state the page's image. The given page id takes number 1 of the minted
  ids.
- **Zones are Boxes of their accolade** with the zone's `xml:id`, shape as
  written (MEI corners are points, so a width is `lrx - ulx`), start =
  `ulx` less the accolade's left edge. All nine are asserted literally:
  accolade 1 holds zones 1 to 5 at 0, 454, 991, 1544, 2179 with durations
  483, 579, 601, 660, 627; accolade 2 zones 6 to 9 at 0, 967, 1612, 2206
  with durations 998, 681, 638, 598. Zone 6 at 0 (not 498) is what a
  page-wide timeline gets wrong. The image-space invariant (principle 3) is
  checked on every accolade with the tests' own helper. The two strips
  overlap in y (858 to 916); each zone's centre is nearer its own line.
- **A zone stays in the accolade it was grouped into.** Three zones a (100,
  0, 400, 100), b (400, 300, 800, 400), c (100, 200, 800, 600): the left
  edge falls back 400 to 100 = 300 against a span of 300, so a and b are
  accolade 1 (x 100 to 800, y 0 to 400, line at y 200) and c accolade 2 (y
  200 to 600, line at y 400). The strips overlap and b's centre (y 350) is
  nearer the second line; b must still be a Box of accolade 1 at 300, where
  a loader placing grouped zones by the nearest line puts it on accolade 2.
- **The volume** (`create_timeline()`) is a locked
  `SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]` of the pages: id
  `dgt_…` (or the given `uid`, with `name`), page as above, accolades then
  `dgt_…`, `dgt_…`. With a second surface holding one zone (100, 50, 400,
  90) its length is 5610 + 300 = 5910 and the second page lies at offset
  5610.
- **Which zones take part.** A surface holds, in this order, a `staff` zone
  (300, 300, 3400, 950), zones 1 and 2 as `measure`, and a zone without
  `@type` (1000, 400, 1100, 500). With `zone_types="measure"` only the two
  measures are Boxes and the accolade runs x 490 to 1523 (length 1033, line
  at y 1221/2, tolerance 533/2), not 300 to 3400; without `zone_types` all
  four take part (left edges 300, 490, 944, 1000 never fall back: one
  accolade x 300 to 3400, length 3100, zone 1 at 190, the untyped zone at
  700); a collection of types (`["measure", "staff"]`) takes three. A
  surface none of whose zones takes part is left out of timelines and
  paths, as a surface without zones is.
- **Given paths** replace the grouping. For the nine zones, two lines
  written by hand, x 400 to 3400 at y 630 and y 1120, tolerance 300 each:
  two accolades of length 3000 (page 6000), `meta["box"]` (400, 330, 3000,
  600) and (400, 820, 3000, 600); zone 2 then starts at 944 - 400 = 544 and
  zone 6 (centre y 1097.5, outside the first strip 330 to 930) at 98 on the
  second. The same result whether the path is given alone, under the
  surface's `xml:id` or under position 0. A single `HorizontalLinePath`
  makes the page one plain timeline with the Boxes on it. Refused, each
  message asserted whole: a `VerticalLinePath` where zones take part; one
  path for a file of two surfaces; a mapping naming a surface the file does
  not hold (`KeyError` listing the surfaces); one surface named twice, by id
  and by position; a zone lying on no piece of the given path. A surface
  without zones that is given a path is built from it (a `VerticalLinePath`
  of 0 to 2000 is a plain timeline of length 2000).
- **Identity.** `box_ids="m{n}"` names the nine Boxes `m1` to `m9` through
  both accolades; `accolade_ids="p{page}_s{accolade}"` gives `p1_s1`,
  `p1_s2`; `page_ids=["folio"]` replaces the surface's id. `facs` is the
  file's and unchanged by `box_ids`. A sequence of eight Box ids for nine
  zones is refused. A surface without `xml:id` is minted `dgt_…` and, without
  `@n`, named `page_1`; a zone without `xml:id` is `box_…` on its accolade.
- **`facs`** maps `m-43` to `("zone_01M7DF2D8RNCKHKF6SMCWB94Z5",)` and `m-57`
  to `("zone_01M7DF2D8R00HP8JYGXRTE13G3",)`, the leading `#` of each
  reference dropped; an element pointing to two zones keeps both in the
  order written.
- **`@points`.** MEI, as SVG, fixes neither a first corner nor a direction,
  so a zone's four points are read in either direction from any corner and
  held from the top-left corner (least x + y), clockwise. The outline of
  DMOS measure 1 is listed eight ways (each of its four corners first, each
  direction): every listing reads as the same quadrilateral, asserted
  literally as `top_left (1221, 1589)`, `top_right (2151, 1589)`,
  `bottom_right (2161, 2266)`, `bottom_left (1221, 2266)` in its dict form.
  Alone on its surface it is its accolade: start 0, duration 940,
  `meta["box"]` (1221, 1589, 940, 677), the bounding box of the four points.
  A reader that kept the written order would hold `(2161, 2266)` as the top
  left of the listing starting there (or refuse it); one that refused a
  counter-clockwise listing refuses valid MEI.
- **Known issue: `@points` that are no convex quadrilateral.** A zone is a
  Box holding a `Quadrilateral`, which has four corners bounding a convex
  area; any other `@points` cannot be one, and nothing is put in its place
  (no bounding box is inferred), so the file is refused at loading, the
  message naming the zone's `xml:id` and the reason. Separate cases, each
  asserted whole: five points (`1221,1589 2151,1589 2161,2266 1221,2266
  1221,1900`): `MEI zone 'm1': The SVG points '...' hold 5 points; a
  Quadrilateral's are four.`; a concave outline (`0,0 10,0 4,4 0,10`):
  `... bound no convex quadrilateral: the outline turns clockwise at points
  1, 2 and 4 and counter-clockwise at point 3.`; the crossed order an IIIF
  `SvgSelector` copies from DMOS's measure order (`1221,1589 2151,1589
  1221,2266 2161,2266`): `... bound no convex quadrilateral: the outline
  turns clockwise at points 1 and 2 and counter-clockwise at points 3 and
  4.`
- **The image's right edge.** A zone ending at x 4600 on a graphic 4512
  pixels wide is refused at loading, naming the zone; so is one ending at
  4050 on a graphic of `4000px`. A graphic whose size is in centimetres
  states no pixel width, so nothing is checked against it.
- **Known issue: the surface's corners.** Zone coordinates are taken as the
  graphic's pixels; the surface's `ulx` .. `lry` are not read. The case pins
  this: on a surface `ulx` 100, `uly` 0, `lrx` 4100, `lry` 3000 with a
  graphic 4000 x 3000, a zone ending at x 4050 (inside the surface's
  corners) is refused as past the right edge at x 4000, and a zone at `ulx`
  150 .. `lrx` 200 states `meta["box"]` x 150 on its accolade, as written.
  Should the surface's corners come to state the space, this case is the
  one to change.
- **Zones without coordinates** are not read and are listed, in document
  order, in `zones_without_coordinates`.
- **The package round trip** (principle 4). The excerpt's volume, created
  with `uid="chorale"`, is written in a bundle to `package/package.yaml`
  (the directory then holds exactly `boxes.parquet` and `package.yaml`) and
  read back. Every value is one derived above: the volume `chorale` is a
  locked segment line of length 5610 with empty `meta`; its one page has the
  surface's `xml:id`, name `"15"`, length 5610, `meta` the image alone and
  no event; the accolades `dgt_…` (`accolade_1`, offset 0, length 2806,
  `meta["box"]` (490, 344, 2806, 572)) and `dgt_…` (`accolade_2`, offset
  2806, length 2804, (498, 858, 2804, 529)) state the image; all nine Boxes
  are compared whole with the table of starts, durations and bounding boxes
  above (zone 6 at 0 on accolade 2), under their zones' `xml:id`s.
- **Refusals.** A root other than `<mei>` in the MEI namespace, each of the
  three `create_*` methods before `load()`, and a second source are refused.

## `IIIFManifestLoader`

`test_iiif_loader.py`. A manifest states images and their sizes, never where
time runs through them, so the loader builds nothing without a time path per
canvas. The tests therefore prove two things: that the timelines are the
PATHS' (every length a number of the path, none a canvas's width or height),
and that each one names the CANVAS it lies on.

**Sources.** The SUPRA piano roll manifest (corpus `supra`): one canvas,
4096 x 299400, labelled `display image`. Two published manifests, trimmed by
the test to the members the loader reads (`@context`, the manifest's and
each canvas's id and label, `width`, `height`, `duration`) and written to
`tmp_path`, the values stated literally in the test. Gallica's
`bpt6k11620688` (Presentation 2): six page canvases `f1`...`f6` at positions
0 to 5, all 6174 high, 4564, 4512, 4564, 4512, 4564 and 4577 wide, labelled
`NP`, `2`, `3`, `4`, `5`, `NP`. The BnF disc `bpt6k88448791` (Presentation
3): four sound canvases (positions 0 to 3; `duration` 221.0, 219.0, 257.0,
155.0; no width or height), then four image canvases (positions 4 to 7; two
2677 x 2677 labelled `5`, `6`, two 2940 wide x 2985 high labelled `recto`,
`verso`).

**The paths are written by hand** from the *scan* literals of "Time paths
and pages" below (the systems of the four music pages of this very Gallica
volume, which lie on canvases 1 to 4): one horizontal line per system over
its x extent. Only system 1 of page 1 has its real line (y 3855/2,
half-breadth 677/2, so its strip is y 1589 to 2266, height 677); every other
system k of its page is placed at y = 1000 k with half-breadth 300 (strip
1000 k - 300, height 600), which is on every canvas (5000 <= 6174).

**What proves the volume.** `create_timeline` with these four paths and
`canvases=[1, 2, 3, 4]`:

1. *The lengths are the paths'.* Volume 53123; pages 12311, 15690, 12562,
   12560 at offsets 0, 12311, 28001, 40563; the accolades of page 1 are
   2880, 3141, 3142, 3148 long at offsets 0, 2880, 6021, 9163 (the sums of
   "Time paths and pages"). No canvas width (4512, 4564) or height (6174)
   appears, which a loader measuring the image instead of the path would
   give.
2. *The shape.* The volume is a `SegmentLine[SegmentLine[
   DiscreteGraphicalTimeline]]`, each page a segment line, each accolade a
   plain timeline; the volume and the pages are locked, the accolades are
   not.
3. *Every timeline names what it measures.* The volume is named with the
   manifest's label and states `dcterms:source` with the manifest's `@id`;
   page i is named with its canvas's label (`2`, `3`, `4`, `5`) and states
   `dcterms:source` and `meta["image"]` with its canvas's `@id` (`f2` to
   `f5`: the canvases selected, not the first four), and no `meta["box"]`;
   an accolade states the same `meta["image"]` and its strip as
   `meta["box"]` (page 1: `{x: 1221, y: 1589, width: 2880, height: 677}`
   and `{x: 971, y: 1700, width: 3141, height: 600}`), and no statement.
4. *Identity.* Every timeline id is minted from the loader's minter (a
   `dgt_…`, checked by its shape: prefix and 13 base-36 characters; the
   order of minting is the page builder's, pinned in "Time paths and
   pages"). `uid`, `name`, `page_ids` (a template) and `accolade_ids` (a
   template) replace them; the values expected are spelled out. The file
   read back holds the very ids written.
5. *The file.* Written with the manifest file writer and read back, the
   volume has the same class, ids, offsets, lengths, names, `meta` and
   statements on all three levels, so the canvas of every page survives in
   the format.

`create_timelines` with the same arguments answers the four pages as roots:
the same lengths, names, `meta` and statements, each locked, and nothing
states the manifest (there is no volume); each page's id a minted `dgt_…`.

**What proves the selection.** Each case is decided by which canvases'
labels and `@id`s come back, in which order:

- canvases named by id, by position, mixed and in reverse order give the
  same pages in manifest order; paths keyed by id and by position are the
  same paths;
- `canvases=None` on the BnF disc selects positions 4 to 7 (the images) and
  never a sound canvas: with one horizontal line each (x 100 to 2600, 2500
  long; x 0 to 2677; x 0 to 2940; x 40 to 2940, 2900 long) the volume is a
  `SegmentLine[DiscreteGraphicalTimeline]` 11017 long (2500 + 2677 + 2940 +
  2900), pages at 0, 2500, 5177, 8117 named `5`, `6`, `recto`, `verso`; a
  page along one horizontal line states its strip as `meta["box"]`;
- one path with one selected canvas needs no mapping. The piano roll is
  this case: SUPRA with `VerticalLinePath(x=2048, y0=0, y1=299400,
  tolerance=2048)` gives ONE plain `DiscreteGraphicalTimeline` of length
  299400 (the path's length, here also the image's height), locked, named
  `display image`, stating its canvas in `meta["image"]` and
  `dcterms:source`, with no `meta["box"]` (the path is no horizontal line)
  and no child. The same roll read along the half 0 to 149700 is 149700
  long, which a loader reading the canvas's height would not give.

**Refusals**, each asserted whole, and each followed by the proof that
nothing was built: the next successful call mints the id a fresh loader of
the same file mints first.

- Gallica with the four paths and no `canvases`: `canvases 0, 5 have no time
  path: give each one or leave them out with canvases=`; one canvas missing
  reads `canvas 0 has no time path: give it one or leave it out with
  canvases=`.
- A path for a canvas that is unknown (position 9, id `nope`), time-based
  (BnF position 0) or not selected (Gallica position 0 beside
  `canvases=[1]`): `KeyError` naming each with its reason and listing the
  selected canvases as `position: 'id'`. One path given with `canvases=`
  selecting a time-based canvas alone (BnF position 0, by position and by
  id) is refused with the same `KeyError` as the mapping form, `paths gives
  a time path to 0 (time-based: no width and height). Selected canvases: 0:
  '<id>'`, before any bound is compared with the absent width.
- One path for several selected canvases; one canvas given two paths (by
  position and by id); `paths` that is neither a path nor a mapping
  (`TypeError`).
- `canvases` as a bare string (`TypeError`: it would read as its
  characters); `canvases` naming a canvas the manifest does not hold
  (`KeyError` listing all canvases).
- A line leaving its canvas, the message naming the canvas (position, id,
  size), the piece and what a line needs. Pinned on both sides of each
  bound on Gallica canvas 1 (4512 x 6174): `x1` 4512 and `y` 6174 and `y` 0
  are on it; `x1` 4513, `y` 6175, `x0` -1 are refused; for a vertical line
  `y1` 6174 and `x` 4512 are on it, `y1` 6175 and `x` 4513 refused. A piece
  of a piecewise path is named by its number from 1.
- The removed arguments have no alias: `create_timeline()` without paths
  and `axis=` are a `TypeError`; `canvases="all"` is the bare-string
  refusal.
- A manifest of sound canvases only selects nothing: `create_timelines({})`
  is `[]` and `create_timeline({})` is refused (`A volume needs at least one
  page.`).

**Time-based canvases and parsing** are unchanged: the BnF disc parses, its
canvases expose their durations and no width or height, `dimensions`,
`width` and `height` refuse a first canvas that is time-based with the
time-based message, `repr` works, and a canvas with neither extent nor
duration is refused at parsing.

### Manifests without `@context`

The synchronisation manifests Neuma publishes for CollabScore are
Presentation 3 in their structure (`id`, `type`, language maps, canvases in
`items`) but state no `@context`, which the specifications require. The
loader read every manifest without a Presentation 3 context as
Presentation 2: one with `metadata` raised `TypeError: unhashable type:
'dict'` (a Presentation 3 metadata label is a language map, used as a
Presentation 2 key), one without raised `Manifest contains no canvases`
(its `items` were never read). Readers accept valid data, so the version is
told from the structure where `@context` names none. What proves it:

- **The failing shape opens.** A reduced copy of
  `royaumont_concert/concert_C130_0/combined_manifest.json` (Neuma, *Tristesse*;
  kept: `id`, `type`, `label`, the four `metadata` pairs with their language
  maps, and the one canvas with its `label`, `width`, `height`, `duration`,
  its painting annotations of a `Video` and an `Image`, and one synchronisation
  annotation of its `annotations`; the summary, rights, thumbnail and the other
  annotations dropped) is written to `tmp_path` with no `@context`. It loads
  as Presentation 3: manifest label `[Tristesse] `, metadata `title`
  `[Tristesse] ` and `creator` `Camille  Saint-Saëns (1835, 1921)` (two spaces,
  verbatim); one canvas, id
  `https://neuma.huma-num.fr/home/opus/all:collabscore:royaumont:concert:C130_0/canvas`,
  4772 wide, 6248 high, `duration` 167.0, label `Combined image-audio
  canvas`, spatial. A Presentation 2 reading could give none of these values.
- **`@context` still decides where it names one version.** The existing
  manifests with a Presentation 2 or 3 context load as before (SUPRA, Gallica,
  the BnF disc above). A Presentation 2 manifest with `sequences` and no
  `@context` loads as Presentation 2 (two canvases, 100 x 200 and 300 x 400,
  labels `1` and `2`).
- **A manifest that fits neither is refused, naming what was looked for.**
  The shape of `royaumont_concert/concert_C144_0/reverie_audio_manifest.json`
  (Neuma's frame list: `opus_ref`, `source_ref`, `nb_frames`, `time_frames`;
  no IIIF manifest) raises `ValueError`: `Cannot tell the IIIF Presentation
  version of <path>: its @context names none, and it lists neither 'items'
  (Presentation 3) nor 'sequences' (Presentation 2)`. One listing both raises
  the same message with `both 'items' (Presentation 3) and 'sequences'
  (Presentation 2)`: either reading would drop the other's canvases.

Annotations and the bodies of a canvas are not read: a canvas is what it
states (`width`, `height`, `duration`), as for every manifest.

## Time paths and pages

`test_paths.py` covers `loader/graphical/paths.py` (the accolade rule,
`PiecewisePath`, `TimeAxisPath.from_boxes`, the numbers a path holds, drawing
a piecewise path); `test_pages.py` covers the builder that turns pages with
paths into timelines (`loader/graphical/_pages.py`). No file is read: every
number is a literal taken by hand from two real sources, and the expected
values are derived here, on paper, from those literals.

**Sources of the literals.**

- *Scan* (the DMOS output for Gallica `ark:/12148/bpt6k11620688`, 4 pages, 17
  systems, 67 measures). The x extent of each system's region, per page:
  page 1: 1221–4101, 971–4112, 983–4125, 994–4142 (lengths 2880, 3141, 3142,
  3148; sum 12311); page 2: 519–3661, 522–3659, 522–3659, 523–3658, 522–3661
  (3142, 3137, 3137, 3135, 3139; sum 15690); page 3: 913–4060, 918–4054,
  913–4051, 907–4048 (3147, 3136, 3138, 3141; sum 12562); page 4: 508–3650,
  504–3643, 494–3634, 489–3628 (3142, 3139, 3140, 3139; sum 12560). Volume:
  12311 + 15690 + 12562 + 12560 = 53123; the pages start at 0, 12311, 28001,
  40563. System 1 of page 1 is the quadrilateral (1221,1589) (4097,1589)
  (4101,2266) (1221,2266): y 1589 to 2266, so its line is at y (1589 + 2266)
  / 2 = 3855/2 with half-breadth (2266 − 1589) / 2 = 677/2. Its measures 1
  and 2 span x 1221–2161 and 2151–2775: starts 0 and 930, durations 940 and
  624. Systems of which only the extent matters are written as rectangles.
- *Chorale page* (the facsimile of an MEI chorale edition: one surface
  `n="15"`, nine measure zones, given as `ulx, uly, lrx, lry`): (490, 344,
  973, 877), (944, 368, 1523, 864), (1481, 388, 2082, 912), (2034, 402, 2694,
  884), (2669, 389, 3296, 916), (498, 858, 1496, 1337), (1465, 885, 2146,
  1353), (2110, 886, 2748, 1357), (2704, 886, 3302, 1387).

**What proves the accolade rule.** The rule is stated as an inequality, `2 ·
(previous − x) > span`, so the tests pin both of its sides, not only a
typical page: the chorale page's left edges (span 2704 − 490 = 2214; the one
fall, 2669 → 498 = 2171, doubles to 4342 > 2214: zones 1–5 and 6–9); a fall
of exactly half the span, which begins nothing (0, 10, 5); one unit more,
which does (0, 10, 4); the same in fractions and floats; no box and one box.
That IEEE 1599 files group as before is proved by that loader's own tests,
which must pass unchanged.

**What proves a path keeps its numbers.** For each line class, a line of
integers and a line of `Fraction` values answer `length`, `to_2d` and
`from_2d` with values that are equal AND of the same Python type as exact
arithmetic gives; a float line answers floats. The strip is pinned at its
edges: a point at exactly `y ± tolerance` is on the line, one beyond is not.

**What proves `PiecewisePath`.** Built from the chorale page's two lines
(below), every answer is one a reader can compute from the offsets (0, 2806)
and length 5610: the last coordinate of the first piece and the first of the
second map to different lines; the path's end maps to the end of the last
piece; a point in the y range both strips share (858 to 916) goes to the
nearer line, checked with a point nearer each; a point in no strip gives
`None`; the distance is the least over the pieces. `piece_of` places all nine
zones (centres at y 610.5 to 652.5 for zones 1–5, on piece 0; 1097.5 to
1136.5 for zones 6–9, on piece 1). Two made-up shapes centred in the shared
range decide by the nearer line: centre 870 is 240 from line 630 and 252.5
from line 2245/2 (piece 0), centre 880 is 250 and 242.5 (piece 1). A shape
reaching past every piece's x extent, or centred in no strip, is refused; so
is a centre exactly between two lines that both hold it (lines at 100 and
200 of half-breadth 60, centre 150).

**What proves `from_boxes`.** The expected pieces are computed by hand:
chorale accolade 1 spans x 490–3296 and y 344–916, so `y` = 630 and
`tolerance` = 286 (integers); accolade 2 spans x 498–3302 and y 858–1387, so
`y` = 2245/2 and `tolerance` = 529/2 (fractions, never floats). With stated
accolades (the scan's systems, one label each) nothing is grouped: the pieces
are the systems' extents even where a later system starts further left. Box
events give the same path as their shapes. Labels group in order of first
appearance. Refusals are asserted whole.

**What proves the page builder.** A test passes only if the tree a loader
would return is right in every respect a user can observe, each checked
against the hand-derived numbers above:

1. *Shape of the tree*: the class of every timeline (volume, page, accolade),
   segment order, each offset and length, and that the length of a page is
   the sum of its accolades and of a volume the sum of its pages.
2. *Locks*: the returned root is locked, a page inside a volume is locked,
   an accolade is not.
3. *Areas*: an accolade states `meta["box"]` (the stated area, or the strip
   of its line: chorale accolade 1 is `{x: 490, y: 344, width: 2806, height:
   572}`) and `meta["image"]`; a page of accolades and a volume state no
   `meta["box"]`; a page along a vertical line states none either.
4. *Boxes*: each Box sits on the right accolade with `start = x_min − x0` and
   `duration = x_max − x_min` (chorale zone 2: 454 and 579; zone 6: 0 and
   998), and its stored shape is the source's.
5. *Identity*: every id the call mints comes from the loader's one minter
   in document order (the volume, then each page followed by its
   accolades, each accolade followed by its Boxes without id), and a
   timeline or Box the call names draws none. The test loader holds
   `IdMinter(0)`, so the expected ids are the minter's own sequence,
   computed by a fresh `IdMinter(0)` in the test, never typed: scan volume
   `scan` given, page 1 is the 1st id, its accolades the 2nd–5th, page 2
   the 6th, …; chorale page `surface_15` given, its accolades are the 1st
   and 2nd; without a source id the page is the 1st, accolade 1 the 2nd,
   its five Boxes the 3rd–7th (`box_…`), accolade 2 the 8th, its Boxes the
   9th–12th. Source ids are used unless the argument replaces them;
   sequences are read in document order and templates filled from `page`,
   `accolade`, `box`, `n` (`m{n}` numbers the scan's measures through the
   pages; with zones given in an order that alternates between accolades, a
   sequence still names the zones in the order given).
   *A Box id is unique in its accolade, and only there.* The Boxes of one
   accolade are the events of one timeline, so two of them cannot share an
   id; two accolades are two timelines and may. Both sides are pinned: a
   template without a position (`box_ids="m"`) and a sequence naming the
   chorale's zones 1 and 3 alike are refused, naming the id and accolade
   `accolade_1` of page `15` (a page along one line is named itself), while
   `box_ids="m{box}"` gives `m1`–`m5` on the first accolade and `m1`–`m4`
   on the second.
   *Given ids draw nothing.* A call naming every timeline mints nothing
   (the loader's next id is the minter's first); with page ids given, the
   volume is the 1st id and the accolades of the four pages the 2nd–5th,
   6th–10th, 11th–14th and 15th–18th, so a builder that drew an id for a
   named page would be off by one from page 2 on.
6. *Nothing happens on a refusal*: each refusal (wrong sequence length,
   unknown template field, invalid id, one id for two timelines, one id for
   two Boxes of an accolade, a Box on no
   piece, a Box leaving its accolade, a stated area that is not its piece's x
   extent, a length that is no whole pixel, pages of two kinds in one volume,
   no page) is asserted whole, and after it the next id minted from the
   same loader is still the minter's first, so no id was drawn.

A diagonal of length 500.0 (300, 400) is a page of 500 pixels; one of length
141.42… (100, 100) is refused: lengths are never rounded.

## Drawing on a page image (`test_drawing.py`)

`ImageSource.draw_rectangle`, `draw_line` and `draw_path` and the drawing
methods of `GraphicalStore` (`draw_segments_on_source`, `draw_interval`,
`draw_rectangle_at_coord`) draw with Pillow on a copy of the page. A page
is converted from its pixmap to a Pillow image once, any number of shapes
are drawn on that image, and the image becomes a pixmap once, from its raw
samples, when the pixmap is needed (`pixmap`, `to_pil`, `to_bytes`,
`save`, `copy`). Drawing on the result of a drawing starts from the image
held, without a conversion.

A pixmap stores colour multiplied by alpha, so every conversion out and
back in multiplies a semi-transparent pixel once more. On a translucent
image (an alpha channel with values other than 0 and 255, or a filled
rectangle drawn) the conversion is therefore kept between two shapes; on
every other image it changes no pixel and is left out.

What proves it:

1. *The pixels are those of drawing shape by shape through a PNG file.* The
   test keeps, as its reference, the drawing route that converts the whole
   page for every shape: pixmap to Pillow image, one shape, PNG encoding, a
   pixmap read from the PNG bytes (and, for the store, a `copy()` of the
   page first, which adds an alpha channel). The same shapes (rectangles
   outlined and filled, lines, a `PiecewisePath`, a chain of four calls with
   a fill in the middle, the four segments of a store, an interval) are
   drawn by both routes and the results compared whole: width, height,
   number of components, alpha, colorspace, resolution and the sample
   bytes, exactly. Pages: a synthetic grey page (Pillow mode `L`, so the
   conversion to RGB is covered) and a synthetic RGB page, both 100 x 100,
   and two real page images of the `ieee1599` corpus: the scan of *Serie
   in 9/8* (greyscale JPEG, 1248 x 1745 at 600 dpi) and Satie's first
   *Gymnopédie* (RGBA PNG, 475 x 638 at 300 dpi, 2,218 semi-transparent
   pixels), so the translucent case is pinned on real data. Run against the
   drawing code that converted once per shape, these comparisons pass: the
   reference is that code.
2. *The number of conversions is counted, not timed.* Both conversion
   functions of `timetoalign.loader.graphical.source` are wrapped by
   counting spies. One shape and then the pixmap: one in, one out; reading
   the pixmap again or `to_bytes` converts nothing more. A chain of 27
   public calls (the nine measure zones of a chorale page as rectangles,
   then two diagonals each) saved to a file: one in, one out. The four
   segments of a store, and an interval of 67.5 pixels (67 lines) on one of
   three pages: one in, one out. On the RGBA page the four segments convert
   four times each way.
3. *The source drawn on is unchanged and each call returns a new
   `ImageSource`*: the source's samples are compared before and after, two
   drawings from one drawing are each equal to the reference drawn on it,
   and the three results differ.
4. *Nothing to draw, nothing converted*: `draw_segments_on_source` on a
   page without segments returns the `copy()` it always returned (grey with
   an alpha channel) without a conversion, and `draw_interval` copies the
   pages it does not touch.

`draw_interval` reaches a segment's end only when the interval ends inside
that segment (`GraphicalSegment.to_image` refuses the end of its own span),
so every interval tested lies within one segment.

No timing is measured anywhere: a timing assertion fails under load.
