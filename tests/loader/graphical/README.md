# Graphical loader tests: areas of page images

This README documents the validation logic of the two readers of areas drawn
on page images, `DmosLoader` (`test_dmos_loader.py`) and `MeiFacsimileLoader`
(`test_mei_facsimile_loader.py`). Both turn each area into a `Box` event: its
shape (a `Quadrilateral` or a `BoundingBox`) in the image's pixel space, its
interval on the timeline that runs along the image's x axis.

The fixtures are small excerpts of real files, written by the tests to
`tmp_path`; no corpus is fetched. All expected values are exact integers read
from the source files by hand (named below), never computed by the code under
test.

## What proves a reader right

A reader is right when three things hold, and each test checks all three on
what the loader returns, not on its internals:

1. **The tree is the source's.** Every container the file states (a page, a
   system, an MEI surface) is a timeline, in document order, with the name
   the file gives it, and nothing the file does not state is invented: a
   page size the file does not give is unknown (`length` `None`, no
   `meta["box"]`), never guessed.
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
   that placed a measure on its page's axis instead of its system's fails
   even where the single expected values were not spelled out.

Refusals the library composes are asserted whole.

## `DmosLoader`

**Source:** the DMOS optical music recognition output for the Gallica score
`ark:/12148/bpt6k11620688`. The excerpt holds page 1 (`no_page` 1,
`page_url` `https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688/f2`, an
image of 4512 x 6174 pixels that the file does not state), its systems 1
and 2, and measures 1 and 2 of system 1. System 2 keeps no measures: DMOS
output holds systems with no measure, and the reader must take one.

- **System corners** are written top left, bottom left, top right, bottom
  right. System 1 `[[1221,1589],[1221,2266],[4097,1589],[4101,2266]]` is the
  quadrilateral (1221,1589) (4097,1589) (4101,2266) (1221,2266): a child of
  the page at offset 1221 (its least x) of length 2880 (4101 - 1221). System 2
  `[[971,2493],[971,3192],[4104,2493],[4112,3192]]`: offset 971, length 3141;
  it overlaps system 1 in x, and both are children of the page.
- **Measure corners** are written top left, top right, bottom left, bottom
  right. Measure 1 `[[1221,1589],[2151,1589],[1221,2266],[2161,2266]]` is
  (1221,1589) (2151,1589) (2161,2266) (1221,2266), the Box `box_1` with start
  0 and duration 940 on system 1. Measure 2 is (2151,1589) (2766,1589)
  (2775,2266) (2161,2266), `box_2`, start 930 (2151 - 1221), duration 624.
  The two measures overlap by 10 pixels at the bottom (the right edge of
  measure 1 leans), so the intervals are [0, 940) and [930, 1554).
- **Page size.** With `page_sizes={1: (4512, 6174)}` the page has length
  4512, is locked (the image's width is a fact, never grown) and `meta ==
  {"image": page_url, "box": {"x": 0, "y": 0, "width": 4512, "height":
  6174}}`; without, `length` is `None` and `meta == {"image": page_url}`.
  A `page_sizes` key naming no page is a `KeyError` listing the pages the
  file holds. A width of 4101 takes system 1 (ending at 4101: the edge
  itself is inside) and refuses system 2 (ending at 4112), naming page and
  system, so the boundary is exercised from both sides.
- **Ids.** Timelines are minted `dgt_1` (the page), `dgt_2`, `dgt_3` (the
  systems, in document order); Boxes are minted `box_1`, `box_2` per system.
  System names are DMOS's system ids as strings: DMOS output repeats a
  system id on some pages, so on a page whose systems are both numbered 1
  the two children are named `"1"` and `"1"` and keep distinct ids.
- **What is not read.** Measure staff headers and voices are left in the
  excerpt and must leave no trace: each system holds exactly its Boxes.
- **Refusals.** A measure whose corners, read in DMOS's measure order, do not
  run around a convex outline is refused with the shape's own message,
  prefixed by where it stands (`DMOS page 1, system 1, measure 1: ...`),
  since DMOS measures have no ids; so is a corner that is no whole number
  of pixels (`1221.5`: geometry is never rounded). `create_timelines()`
  before `load()` and a second source are refused (the loader reads one
  file).
- **Known issue: a measure leaving its system.** A measure is a Box of its
  system's timeline, which spans exactly the system region's x extent, so
  a measure reaching left or right of that extent has no place there;
  optical music recognition can produce one (measure edges lean: measure
  1's right edge runs from x 2151 to 2161). One rule for both edges: the file is refused at
  loading with one message naming page, system and measure and both
  extents; no negative start is produced and the system is not widened.
  Separate cases, each with the excerpt's system 1 (x 1221 to 4101):
  measure 1 at `[[1200,1589],[2151,1589],[1200,2266],[2161,2266]]` (x 1200
  to 2161, left of the system) gives `DMOS page 1, system 1, measure 1
  spans x 1200 to 2161, which leaves its system's region, spanning x 1221
  to 4101.`; measure 2 at `[[3500,1589],[4110,1589],[3500,2266],
  [4120,2266]]` after measure 1 (x 3500 to 4120, right of it) gives the
  same sentence for `measure 2` and `3500 to 4120`. A measure spanning the
  system exactly (x 1221 to 4101, corners those of system 1 in measure
  order) loads as `box_1`, start 0, duration 2880: both edges are inside.

## `MeiFacsimileLoader`

**Source:** `MEI/061_Gelobet seist du Jesu Christ_facsimile.mei` of the
Anhaltisches Choralbuch digital edition (CC BY 4.0): one `surface`
(`xml:id` `surface_01M7DF2DF3J8BVY6RSGDQBJ6W6`, `n` 54, `ulx` 0 `uly` 0
`lrx` 3999 `lry` 2999) with a `graphic` (`target`
`../facsimile/20251230_181354.jpg`, `width` 4000, `height` 3000) and its first
two zones, and the two measures pointing to them by `@facs`.

- **The surface** is a root timeline with the surface's `xml:id`, named
  `"54"`, length 4000 (the graphic's width, not `lrx - ulx` = 3999: the
  graphic states the image), locked, `meta == {"image":
  "../facsimile/20251230_181354.jpg", "box": {"x": 0, "y": 0, "width": 4000,
  "height": 3000}}`.
- **Zones** are Boxes with the zone's `xml:id`. MEI corners are points, so a
  zone's width is `lrx - ulx`: `zone_01M7DF2DF4MQT5E96R26XFVEYG` (`ulx` 520,
  `uly` 390, `lrx` 874, `lry` 1019) is the BoundingBox (520, 390, 354, 629),
  start 520, duration 354; `zone_01M7DF2DF4R34CYF6DVH8GZD9P` is (814, 408,
  315, 625), start 814, duration 315. They overlap in x.
- **`facs`** maps `m-43` to `("zone_01M7DF2DF4MQT5E96R26XFVEYG",)` and `m-57`
  to `("zone_01M7DF2DF4R34CYF6DVH8GZD9P",)`, the leading `#` of each
  reference dropped; an element pointing to two zones keeps both in the
  order written.
- **`@points`.** MEI, as SVG, fixes neither a first corner nor a direction,
  so a zone's four points are read in either direction from any corner and
  held from the top-left corner (least x + y), clockwise. The outline of
  DMOS measure 1 is listed eight ways (each of its four corners first, each
  direction): every listing reads as the same quadrilateral, asserted
  literally as `top_left (1221, 1589)`, `top_right (2151, 1589)`,
  `bottom_right (2161, 2266)`, `bottom_left (1221, 2266)` in its dict form,
  start 1221, duration 940. A reader that kept the written order would
  hold `(2161, 2266)` as the top left of the listing starting there (or
  refuse it); one that refused a counter-clockwise listing refuses valid
  MEI.
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
- **Sizes the file does not give.** A surface without a graphic but with
  `ulx` 100, `uly` 0, `lrx` 600, `lry` 400 has length 500 and `meta["box"]`
  (100, 0, 500, 400), so a zone at `ulx` 150 starts at 50 (the invariant with
  `o(T)` = 100). A graphic whose size is in centimetres states no pixel size:
  with no surface coordinates the length is `None` and `meta` holds the image
  only. A graphic size written `4000px` is 4000. A surface without `xml:id`
  is minted `dgt_N` (and, without `@n`, named by that id); a second surface
  after one with a given id is minted `dgt_2`, since the given id takes a
  number too.
- **The image's right edge.** A zone ending at x 4600 on a 4512-pixel image
  is refused at loading, naming the zone.
- **Known issue: the surface's corners beside the graphic's size.** Where
  the graphic states its pixel width, the image's space is that width from
  x 0 and the surface's `ulx` .. `lry` are not used, even where they state
  another area. The case pins this: a surface `ulx` 100, `uly` 0, `lrx`
  4100, `lry` 3000 with a graphic 4000 x 3000 has length 4000 and
  `meta["box"]` (0, 0, 4000, 3000), not the surface's (100, 0, 4000,
  3000); a zone at `ulx` 150 .. `lrx` 200 starts at 150 (not 50); a zone
  ending at x 4050 (inside the surface's corners) is refused as past the
  right edge at x 4000. Should the surface's corners come to decide the
  space, this case is the one to change.
- **Zones without coordinates** are not read and are listed, in document
  order, in `zones_without_coordinates`.
- **Refusals.** A root other than `<mei>` in the MEI namespace, `create_
  timelines()` before `load()`, and a second source are refused.
