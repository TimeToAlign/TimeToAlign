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

**Source:** `aimons-nous_C055/saintsaens-audio_C055_0/iiif.json` of the
CollabScore specimens (DMOS optical music recognition output). The excerpt
holds page 1 (`no_page` 1, `page_url` `.../bpt6k11620688/f2`, an image of
4512 x 6174 pixels that the file does not state), its systems 1 and 2, and
measures 1 and 2 of system 1. System 2 keeps no measures, which is also the
shape of the defect of one Pelléas page (a system with zero measures).

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
  System names are DMOS's system ids as strings: on a page whose systems are
  both numbered 1 (the Pelléas defect) the two children are named `"1"` and
  `"1"` and keep distinct ids.
- **What is not read.** Measure staff headers and voices are left in the
  excerpt and must leave no trace: each system holds exactly its Boxes.
- **Refusals.** A measure whose corners, read in DMOS's measure order, do not
  run around a convex outline is refused with the shape's own message,
  prefixed by where it stands (`DMOS page 1, system 1, measure 1: ...`),
  since DMOS measures have no ids. `create_timelines()` before `load()` and a
  second source are refused (the loader reads one file).

The full CollabScore files are not a test corpus; whole-file counts (4 pages,
17 systems, 67 measures for C055) are checked by hand, not here.

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
- **`@points`.** A zone with `points="1221,1589 2151,1589 2161,2266
  1221,2266"` (DMOS measure 1 in canonical order) reads as that
  quadrilateral, start 1221, duration 940. A zone with the crossed order
  Neuma's annotations copy from DMOS (`1221,1589 2151,1589 1221,2266
  2161,2266`) is refused with the shape's crossed-corners message, prefixed
  by the zone's id.
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
- **Zones without coordinates** are not read and are listed, in document
  order, in `zones_without_coordinates`.
- **Refusals.** A root other than `<mei>` in the MEI namespace, `create_
  timelines()` before `load()`, and a second source are refused.
