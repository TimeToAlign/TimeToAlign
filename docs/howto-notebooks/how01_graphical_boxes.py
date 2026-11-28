# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How to Represent Areas of Page Images as Boxes
#
# A measure on a scanned page, a zone of an MEI facsimile and a rectangle
# drawn on an analytical diagram are all areas of an image. An image has no
# time axis of its own: neither its width nor its height is one. It has one
# only along a **time path**, the line a reader follows through it. On a
# page of a score the path runs along the first **accolade** (a system, one
# line of the page) from left to right, then along the second, and so on. A
# page is therefore built of its accolades, laid end to end, and a volume
# of its pages. Along one accolade each area is an interval: it starts
# where the area starts in x, counted from the accolade's left end, and
# lasts as long as the area is wide.
#
# The library holds both views in one event, the **Box**. A Box keeps the
# area as the source states it, its **shape**, in the image's own pixels:
# either a **BoundingBox** (`x`, `y`, `width`, `height`, the IIIF and Media
# Fragments `xywh`) or a **Quadrilateral** (four corners, for the skewed
# measures of a scan). Its `start` and `duration` are the interval on the
# timeline. The two always agree, and the library refuses a Box in which
# they do not.
#
# This guide builds shapes and writes them in the forms other formats use,
# makes the time path of a page from the areas drawn on it, builds the page
# from its accolades and puts Boxes on them, shows the rectangles of a
# graphical analysis as Boxes on its image crops, reads pages from a DMOS
# page layout and from an MEI facsimile, lays the pages of an IIIF volume
# along their paths, and writes a bundle holding Boxes as a package and
# reads it back.

# %% [markdown]
# ## Setup
#
# Install the library with `pip install timetoalign`. Every source file this
# guide reads, except the annotation table of the graphical analysis, is
# written by the guide into a temporary directory. A timeline's Boxes come
# back as the events of any type do: `get_events(event_type="Box")` selects
# them and `get_field(Box)` reads them as `Box` objects, each with its ID,
# interval and shape. The helpers: `box_table` shows them one per row with
# the x extent of each shape beside the interval, `segment_table` lists the
# timelines a page or a volume is built of, `refusal` runs a call the
# library refuses and prints its message, `write` writes a source file into
# the temporary directory, and `show` prints one key of a manifest file.

# %%
import json
import tempfile
from fractions import Fraction
from pathlib import Path

import pandas as pd
from timetoshare import dumps, loads

from timetoalign import (
    AlignmentBundle,
    BoundingBox,
    Box,
    ContinuousLogicalTimeline,
    DiscreteGraphicalTimeline,
    DmosLoader,
    IIIFManifestLoader,
    MeiFacsimileLoader,
    Quadrilateral,
)
from timetoalign.loader.graphical import TimeAxisPath
from timetoalign.testdata import ensure_data
from timetoalign.timelines import SegmentLine

FILES = Path(tempfile.mkdtemp())


def box_table(timeline):
    """One row per Box: its interval on the timeline and its shape's x extent."""
    return pd.DataFrame(
        [
            {
                "id": box.id,
                "start": box.start.value,
                "duration": box.duration.value,
                "shape": type(box.shape).__name__,
                "x_min": box.shape.x_min.value,
                "x_max": box.shape.x_max.value,
            }
            for box in timeline.get_events(event_type="Box").get_field(Box)
        ]
    )


def segment_table(line):
    """One row per timeline a page or a volume is built of, in reading order."""
    return pd.DataFrame(
        [
            {
                "id": child_id,
                "name": line.get_child(child_id).name,
                "offset": line.get_child_offset(child_id).value,
                "length": line.get_child(child_id).length.value,
            }
            for child_id in line.list_children()
        ]
    )


def write(name, text):
    """Write a source file into the temporary directory."""
    (FILES / name).write_text(text, encoding="utf-8")


def refusal(call):
    """Run a call the library refuses and print its message."""
    try:
        call()
    except ValueError as error:
        print(error)


def show(manifest_path, key):
    """Print one key of a manifest file as canonical YAML."""
    print(dumps({key: loads(manifest_path.read_text(encoding="utf-8"))[key]}), end="")


# %% [markdown]
# ## Shapes
#
# A shape is an area of an image in **image space**: the origin is the
# image's top-left corner, x runs right and y down, and every number is in
# one unit and one number type. On `pixels` the numbers are integers.
#
# A `BoundingBox` is built from its top-left corner and its extent with
# `from_xywh()`. This one is a rectangle of the graphical analysis used
# below: x 385, y 46, 139 pixels wide, 20 high.

# %%
rect = BoundingBox.from_xywh(385, 46, 139, 20, unit="pixels")
rect

# %% [markdown]
# A shape writes itself in the forms other formats use: its dict form (as a
# manifest holds it), a Media Fragments and IIIF selector, the `points` of
# an SVG polygon (corners in the order top-left, top-right, bottom-right,
# bottom-left), and the attributes of an MEI `zone`, whose `lrx` and `lry`
# are the lower-right corner (`x + width`, `y + height`).

# %%
{
    "dict form": rect.to_dict(),
    "media fragment": rect.as_media_fragment(),
    "SVG points": rect.as_svg_points(),
    "MEI zone": rect.as_mei_zone(),
}

# %% [markdown]
# Each form reads back into the same box: `from_media_fragment()` reads the
# selector, `from_corners()` the corners an MEI zone states.

# %%
(
    BoundingBox.from_media_fragment("xywh=385,46,139,20", unit="pixels") == rect,
    BoundingBox.from_corners(385, 46, 524, 66, unit="pixels") == rect,
)

# %% [markdown]
# A `Quadrilateral` holds four corners, so it can follow a measure whose
# sides are not upright. `from_points()` takes the points in any order the
# source writes them, with `order=` naming the corner each one is; the
# optical music recognition tool DMOS writes a measure top-left, top-right,
# bottom-left, bottom-right. These are the corners of the first measure of
# the first system on page 2 of a printed score in Gallica
# (`ark:/12148/bpt6k11620688`), as DMOS states them; measure 2 follows it
# on that system, and measure 3 is the first of the second system. The
# quadrilateral keeps the corners in the canonical order.

# %%
DMOS_MEASURE_ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")
MEASURE_1 = [[1221, 1589], [2151, 1589], [1221, 2266], [2161, 2266]]
MEASURE_2 = [[2151, 1589], [2766, 1589], [2161, 2266], [2775, 2266]]
MEASURE_3 = [[971, 2493], [1935, 2493], [971, 3192], [1945, 3192]]

measure_1, measure_2, measure_3 = (
    Quadrilateral.from_points(corners, order=DMOS_MEASURE_ORDER, unit="pixels")
    for corners in (MEASURE_1, MEASURE_2, MEASURE_3)
)
measure_1

# %% [markdown]
# Its top edge is 930 pixels long and its bottom edge 940: the scan is
# slightly skewed. The quadrilateral's bounding box is the smallest
# upright box holding it, and its MEI zone carries both, the box as `ulx`
# to `lry` and the corners as `points`.

# %%
measure_1.bounding_box

# %%
{"SVG points": measure_1.as_svg_points(), "MEI zone": measure_1.as_mei_zone()}

# %% [markdown]
# SVG and MEI `points` may start at any corner and run in either direction.
# `from_svg_points()` reads such a list and brings it into the canonical
# order; here the list starts at the bottom-right corner and runs
# anticlockwise.

# %%
Quadrilateral.from_svg_points(
    "2161,2266 2151,1589 1221,1589 1221,2266", unit="pixels"
) == (measure_1)

# %% [markdown]
# A shape's numbers are never rounded. A whole number given as a float is
# taken (`385.0` is 385); a pixel position of 385.5 cannot be held on an
# integer axis and is refused.

# %%
BoundingBox.from_xywh(385.0, 46, 139, 20, unit="pixels") == rect

# %%
refusal(lambda: BoundingBox.from_xywh(385.5, 46, 139, 20, unit="pixels"))

# %% [markdown]
# ## The time path of a page
#
# **No image has a time axis until its path through the image is defined.**
# A set of boxes makes one: `TimeAxisPath.from_boxes()` takes the areas of
# one page in reading order and returns a `PiecewisePath`, one horizontal
# piece per accolade.
#
# DMOS states the systems of the page as areas too (corners top-left,
# bottom-left, top-right, bottom-right). Each system is one accolade, so
# `accolades=` gives every box its own label.

# %%
PAGE_URL = "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688/f2"
DMOS_SYSTEM_ORDER = ("top_left", "bottom_left", "top_right", "bottom_right")
SYSTEM_1 = [[1221, 1589], [1221, 2266], [4097, 1589], [4101, 2266]]
SYSTEM_2 = [[971, 2493], [971, 3192], [4104, 2493], [4112, 3192]]
system_areas = [
    Quadrilateral.from_points(corners, order=DMOS_SYSTEM_ORDER, unit="pixels")
    for corners in (SYSTEM_1, SYSTEM_2)
]

path = TimeAxisPath.from_boxes(system_areas, accolades=[1, 2])
path.pieces

# %% [markdown]
# A piece spans its boxes exactly: `x0` is their least x and `x1` their
# greatest, `y` is the middle of their y extent and `tolerance` half of it,
# so the strip `y ± tolerance` is the accolade's height. Nothing is
# rounded: the first system spans y 1589 to 2266, whose middle is 3855/2.
#
# The path's length is the sum of its pieces, 2880 + 3141 pixels, and each
# piece starts on the path where the ones before it end.

# %%
(path.offsets, path.length)

# %% [markdown]
# The path converts between its own coordinate and the image's (x, y).
# Coordinate 2880 is the left end of the second accolade, x 971; the point
# (2151, 1900) lies in the first accolade's strip, 930 pixels from its left
# end; a point between the two strips is on no piece.

# %%
(path.to_2d(2880), path.from_2d(2151, 1900), path.from_2d(2151, 2400))

# %% [markdown]
# Where a source states no accolades, `accolades=` is left out and
# `from_boxes()` finds them: reading on, the left edges of the boxes grow
# within an accolade and fall back at the start of the next, so a box
# begins a new accolade when its left edge lies left of the previous box's
# by more than half the span of all left edges. Of the three measures, the
# third falls back from x 2151 to 971 and begins the second piece; each
# piece spans the measures given, not the whole system.

# %%
TimeAxisPath.from_boxes([measure_1, measure_2, measure_3]).pieces

# %% [markdown]
# ## A page, its accolades and their measures
#
# The timelines mirror the path. An accolade is a timeline as long as its
# piece, and it says which area of which image it stands for in its `meta`:
# `meta["box"]` is that area as a shape's dict form, in the image's space,
# and `meta["image"]` names the image. Identifiers are given at creation:
# `uid=` for a timeline, `id=` for a Box.
#
# `Box.from_shape()` makes the Box of a shape on a timeline. Its `origin` is
# the x at which the timeline stands in the image, here the accolade's
# `x_min`. The Box's interval follows from the shape:
#
# - `start` = the shape's `x_min` − the timeline's origin,
# - `duration` = the shape's `x_max` − `x_min`.

# %%
system_1, system_2 = (
    DiscreteGraphicalTimeline(
        length=(area.x_max - area.x_min).value,
        unit="pixels",
        uid=f"system_{number}",
        name=f"System {number}",
        meta={"box": area.to_dict(), "image": PAGE_URL},
    )
    for number, area in enumerate(system_areas, start=1)
)
system_1.add_events(
    [
        Box.from_shape(measure_1, origin=system_areas[0].x_min, id="m1"),
        Box.from_shape(measure_2, origin=system_areas[0].x_min, id="m2"),
    ]
)
system_2.add_events([Box.from_shape(measure_3, origin=system_areas[1].x_min, id="m3")])
box_table(system_1)

# %% [markdown]
# Measure 1 starts at x 1221 of the image and the accolade stands at x 1221,
# so the measure starts at 0 on the accolade; measure 2 starts at 2151,
# which is 930 on the accolade.
#
# The page is the segment line of its accolades, in reading order. It names
# its image and states no `meta["box"]`: its axis is the accolades laid end
# to end, which is no area of the image.

# %%
page = SegmentLine[DiscreteGraphicalTimeline](
    length=0, unit="pixels", uid="page", name="Page 2", meta={"image": PAGE_URL}
)
page.append_segment(system_1, name="System 1")
page.append_segment(system_2, name="System 2")
page

# %% [markdown]
# The page is as long as the path, 6021 pixels, and its accolades start at
# the path's offsets. A coordinate on the page is therefore a position along
# the path, not an x of the image: measure 3 starts at x 971 of the image,
# at 0 on its accolade, and at 2880 on the page.

# %%
print(page.get_timestamp_for("m3"))

# %% [markdown]
# A Box whose interval contradicts its shape is refused, on every path that
# adds events. Forgetting the origin is the common case: the Box below
# starts at 1221, the shape's x in the image, but on an accolade standing at
# x 1221 of its image the shape starts at 0.

# %%
refusal(lambda: system_1.add_events([Box.from_shape(measure_1, id="m1_again")]))

# %% [markdown]
# ## The rectangles of a graphical analysis
#
# [How to Transfer Annotations Between Graphical Analyses](how01_thoresen_annotation_transfer.ipynb)
# places eleven rectangles, drawn on five image crops of an analytical
# diagram, on a segment line of those crops. The diagram is read along one
# horizontal line per crop, one crop after the other, so the crops are to
# the analysis what accolades are to a page. Each rectangle is stated in its
# image's pixels, and each crop starts a few pixels right of its image's
# left edge. As Boxes, the rectangles keep their shapes and each crop states
# where it stands in its image.

# %%
DATA_DIR = ensure_data("thoresen")
annotations = pd.read_csv(DATA_DIR / "thoresen_test.tsv", sep="\t")

# Each crop: its image, its left and right x in the image, the image's height
CROPS = [
    ("thoresen_2010_form-building-patterns_p90-91_page1_1.jpeg", 8, 874, 264),
    ("thoresen_2010_form-building-patterns_p90-91_page1_2.jpeg", 7, 874, 270),
    ("thoresen_2010_form-building-patterns_p90-91_page1_3.jpeg", 7, 874, 266),
    ("thoresen_2010_form-building-patterns_p90-91_page1_4.jpeg", 8, 872, 272),
    ("thoresen_2010_form-building-patterns_p90-91_page2_1.jpeg", 9, 873, 278),
]

analysis = SegmentLine[DiscreteGraphicalTimeline](
    length=0, unit="pixels", uid="dgt_2", name="Thoresen 2010"
)
for number, (image, x0, x1, height) in enumerate(CROPS, start=1):
    area = BoundingBox.from_xywh(x0, 0, x1 - x0, height, unit="pixels")
    crop = DiscreteGraphicalTimeline(
        length=x1 - x0,
        unit="pixels",
        uid=f"crop_{number}",
        meta={"box": area.to_dict(), "image": image},
    )
    on_crop = annotations[annotations["image_filename"] == image]
    crop.add_events(
        [
            Box.from_shape(
                BoundingBox.from_xywh(**json.loads(rectangle), unit="pixels"),
                origin=x0,
                id=rectangle_id,
            )
            for rectangle_id, rectangle in zip(
                on_crop["graphical_element_id"], on_crop["rect_coords_json"]
            )
        ]
    )
    analysis.append_segment(crop, name=f"seg_{number}")

crop_2 = analysis.get_child("crop_2")
crop_2.meta

# %% [markdown]
# Crop 2 stands at x 7 of its image. Rectangle `rect_h2` is the box
# (385, 46, 139, 20), so on the crop it starts at 385 − 7 = 378 and lasts
# 139 pixels.

# %%
box_table(crop_2)

# %% [markdown]
# On the segment line, crop 2 follows crop 1 (866 pixels), so `rect_h2`
# spans 1244 to 1383 there: the same interval the annotation-transfer guide
# computes, now with the rectangle's y and height kept beside it.

# %%
print(analysis.get_timestamp_for("rect_h2"))

# %% [markdown]
# ## A DMOS page layout
#
# DMOS writes the layout it recognises on each page as JSON: per page its
# image (`page_url`) and its systems, per system its region and its
# measures, each with its region. Region corners are in the orders seen
# above. The excerpt below is pages 2 and 3 of the Gallica score, trimmed
# to the first two systems of page 2 and the first of page 3, to the
# measures used above and the first measure of page 3, and to the keys the
# loader reads.

# %%
SCORE_URL = "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688"
SYSTEM_3 = [[519, 896], [519, 1647], [3661, 896], [3655, 1647]]
MEASURE_4 = [[519, 896], [1520, 896], [519, 1647], [1522, 1647]]

write(
    "dmos_pages.json",
    json.dumps(
        {
            "score_image_url": SCORE_URL,
            "pages": [
                {
                    "page_url": PAGE_URL,
                    "no_page": 1,
                    "systems": [
                        {
                            "id": 1,
                            "region": SYSTEM_1,
                            "measures": [{"region": MEASURE_1}, {"region": MEASURE_2}],
                        },
                        {
                            "id": 2,
                            "region": SYSTEM_2,
                            "measures": [{"region": MEASURE_3}],
                        },
                    ],
                },
                {
                    "page_url": f"{SCORE_URL}/f3",
                    "no_page": 2,
                    "systems": [
                        {
                            "id": 1,
                            "region": SYSTEM_3,
                            "measures": [{"region": MEASURE_4}],
                        }
                    ],
                },
            ],
        },
        indent=1,
    ),
)

# %% [markdown]
# DMOS gives each page its time path: the systems, read one after the
# other. `create_paths()` returns it per page number (`no_page`), every
# stated system one piece over the x extent of its region. Page 1 has the
# path made by hand above.

# %%
dmos = DmosLoader.from_file(FILES / "dmos_pages.json")
dmos_paths = dmos.create_paths()
(dmos_paths[1] == path, dmos_paths[2].pieces)

# %% [markdown]
# `create_timeline()` reads the whole file as one volume: a segment line of
# its pages, each page a segment line of its systems' accolades, each
# measure a Box of its system's accolade with its quadrilateral. The
# identifiers are given in the same call. `uid` names the volume; `page_ids`,
# `accolade_ids` and `box_ids` each name every element of their kind, as a
# sequence of IDs in document order or as a template filled from `page`
# (the page's position, from 1), `accolade` (its position on its page),
# `box` (its position in its accolade) and `n` (the running count). DMOS
# states no measure numbers, so `box_ids="m{n}"` numbers the measures
# through the score.

# %%
scan = dmos.create_timeline(
    uid="scan",
    name="Gallica bpt6k11620688",
    page_ids="page_{page}",
    accolade_ids="p{page}_s{accolade}",
    box_ids="m{n}",
)
scan

# %% [markdown]
# Pages are named by their `no_page`, accolades by their DMOS system ID.
# The volume and the pages name their images and state no area; each
# accolade states its system's region.

# %%
dmos_page = scan.get_child("page_1")
{
    "volume": scan.meta,
    "page": dmos_page.meta,
    "accolade": dmos_page.get_child("p1_s2").meta,
}

# %%
segment_table(dmos_page)

# %%
box_table(dmos_page.get_child("p1_s1"))

# %% [markdown]
# The loader builds what the previous section built by hand: the same
# lengths and offsets and the same measure shapes.

# %%
system_events = dmos_page.get_child("p1_s1").get_events(event_type="Box")
(
    segment_table(dmos_page)[["offset", "length"]].equals(
        segment_table(page)[["offset", "length"]]
    ),
    [box.shape for box in system_events.get_field(Box)]
    == [box.shape for box in system_1.get_events(event_type="Box").get_field(Box)],
)

# %% [markdown]
# On the volume, measure 4, the first of the second page, starts where the
# first page ends, at 6021.

# %%
print(scan.get_timestamp_for("m4"))

# %% [markdown]
# `create_timelines()` returns one root timeline per page instead. Without
# the identity parameters, timelines get minted IDs (`dgt_…`) and Boxes too
# (`box_…`), seeded from the file the loader read.

# %%
{
    root.id: {
        accolade_id: [
            box.id
            for box in root.get_child(accolade_id)
            .get_events(event_type="Box")
            .get_field(Box)
        ]
        for accolade_id in root.list_children()
    }
    for root in DmosLoader.from_file(FILES / "dmos_pages.json").create_timelines()
}

# %% [markdown]
# ## An MEI facsimile
#
# An MEI `facsimile` holds one `surface` per page image, with a `graphic`
# naming the image and one `zone` per area; elements of the encoding point
# to their zones with `@facs`. The first surface below is a page of a
# chorale of the *Anhaltisches Choralbuch digital* edition (CC BY 4.0) with
# its nine measure zones, and two measures pointing to the first two. The
# second surface is page 2 of the Gallica score, holding measure 1 as the
# zone `as_mei_zone()` wrote above, its corners as `points`.

# %%
SURFACE = "surface_01M7DF2D8Q8FYGETPJXXMG03QP"
ZONES = [
    ("zone_01M7DF2D8RNCKHKF6SMCWB94Z5", 490, 344, 973, 877),
    ("zone_01M7DF2D8R00HP8JYGXRTE13G3", 944, 368, 1523, 864),
    ("zone_01M7DF2D8R4E9KQCXCPC9RV59F", 1481, 388, 2082, 912),
    ("zone_01M7DF2D8S9S18YJ6CYJVF4QCJ", 2034, 402, 2694, 884),
    ("zone_01M7DF2D8SY6CN7SFAZFPE58FT", 2669, 389, 3296, 916),
    ("zone_01M7DF2D8SF9X38YEGDA0JCD04", 498, 858, 1496, 1337),
    ("zone_01M7DF2D8S1F9S83VHS68E6N7W", 1465, 885, 2146, 1353),
    ("zone_01M7DF2D8TV0F9DCHY6QNWYTDX", 2110, 886, 2748, 1357),
    ("zone_01M7DF2D8TSXCHCSA07C235VTV", 2704, 886, 3302, 1387),
]
chorale_zones = "\n        ".join(
    f'<zone xml:id="{zone_id}" type="measure" ulx="{ulx}" uly="{uly}" '
    f'lrx="{lrx}" lry="{lry}"/>'
    for zone_id, ulx, uly, lrx, lry in ZONES
)
zone_attributes = " ".join(
    f'{name}="{value}"' for name, value in measure_1.as_mei_zone().items()
)

write(
    "facsimile.mei",
    f"""<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.1">
  <meiHead><fileDesc><titleStmt><title>Ach bleib bei uns, Herr Jesu Christ</title></titleStmt>
  <pubStmt/></fileDesc></meiHead>
  <music>
    <facsimile>
      <surface xml:id="{SURFACE}" n="15" ulx="0" uly="0" lrx="3999" lry="2999">
        <graphic xml:id="graphic_01M7DF2D8RMG9SGXTZRR3S1BA3"
                 target="facsimile/20251230_180023.jpg" width="4000" height="3000"/>
        {chorale_zones}
      </surface>
      <surface xml:id="gallica_f2" n="2">
        <graphic target="{PAGE_URL}" width="4512" height="6174"/>
        <zone xml:id="m1" type="measure" {zone_attributes}/>
      </surface>
    </facsimile>
    <body>
      <mdiv>
        <score>
          <section>
            <measure xml:id="m-43" metcon="false" n="0" facs="#{ZONES[0][0]}"/>
            <measure xml:id="m-57" n="1" facs="#{ZONES[1][0]}"/>
          </section>
        </score>
      </mdiv>
    </body>
  </music>
</mei>
""",
)

# %% [markdown]
# MEI zones name no accolade, so `MeiFacsimileLoader` makes each surface's
# time path from its zones, with `TimeAxisPath.from_boxes()` and the rule
# of the left edges. On the chorale page the sixth zone falls back from
# x 2669 to 498: the page has two accolades, of five and four measures.

# %%
mei = MeiFacsimileLoader.from_file(FILES / "facsimile.mei")
mei.create_paths()[SURFACE].pieces

# %% [markdown]
# `create_timelines()` reads one root timeline per surface, a page built of
# the accolades of that path: its ID is the surface's `xml:id`, its name the
# surface's `@n`, its `meta["image"]` the graphic's target. The graphic's
# size is on no timeline. Each accolade is named `accolade_K` and states
# the strip of its piece as `meta["box"]`. Each zone becomes a Box of the
# accolade it lies on, with the zone's `xml:id` as its ID: a `BoundingBox`
# from `ulx` … `lry`, or a `Quadrilateral` where the zone has `points`.
# Here the accolades are named in the call, with `accolade_ids`.

# %%
chorale, gallica_page = mei.create_timelines(accolade_ids="p{page}_a{accolade}")
{
    surface.id: (surface.name, surface.length, surface.meta)
    for surface in (chorale, gallica_page)
}

# %%
segment_table(chorale)

# %%
second_accolade = chorale.get_child("p1_a2")
second_accolade.meta

# %% [markdown]
# The second accolade stands at x 498 of the image, so its first zone
# (x 498 to 1496) starts at 0 and lasts 998 pixels. On the page that zone
# starts at 2806, the length of the first accolade.

# %%
box_table(second_accolade)

# %%
print(chorale.get_timestamp_for(ZONES[5][0]))

# %% [markdown]
# The `@facs` links are kept per element, so a measure of the encoding finds
# its zone, and through it its Box.

# %%
mei.facs

# %% [markdown]
# The zone written from measure 1 reads back as the same quadrilateral.

# %%
accolade_boxes = gallica_page.get_child("p2_a1").get_events(event_type="Box")
accolade_boxes.get_field(Box)[0].shape == measure_1

# %% [markdown]
# Where the rule of the left edges does not find a page's accolades (zones
# not listed accolade by accolade, say), the path returned by
# `create_paths()` is corrected or written anew and given back as `paths=`,
# keyed by the surface's `xml:id`. `zone_types=` selects the zones taking
# part by their `@type`, and `box_ids=` replaces the zones' `xml:id`s, as in
# the DMOS section; `create_timeline()` gives the whole facsimile as one
# volume.

# %% [markdown]
# ## An IIIF volume
#
# An IIIF Presentation manifest lists the page images of a volume as
# canvases, each with its width and height. It states no areas, so nothing
# in it says where the page is read: `IIIFManifestLoader` builds a timeline
# only along the time path it is given for a canvas. The excerpt below is
# the Gallica manifest of the score above (Presentation 2), trimmed to its
# first three canvases.

# %%
MANIFEST_ID = f"{SCORE_URL}/manifest.json"
CANVAS = SCORE_URL + "/canvas/f{}"
write(
    "manifest.json",
    json.dumps(
        {
            "@context": "http://iiif.io/api/presentation/2/context.json",
            "@id": MANIFEST_ID,
            "@type": "sc:Manifest",
            "label": "Exemplaire localisé à la Médiathèque Hector Berlioz (CNSMDP), D-9658 (1)",
            "sequences": [
                {
                    "@type": "sc:Sequence",
                    "canvases": [
                        {
                            "@id": CANVAS.format(number),
                            "@type": "sc:Canvas",
                            "label": label,
                            "width": width,
                            "height": height,
                        }
                        for number, label, width, height in [
                            (1, "NP", 4564, 6174),
                            (2, "2", 4512, 6174),
                            (3, "3", 4564, 6174),
                        ]
                    ],
                }
            ],
        },
        indent=1,
    ),
)

# %% [markdown]
# `create_timeline()` takes the paths as a mapping from a canvas, named by
# its position from 0 or by its ID, to its path, and `canvases=` selects
# the canvases to read. The first canvas is an unnumbered page without
# music; the second and third are the pages DMOS read, and DMOS's
# `no_page` is the canvas's position, so the paths of `create_paths()` fit
# as they are. The volume is the segment line of the selected pages, each
# page the segment line of its accolades. IDs are given in the call, as
# above.

# %%
volume = IIIFManifestLoader.from_file(FILES / "manifest.json").create_timeline(
    dmos_paths,
    canvases=[1, 2],
    uid="volume",
    page_ids=["f2", "f3"],
    accolade_ids="p{page}_s{accolade}",
)
volume

# %% [markdown]
# Each page is named with its canvas's label and is as long as its path;
# no page is as long as its canvas is wide. Each page, and the volume,
# states the IIIF resource it stands for as a `dcterms:source` statement,
# and each accolade the strip of its piece as `meta["box"]`.

# %%
pages = segment_table(volume)
pages["dcterms:source"] = [
    volume.get_child(page_id).statements["dcterms:source"][0]["id"]
    for page_id in pages["id"]
]
pages

# %%
(volume.length, volume.statements, volume.get_child("f2").get_child("p1_s1").meta)

# %% [markdown]
# A canvas read along a single line is one plain timeline of the line's
# length: a piano roll, read from top to bottom, takes a `VerticalLinePath`
# down its image (see
# [How to Load Anything](how04_load_anything.ipynb)).

# %% [markdown]
# ## Boxes in a package
#
# A bundle keeps its timelines in a TimeToShare manifest; the Boxes are
# events and live beside it. The registry (`registry.parquet`) holds one
# row per Box with its timeline's ID and its position on the axis of the
# timeline's root (a measure of an accolade at its place on the page), and
# the Box table (`boxes.parquet`) its shape, keyed by its ID; the manifest
# names the registry among its `resources` with the type `Registry`. The
# bundle below holds the page built by hand, with its two accolades and
# their three measures, and a logical timeline beside it.

# %%
score = ContinuousLogicalTimeline(length=Fraction(8), uid="score", name="Score")
bundle = AlignmentBundle(name="Page and score")
bundle.add_timeline(page)
bundle.add_timeline(score)

manifest_path = bundle.to_file(FILES / "package" / "page.tts.yaml")
sorted(path.name for path in manifest_path.parent.iterdir())

# %% [markdown]
# The manifest holds the timelines and their `meta`, and names the
# registry:

# %%
show(manifest_path, "resources")

# %% [markdown]
# `from_file()` reads the manifest and the tables. Every Box comes back on
# its timeline with the same ID, interval (on the timeline's own axis) and
# shape, and is checked against its shape on the way in.

# %%
read = AlignmentBundle.from_file(manifest_path)
read_page = read.timelines["page"]
[
    (
        accolade.id,
        list(
            read_page.get_child(accolade.id).get_events(event_type="Box").get_field(Box)
        )
        == list(accolade.get_events(event_type="Box").get_field(Box)),
        read_page.get_child(accolade.id).meta == accolade.meta,
    )
    for accolade in (system_1, system_2)
]

# %% [markdown]
# ## Next
#
# - The same rectangles carried from one analysis to another through
#   seconds, with MatchClaims and Projections:
#   [How to Transfer Annotations Between Graphical Analyses](how01_thoresen_annotation_transfer.ipynb).
# - Time paths of every kind, and timelines laid along drawn paths in an
#   image:
#   [How to Work with Graphical Timelines](how01_graphical_timelines.ipynb).
# - Manifests, files and packages in depth:
#   [How to Save and Load Timelines, Groups and Bundles](how02_group_and_bundle_files.ipynb)
#   and [How to Keep MatchClaims, Projections and Beliefs in a Package](how03_match_claims_package.ipynb).
