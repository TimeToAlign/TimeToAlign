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
# drawn on an analytical diagram are all areas of an image. A graphical
# timeline runs along the image's x axis, so each such area is also an
# interval on that timeline: it starts where the area starts in x and lasts
# as long as the area is wide.
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
# puts Boxes on a page and on a system of that page, shows the rectangles of
# a graphical analysis as Boxes on its image crops, reads Boxes from a DMOS
# page layout and from an MEI facsimile, lays out the pages of an IIIF
# volume, and writes a bundle holding Boxes as a package and reads it back.

# %% [markdown]
# ## Setup
#
# Install the library with `pip install timetoalign`. Every source file this
# guide reads, except the annotation table of the graphical analysis, is
# written by the guide into a temporary directory. The helpers: `boxes`
# reads the Boxes a timeline holds (ID, interval and shape, straight from
# its event table), `box_table` shows them one per row with the x extent of
# each shape beside the interval, `refusal` runs a call the library refuses
# and prints its message, `write` writes a source file into the temporary
# directory, and `show` prints one key of a manifest file.

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
from timetoalign.testdata import ensure_data
from timetoalign.timelines import SegmentLine

FILES = Path(tempfile.mkdtemp())
SHAPES = {"bounding_box": BoundingBox, "quadrilateral": Quadrilateral}


def boxes(timeline):
    """The Boxes a timeline holds itself: ID, start, duration and shape, in table order."""
    events = timeline.get_events(include_children=False)
    columns = [
        events.get_field(kind)
        for name, kind in SHAPES.items()
        if name in events.table.column_names
    ]
    starts, durations = events.get_field("start"), events.get_field("duration")
    return [
        {
            "id": event_id,
            "start": starts[row],
            "duration": durations[row],
            "shape": next(c[row] for c in columns if c[row] is not None),
        }
        for row, (event_id, kind) in enumerate(
            zip(events.column_values("id"), events.column_values("event_type"))
        )
        if kind == "Box"
    ]


def box_table(timeline):
    """One row per Box: its interval on the timeline and its shape's x extent."""
    return pd.DataFrame(
        [
            {
                "id": box["id"],
                "start": box["start"].value,
                "duration": box["duration"].value,
                "shape": type(box["shape"]).__name__,
                "x_min": box["shape"].x_min.value,
                "x_max": box["shape"].x_max.value,
            }
            for box in boxes(timeline)
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
# (`ark:/12148/bpt6k11620688`), as DMOS states them. The quadrilateral keeps
# them in the canonical order.

# %%
DMOS_MEASURE_ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")
MEASURE_1 = [[1221, 1589], [2151, 1589], [1221, 2266], [2161, 2266]]
MEASURE_2 = [[2151, 1589], [2766, 1589], [2161, 2266], [2775, 2266]]

measure_1 = Quadrilateral.from_points(
    MEASURE_1, order=DMOS_MEASURE_ORDER, unit="pixels"
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
# ## A page, a system and its measures
#
# A timeline standing for an area of an image says so in its `meta`:
# `meta["box"]` is that area as a shape's dict form, in the image's space,
# and `meta["image"]` names the image. A page spans its whole image, so its
# box starts at x 0. The page image of the score above is 4512 × 6174
# pixels, as Gallica's IIIF manifest states.

# %%
PAGE_URL = "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688/f2"
page_area = BoundingBox.from_xywh(0, 0, 4512, 6174, unit="pixels")
page = DiscreteGraphicalTimeline(
    length=4512,
    unit="pixels",
    uid="page",
    name="Page 2",
    meta={"box": page_area.to_dict(), "image": PAGE_URL},
)

# %% [markdown]
# A system of that page is a child timeline. It starts where the system
# starts in x, so its offset on the page is the system's `x_min`, and it
# states its own area, the system's quadrilateral as DMOS gives it (corners
# top-left, bottom-left, top-right, bottom-right).

# %%
DMOS_SYSTEM_ORDER = ("top_left", "bottom_left", "top_right", "bottom_right")
SYSTEM_1 = [[1221, 1589], [1221, 2266], [4097, 1589], [4101, 2266]]
system_area = Quadrilateral.from_points(
    SYSTEM_1, order=DMOS_SYSTEM_ORDER, unit="pixels"
)

system = DiscreteGraphicalTimeline(
    length=(system_area.x_max - system_area.x_min).value,
    unit="pixels",
    uid="system_1",
    name="System 1",
    meta={"box": system_area.to_dict()},
)
page.add_child(system, offset=system_area.x_min)
{"offset": page.get_child_offset("system_1"), "length": system.length}

# %% [markdown]
# `Box.from_shape()` makes the Box of a shape on a timeline. Its `origin` is
# the x at which the timeline stands in the image, here the system's
# `x_min`. The Box's interval follows from the shape:
#
# - `start` = the shape's `x_min` − the timeline's origin,
# - `duration` = the shape's `x_max` − `x_min`.

# %%
measure_2 = Quadrilateral.from_points(
    MEASURE_2, order=DMOS_MEASURE_ORDER, unit="pixels"
)
system.add_events(
    [
        Box.from_shape(measure_1, origin=system_area.x_min, id="m1"),
        Box.from_shape(measure_2, origin=system_area.x_min, id="m2"),
    ]
)
box_table(system)

# %% [markdown]
# Measure 1 starts at x 1221 of the image and the system stands at x 1221,
# so the measure starts at 0 on the system; measure 2 starts at 2151, which
# is 930 on the system. Asked from the page, which stands at x 0, a Box's
# interval is again the image's x: the timestamp of measure 2 gives 2151 on
# the page and 930 on the system.

# %%
print(page.get_timestamp_for("m2"))

# %% [markdown]
# A Box whose interval contradicts its shape is refused, on every path that
# adds events. Forgetting the origin is the common case: the Box below
# starts at 1221, the shape's x in the image, but on a system standing at
# x 1221 of its image the shape starts at 0.

# %%
refusal(lambda: system.add_events([Box.from_shape(measure_1, id="m1_again")]))

# %% [markdown]
# ## The rectangles of a graphical analysis
#
# [How to Transfer Annotations Between Graphical Analyses](how01_thoresen_annotation_transfer.ipynb)
# places eleven rectangles, drawn on five image crops of an analytical
# diagram, on a segment line of those crops. Each rectangle is stated in its
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
# above. The excerpt below is page 2 of the Gallica score, trimmed to its
# first system and that system's first two measures and to the keys the
# loader reads.

# %%
write(
    "dmos_page.json",
    json.dumps(
        {
            "score_image_url": "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688",
            "pages": [
                {
                    "page_url": PAGE_URL,
                    "no_page": 1,
                    "systems": [
                        {
                            "id": 1,
                            "region": SYSTEM_1,
                            "measures": [{"region": MEASURE_1}, {"region": MEASURE_2}],
                        }
                    ],
                }
            ],
        },
        indent=1,
    ),
)

# %% [markdown]
# `DmosLoader` reads one root timeline per page, each system as a child at
# its left x, and each measure as a Box with its quadrilateral; measures
# have no IDs in DMOS, so their Boxes are minted `box_1`, `box_2`, … per
# system. DMOS states no page size: `page_sizes` gives it (page number →
# width, height), and the page then has a length and a `meta["box"]`.

# %%
(dmos_page,) = DmosLoader.from_file(FILES / "dmos_page.json").create_timelines(
    page_sizes={1: (4512, 6174)}
)
dmos_system = dmos_page.get_child(dmos_page.list_children()[0])
{
    "page": (dmos_page.id, dmos_page.length, dmos_page.meta),
    "system": (
        dmos_system.id,
        dmos_page.get_child_offset(dmos_system.id),
        dmos_system.meta,
    ),
}

# %%
box_table(dmos_system)

# %% [markdown]
# The loader builds what the previous section built by hand: the same
# system offset and the same measure shapes.

# %%
[box["shape"] for box in boxes(dmos_system)] == [box["shape"] for box in boxes(system)]

# %% [markdown]
# Without `page_sizes`, a page's length is unknown and it states no
# `meta["box"]`; its systems and Boxes are read all the same.

# %%
(unsized,) = DmosLoader.from_file(FILES / "dmos_page.json").create_timelines()
(unsized.length, unsized.meta)

# %% [markdown]
# ## An MEI facsimile
#
# An MEI `facsimile` holds one `surface` per page image, with a `graphic`
# naming the image and its size and one `zone` per area; elements of the
# encoding point to their zones with `@facs`. The first surface below is an
# excerpt of a chorale of the *Anhaltisches Choralbuch digital* edition
# (CC BY 4.0): two measure zones and the two measures pointing to them. The
# second surface is page 2 of the Gallica score, holding measure 1 as the
# zone `as_mei_zone()` wrote above, its corners as `points`.

# %%
SURFACE = "surface_01M7DF2DF3J8BVY6RSGDQBJ6W6"
ZONE_1 = "zone_01M7DF2DF4MQT5E96R26XFVEYG"
ZONE_2 = "zone_01M7DF2DF4R34CYF6DVH8GZD9P"
zone_attributes = " ".join(
    f'{name}="{value}"' for name, value in measure_1.as_mei_zone().items()
)

write(
    "facsimile.mei",
    f"""<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.1">
  <meiHead><fileDesc><titleStmt><title>Gelobet seist du Jesu Christ</title></titleStmt>
  <pubStmt/></fileDesc></meiHead>
  <music>
    <facsimile>
      <surface xml:id="{SURFACE}" n="54" ulx="0" uly="0" lrx="3999" lry="2999">
        <graphic xml:id="graphic_01M7DF2DF43DZBVPP3758RC994"
                 target="../facsimile/20251230_181354.jpg" width="4000" height="3000"/>
        <zone xml:id="{ZONE_1}" type="measure" ulx="520" uly="390" lrx="874" lry="1019"/>
        <zone xml:id="{ZONE_2}" type="measure" ulx="814" uly="408" lrx="1129" lry="1033"/>
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
            <measure xml:id="m-43" metcon="false" n="0" facs="#{ZONE_1}"/>
            <measure xml:id="m-57" n="1" facs="#{ZONE_2}"/>
          </section>
        </score>
      </mdiv>
    </body>
  </music>
</mei>
""",
)

# %% [markdown]
# `MeiFacsimileLoader` reads one root timeline per surface: its ID is the
# surface's `xml:id`, its name the surface's `@n`, its length the graphic's
# width, and its `meta` the graphic's area and target. Each zone becomes a
# Box with the zone's `xml:id`: a `BoundingBox` from `ulx` … `lry`, or a
# `Quadrilateral` where the zone has `points`.

# %%
mei = MeiFacsimileLoader.from_file(FILES / "facsimile.mei")
chorale, gallica_page = mei.create_timelines()
{
    surface.id: (surface.name, surface.length, surface.meta)
    for surface in (chorale, gallica_page)
}

# %%
box_table(chorale)

# %% [markdown]
# The `@facs` links are kept per element, so a measure of the encoding finds
# its zone, and through it its Box.

# %%
mei.facs

# %% [markdown]
# The zone written from measure 1 reads back as the same quadrilateral.

# %%
boxes(gallica_page)[0]["shape"] == measure_1

# %% [markdown]
# ## An IIIF volume
#
# An IIIF Presentation manifest lists the page images of a volume as
# canvases, each with its width and height. `IIIFManifestLoader` reads the
# first canvas by default; `canvases="all"` lays every canvas end to end as
# a segment line, one child per canvas. The excerpt below is the Gallica
# manifest of the score above (Presentation 2), trimmed to its first three
# canvases.

# %%
MANIFEST_ID = "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688/manifest.json"
CANVAS = "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688/canvas/f{}"
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
# By default the pages are laid along their height. `axis="width"` lays
# them along their width, the axis Boxes run along, so each child is as
# long as its canvas is wide. Each child, and the volume, states the IIIF
# resource it stands for as a `dcterms:source` statement.

# %%
volume = IIIFManifestLoader.from_file(FILES / "manifest.json").create_timeline(
    axis="width", canvases="all"
)
pd.DataFrame(
    [
        {
            "id": child_id,
            "name": volume.get_child(child_id).name,
            "offset": volume.get_child_offset(child_id).value,
            "length": volume.get_child(child_id).length.value,
            "dcterms:source": volume.get_child(child_id).statements["dcterms:source"][
                0
            ]["id"],
        }
        for child_id in volume.list_children()
    ]
)

# %%
(volume.length, volume.statements)

# %% [markdown]
# ## Boxes in a package
#
# A bundle keeps its timelines in a TimeToShare manifest; the Boxes are data
# and live beside it in a table, `boxes.parquet`, one row per Box with its
# timeline's ID, which the manifest names among its `resources` with the
# type `Box`. The bundle below holds the page built by hand, with its
# system and the two measures, and a logical timeline beside it.

# %%
score = ContinuousLogicalTimeline(length=Fraction(8), uid="score", name="Score")
bundle = AlignmentBundle(name="Page and score")
bundle.add_timeline(page)
bundle.add_timeline(score)

manifest_path = bundle.to_file(FILES / "package" / "page.yaml")
sorted(path.name for path in manifest_path.parent.iterdir())

# %% [markdown]
# The manifest holds the timelines and their `meta`, and names the table:

# %%
show(manifest_path, "resources")

# %% [markdown]
# `from_file()` reads the manifest and the table. Every Box comes back on
# its timeline with the same ID, interval and shape, and is checked against
# its shape on the way in.

# %%
read = AlignmentBundle.from_file(manifest_path)
read_system = read.timelines["page"].get_child("system_1")
(boxes(read_system) == boxes(system), read_system.meta == system.meta)

# %% [markdown]
# ## Next
#
# - The same rectangles carried from one analysis to another through
#   seconds, with MatchClaims and Projections:
#   [How to Transfer Annotations Between Graphical Analyses](how01_thoresen_annotation_transfer.ipynb).
# - Timelines laid along drawn paths in an image:
#   [How to Work with Graphical Timelines](how01_graphical_timelines.ipynb).
# - Manifests, files and packages in depth:
#   [How to Save and Load Timelines, Groups and Bundles](how02_group_and_bundle_files.ipynb)
#   and [How to Keep MatchClaims, Projections and Beliefs in a Package](how03_match_claims_package.ipynb).
