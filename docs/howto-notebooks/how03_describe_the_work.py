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
# # How to Describe the Work a Bundle Is About
#
# An alignment bundle holds the timelines of some music, a score or a
# recording, and nothing in their units and lengths says which music it is.
# A TimeToShare manifest says so in its **description layer**: a
# {{< glossary "description link" >}} leads from the manifest, a timeline or
# a region to an entity outside it, at one of five
# {{< glossary level >}}s (a work, an expression, a manifestation, an item or
# an agent), and names that entity by the addresses catalogues give it. The
# element *pertains to* the entity; the two never become one, and a timeline
# keeps its own id, unit and length.
#
# This guide describes Beethoven's Piano Sonata No. 29 in B♭ major, Op. 106.
# It builds the link to the work from the addresses of its catalogue pages,
# lists the four movements, states who made the link, adds a second reading
# of the movements' order, links a recording and its first movement, finds
# and edits links, and writes the manifest whole and in part.
#
# The format of the description layer is specified in TimeToShare's
# [description of a manifest](https://github.com/TimeToAlign/TimeToShare/blob/main/docs/description.md);
# this guide shows how the library carries it.

# %% [markdown]
# ## Setup
#
# Install the library with `pip install timetoalign`; the TimeToShare
# package, whose classes the library uses for links, comes with it. No data
# is downloaded: every address below is read offline. `show` prints a
# manifest as the text a file would hold, in canonical YAML, the
# serialization `to_file()` writes by default, and `summary` lists links by
# their level, their first URI and the agent who asserted them.

# %%
import copy

from timetoshare import dumps

import timetoalign as tta

EDITOR = "https://example.org/people/editor"
MUSICOLOGIST = "https://example.org/people/musicologist"

WORK = "https://musicbrainz.org/work/4a622b0d-b0c0-405e-b49e-05b70c108284"
IMSLP = "https://imslp.org/wiki/Piano_Sonata_No.29,_Op.106_(Beethoven,_Ludwig_van)"
MOVEMENTS = {
    "I. Allegro": "https://musicbrainz.org/work/80cbaf7b-a1da-3fc5-b650-6eb72a803980",
    "II. Scherzo. Assai vivace": (
        "https://musicbrainz.org/work/7813c9cf-6054-3ef3-b9ed-8ae4690581c9"
    ),
    "III. Adagio sostenuto": (
        "https://musicbrainz.org/work/7d110747-f011-41d4-9ce5-a1fad9abed2f"
    ),
    "IV. Introduzione. Largo – Allegro risoluto": (
        "https://musicbrainz.org/work/cfa82cd6-bcb5-39dd-9beb-c98e275c7c84"
    ),
}


def show(data):
    """Print a manifest as canonical YAML."""
    print(dumps(data), end="")


def summary(links):
    """List each link as its level, its first URI and who asserted it."""
    return [
        (
            link.level,
            link.uri[0],
            link.assertion.asserted_by[0] if link.assertion else None,
        )
        for link in links
    ]


def slots(link):
    """List each part slot of a link: its label and what represents it."""
    return [
        (slot.label, list(slot.represented_by) or slot.absent) for slot in link.parts
    ]


# %% [markdown]
# ## Name the work by its catalogue pages
#
# `Link.from_urls()` takes the addresses you copy from catalogue pages: here
# the work's pages at MusicBrainz, Wikidata, VIAF and IMSLP. Its opus number
# is given as an identifier in the scheme `mb_beethoven_opus`, which the
# bundle declares below.

# %%
work = tta.Link.from_urls(
    WORK,
    "https://www.wikidata.org/wiki/Q146730",
    "https://viaf.org/viaf/186299955",
    IMSLP,
    identifiers={"mb_beethoven_opus": "op. 106"},
    label="Piano Sonata No. 29 in B♭ major, Op. 106",
)
work.level, work.relation

# %% [markdown]
# Each address is read through TimeToShare's registry of identifier schemes,
# which knows what kind of entity each catalogue's pages name. MusicBrainz
# and IMSLP pages name works, so the link is at the level `work` without your
# stating it; the relation of a link at a bibliographic level is
# `pertains_to`.
#
# MusicBrainz, Wikidata and VIAF each give the work an **entity URI**, a URI
# identifying the work itself, and every address is rewritten to that form
# (`http`, Wikidata's `/entity/` path). These go to `uri`, in the order given,
# so the MusicBrainz work comes first and is the preferred one.

# %%
work.uri

# %% [markdown]
# IMSLP gives no URI for a work, so its page name becomes an
# {{< glossary identifier >}}: a value in a {{< glossary scheme >}}, here the
# scheme of IMSLP page names, which the Wikidata property P839 identifies.
# The opus number has no URI anywhere, so it stays an identifier in the
# scheme you named. Each fact is stored once: an identifier whose scheme
# gives URIs would have been stored in `uri` as its URI instead.

# %%
work.identifiers

# %% [markdown]
# ## List the movements
#
# A link may state the parts of its entity as it reads them, a
# {{< glossary "part reading" >}}: every part, in order, each as a
# {{< glossary "part slot" >}}. `PartSlot.from_url()` names a movement by its
# MusicBrainz page. A slot is either represented by elements of the manifest
# or absent with an {{< glossary "absence reason" >}}; nothing in the bundle
# represents a movement yet, so each is `not_represented`: the movement
# exists, and no element of this manifest represents it.
#
# `with_parts()` returns a new link; the link built above stays as it was.

# %%
work = work.with_parts(
    "has_part",
    [
        tta.PartSlot.from_url(url, label=label, absent="not_represented")
        for label, url in MOVEMENTS.items()
    ],
)
slots(work)

# %% [markdown]
# ## Declare the scheme and add the link
#
# The bundle stands for the manifest as a whole, so a link of the manifest
# is a link of the bundle. A scheme without a URI of its own is declared once
# in the manifest and named by its id. `add_scheme()` takes the id, a name and the
# URI of the MusicBrainz series of Beethoven's works by opus number. The
# scheme id is an id of the manifest like any other, unique beside its
# timelines, regions and groups.

# %%
bundle = tta.AlignmentBundle(id="op106", name="Beethoven, Piano Sonata No. 29, Op. 106")
bundle.add_scheme(
    "mb_beethoven_opus",
    name="Works of Ludwig van Beethoven by opus number",
    uri="https://musicbrainz.org/series/004f17aa-0f46-4fec-b939-c5dfc849e5c6",
)
bundle.schemes

# %% [markdown]
# An {{< glossary assertion >}} records who stated a link, when and how.
# Rather than build one for every link, set a default: inside the `with`
# block, every link added without an assertion takes the default, dated the
# day it is added (for this page, the day it was built). The method is
# `manual` unless you state another.

# %%
with tta.set_default_assertion(asserted_by=EDITOR) as default:
    bundle.add_link(work)
default

# %% [markdown]
# The default itself carries no date; the link took it dated the day it was
# added.

# %%
bundle.links[0].assertion

# %% [markdown]
# Leaving the block restores the default it replaced, here none: a link added
# now keeps no assertion unless it brings its own.

# %%
tta.get_default_assertion() is None

# %% [markdown]
# ## Add a second reading
#
# Sources may disagree about an entity. The first edition published in London
# in 1819 printed the sonata in two parts: the first holds the first three
# movements in the order I–III–II, the Adagio sostenuto before the Scherzo,
# and the second the introduction and fugue. Both first editions, Vienna's
# and London's, are on [IMSLP](https://imslp.org/wiki/Piano_Sonata_No.29,_Op.106_(Beethoven,_Ludwig_van)).
#
# One link is one attributed reading, so the London order is a second link
# to the same work, with an assertion of its own. Here a second person states
# it, and the assertion names the IMSLP page as the source it rests on.

# %%
london_order = [
    "I. Allegro",
    "III. Adagio sostenuto",
    "II. Scherzo. Assai vivace",
    "IV. Introduzione. Largo – Allegro risoluto",
]
london = tta.Link.from_url(
    WORK, label="Piano Sonata No. 29 in B♭ major, Op. 106"
).with_parts(
    "has_part",
    [
        tta.PartSlot.from_url(MOVEMENTS[label], label=label, absent="not_represented")
        for label in london_order
    ],
)
with tta.set_default_assertion(asserted_by=MUSICOLOGIST, based_on=IMSLP):
    bundle.add_link(london)
slots(bundle.links[1])

# %% [markdown]
# Both readings keep four slots. That the London edition printed the fourth
# movement as a part of its own is a fact of that edition, which a link at
# the level `manifestation` could state with parts of its own; the readings
# of the work are left as they are.

# %% [markdown]
# ## Link a recording and its first movement
#
# Timelines and regions carry links as the bundle does. The recording below
# is invented for this guide, and so are its timings and its release:
# `https://example.org/releases/hammerklavier` is an address no catalogue
# recognises, so the link keeps it as it is and you state the level. A
# MusicBrainz release page would be recognised at the level `manifestation`
# by itself. A second link names the pianist at the level `agent`, whose
# relation is the agent's role, here `prf`, the performer, of the MARC
# relators.

# %%
RELEASE = "https://example.org/releases/hammerklavier"
PERFORMER = "http://id.loc.gov/vocabulary/relators/prf"

recording = tta.ContinuousPhysicalTimeline(
    length=2700.0, uid="recording", name="Recording"
)
recording.add_link(RELEASE, level="manifestation")
recording.add_link(
    "https://example.org/people/pianist", level="agent", relation=PERFORMER
)
[(link.level, link.relation) for link in recording.links]

# %% [markdown]
# The first movement takes the first 690 seconds. Its region pertains to the
# movement, a work of its own, which belongs to the sonata at position 1: a
# `part_of` entry names the sonata by its MusicBrainz URI and its opus number.
#
# A region is frozen. `with_link()` returns a copy holding one more link, and
# `replace_region()` stores that copy in the place of the region with its id.

# %%
allegro = recording.create_region(0.0, 690.0, id="mvt_1", name="I. Allegro")
movement = tta.Link.from_url(MOVEMENTS["I. Allegro"], label="I. Allegro").with_part_of(
    tta.PartOf(
        "has_part",
        uri=work.uri[:1],
        identifiers=(tta.Identifier("mb_beethoven_opus", "op. 106"),),
        position=1,
    )
)
recording.replace_region(allegro.with_link(movement))
allegro.links, summary(recording.get_region("mvt_1").links)

# %% [markdown]
# Once the recording is in the bundle, its region can represent the first
# movement in both readings. A link is frozen too: build the reading with its
# first slot represented by `mvt_1`, and `replace_link()` puts it in the
# place of the old one.

# %%
bundle.add_timeline(recording)
represented = tta.PartSlot.from_url(
    MOVEMENTS["I. Allegro"], label="I. Allegro", represented_by=["mvt_1"]
)
for reading in bundle.get_links(level="work"):
    bundle.replace_link(
        reading, reading.with_parts("has_part", [represented, *reading.parts[1:]])
    )
[slots(reading) for reading in bundle.links]

# %% [markdown]
# ## Find links
#
# `get_links()` answers the links matching every filter given, in their
# order: `level`, `relation`, `asserted_by` (one of the agents asserting
# the link) and `scheme`. A scheme may be named by the id of a scheme of the
# manifest, by a scheme's URI, or by the id of a scheme of TimeToShare's
# registry, which also matches a URI that the registry recognises in that
# scheme.

# %%
summary(bundle.get_links(level="work"))

# %%
summary(bundle.get_links(asserted_by=MUSICOLOGIST))

# %% [markdown]
# Only the editor's reading carries the opus number and the IMSLP page name;
# both readings name the work by its MusicBrainz URI.

# %%
{
    scheme: summary(bundle.get_links(scheme=scheme))
    for scheme in ("mb_beethoven_opus", "imslp", "musicbrainz_work")
}

# %% [markdown]
# Filters combine, and every element carrying links answers the same
# query: the bundle, a timeline and a region.

# %%
(
    summary(bundle.get_links(level="work", asserted_by=EDITOR)),
    summary(recording.get_links(relation=PERFORMER)),
    summary(recording.get_region("mvt_1").get_links(scheme="musicbrainz_work")),
)

# %% [markdown]
# ## Write and read the manifest
#
# `to_dict()` writes the links where they stand: the bundle's in the
# manifest's `links`, the recording's and its region's in theirs, and the
# scheme in `schemes`. It writes the minimal form, which leaves out every
# key at its default; `to_dict(minimal=False)` writes the complete form.

# %%
manifest = bundle.to_dict()
show(manifest)

# %% [markdown]
# `from_dict()` reads the links and schemes back, and the bundle read writes
# both forms exactly as the original does.

# %%
read_back = tta.AlignmentBundle.from_dict(manifest)
(
    read_back.links == bundle.links,
    read_back.schemes == bundle.schemes,
    read_back.to_dict() == manifest,
    read_back.to_dict(minimal=False) == bundle.to_dict(minimal=False),
)

# %% [markdown]
# ## Write one timeline
#
# A timeline written on its own writes a manifest of its own, which must be
# valid by itself. The region's link names the scheme `mb_beethoven_opus` in
# its `part_of` entry, so the scheme is written with the timeline; the
# bundle's own links pertain to the bundle and stay behind. A part slot
# naming an element that such a write leaves out would be written as absent,
# `not_represented`.

# %%
show(bundle.get_timeline("recording").to_dict())

# %% [markdown]
# ## Change and remove links
#
# `replace_link()` takes a link equal to the one to replace and the new link,
# or a URL with the options of `Link.from_url()`. Here the release gets a
# label.

# %%
recording.replace_link(
    recording.links[0], RELEASE, level="manifestation", label="Hammerklavier"
)
recording.links[0].label

# %% [markdown]
# A scheme stays while any link of the bundle names it, at any depth: in an
# identifier of its own, of a part slot or of a `part_of` entry.
# `remove_scheme()` refuses and names the first such link.

# %%
try:
    bundle.remove_scheme("mb_beethoven_opus")
except ValueError as error:
    print(f"ValueError: {error}")

# %% [markdown]
# To remove the scheme, remove or change those links first. On the bundle read
# back above, `remove_link()` removes the editor's reading, and the refusal
# then names the next link, the region's. `without_link()` returns the region
# without its link, `replace_region()` stores it, and the scheme can go.

# %%
read_back.remove_link(read_back.links[0])
try:
    read_back.remove_scheme("mb_beethoven_opus")
except ValueError as error:
    print(f"ValueError: {error}")

read_back_recording = read_back.get_timeline("recording")
first_movement = read_back_recording.get_region("mvt_1")
read_back_recording.replace_region(first_movement.without_link(first_movement.links[0]))
read_back.remove_scheme("mb_beethoven_opus").schemes

# %% [markdown]
# ## A slot names elements of the manifest
#
# A part slot names the timelines and regions representing its part by
# their ids, and each id must belong to the manifest. The library checks the
# description layer by TimeToShare's rules whenever it reads a manifest and
# whenever it writes one, and raises the first problem found, its location
# in the manifest first. Here the Scherzo's slot names a region `mvt_2` that
# the recording does not hold: a bundle holding that link refuses to write
# its manifest, and a manifest stating it is refused when read. A slot on
# its own writes its ids bare; inside a manifest every id is written `#`
# followed by the id, so the edited manifest names `#mvt_2`.

# %%
trial = tta.AlignmentBundle.from_dict(manifest)
reading = trial.links[0]
ghost = tta.PartSlot.from_url(
    MOVEMENTS["II. Scherzo. Assai vivace"],
    label="II. Scherzo. Assai vivace",
    represented_by=["mvt_2"],
)
trial.replace_link(
    reading,
    reading.with_parts("has_part", [reading.parts[0], ghost, *reading.parts[2:]]),
)
edited = copy.deepcopy(manifest)
edited["links"][0]["parts"][1] = {**ghost.to_dict(), "represented_by": ["#mvt_2"]}
for operation in (trial.to_dict, lambda: tta.AlignmentBundle.from_dict(edited)):
    try:
        operation()
    except ValueError as error:
        print(f"ValueError: {error}")

# %% [markdown]
# ## Next
#
# - The format of links, schemes, assertions, part readings and absence
#   reasons: TimeToShare's
#   [description of a manifest](https://github.com/TimeToAlign/TimeToShare/blob/main/docs/description.md).
# - Saving manifests to files, selecting one timeline or group, and the
#   complete and minimal forms:
#   [How to Save and Load Timelines, Groups and Bundles](how02_group_and_bundle_files.ipynb).
# - Alignments between the timelines a bundle holds: the
#   [Alignment Bundles and MatchClaims](../tutorials/tut06_alignment_bundles.ipynb)
#   tutorial.
