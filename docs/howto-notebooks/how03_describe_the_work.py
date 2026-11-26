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
# # How to Describe What a Bundle and Its Timelines Stand For
#
# A TimeToShare manifest holds two planes. Its temporal structure states how
# coordinates relate: timelines, their nesting, maps, regions and flow
# control. Its {{< glossary statement >}}s state what the elements stand
# for and how those things relate to others, and never nest, order or align
# anything.
#
# Three kinds of element are subjects of statements: the bundle, which is
# the manifest itself and names the work it is about; every timeline, at any
# depth, for the manifestation it is the axis of (an edition, an encoding, a
# recording file, a video); and every region, for the stretch of that
# manifestation it marks. A statement is a pair of a property and a value,
# both identifiers in published vocabularies where they name things; the
# property `type` reads "is a". Works and expressions are never elements:
# they appear as values, the work under the bundle's `schema:about`.
#
# A manifest carrying statements is YAML-LD: its inline `@context` declares
# the prefixes its compact IRIs use, so it reads as linked data without
# fetching anything. The {{< glossary catalogue >}} shipped with the
# TimeToShare package knows the vocabularies, the identifier systems and the
# verbs that state one relation in each of them.
#
# This guide describes the LayerIt example: an MEI encoding of the first six
# bars of Debussy's *Clair de lune*, a concert video of Maria João Pires and
# a WAV file cut from its sound. It states what each timeline is, who
# performs, who said so and who believes it, offers candidates, describes a
# region, writes the manifest, and imports the description of an MEI header
# and of a MeLODy repository. The format is specified in TimeToShare's
# [description of a manifest](https://github.com/TimeToAlign/TimeToShare/blob/main/docs/description.md).

# %% [markdown]
# ## Setup
#
# Install the library with `pip install timetoalign`; the MeLODy import also
# needs rdflib, `pip install "timetoalign[rdf]"`. No data is downloaded: the
# MEI header and the MeLODy repository are written below into a temporary
# directory. The identifiers are those of public catalogues; the editor who
# states the pairs and the days are invented.
#
# `show` prints the pairs of a subject, or of a value node inside it, and
# the {{< glossary "provenance entry" >}} of each pair that has one, marked
# `stated` or `candidate`; a value node's own entries are shown when `show`
# is given that node.
# `show_manifest` prints a manifest as canonical YAML, leaving the format's
# own terms out of `@context` so that only the prefixes remain.

# %%
import json
import tempfile
from pathlib import Path

from timetoshare import FORMAT_CONTEXT, dumps, to_file_id

import timetoalign as tta
from timetoalign.loader import MeiHeadLoader, MelodyLoader

EDITOR = "https://example.org/people/editor"
REVIEWER = "https://example.org/people/reviewer"
DAY = "2026-11-18"
FOLDER = Path(tempfile.mkdtemp())


def text(value):
    """Write a JSON value on one line."""
    return json.dumps(value, ensure_ascii=False)


def show(holder, node=()):
    """Print the pairs of a holder or of one of its value nodes, then their entries.

    A value node's own entries are left out of its pair; ``node=`` shows them.
    """
    for property, value in holder.get_pairs(node=node):
        if isinstance(value, dict):
            value = {key: item for key, item in value.items() if key != "provenance"}
        print(f"{property}: {text(value)}")
    candidates = holder.get_candidates(node=node)
    for entry in holder.get_provenance(node=node):
        kind = "candidate" if entry in candidates else "stated"
        rest = {
            key: value
            for key, value in entry.items()
            if key not in ("property", "value")
        }
        print(f"[{kind}] {entry['property']}: {text(entry['value'])}")
        print(f"    {text(rest)}")


def show_manifest(manifest):
    """Print a manifest as canonical YAML, its context reduced to the prefixes."""
    prefixes = {
        key: value
        for key, value in manifest["@context"].items()
        if key not in FORMAT_CONTEXT
    }
    print(dumps({**manifest, "@context": prefixes}), end="")


# %% [markdown]
# The bundle holds the three timelines: the encoding `clt_6` in quarters,
# 27 of them for six bars of 9/8; the video `cpt_1` in seconds, of unknown
# length, with a child `cpt_2` for the part the example uses; and the WAV
# file `dpt_1` in samples.

# %%
bundle = tta.AlignmentBundle(
    id="layerit", name="LayerIt, Clair de lune mm. 1–6 (Pires)"
)
score = tta.ContinuousLogicalTimeline(
    length=27, uid="clt_6", name="Clair de lune, mm. 1–6 (MEI)"
)
video = tta.ContinuousPhysicalTimeline(
    length=None, uid="cpt_1", name="Maria João Pires live at Jardin Musical"
)
video.create_child(length=33.353016, uid="cpt_2")
wav = tta.DiscretePhysicalTimeline(
    length=1470868, uid="dpt_1", name="ClairDeLune_MariaJoaoPires_untilM6.wav"
)
for timeline in (score, video, wav):
    bundle.add_timeline(timeline)
bundle.timeline_ids

# %% [markdown]
# ## Say what the bundle and a timeline are
#
# `add_statement()` states one pair on a timeline or a bundle, in place. The
# bundle is about the work, named here by its Wikidata item; the value is a
# node, a thing with an `id` and a `label` cached for display. The video's
# timeline is a video: `type` takes a class, and the catalogue's verb
# `is_a` states the same in Wikidata's vocabulary, as an instance of
# (`wdt:P31`) the item for a video, `wd:Q98069877`. `add_verb()` takes the
# verb, the system and the value, and states the pairs the catalogue's
# pattern for them gives.

# %%
bundle.add_statement("schema:about", {"id": "wd:Q29117946", "label": "Clair de lune"})
video.add_statement("type", "schema:VideoObject")
video.add_verb("is_a", "wikidata", "wd:Q98069877")
video.statements

# %% [markdown]
# ## One verb, two vocabularies
#
# The pianist is named in two systems: by a Wikidata item through
# Wikidata's property "performer" (`wdt:P175`), and by a VIAF identifier
# through the MARC relator "performer" (`rel:prf`), the fallback for a
# system without a property of its own. Each pair holds a node of its own.

# %%
video.add_verb("performed_by", "wikidata", "wd:Q235650", label="Maria João Pires")
video.add_verb("performed_by", "fallback", "viaf:113931950")
[(rectangle.verb, rectangle.things) for rectangle in video.get_rectangles()]

# %% [markdown]
# `get_rectangles()` gathers the pairs the catalogue reads as one verb about
# one thing. So far the two performers are two things. A value node is a
# subject too: `node=` addresses it by its path, the property and the
# index, and `schema:sameAs` on the Wikidata node ties the two. The pairs
# then form one rectangle, labelled by the node's label.

# %%
video.add_statement("schema:sameAs", {"id": "viaf:113931950"}, node=("wdt:P175", 0))
[
    (rectangle.verb, rectangle.label, rectangle.things)
    for rectangle in video.get_rectangles()
]

# %% [markdown]
# ## Who said so, and who believes it
#
# A pair without a provenance entry is stated by the manifest's publisher.
# `provenance=` gives the pair an entry naming who stated it, when, how and
# why; the keys are those the file writes. `actor=` records the person
# stating the pair as believing it, with the belief `"true"` and the day.
# A loader or a service records how a pair was made and holds no belief.
# `get_provenance()` answers the entries.

# %%
video.add_statement(
    "schema:sameAs",
    {"id": "https://www.youtube.com/watch?v=Ch2mrPm1JnM"},
    provenance={
        "stated_by": [EDITOR],
        "date": DAY,
        "method": "manual",
        "note": "pasted URL",
    },
    actor=EDITOR,
    date=DAY,
)
video.get_provenance()

# %% [markdown]
# ## Candidates and beliefs
#
# A candidate is a pair held only as its entry: the subject does not state
# it, and it waits for someone to. `add_candidate()` offers one. The WAV
# file was cut from the video's sound, which is an inference. A value
# naming an element of the manifest is the node of its reference, `#`
# followed by its ID, which `to_file_id()` builds.

# %%
source = {"id": to_file_id(video.id)}
wav.add_candidate(
    "dcterms:source",
    source,
    stated_by=EDITOR,
    method="inferred",
    date=DAY,
    note="assumed ripped from the video's sound and cut",
)
show(wav)

# %% [markdown]
# Stating the candidate with `add_statement()` makes it a pair of the
# subject; its entry keeps every key and records the person stating it as
# believing it.

# %%
wav.add_statement("dcterms:source", source, actor=EDITOR, date="2026-11-19")
show(wav)

# %% [markdown]
# Any actor may record a belief in any pair with `add_belief()`: `"true"`,
# `"false"` or `"unlikely"`, at most one per actor, a later one replacing
# the earlier. A belief never changes what is stated: a reviewer who
# believes the source `"false"` leaves the pair in place, and nothing
# derives a truth value from the beliefs. The belief filter reads them for
# one actor, the believer, keeping the states listed in `beliefs=`
# (`"true"`, `"undecided"` where the believer holds no belief,
# `"unlikely"`, `"false"`); by default every state but `"false"`.

# %%
wav.add_belief(
    "dcterms:source", source, actor=REVIEWER, belief="false", date="2026-11-20"
)
(
    wav.statements,
    len(wav.get_provenance(believer=REVIEWER)),
    len(wav.get_provenance(believer=EDITOR, beliefs=["true"])),
)

# %% [markdown]
# A candidate an actor believes `"false"` was examined and refused; it is
# kept, so that it is not offered to that actor again. A lookup in the GND
# offers its entry "Clair de lune" as the work the encoding is an example
# of; it is Debussy's song on Verlaine's poem, not the piano piece. The
# entry keeps how the candidate was found beside the belief.
# `get_candidates()` answers every candidate, and with `believer=` those the
# believer does not believe `"false"`.

# %%
score.add_statement("type", "melod:Manifestation")
score.add_candidate(
    "schema:exampleOfWork",
    {"id": "gnd:1156628156"},
    stated_by=EDITOR,
    method="reconciled",
    service="https://lobid.org/gnd/search",
    date=DAY,
)
score.add_belief(
    "schema:exampleOfWork",
    {"id": "gnd:1156628156"},
    actor=EDITOR,
    belief="false",
    date=DAY,
    note="GND 'Clair de lune' is Debussy's song L 32 on Verlaine, not the piano piece",
)
show(score)
len(score.get_candidates()), score.get_candidates(believer=EDITOR)

# %% [markdown]
# Candidates and beliefs take a node path as statements do. The encoding
# is an edition of the work (`wdt:P629`), and the GND entry for the piano
# piece is offered as the same work, inside that node.

# %%
score.add_statement("wdt:P629", {"id": "wd:Q29117946", "label": "Clair de lune"})
score.add_candidate(
    "schema:sameAs",
    {"id": "gnd:300042558"},
    node=("wdt:P629", 0),
    stated_by=EDITOR,
    method="reconciled",
    service="https://lobid.org/gnd/search",
    match="exact",
    date=DAY,
)
show(score, node=("wdt:P629", 0))

# %% [markdown]
# ## Describe a region
#
# A region stands for the stretch of its timeline's manifestation it marks.
# Regions are frozen: `with_statement()` returns a copy holding the pair,
# and `replace_region()` stores it in the place of the region with its ID.
# The region's siblings `without_statement()`, `with_candidate()`,
# `with_belief()`, `with_verb()` and `with_prefix()` mirror the timeline's
# `remove_statement()`, `add_candidate()`, `add_belief()`, `add_verb()` and
# `add_prefix()`.

# %%
opening = score.create_region(0, 9, id="opening", name="mm. 1–2")
score.replace_region(
    opening.with_statement("rdfs:comment", "the opening, before the melody enters")
)
opening.statements, score.get_region("opening").statements

# %% [markdown]
# ## Write the manifest
#
# `to_dict()` writes each element's `statements` and `provenance` where the
# element stands, and an `@context` holding the format's terms and the
# prefixes the statements use, no others. Every prefix used here is the
# catalogue's; a prefix it lacks is declared on the element with
# `add_prefix()`, and a pair using an undeclared one is refused.

# %%
manifest = bundle.to_dict()
show_manifest(manifest)

# %% [markdown]
# The bundle read back writes the same manifest, and so does a manifest file.

# %%
bundle.to_file(FOLDER / "layerit.yaml")
(
    tta.AlignmentBundle.from_dict(manifest).to_dict() == manifest,
    tta.AlignmentBundle.from_file(FOLDER / "layerit.yaml").to_dict() == manifest,
)

# %% [markdown]
# A timeline written on its own writes a manifest of its own. The WAV
# file's source names the video, which that manifest does not hold, so the
# pair is left out together with its entry; the bundle's own statements
# stay behind, and the `@context` holds the format's terms alone.

# %%
show_manifest(bundle.get_timeline("dpt_1").to_dict())

# %% [markdown]
# ## Import an MEI header
#
# An MEI file's `meiHead` describes the work it encodes, the work's
# expressions, the manifestations embodying them and the items exemplifying
# those. `MeiHeadLoader` states that description in LRMoo's and MeLODy's
# vocabulary. The header below is a short one modelled on the Carl Nielsen
# Edition's header of the song *Hr. Olufs Sang*, CNW 133; the autograph's
# address is invented.

# %%
HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.1">
 <meiHead>
  <fileDesc><titleStmt><title>Hr. Olufs Sang</title></titleStmt><pubStmt/></fileDesc>
  <workList>
   <work xml:id="work_1">
    <identifier xml:id="identifier_1" label="CNW">133</identifier>
    <title xml:id="title_1" type="main" xml:lang="da">Hr. Olufs Sang</title>
    <composer><persName xml:id="persName_1" role="composer">Carl Nielsen</persName></composer>
    <lyricist><persName xml:id="persName_2"
      auth.uri="https://d-nb.info/gnd/" codedval="119018934">Holger Drachmann</persName></lyricist>
    <expressionList>
     <expression xml:id="expression_1">
      <title xml:id="title_3">Version for voice and piano</title>
      <key xml:id="key_1" pname="f" accid="s" mode="minor"/>
     </expression>
    </expressionList>
   </work>
  </workList>
  <manifestationList>
   <manifestation xml:id="source_1">
    <titleStmt><title xml:id="title_4">Score, print</title></titleStmt>
    <relationList>
     <relation xml:id="relation_2" rel="isEmbodimentOf" target="#expression_1"/>
    </relationList>
   </manifestation>
   <manifestation xml:id="source_2" sameas="https://example.org/sources/autograph">
    <titleStmt><title xml:id="title_5" type="main">Score, autograph</title></titleStmt>
    <itemList>
     <item xml:id="item_1">
      <identifier xml:id="identifier_4" label="RISM">DK-Kk CNS 342e</identifier>
     </item>
    </itemList>
   </manifestation>
  </manifestationList>
 </meiHead>
</mei>
"""
(FOLDER / "cnw0133.mei").write_text(HEADER, encoding="utf-8")
song = MeiHeadLoader.from_file(FOLDER / "cnw0133.mei").create_bundle(date="2026-11-19")
song.name, [(tl_id, song.get_timeline(tl_id).name) for tl_id in song.timeline_ids]

# %% [markdown]
# The bundle takes the file's stem as its name; its ID is minted, as are the
# timelines' IDs (`cgt_…`), seeded from the file, so the same file gives the
# same IDs on every run. Each manifestation becomes a
# graphical timeline in millimetres of unknown length, named by its title,
# the axis along which pages of the print or the autograph can be laid out.
# The work lands as a value node under the bundle's `schema:about`, its IRI the
# file's address followed by `#` and the work's `xml:id`, and the bundle
# points to the MEI file with `rdfs:seeAlso`.

# %%
show(song)

# %% [markdown]
# Every pair carries an entry: method `imported`, the agent
# `timetoalign.timelines.LIBRARY_AGENT` (the library's address) unless
# `stated_by=` names another, and as its basis
# the MEI element it comes from. What the header encodes with an IRI is
# stated: the lyricist, whom the header names by a GND number. A name
# without an IRI is a candidate, for reconciliation: the composer.

# %%
WORK = ("schema:about", 0)
show(song, node=WORK)

# %% [markdown]
# The expression is a value node too, under `lrmoo:R4_embodies` of the
# manifestation that embodies it, the print; the autograph holds its item.

# %%
print_timeline = next(
    song.get_timeline(tl_id)
    for tl_id in song.timeline_ids
    if song.get_timeline(tl_id).name == "Score, print"
)
show(print_timeline)

# %% [markdown]
# An imported candidate is stated as any other. Here the editor states the
# composer's name as the header writes it; a lookup service could instead
# offer the person's IRI, as the GND entry was offered above.

# %%
song.add_statement(
    "rel:cmp", {"label": "Carl Nielsen"}, actor=EDITOR, date="2026-11-19", node=WORK
)
song.get_candidates(node=WORK), song.statements["schema:about"][0]["rel:cmp"]

# %% [markdown]
# `create_bundle(into=…)` states the description on an existing bundle, and
# `subjects=` maps the `xml:id` of a manifestation or an item to one of its
# timelines or regions. Importing again adds nothing: a pair already
# stated or held as a candidate is passed over, so a decision stays.
# `base=` gives the MEI file's published address, from which every IRI and
# basis built from an `xml:id` then begins.

# %% [markdown]
# ## Import a MeLODy repository
#
# A MeLODy data repository keeps one Turtle file per entity: a work, an
# expression, a manifestation, an item, a person. Its statements are pairs
# already, so `MelodyLoader` carries them as they are, in LRMoo, MeLODy's
# own terms and schema.org. The repository below is written for this guide
# in the shape of MeLODy's template.

# %%
PREFIXES = """\
@prefix melod: <https://lod.academy/melod/vocab/ontology#> .
@prefix melodTerms: <https://lod.academy/melod/vocab/terms#> .
@prefix lrmoo: <http://iflastandards.info/ns/lrm/lrmoo/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix schema: <https://schema.org/> .
"""
REPOSITORY = {
    "works/1.ttl": """\
<urn:uuid:works/1> a melod:Work ;
  melod:hasTitle [ a melod:Title ; rdfs:label "Clair de lune"@fr ] ;
  owl:sameAs <http://www.wikidata.org/entity/Q29117946> ;
  lrmoo:R3_is_realised_in <urn:uuid:expressions/2> .
""",
    "expressions/2.ttl": """\
<urn:uuid:expressions/2> a melod:Expression ;
  rdfs:label "Clair de lune, piano" ;
  melod:hasContribution [ a melod:Contribution ; melod:hasAgent <urn:uuid:persons/3> ;
    melod:hasRole melodTerms:Composer ] .
""",
    "persons/3.ttl": """\
<urn:uuid:persons/3> a melod:Person ; schema:givenName "Claude" ;
  schema:familyName "Debussy" ; owl:sameAs <http://www.wikidata.org/entity/Q4700> .
""",
    "manifestations/4.ttl": """\
<urn:uuid:manifestations/4> a melod:Manifestation ;
  melod:hasTitle [ a melod:Title ; rdfs:label "Suite bergamasque, first edition" ] ;
  lrmoo:R4_embodies <urn:uuid:expressions/2> .
""",
    "items/5.ttl": """\
<urn:uuid:items/5> a melod:Item ; rdfs:label "Copy 1" ; melod:hasShelfmark "TEST 1" ;
  lrmoo:R7_exemplifies <urn:uuid:manifestations/4> .
""",
}
repository = FOLDER / "melody"
for relative, turtle in REPOSITORY.items():
    path = repository / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PREFIXES + turtle, encoding="utf-8")

suite = MelodyLoader.from_file(repository).create_bundle(
    id="clair_de_lune", date="2026-11-19"
)
[(tl_id, suite.get_timeline(tl_id).name) for tl_id in suite.timeline_ids]

# %% [markdown]
# The one manifestation embodying the work's expression became a timeline.
# The work is the value of the bundle's `schema:about`, with the triples of
# its file. IRIs are compacted with the catalogue's prefixes, which read
# MeLODy's `https://schema.org/` as `schema:`.

# %%
show(suite, node=("schema:about", 0))

# %% [markdown]
# The manifestation's timeline states its own IRI as `owl:sameAs`, the
# expression it embodies as a value node and its item under
# `lrmoo:R7i_is_exemplified_by`; the item names its manifestation by the
# timeline's reference. A person who is no subject is a value, labelled
# from the triples of its file. Each entry's basis is the file the pair was
# read from. The bundle holds one graphical timeline, so a regular expression
# matching every `cgt_` ID finds it.

# %%
show(suite.get_timeline("cgt_.*"))

# %% [markdown]
# Both importers' bundles write manifests as any other, their `@context`
# holding the prefixes of LRMoo, MeLODy and the other vocabularies used.

# %%
sorted(key for key in suite.to_dict()["@context"] if key not in FORMAT_CONTEXT)

# %% [markdown]
# ## Next
#
# - Statements, provenance, the inline context and the catalogue as the
#   format specifies them: TimeToShare's
#   [description of a manifest](https://github.com/TimeToAlign/TimeToShare/blob/main/docs/description.md).
# - Saving manifests to files, selecting one timeline or group, and the
#   complete and minimal forms:
#   [How to Save and Load Timelines, Groups and Bundles](how02_group_and_bundle_files.ipynb).
# - Alignments between the timelines a bundle holds: the
#   [Alignment Bundles and MatchClaims](../tutorials/tut06_alignment_bundles.ipynb)
#   tutorial.
