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
# # How to Keep MatchClaims, Projections and Beliefs in a Package
#
# A TimeToShare manifest holds the structure of its timelines. The
# MatchClaims that relate them are data and live beside it, in the
# **MatchClaim table** (`match_claims.parquet`). The events that stand in
# for an event a timeline does not hold, Projections, are events like any
# other: each has a row in the **registry** (`registry.parquet`), which
# holds every event's ID, timeline and position, and its content, the
# event it stands for, in the pre-parsed **Projection table**
# (`projections.parquet`); the **Extraction table** (`extractions.parquet`)
# records what made them. The manifest names the registry and the
# MatchClaim table in its `resources`; the manifest and the files beside it
# make a **package**.
#
# This guide states MatchClaims between the events of a score and a
# recording, gives a side without an event a Projection, records who
# believes which claim and filters by those beliefs, writes the bundle as a
# package with `to_file()` and reads it back with `from_file()`, and shows
# why a package's claims can be added to but never removed.

# %% [markdown]
# ## Setup
#
# Install the library with `pip install timetoalign`. Two people take part:
# an editor, who states claims and believes what they state, and an
# automatic aligner, which states how it made its claims and believes
# nothing. Every date is a fixed string. The helper `claim_rows` shows
# claims one per row with the editor's belief, `files` lists a directory,
# and `show` prints part of a manifest as the file holds it.

# %%
import tempfile
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from timetoshare import dumps, loads

from timetoalign import (
    AlignmentBundle,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    Coordinate,
    EventReference,
    Projection,
    Provenance,
)

FILES = Path(tempfile.mkdtemp())
EDITOR = "https://example.org/people/editor"
ALIGNER = "https://example.org/tools/onset-aligner"
DAY = "2026-11-21"


def claim_rows(claims):
    """One row per claim: its id, its two events, its kind, the editor's belief."""

    def editor_belief(claim):
        beliefs = claim.provenance.beliefs if claim.provenance else ()
        return next((b.belief for b in beliefs if b.actor == EDITOR), None)

    return pd.DataFrame(
        [
            {
                "id": claim.id,
                "event_a_id": claim.event_a_id,
                "event_b_id": claim.event_b_id,
                "claim_type": claim.claim_type.value,
                "editor believes": editor_belief(claim),
            }
            for claim in claims
        ]
    )


def files(directory):
    """The names of the files in a directory, sorted."""
    return sorted(path.name for path in directory.iterdir())


def show(manifest_path, key):
    """Print one key of a manifest file as canonical YAML."""
    print(dumps({key: loads(manifest_path.read_text(encoding="utf-8"))[key]}), end="")


# %% [markdown]
# ## Two timelines with events
#
# A score of eight quarters with four notes, and a recording of six seconds
# in which an onset detector found three onsets. The fourth note, held at
# the end, was played too softly for the detector.

# %%
score = ContinuousLogicalTimeline(length=Fraction(8), uid="score", name="Score")
notes = {"n1": (0, 1), "n2": (1, 1), "n3": (2, 2), "n4": (4, 4)}
score.add_events(
    [
        {
            "id": note,
            "event_type": "Note",
            "start": Fraction(start),
            "duration": Fraction(length),
        }
        for note, (start, length) in notes.items()
    ]
)
recording = ContinuousPhysicalTimeline(length=6.0, uid="recording", name="Recording")
recording.add_events(
    [
        {"id": "o1", "event_type": "Onset", "instant": 0.25},
        {"id": "o2", "event_type": "Onset", "instant": 1.0},
        {"id": "o3", "event_type": "Onset", "instant": 1.75},
    ]
)

bundle = AlignmentBundle(name="Song")
bundle.add_timeline(score, as_group="score")
bundle.add_timeline(recording, as_group="recording")

# %% [markdown]
# ## MatchClaims between events
#
# A MatchClaim connects two events, one on each timeline. Give
# `create_match_claims()` one tuple per claim, `(event_a, timeline_a_id,
# event_b, timeline_b_id)`, each event by its id. Its `provenance` records
# how the claims were made, by whom and when. `actor=` names the person
# stating them: a person who states a claim believes it, so the editor's
# belief `"true"` is recorded in the provenance, dated `date=`. Every claim
# is minted its id when it is made, `claim_…`, which never changes.

# %%
stated = Provenance(stated_by=(EDITOR,), method="manual", date=DAY)
event_claims = bundle.create_match_claims(
    [
        ("n1", score.id, "o1", recording.id),
        ("n2", score.id, "o2", recording.id),
        ("n3", score.id, "o3", recording.id),
    ],
    provenance=stated,
    actor=EDITOR,
    date=DAY,
)
claim_rows(event_claims)

# %% [markdown]
# Each claim takes its coordinates from its two events, each on its own
# timeline's axis: quarters on the score, seconds on the recording.

# %%
event_claims[2]

# %% [markdown]
# ## A side without an event: a Projection
#
# The recording holds no event for `n4`, but the editor hears it begin at
# 3.5 seconds. Give that side as a position, a dict with `start` (and `end`
# for a span). A **Projection** is added to the recording there: an event
# that stands on the recording for `n4`, which its `projection_of` names.
# The claim then joins two events like any other.

# %%
projected = bundle.create_match_claims(
    [("n4", score.id, {"start": 3.5}, recording.id)],
    provenance=stated,
    actor=EDITOR,
    date=DAY,
)[0]
recording.get_event(projected.event_b_id)

# %% [markdown]
# The timeline keeps one Projection per event it stands for and position.
# Stating the same pair again makes a second claim with its own id, and it
# reuses the Projection rather than adding another.

# %%
again = bundle.create_match_claims(
    [("n4", score.id, {"start": 3.5}, recording.id)],
    provenance=stated,
    actor=EDITOR,
    date=DAY,
)[0]
{"claim": again.id, "its recording event": again.event_b_id}

# %% [markdown]
# `Timeline.add_projections()` is the same operation, for adding
# Projections yourself: a `Projection` holds its position (a `Coordinate`,
# converted onto the timeline's axis like any event's) and, in
# `projection_of`, an `EventReference` to the event it stands for. It
# returns the id of the Projection standing for each one, here the one
# already held.

# %%
recording.add_projections(
    Projection(
        start=Coordinate(3.5, "seconds"),
        projection_of=EventReference(timeline_id=score.id, event_id="n4"),
    )
)

# %% [markdown]
# ## Who believes a claim
#
# The aligner proposes two claims. Software records how it made a claim and
# believes nothing, so the aligner gives a `provenance` and no `actor`. One
# claim matches the third note to the second onset. The other places
# quarter 6, where the score has no event, at 5 seconds, where the recording
# has none: both sides are positions, so neither gets a Projection. Each side
# is the sentinel CHUNK, whose coordinate the claim's anchor states.

# %%
proposed = Provenance(service=ALIGNER, method="inferred", date=DAY)
rejected, doubted = bundle.create_match_claims(
    [
        ("n3", score.id, "o2", recording.id),
        ({"start": Fraction(6)}, score.id, {"start": 5.0}, recording.id),
    ],
    provenance=proposed,
)
claim_rows([rejected, doubted])

# %% [markdown]
# The editor reviews the proposals later. `add_match_claim_belief()`
# records a belief in a claim the bundle holds, named by its id: the editor
# believes the first `"false"` and finds the second `"unlikely"`. A belief
# replaces the actor's earlier one in that claim and leaves every other
# actor's belief as it is. It changes only the claim's provenance and never
# removes the claim.

# %%
bundle.add_match_claim_belief(rejected.id, actor=EDITOR, belief="false", date=DAY)
bundle.add_match_claim_belief(doubted.id, actor=EDITOR, belief="unlikely", date=DAY)
claim_rows(bundle.get_match_claims())

# %% [markdown]
# Every reader of claims takes the belief filter: `believer=` names an
# actor, and `beliefs=` lists the states kept for that actor: `"true"`,
# `"undecided"` (the actor holds no belief on the claim), `"unlikely"` and
# `"false"`. The default keeps every state but `"false"`, so `"unlikely"`
# passes. Without a `believer=` nothing is filtered.

# %%
{
    "every claim": len(bundle.get_match_claims()),
    "not believed false by the editor": len(bundle.get_match_claims(believer=EDITOR)),
    "believed true by the editor": len(
        bundle.get_match_claims(believer=EDITOR, beliefs=["true"])
    ),
    "undecided for the editor": len(
        bundle.get_match_claims(believer=EDITOR, beliefs=["undecided"])
    ),
}

# %% [markdown]
# MatchStamps read the same filter, and so do the bundle's coordinate
# getters (`get_coordinate_at` and the rest of its row). At quarter 6, the
# doubted claim reaches the recording under the default filter; with
# `beliefs=["true"]` the stamp holds the score alone.

# %%
[
    bundle.get_matchstamp_at(Fraction(6), score.id, believer=EDITOR),
    bundle.get_matchstamp_at(Fraction(6), score.id, believer=EDITOR, beliefs=["true"]),
]

# %% [markdown]
# ## Write the package
#
# `to_file()` writes the MatchClaim table, the registry, the Extraction
# table and the Projection table beside the manifest, then the manifest
# naming the registry and the MatchClaim table. A manifest names no event:
# the claims name the score's notes and the recording's onsets by their
# ids, and by default those events stay with the sources they came from,
# here the score and the onset list. The package holds the events it made
# itself, the Projections. To keep the notes and onsets in the package as
# well, `to_file(events=True)` writes them into the registry and a
# pre-parsed table per event type, never into the manifest itself:
# see [How to Write the Registry and Share Tables in a Package](how03_registry_and_packages.ipynb).

# %%
package = FILES / "song"
package.mkdir()
manifest_path = bundle.to_file(package / "song.yaml")
files(package)

# %% [markdown]
# The manifest's `resources` names the registry and the MatchClaim table:
# each its `id`, its `type` (what its rows are: `Registry` or
# `MatchClaim`), the `timelines` it holds rows for, its `url` relative to
# the manifest, and its media type, size in bytes and SHA-256 digest.

# %%
show(manifest_path, "resources")

# %% [markdown]
# ## What the tables hold
#
# The MatchClaim table has one row per claim and one column per property of
# a MatchClaim: its two timelines and two events, the anchors' coordinates,
# its provenance with the beliefs, and more. An event ID cell is never empty:
# a side without an event holds a sentinel, `NOMATCH_SENTINEL` for the absent
# side of a NOMATCH and `CHUNK_SENTINEL` for a side whose coordinate the
# anchor states, as on both sides of the doubted claim. Its identifying
# columns:

# %%
pq.read_table(package / "match_claims.parquet").select(
    ["id", "timeline_a_id", "event_a_id", "timeline_b_id", "event_b_id"]
).to_pandas()

# %% [markdown]
# A Projection's registry row holds its id, its timeline and its position
# on the axis of the timeline's root (`start`, and `duration` for a span),
# with that axis's unit and number type once, each number written once, as
# that number type says: an int as its numerator alone, a fraction as
# numerator and denominator, a float as its value alone. The registry's
# other rows, its non-event rows, name the manifest's elements, the
# MatchClaims and the extraction.

# %%
[
    {
        name: row[name]
        for name in ("id", "kind", "timeline_id", "unit", "number_type", "start")
    }
    for row in pq.read_table(package / "registry.parquet").to_pylist()
    if row["kind"] == "Projection"
]

# %% [markdown]
# The Projection table holds the content of each Projection, keyed by its
# id: the event it stands for, if any.

# %%
pq.read_table(package / "projections.parquet").to_pylist()

# %% [markdown]
# An event's id is unique among all the events of the package, whichever
# timeline holds it, so the registry and the table name each id once. Here
# they hold one Projection, the recording's for the fourth note (its
# `projection_of` names the note); the claim between two positions added
# none. Each id is minted (`projection_…`) and never re-minted, and a
# Projection given an id another timeline's Projection holds is refused.

# %% [markdown]
# ## Read the package
#
# `from_file()` reads the manifest, then the tables it names: each
# Projection's registry row onto its timeline (its content read from the
# Projection table when the timeline's events are read), then the
# MatchClaim table. The bundle read
# holds the same claims and the same manifest, and its belief filter gives
# the same answers. Its timelines hold their Projections but not the notes
# and onsets, which the manifest does not carry (the score read holds no
# event at all). Each claim keeps the coordinates of its two
# sides, so MatchStamps need no event: under the editor's beliefs the
# third note, at quarter 2, still lies at 1.75 seconds.

# %%
read = AlignmentBundle.from_file(manifest_path)
{
    "same MatchClaim table": read.match_claim_table == bundle.match_claim_table,
    "same manifest": read.to_dict() == bundle.to_dict(),
    "believed true by the editor": [
        claim.id for claim in read.get_match_claims(believer=EDITOR, beliefs=["true"])
    ],
    "events of the score read": len(read.get_timeline(score.id).get_events()),
}

# %%
read.get_matchstamp_at(Fraction(2), score.id, believer=EDITOR)

# %% [markdown]
# Writing the bundle read gives the same bytes, file by file.

# %%
copy = FILES / "copy"
copy.mkdir()
read.to_file(copy / "song.yaml")
{
    name: (package / name).read_bytes() == (copy / name).read_bytes()
    for name in files(package)
}

# %% [markdown]
# ## Add to a package, never remove from it
#
# A MatchClaim is never deleted and its id never changes; whoever disagrees
# states a belief instead. So writing over a package's MatchClaim table
# obeys the succession rule: every claim of the old table is in the new
# one, with the same timelines, events and anchors; only its provenance
# may differ. Adding a claim, stating a belief and writing the bundle back
# is fine. The editor goes on in the bundle that holds the events: a
# conceptual claim between the second note and the second onset, and a
# change of mind about the doubted claim.

# %%
bundle.create_match_claims(
    [("n2", score.id, "o2", recording.id)], synchronous=False, actor=EDITOR, date=DAY
)
bundle.add_match_claim_belief(doubted.id, actor=EDITOR, belief="false", date=DAY)
bundle.to_file(manifest_path)
claim_rows(AlignmentBundle.from_file(manifest_path).get_match_claims())

# %% [markdown]
# A bundle that lacks claims the table holds cannot be written over it.
# Here a draft is built from the manifest alone (`from_dict()` reads no
# tables, and the manifest holds no event) and states one claim between two
# positions: the write is refused before any file is touched, and the
# package keeps its claims.

# %%
draft = AlignmentBundle.from_dict(read.to_dict())
draft.create_match_claims([({"start": 0}, score.id, {"start": 0.25}, recording.id)])
try:
    draft.to_file(manifest_path)
except ValueError as refusal:
    print(refusal)
len(AlignmentBundle.from_file(manifest_path).get_match_claims())

# %% [markdown]
# ## Next
#
# - The kinds of claim (event match, NOMATCH, conceptual) and how
#   claims become MatchStamps: the
#   [Alignment Bundles and MatchClaims](../tutorials/tut06_alignment_bundles.ipynb)
#   tutorial.
# - Events in the registry and pre-parsed tables, and several manifests
#   sharing their tables in one package:
#   [How to Write the Registry and Share Tables in a Package](how03_registry_and_packages.ipynb).
# - Manifests in depth (minimal and complete form, serializations, editing
#   a file): [How to Save and Load Timelines, Groups and Bundles](how02_group_and_bundle_files.ipynb).
# - Beliefs on statements as well as claims:
#   [How to Describe What a Bundle and Its Timelines Stand For](how03_describe_the_work.ipynb).
# - The format of the package, its resources and the two tables:
#   [TimeToShare's README, "Resources and tables"](https://github.com/TimeToAlign/TimeToShare#resources-and-tables).
