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
# # How to Write the Registry and Share Tables in a Package
#
# A TimeToShare manifest holds the structure of its timelines and names no
# event. What lies on the timelines lies in Parquet tables beside it:
#
# - the **registry**, one row per ID the package can resolve: for every
#   event its ID, its type, its timeline, its position on the axis of its
#   timeline's root and the extraction that made it; and a non-event row
#   for every element of the manifest, every MatchClaim and every
#   extraction;
# - the **Extraction table**, one row per run of a parser over a source
#   file, or of a tool that made events without one (here the library,
#   since the events are made in code);
# - one **pre-parsed table** per event type, holding what is particular to
#   events of that type, keyed by their IDs and without positions (the
#   Projection table, a Note table); an event whose content can be read
#   again from its source file is **re-parsed** instead and has no row
#   there;
# - the **MatchClaim table**.
#
# The manifest names the registry and the MatchClaim table in its
# `resources`. Several manifests can share their tables in a **package**: a
# folder whose `package.yaml` lists the manifests and the tables: the
# registry one Parquet file holding every row of every manifest, the
# Extraction table one file, and the MatchClaim table and every pre-parsed
# table a folder `tables/<Type>/` of the rows of every manifest. An ID
# combined with the location of a manifest, or of the package, resolves to
# what it names.
#
# This guide writes a bundle with its tables, resolves IDs through its
# manifest, builds a package of two bundles with `Package`, opens one of its
# manifests, checks the package with TimeToShare's package check, and
# writes one bundle into the package again.

# %% [markdown]
# ## Setup
#
# Install the library with `pip install timetoalign`; the TimeToShare
# package, which defines the format and checks it, comes with it. Every file
# is written to a temporary directory.
#
# The helper `piece` builds the bundle of one piece: a score of four notes,
# a recording with an onset for each note but the last, a MatchClaim for each
# note and onset, and one claim giving the last note a position on the
# recording, where it gets a Projection. Every ID starts with the piece's
# key, so that the IDs of two pieces differ. `files` lists a directory's files
# and folders, `show` prints one key of a YAML file as the file holds it,
# `values_per_file` names the values of one column in each file of a table
# folder, and `event_rows` reads the event rows of a registry file.

# %%
import tempfile
from fractions import Fraction
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from timetoshare import dumps, loads
from timetoshare.tables import package_problems

from timetoalign import (
    AlignmentBundle,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    Package,
    Provenance,
    resolve,
)

FILES = Path(tempfile.mkdtemp())
EDITOR = "https://example.org/people/editor"
STATED = Provenance(stated_by=(EDITOR,), method="manual", date="2026-11-27")


def piece(key, onsets):
    """The bundle of one piece: a score, a recording, and their MatchClaims."""
    score = ContinuousLogicalTimeline(
        length=Fraction(4), uid=f"{key}_score", name="Score"
    )
    score.add_events(
        [
            {
                "id": f"{key}_n{number}",
                "event_type": "Note",
                "start": Fraction(number - 1),
                "duration": Fraction(1),
            }
            for number in range(1, 5)
        ]
    )
    recording = ContinuousPhysicalTimeline(
        length=4.0, uid=f"{key}_recording", name="Recording"
    )
    recording.add_events(
        [
            {"id": f"{key}_o{number}", "event_type": "Onset", "instant": seconds}
            for number, seconds in enumerate(onsets, start=1)
        ]
    )
    bundle = AlignmentBundle(id=key, name=key.title())
    bundle.add_timeline(score, as_group=f"{key}_score")
    bundle.add_timeline(recording, as_group=f"{key}_recording")
    pairs = [
        (f"{key}_n{number}", score.id, f"{key}_o{number}", recording.id)
        for number in range(1, len(onsets) + 1)
    ]
    pairs.append((f"{key}_n4", score.id, {"start": 3.5}, recording.id))
    bundle.create_match_claims(pairs, provenance=STATED)
    return bundle


def files(directory):
    """Every file and folder under a directory, folders ending in '/', sorted."""
    return sorted(
        str(path.relative_to(directory)) + ("/" if path.is_dir() else "")
        for path in directory.rglob("*")
    )


def show(path, key):
    """Print one key of a YAML file as canonical YAML."""
    print(dumps({key: loads(path.read_text(encoding="utf-8"))[key]}), end="")


def values_per_file(folder, column):
    """For each Parquet file of a table folder, the values of one column."""
    return {
        file.name: sorted(
            value
            for value in pc.unique(
                pq.read_table(file, columns=[column])[column]
            ).to_pylist()
            if value is not None
        )
        for file in sorted(folder.glob("*.parquet"))
    }


def event_rows(path, columns=("id", "kind", "timeline_id", "start", "location")):
    """The event rows of a registry file: the rows with a position."""
    table = pq.read_table(path)
    return [
        {name: row[name] for name in columns}
        for row in table.to_pylist()
        if row["start"] is not None
    ]


# %% [markdown]
# ## Two bundles
#
# Two pieces, each with its score and recording. Their onsets were detected
# at slightly different times.

# %%
bach = piece("bach", [0.0, 0.9, 1.8])
handel = piece("handel", [0.1, 1.1, 2.0])
bach

# %% [markdown]
# ## Write a bundle with its tables
#
# `to_file()` always writes the Projections and Boxes the bundle holds,
# with the MatchClaim table. With `events=True` it also writes every other
# event type its timelines hold: the registry gets a row for each note and
# onset, and each type its pre-parsed table, `Note.parquet` and
# `Onset.parquet`. The manifest itself still names no event.

# %%
single = FILES / "bach"
single.mkdir()
manifest_path = bach.to_file(single / "bach.yaml", events=True)
files(single)

# %% [markdown]
# The manifest names the registry and the MatchClaim table in its
# `resources`: their `type`, the timelines whose rows they hold for it, and
# their location relative to the manifest, here files beside it, with their
# size and SHA-256 digest. The pre-parsed tables and the Extraction table
# are not named: the registry's rows point to them.

# %%
show(manifest_path, "resources")

# %% [markdown]
# Each event row states the event's ID, its type (`kind`), its timeline and
# its position, each number written once in the member its number type
# names (the score's quarters as a fraction's numerator and denominator,
# the recording's seconds as a float's `value`), and the `location` of the
# pre-parsed table holding its content. The rows are sorted by extraction,
# timeline, exact start and ID.

# %%
event_rows(single / "registry.parquet")[:6]

# %% [markdown]
# The other rows of the registry are its non-event rows: one per element
# of the manifest (its timelines, groups and resources, the manifest
# itself), per MatchClaim and per extraction, each with its kind and its
# location, so that an ID found in the registry leads to the file holding
# it.

# %%
sorted(
    (row["kind"], row["location"])
    for row in pq.read_table(single / "registry.parquet").to_pylist()
    if row["start"] is None
)[:8]

# %% [markdown]
# Every event names the extraction that made it. These events were made in
# code, so each timeline family (a root and its children) names one
# extraction of the library itself: no source file, the parser
# `timetoalign` and its version, and the ID minter. A loader records the
# source file it read and its digest instead.

# %%
pq.read_table(single / "extractions.parquet").select(
    ["id", "source", "parser", "minter", "id_length"]
).to_pylist()

# %% [markdown]
# A pre-parsed table holds the content of each event, keyed by ID, without
# its position: the Projection table states which event each Projection
# stands for, here the last note, whose position on the recording the claim
# states.

# %%
pq.read_table(single / "projections.parquet").to_pylist()

# %% [markdown]
# `from_file()` reads the manifest and its tables, so the bundle read holds
# the notes and onsets again, beside the recording's Projection and the
# MatchClaims; each event's content is read from its pre-parsed table when
# its timeline's events are first read. Written again, without `events=` (a
# manifest read with notes and onsets is written with them), it gives the
# same bytes, file by file.

# %%
read = AlignmentBundle.from_file(manifest_path)
copy = FILES / "copy"
copy.mkdir()
read.to_file(copy / "bach.yaml")
{
    "events per timeline": {
        timeline_id: len(read.get_timeline(timeline_id).get_events())
        for timeline_id in read.timeline_ids
    },
    "same MatchClaim table": read.match_claim_table == bach.match_claim_table,
    "same manifest": read.to_dict() == bach.to_dict(),
    "same bytes": all(
        (single / name).read_bytes() == (copy / name).read_bytes()
        for name in files(single)
    ),
}

# %% [markdown]
# ## Resolve an ID
#
# `resolve(target, id)` answers what an ID names, through a manifest (a
# path or a `file:` URL) or a package root, always as a list: a timeline,
# an event as its scalar, a MatchClaim, an extraction. An event's position
# is the registry's, on the axis of its timeline's root. An ID nothing
# holds gives the empty list.

# %%
{
    "bach_n3": resolve(manifest_path, "bach_n3"),
    "bach_recording": resolve(manifest_path.as_uri(), "bach_recording"),
    "nothing": resolve(manifest_path, "bach_n9"),
}

# %% [markdown]
# ## Build a package of two bundles
#
# A `Package` takes a `name` (and an `id`, its IRI where it is published).
# `write_bundle(bundle, url)` adds a bundle as the manifest at `url`,
# relative to the package's root. A new package only stages the bundles;
# `to_file()` writes them all at once, then `package.yaml`, then the
# registry's non-event rows, into a folder that holds no package yet.

# %%
package = Package(name="Two pieces")
package.write_bundle(bach, "bach/manifest.yaml")
package.write_bundle(handel, "handel/manifest.yaml")
root = FILES / "pieces"
root.mkdir()
package_path = package.to_file(root / "package.yaml")
files(root)

# %% [markdown]
# A package writes every event type of its bundles, with no `events=`
# needed. The registry is one file at the root, `registry.parquet`, holding
# every row: the non-event rows first, then the event rows of both pieces.
# Every other table has one folder under `tables/` but the Extraction table,
# one file; each write adds one
# file per pre-parsed type and one of MatchClaims, named from its content,
# holding the rows of every bundle it writes. `package.yaml` lists the
# manifests and the tables.

# %%
print(package_path.read_text(encoding="utf-8"), end="")

# %% [markdown]
# Each manifest names the registry and the MatchClaim folder, relative to
# itself, with the timelines whose rows it keeps. Neither has a size or a
# digest: the rows of other manifests share them.

# %%
show(root / "bach" / "manifest.yaml", "resources")

# %% [markdown]
# The registry holds the event rows of the four timelines of both pieces;
# the one file of notes holds the notes of both scores, and the one file of
# MatchClaims the claims of both pieces.

# %%
{
    "registry": sorted(
        {row["timeline_id"] for row in event_rows(root / "registry.parquet")}
    ),
    "Note": values_per_file(root / "tables" / "Note", "id"),
    "MatchClaim": values_per_file(root / "tables" / "MatchClaim", "timeline_a_id"),
}

# %% [markdown]
# ## Open one manifest from the package
#
# `Package.from_file()` opens the package at its `package.yaml`;
# `manifests` lists the manifests' URLs and `tables` the table entries.
# `bundle(url)` reads one manifest as an `AlignmentBundle`, every table
# filtered to the timelines its resources list, so the Bach bundle holds
# its own events and claims and none of Handel's. Its MatchClaims are the
# ones written; the package's file orders them by timeline and ID and states
# each claim's provenance in its own row, so compare the claims, not the
# tables.

# %%
opened = Package.from_file(package_path)
from_package = opened.bundle("bach/manifest.yaml")
{
    "manifests": opened.manifests,
    "table types": [table["type"] for table in opened.tables],
    "timelines read": from_package.timeline_ids,
    "events per timeline": {
        timeline_id: len(from_package.get_timeline(timeline_id).get_events())
        for timeline_id in from_package.timeline_ids
    },
    "same MatchClaims": set(from_package.get_match_claims())
    == set(bach.get_match_claims()),
}

# %% [markdown]
# The claims read answer as before: the third note, at quarter 2, lies at
# 1.8 seconds on the Bach recording.

# %%
from_package.get_matchstamp_at(Fraction(2), "bach_score")

# %% [markdown]
# Through the package root, `resolve` finds every row of an ID in the
# registry: Handel's first onset, and the manifest itself, with no manifest
# named.

# %%
{
    "handel_o1": resolve(root, "handel_o1"),
    "bach": resolve(root, "bach"),
}

# %% [markdown]
# ## Check the package
#
# `timetoshare.tables.package_problems()` checks a package whole from its
# `package.yaml`: every manifest valid, every table file of its type and
# version, every resource naming a table of the package, every ID held once,
# every event row naming an extraction and a table holding its content, the
# non-event rows agreeing with the manifests and tables, and the rows of
# each manifest checked against it. It returns the list of problems, here
# none. `Package` runs the same check after every write and raises its
# first problem, with every file restored.

# %%
package_problems(package_path)

# %% [markdown]
# ## IDs are the package's
#
# Within a package, an event ID is held once among all events of every
# type, and a MatchClaim ID once among its MatchClaims; as `Package` writes
# a package, a timeline ID is held by one manifest. A bundle colliding with
# what the package holds is refused before anything is written, naming the
# ID and both holders. Writing the Bach piece a second time, under another
# URL, collides at its timelines.

# %%
try:
    opened.write_bundle(piece("bach", [0.0, 0.9, 1.8]), "bach_again/manifest.yaml")
except ValueError as refusal:
    print(refusal)

# %% [markdown]
# ## Write one bundle again
#
# The editor adds a conceptual claim to the Bach bundle read from the
# package: the last note, which the detector missed, relates to the third
# onset. `write_bundle()` on a package lying on disk writes at once and
# replaces the manifest at that URL. The rows of its timelines leave the
# files holding them: the registry is rewritten whole with its rows, the
# pre-parsed and MatchClaim files keep the other manifests' rows alone, and
# one new file per type holds the bundle's rows. As with a single
# MatchClaim table, every claim the package held for the manifest must
# still be in the bundle, and the bundle may add claims.

# %%
from_package.create_match_claims(
    [("bach_n4", "bach_score", "bach_o3", "bach_recording")],
    synchronous=False,
    provenance=STATED,
)
opened.write_bundle(from_package, "bach/manifest.yaml")
{
    "Note": values_per_file(root / "tables" / "Note", "id"),
    "MatchClaim": values_per_file(root / "tables" / "MatchClaim", "timeline_a_id"),
}

# %% [markdown]
# The package still checks clean, and each manifest reads its own claims:
# five for Bach now, four for Handel as before.

# %%
{
    "problems": package_problems(package_path),
    "MatchClaims": {
        url: len(opened.bundle(url).get_match_claims()) for url in opened.manifests
    },
}

# %% [markdown]
# A bundle read from a package names the package's tables, which other
# manifests share. `to_file()` writes a manifest's tables as single files
# beside it and refuses to write into a package's tables; it names
# `Package.write_bundle` instead.

# %%
elsewhere = FILES / "elsewhere"
elsewhere.mkdir()
try:
    from_package.to_file(elsewhere / "bach.yaml")
except NotImplementedError as refusal:
    print(refusal)

# %% [markdown]
# ## Next
#
# - MatchClaims, Projections and beliefs, written as a single manifest
#   with its tables:
#   [How to Keep MatchClaims, Projections and Beliefs in a Package](how03_match_claims_package.ipynb).
# - Manifests in depth (minimal and complete form, serializations, editing
#   a file): [How to Save and Load Timelines, Groups and Bundles](how02_group_and_bundle_files.ipynb).
# - The format of the registry, the Extraction table, pre-parsed tables and
#   packages, and every rule the package check applies:
#   [TimeToShare's README, "Resources and tables"](https://github.com/TimeToAlign/TimeToShare#resources-and-tables).
