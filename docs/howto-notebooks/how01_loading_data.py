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
# # How to Load Data
#
# This tutorial introduces the Loader pattern and EventStores - the foundation
# for bringing music data into TimeToAlign!
#
# **Learning Objectives:**
# - Use Loaders to ingest music data from various formats
# - Navigate EventStores and access event data
# - Understand the harmonized schema that unifies different data sources
#
# **Prerequisites:**
# - Basic Python and pandas knowledge
# - TimeToAlign! installed (`pip install timetoalign`)

# %% [markdown]
# ## Why Loaders Matter
#
# Music data comes in many formats: MusicXML, MIDI, MEI, Humdrum, proprietary
# TSV exports, and more. Each format has its own structure, terminology,
# and quirks.
#
# **The problem:** Without a unified approach, you'd need format-specific code
# for every data source, making cross-format analysis difficult and error-prone.
#
# **The TimeToAlign! solution:** Loaders normalize heterogeneous formats into
# a consistent `EventStore`, enabling downstream processing without
# format-specific code.
#
# ```
# MusicXML ─┐
# MIDI ─────┼──> Loader ──> EventStore ──> DataFrame
# TSV ──────┘
# ```

# %% [markdown]
# ## Setup

# %%
import json
import os
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from timetoalign import Ms3Loader
from timetoalign.core.events import EnharmonicPitch
from timetoalign.loader import materialize
from timetoalign.loader.score.music21 import Music21Loader
from timetoalign.loader.score.partitura import PartituraLoader
from timetoalign.testdata import ensure_data

DATA_DIR = ensure_data("vienna_1x22")

# Our test piece: Chopin Etude Op.10 No.3
CHOPIN_XML = DATA_DIR / "Chopin_op10_no3.musicxml"
CHOPIN_TSV = DATA_DIR / "ms3" / "chopin_op10_no3.notes.tsv"


def source_entry(loader):
    """The `path` and `member` a loader records for its one source, the path by name."""
    entry = loader.metadata["sources"][0]
    return {"path": Path(entry["path"]).name, "member": entry.get("member", "—")}


CHOPIN_XML.name, CHOPIN_TSV.name

# %% [markdown]
# ## The Loader Pattern
#
# All TimeToAlign! loaders follow the same three-step pattern:
#
# 1. **Create** a loader instance
# 2. **Load** a file using `.load(path)`
# 3. **Access** the store containing EventStores
#
# Let's see this in action with three different loaders, all loading the same Chopin piece:

# %%
# Load from three different sources
tsv_loader = Ms3Loader()
tsv_loader.load(CHOPIN_TSV)

partitura_loader = PartituraLoader()
partitura_loader.load(CHOPIN_XML)

music21_loader = Music21Loader()
music21_loader.load(CHOPIN_XML)

# All produce ScoreStores
{
    "TSV": type(tsv_loader.store).__name__,
    "Partitura": type(partitura_loader.store).__name__,
    "Music21": type(music21_loader.store).__name__,
}

# %% [markdown]
# ## Cross-Loader Validation
#
# One of the key benefits of TimeToAlign! is that different loaders produce
# comparable output. Let's verify that all three loaders found the same
# number of notes:

# %%
# Convert to DataFrames
tsv_df = tsv_loader.store.notes.to_dataframe()
partitura_df = partitura_loader.store.notes.to_dataframe()
music21_df = music21_loader.store.notes.to_dataframe()

# Count only Note events (not rests or other event types)
counts = {
    "TSV": len(tsv_df[tsv_df["event_type"] == "Note"]),
    "Partitura": len(partitura_df[partitura_df["event_type"] == "Note"]),
    "Music21": len(music21_df[music21_df["event_type"] == "Note"]),
}

# Validate against gold standard
assert all(c == 498 for c in counts.values()), f"Note count mismatch: {counts}"

pd.Series(counts, name="note_count")

# %% [markdown]
# ## The EventStore
#
# Each ScoreStore contains **EventStores** - efficient, PyArrow-backed tables
# that hold musical events.
#
# Key characteristics:
# - **High Performance**: Built on Apache Arrow for fast columnar operations
# - **Type Safety**: Schema metadata preserves units and types
# - **Pandas Interop**: Easy conversion with `.to_dataframe()`

# %%
notes_store = tsv_loader.store.notes

{
    "type": type(notes_store).__name__,
    "n_events": len(notes_store),
    "storage": type(notes_store.table).__name__,
}

# %%
# Examine the schema with metadata
schema_info = []
for field in notes_store.table.schema:
    meta = field.metadata or {}
    # Column metadata is one `timetoalign` JSON blob; unpack it so the table
    # shows the declared unit and number type rather than a serialized string.
    blob = json.loads(meta[b"timetoalign"]) if b"timetoalign" in meta else {}
    extra = {k.decode(): v.decode() for k, v in meta.items() if k != b"timetoalign"}
    meta_str = ", ".join(
        f"{k}={v}" for k, v in {**blob, **extra}.items() if k != "version"
    )
    schema_info.append(
        {"name": field.name, "type": str(field.type)[:30], "metadata": meta_str}
    )

pd.DataFrame(schema_info)

# %% [markdown]
# ## The Harmonized Schema
#
# TimeToAlign! uses a harmonized schema to represent events consistently across formats:
#
# | Column | Description |
# |--------|-------------|
# | `id` | Unique identifier for the event |
# | `temporal_type` | "instant" or "interval" |
# | `event_type` | Type of event (Note, Rest, etc.) |
# | `start`, `end`, `duration` | Temporal coordinates in quarter notes |
# | `mc`, `mn` | Measure count and measure number |
# | `specific_pitch` | Fully spelled pitch (step + alter + octave) — the default pitch field |
# | `midi` | The raw source MIDI pitch as an integer (affords an `EnharmonicPitch` view on request) |

# %%
# Show selected fields for the first few notes
display_cols = [
    "id",
    "name",
    "temporal_type",
    "event_type",
    "start",
    "duration",
    "mc",
    "mn",
    "octave",
]
tsv_df[display_cols].head(10)

# %% [markdown]
# ## Pitch Information
#
# A spelled score faithfully supports a fully specific pitch, so pitch is
# **represented exactly once**: `specific_pitch` (step + alter + octave) is
# the single default pitch field, and it preserves the enharmonic spelling
# (e.g. G♯ vs A♭) that a bare MIDI pitch cannot.
#
# The source MIDI pitch survives as a plain `midi` integer column. It is
# redundant with the spelling, so it is not stored as a second pitch field;
# instead the EventStore *affords* an `EnharmonicPitch` view over it on
# request via `get_field(EnharmonicPitch)` (or the
# `enharmonic_pitch_field` accessor). The spelled and enharmonic views
# diverge exactly where it matters — an accidental-bearing note.

# %%
# The default pitch field: fully spelled SpecificPitch scalars.
specific_pitch = notes_store.specific_pitch_field

# The afforded EnharmonicPitch view over the raw `midi` integer column.
enharmonic_pitch = notes_store.get_field(EnharmonicPitch)

# Index 3 is an accidental-bearing note, where spelling (SP) and the
# MIDI pitch (EP) part ways.
{
    "specific_pitch[3] (default)": repr(specific_pitch[3]),
    "enharmonic_pitch[3] (afforded view)": repr(enharmonic_pitch[3]),
    "raw midi int[3]": tsv_df.iloc[3]["midi"],
}

# %% [markdown]
# ## Duration Analysis
#
# TimeToAlign! stores durations in quarter notes. Let's analyze the rhythmic content of our piece:

# %%
# Duration distribution
tsv_df["duration"].value_counts().sort_index().to_frame("count")

# %%
# Summary statistics
tsv_df["duration"].describe()

# %% [markdown]
# ## Navigating by Measure
#
# The `mc` (measure count) and `mn` (measure number) fields allow easy
# navigation through the score. Note that `mn` is stored as a string
# (to support labels like "1a", "1b"), so we convert to int for proper sorting:

# %%
# Notes per measure, sorted numerically
notes_per_measure = tsv_df.groupby("mn").size()

# Convert index to int for proper sorting (works for simple numeric measure numbers)
notes_per_measure.index = notes_per_measure.index.astype(int)
notes_per_measure = notes_per_measure.sort_index()

notes_per_measure.to_frame("notes")

# %%
# Get all notes in a specific measure
measure_5 = tsv_df[tsv_df["mn"] == "5"]
measure_5[["name", "duration", "voice", "staff"]]

# %% [markdown]
# ## Comparing Loader Outputs
#
# While all loaders produce the same number of notes, there can be subtle
# differences in how they interpret the score. Let's compare the first
# few notes:

# %%
# Compare ID schemes across loaders
pd.DataFrame(
    {
        "TSV_id": tsv_df["id"].head(5).values,
        "TSV_name": tsv_df["name"].head(5).values,
        "Partitura_id": partitura_df["id"].head(5).values,
        "Music21_id": music21_df["id"].head(5).values,
    }
)

# %% [markdown]
# ## Unit Metadata
#
# TimeToAlign! stores unit information in the PyArrow schema metadata, under a
# single `timetoalign` JSON blob per column. Reading it back through the
# semantic field is the ergonomic route — the field exposes the unit, the
# declared number type, and the domain as typed attributes:

# %%
# Read unit and number type back off the semantic fields
temporal_cols = ["start", "end", "duration"]
note_events = tsv_loader.get_events()
{
    name: {
        "unit": note_events.get_field(name).unit.value,
        "number_type": note_events.get_field(name).number_type.name,
    }
    for name in temporal_cols
}

# %% [markdown]
# ## Voice and Staff Information
#
# In scores with multiple staves or multiple voices per staff, notes are
# distributed accordingly:

# %%
# Notes by staff and voice
tsv_df.groupby(["staff", "voice"]).size().unstack(fill_value=0)

# %% [markdown]
# ## Loading from an Archive
#
# Corpora often travel as ZIP archives. Every loader reads a file inside a ZIP
# archive wherever it reads a path, in `load()` and `from_file()` alike,
# without extracting the archive. Name the file with the standard library's
# `zipfile.Path`: the archive, opened from its path, and the member's path
# inside it. Here we pack the two Chopin files into an archive in a temporary
# folder:

# %%
ARCHIVE_DIR = Path(tempfile.mkdtemp())
ARCHIVE = ARCHIVE_DIR / "chopin.zip"
with zipfile.ZipFile(ARCHIVE, "w") as archive:
    archive.write(CHOPIN_TSV, "ms3/chopin_op10_no3.notes.tsv")
    archive.write(CHOPIN_XML, "score/Chopin_op10_no3.musicxml")

TSV_MEMBER = zipfile.Path(ARCHIVE, at="ms3/chopin_op10_no3.notes.tsv")
XML_MEMBER = zipfile.Path(ARCHIVE, at="score/Chopin_op10_no3.musicxml")

# Copies of members (see `materialize` below) go to a folder of our choice.
os.environ["TTA_CACHE_DIR"] = str(ARCHIVE_DIR / "cache")

ARCHIVE.name, TSV_MEMBER.at, XML_MEMBER.at

# %% [markdown]
# `Ms3Loader` reads the member in place. Partitura needs a file on disk, so
# `PartituraLoader` reads a copy of the member; so do the other loaders built
# on a parser of that kind (music21, Verovio, the match-file reader, Verovio
# timemaps). Either way, the events equal those loaded from disk above:

# %%
archived_tsv = Ms3Loader.from_file(TSV_MEMBER)
archived_xml = PartituraLoader.from_file(XML_MEMBER)

{
    "Ms3Loader": archived_tsv.store.notes.table.equals(tsv_loader.store.notes.table),
    "PartituraLoader": archived_xml.store.notes.table.equals(
        partitura_loader.store.notes.table
    ),
}

# %% [markdown]
# Each loader records where its sources came from in `metadata["sources"]`.
# For a member, `path` is the archive's absolute path (shown here by name) and
# `member` the member's path inside it; a filesystem source records its path
# and no `member` (— below). Messages and warnings name a member the same way,
# `<archive> member <name>`, and `loader.sources` holds the `zipfile.Path`
# itself.

# %%
pd.DataFrame(
    {
        "Ms3Loader, disk": source_entry(tsv_loader),
        "Ms3Loader, archive": source_entry(archived_tsv),
        "PartituraLoader, disk": source_entry(partitura_loader),
        "PartituraLoader, archive": source_entry(archived_xml),
    }
).T

# %% [markdown]
# A few things to keep in mind:
#
# - Only a `zipfile.Path` names a member. A string such as
#   `"chopin.zip/ms3/chopin_op10_no3.notes.tsv"` is a filesystem path, and
#   loading it raises `FileNotFoundError` because no such file exists.
# - A member the archive does not hold raises `FileNotFoundError` naming both
#   the archive and the member.
# - RDF read from a member (`MelodyLoader`) resolves relative IRIs against the
#   member's URI, `jar:<archive file URI>!/<member>`.
#
# To hand a member to your own code that needs a file on disk, call
# `materialize`. It copies a member once into the cache, in the folder the
# environment variable `TTA_CACHE_DIR` names (by default
# `~/.cache/timetoalign`), and returns the copy's path; a later call returns
# the same copy, and a filesystem path comes back unchanged. The cache is
# never pruned: delete the folder by hand to reclaim the space.

# %%
copy = materialize(XML_MEMBER)
{
    "copy in the cache": copy.is_relative_to(ARCHIVE_DIR / "cache"),
    "same bytes as the member": copy.read_bytes() == XML_MEMBER.read_bytes(),
    "second call, same copy": materialize(XML_MEMBER) == copy,
    "a path comes back unchanged": materialize(CHOPIN_XML) == CHOPIN_XML,
}

# %% [markdown]
# ## Summary
#
# In this tutorial, we learned:
#
# 1. **The Loader Pattern**: Create -> Load -> Access Bundle
# 2. **Three Score Loaders**: Ms3Loader, PartituraLoader, Music21Loader
# 3. **EventStore**: PyArrow-backed, high-performance event storage
# 4. **Harmonized Schema**: Consistent fields across all loaders
# 5. **Cross-Validation**: Same piece from different sources yields same note count
# 6. **Archives**: A `zipfile.Path` loads a file from a ZIP archive as from disk
#
# **Key Takeaway:**
# > Loaders normalize heterogeneous formats into a consistent EventStore,
# > enabling downstream processing without format-specific code.

# %% [markdown]
# ## Next Steps
#
# - **how01_advanced_cmaps.ipynb**: Learn how to convert between coordinate systems
# - **how01_manual_timeline_construction.ipynb**: Create Timeline objects from EventStores

# %% [markdown]
# ---
#
# ## Exercise: Load Another Score
#
# **Task:** Load the Beethoven String Quartet from `beethoven_op18.mid` and analyze its structure.
#
# **Hints:**
# 1. Use `PartituraLoader` for MIDI files
# 2. Check how many parts are in the score
# 3. Count notes per part
#
# <details>
# <summary>Solution</summary>
#
# ```python
# # Load the Beethoven quartet
# loader = PartituraLoader()
# loader.load(DATA_DIR / "beethoven_op18.mid")
#
# df = loader.store.notes.to_dataframe()
# {"total_notes": len(df), "notes_per_part": df.groupby("part_id").size().to_dict()}
# ```
#
# </details>

# %%
# Your solution here
