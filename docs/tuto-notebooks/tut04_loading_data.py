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
# # Loading Real Data
#
# ## What you will build
#
# You will load Chopin's Étude Op. 10 No. 3 from MusicXML and a note table
# exported by ms3, a Python toolkit for working with score corpora. The TSV has
# one note per tab-separated row; the [loading-data
# guide](../howto/how01_loading_data.ipynb) compares its `Ms3Loader` with the
# MusicXML loaders. You will finish with the same kind of
# {{< glossary Timeline >}} you previously built by hand, including
# {{< glossary Child >}} timelines and an exact {{< glossary Coordinate >}}
# conversion derived from the file.
#
# ## Before you start
#
# Complete the tutorial on {{< glossary Event >}}s,
# [Events on a Timeline](tut03_events.ipynb), first.

# %%
from fractions import Fraction

import pandas as pd

from timetoalign import (
    Coordinate,
    CsvLoader,
    EventStore,
    Loader,
    Ms3Loader,
    Music21Loader,
    PartituraLoader,
    PerformanceMidiLoader,
    ScoreMidiLoader,
    TimeUnit,
    TsvLoader,
)
from timetoalign.testdata import ensure_data

VIENNA_DATA = ensure_data("vienna_1x22")
MIDI_DATA = ensure_data("midi")

# %% [markdown]
# ## From hand-built to real
#
# The first three tutorials taught the model by constructing timelines by hand;
# here, files replace those construction steps. Nothing new appears in the
# result—only in how it is produced.

# %%
musicxml_path = VIENNA_DATA / "Chopin_op10_no3.musicxml"
ms3_path = VIENNA_DATA / "ms3" / "chopin_op10_no3.notes.tsv"
score_midi_path = MIDI_DATA / "score" / "rachmaninoff_piano.mid"
performance_midi_path = MIDI_DATA / "performance" / "rachmaninoff_perf.mid"
source_files = {
    "MusicXML": musicxml_path,
    "ms3 TSV": ms3_path,
    "score MIDI": score_midi_path,
    "performance MIDI": performance_midi_path,
}
source_files

# %% [markdown]
# These are real corpus files resolved by `ensure_data()`. They are inputs to
# loaders, not a new kind of timeline result.

# %% [markdown]
# ## The two-phase contract
#
# `loader.load(*sources)` ingests files and returns the loader; only
# `loader.create_timeline()` builds objects in the musical
# {{< glossary Domain >}}, and it never takes a file path. `from_file()` is the
# shorthand for creating a loader and performing the first phase.

# %%
two_phase_loader = PartituraLoader()
two_phase_loader.load(musicxml_path)
preview_timeline = two_phase_loader.create_timeline()

partitura_loader = PartituraLoader.from_file(musicxml_path)
phase_objects = {
    "loaded, before timeline creation": two_phase_loader,
    "phase-two result": preview_timeline,
    "from_file result": partitura_loader,
    "follows Loader contract": isinstance(partitura_loader, Loader),
}
phase_objects

# %% [markdown]
# The first object is a loader holding one parsed source, while the phase-two
# object is a timeline. `from_file()` returns another loaded loader, not a
# timeline; the remaining sections use that shorthand loader.

# %% [markdown]
# ## The EventStore
#
# `loader.store` holds the parser's separate note, measure, control, and
# annotation tables before any timeline exists. This inventory uses
# `store.summary()` for counts and reads each table's exact coordinate range.

# %%
score_store = partitura_loader.store
store_contract = isinstance(score_store, EventStore)
store_summary = score_store.summary()
store_tables = store_summary["tables"]

store_rows = []
for category, event_data in score_store.items():
    first_position, last_position = event_data.coordinate_range()
    store_rows.append(
        {
            "category": category,
            "count": store_tables[category]["count"],
            "unit": event_data.unit,
            "first position": Coordinate(first_position, event_data.unit),
            "last position": Coordinate(last_position, event_data.unit),
        }
    )

store_inventory = pd.DataFrame(store_rows)
store_inventory_view = store_inventory.style.format(
    {"first position": repr, "last position": repr}
)
store_inventory_view = store_inventory_view.set_caption(
    f"Parsed table inventory — EventStore contract: {store_contract}"
)
store_inventory_view

# %% [markdown]
# Each row is a parser table with its count, unit, and exact range. The stored
# endpoints are `Fraction` values; placing them in `Coordinate` objects keeps
# both the rational value and `quarters` unit visible. `summary()` is a
# lightweight digest rather than a coordinate accessor: it reports counts,
# units and plain-float extents for a quick look at what a store holds, so it
# is used here for counts while the coordinates come from the typed accessors.
# The object also satisfies the public `EventStore` contract:
# `isinstance(score_store, EventStore)` is `True`.

# %% [markdown]
# ## One parsed record
#
# A store table contains parser records rather than timeline children. Select
# one note now so that its transformation during timeline creation is concrete.

# %%
stored_note_data = score_store.notes
stored_note_frame = stored_note_data.to_dataframe()
stored_note_example = stored_note_frame.loc[
    [0], ["id", "name", "event_type", "start", "end"]
].copy()
stored_note_id = stored_note_example.loc[0, "id"]

for position_column in ("start", "end"):
    stored_note_example[position_column] = [
        Coordinate(value, stored_note_data.unit)
        for value in stored_note_example[position_column]
    ]

stored_note_view = stored_note_example.style.format({"start": repr, "end": repr})
stored_note_view

# %% [markdown]
# This is the first raw note record: B3 begins at zero quarters and ends at one
# half. Its local id is `note:000001`. The timeline will preserve these musical
# fields while placing the record under a queryable child named `notes`.

# %% [markdown]
# ## Making the timeline
#
# `create_timeline(uid=...)` turns the loaded store into the domain object. The
# optional `uid` supplies identity; a separate name can remain readable to a
# human.

# %%
score_timeline = partitura_loader.create_timeline(uid="score:clt_1")
score_timeline.name = "Chopin, Étude Op. 10 No. 3"
timeline_identity = {
    "automatic id": preview_timeline.id,
    "role-prefixed id": score_timeline.id,
    "human-readable name": score_timeline.name,
}
timeline_identity

# %% [markdown]
# `clt_1` means the first continuous logical timeline created by that loader;
# `score:clt_1` adds an optional role prefix. The descriptive title is metadata,
# not identity, so changing it would not change the timeline's `id`.

# %% [markdown]
# ## What you got for free
#
# The loader has already separated each event category into a child timeline and
# attached a quarters-to-ticks {{< glossary ConversionMap >}}.

# %%
child_names = {
    child_id: score_timeline.get_child(child_id).name
    for child_id in score_timeline.list_children()
}
stored_categories = tuple(store_inventory["category"])
quarters_to_ticks = score_timeline.get_conversion_map(TimeUnit.ticks)

quarter_position = score_timeline.make_coordinate(Fraction(13, 2))
tick_position = score_timeline.convert_to(quarter_position, TimeUnit.ticks)
loader_payoff = {
    "children": child_names,
    "stored categories": stored_categories,
    "quarters-to-ticks map": quarters_to_ticks,
    "quarter position": quarter_position,
    "tick position": tick_position,
}
loader_payoff

# %% [markdown]
# Each child is named after one of the four stored categories; its id is minted
# like any other timeline's (`clt_2`, …), because a category is a label, not an
# identity. The map converts the library's
# `Coordinate(Fraction(13, 2), quarters)` directly to `Coordinate(3120, ticks)`;
# no rational value is reconstructed for display. Measures are available as the
# child named `measures`, but this score loader does not also materialize them as
# {{< glossary Region >}} objects.

# %% [markdown]
# ## Querying every child
#
# A loaded timeline uses the same `get_events()` query introduced previously.
# Ask the parent for every event, then select one representative row from each
# child for inspection.

# %%
loaded_events = score_timeline.get_events()
loaded_event_frame = loaded_events.to_dataframe()
event_examples = loaded_event_frame.drop_duplicates(subset="source_timeline")
event_examples = event_examples.loc[
    :, ["source_timeline", "event_type", "name", "start"]
].copy()
event_examples = event_examples.reset_index(drop=True)
event_examples["start"] = [
    score_timeline.make_coordinate(value) for value in event_examples["start"]
]

event_examples_view = event_examples.style.format({"start": repr})
event_examples_view = event_examples_view.set_caption(
    f"One row from each child among {len(loaded_events)} loaded events"
)
event_examples_view

# %% [markdown]
# The four rows come from `notes`, `measures`, `controls`, and `annotations`, so
# the parent query really does gather events from every child. The selected
# fields show each event's source, type, name, and exact start coordinate; the
# caption records that the complete result contains 547 events.

# %% [markdown]
# ## Narrowing by event type
#
# The same query can keep only notes. Convert a few selected fields to a table
# so that both note content and exact positions remain visible.

# %%
loaded_notes = score_timeline.get_events(event_type="Note")
loaded_note_frame = loaded_notes.to_dataframe()
note_examples = loaded_note_frame.loc[
    :3, ["id", "name", "event_type", "start", "end", "staff"]
].copy()

for position_column in ("start", "end"):
    note_examples[position_column] = [
        score_timeline.make_coordinate(value)
        for value in note_examples[position_column]
    ]

timeline_note_id = f"notes:{stored_note_id}"
assert note_examples.loc[0, "id"] == timeline_note_id
note_examples_view = note_examples.style.format({"start": repr, "end": repr})
note_examples_view = note_examples_view.set_caption(
    f"First four of {len(loaded_notes)} Note events"
)
note_examples_view

# %% [markdown]
# These are actual note rows, not an `EventData` summary. The first is the B3
# stored earlier: timeline creation preserves its fields and scopes its local id
# `note:000001` as `notes:note:000001`. The table shows four of the 498 rows kept
# by the familiar `event_type` filter.

# %% [markdown]
# ## The same music, a second way
#
# `PartituraLoader.from_file()` read the MusicXML above; now
# `Ms3Loader.from_file()` reads the neighboring note TSV. Comparing note counts
# tests whether the two provenances describe the same musical content.

# %%
ms3_loader = Ms3Loader.from_file(ms3_path)
ms3_timeline = ms3_loader.create_timeline(uid="ms3:clt_1")
ms3_events = ms3_timeline.get_events()
ms3_notes = ms3_timeline.get_events(event_type="Note")

partitura_note_count = len(loaded_notes)
note_counts = {
    "MusicXML via PartituraLoader": partitura_note_count,
    "TSV via Ms3Loader": len(ms3_notes),
}
timeline_classes = (score_timeline.class_name, ms3_timeline.class_name)
assert len(loaded_notes) == partitura_note_count
assert len(loaded_events) >= partitura_note_count
assert len(ms3_events) == len(ms3_notes)
assert len(set(note_counts.values())) == 1
assert len(set(timeline_classes)) == 1
format_comparison = {
    "timeline class": timeline_classes[0],
    "note counts": note_counts,
}
format_comparison

# %% [markdown]
# Both routes produce a `ContinuousLogicalTimeline` and find 498 notes:
# different provenance, same musical content. File formats do not always
# preserve the same amount of information, however; the later
# pitch-and-harmony tutorial shows where their knowledge differs.

# %% [markdown]
# ## The loader families
#
# Choose a loader family according to what the source represents, then choose a
# concrete loader for its file format.
#
# | Family | Public loaders | What it is for |
# |---|---|---|
# | Score | `Ms3Loader`, `Music21Loader`, `PartituraLoader` | Symbolic notation |
# | MIDI | `ScoreMidiLoader`, `PerformanceMidiLoader` | Quantized scores or performed timing |
# | Tabular | `TsvLoader`, `CsvLoader` | Custom delimited research tables |
# | Alignment | {{< glossary MatchfileLoader >}} | Score-to-performance correspondences; introduced later |
# | Graphical | `GraphicalLoader` | Pages, images, and horizontal layout |
# | Format | `JsonLoader` | Structured JSON and XML |
#
# Each family has a how-to guide of its own:
# [loading data](../howto/how01_loading_data.ipynb) ·
# [tabular loaders](../howto/how01_tabular_loaders.ipynb) ·
# [graphical timelines](../howto/how01_graphical_timelines.ipynb) ·
# [the Vienna corpus](../howto/how03_loading_vienna_corpus.ipynb) ·
# [load anything](../howto/how04_load_anything.ipynb)

# %%
loader_families = {
    "score": {
        "loaders": (Ms3Loader, Music21Loader, PartituraLoader),
        "example": source_files["MusicXML"],
    },
    "MIDI": {
        "loaders": (ScoreMidiLoader, PerformanceMidiLoader),
        "examples": (source_files["score MIDI"], source_files["performance MIDI"]),
    },
    "tabular": {
        "loaders": (TsvLoader, CsvLoader),
        "example": source_files["ms3 TSV"],
    },
}
loader_families

# %% [markdown]
# The class objects show that the score, MIDI, and tabular choices used here are
# importable public loaders, while the example paths distinguish score MIDI
# from performance MIDI. The remaining table rows name a concrete public loader
# and link to the guide that introduces it.

# %% [markdown]
# ## Beyond one timeline
#
# `create_timelines()` opens the door to loaders that yield several timelines;
# this leads to the next tutorial on {{< glossary TimelineGroup >}}s,
# [Timeline Groups](tut05_timeline_groups.ipynb).
# Alignment-capable loaders also provide `create_bundle()`, which produces an
# {{< glossary AlignmentBundle >}} in [Alignment Bundles and
# MatchClaims](tut06_alignment_bundles.ipynb).

# %%
timeline_list = ms3_loader.create_timelines()
future_doors = {
    "timelines returned": timeline_list,
    "loader families sampled above": len(loader_families),
}
future_doors

# %% [markdown]
# This score loader returns a one-item list; multi-source loaders can return
# more. The list-shaped interface is the bridge from loading one timeline here
# to organizing several timelines in the next tutorial.

# %% [markdown]
# ## What you learned
#
# - You can replace hand construction with real files without changing the kind of timeline produced.
# - You can separate `load()` from `create_timeline()` and use `from_file()` as their common shorthand.
# - You can inspect exact table counts, units, and ranges in `loader.store`,
#   and read `summary()` as the counts-and-extent digest it is.
# - You can distinguish a parser record in an `EventStore` from the corresponding event placed under a timeline child.
# - You can assign a type-based timeline id, optionally prefixed by a role, while keeping a readable name separate.
# - You can inspect loader-created children and exact coordinate conversions,
#   and recognise that this loader does not create regions.
# - You can display representative rows from an all-child event query.
# - You can narrow loaded events with the familiar event filter and inspect the resulting note rows.
# - You can compare MusicXML and ms3 TSV note content through the same timeline API.
# - You can choose among score, MIDI, tabular, alignment, graphical, and format loader families.
# - You can use `create_timelines()` and recognize `create_bundle()` as entry points to multi-timeline work.
#
# ## Next
#
# [Timeline Groups](tut05_timeline_groups.ipynb)
#
# ## Go deeper
#
# - [How to load data](../howto/how01_loading_data.ipynb)
# - [How to use tabular loaders](../howto/how01_tabular_loaders.ipynb)
# - [How to load many formats](../howto/how04_load_anything.ipynb)
