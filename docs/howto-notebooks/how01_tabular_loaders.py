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
# # How to Load Tabular Data
#
# The fastest way to get music data into TimeToAlign! is through **tabular
# loaders**. If your data is in CSV or TSV format, you're just 3 lines of
# code away from analysis.
#
# **What you'll learn:**
# - Load music annotations from TSV/CSV files
# - Access event counts, coordinate ranges, and metadata
# - Create timelines from loaded data
# - Write a custom loader that maps your columns to event fields
# - Promote selected columns to typed fields with `column_specs`
# - Reach nested JSON columns with `Field`
#
# **Time:** 15 minutes
#
# This is the gentle introduction. For the full column-to-field mechanism —
# the Step-1 resolution chain, composite columns, and Step-2 field
# promotion — see the in-depth CSV/TSV how-to.

# %% [markdown]
# ## TL;DR
#
# ```python
# from timetoalign.loader.tabular import TsvLoader
#
# loader = TsvLoader()
# loader.load("beethoven.notes.tsv")
#
# df = loader.events.to_dataframe()    # Get as DataFrame
# timeline = loader.create_timeline()  # Create Timeline
# ```

# %% [markdown]
# ## Setup

# %%
from timetoalign.testdata import ensure_data

BEETHOVEN = ensure_data("score") / "beethoven_woo71"
THORESEN = ensure_data("thoresen")

# Available files
{
    "Beethoven files": [f.name for f in BEETHOVEN.glob("WoO71.*.tsv")],
    "Thoresen files": [f.name for f in THORESEN.glob("*.tsv")],
}

# %% [markdown]
# ## Loading Notes from TSV
#
# `TsvLoader` is the generic starting point for a TSV whose columns you
# control. Declare the coordinate columns and promote the values you want as
# typed fields. For a standard ms3 score export, use the score `Ms3Loader`
# instead; it understands its notes, measures, chords, and harmonies facets.
#
# The file gives each note's length twice: `duration_qb` in quarters, written
# as a decimal (a twelfth as `0.08333333333333333`), and `duration` in whole
# notes, written as an exact ratio (`1/48`). On an exact (`Fraction`) axis
# the loader completes each end as start + duration exactly, and for a note
# starting at `529/12` the sum with that decimal's binary value needs a
# numerator wider than the 64 bits an exact value is stored in, so a
# `duration_column = "duration_qb"` is refused rather than guessed back into
# a ratio. Complete the ends from the exact `duration` column instead, with a
# `ComputedField` as below, or declare `coordinate_type = NumberType.float`
# for a file that gives its durations only as decimals.

# %%
from fractions import Fraction  # noqa: E402

import pyarrow as pa  # noqa: E402

from timetoalign.core import NumberType, RedundantNumberField, TimeUnit  # noqa: E402
from timetoalign.loader import ComputedField  # noqa: E402
from timetoalign.loader.tabular import TsvLoader  # noqa: E402


def end_in_quarters(table: pa.Table) -> pa.Array:
    """A note's end in quarters: its start plus four times its whole-note duration."""
    starts = table["quarterbeats"].to_pylist()
    wholes = table["duration"].to_pylist()
    # Written as "n/d" text, which the loader reads exactly.
    return pa.array(
        [
            None if s is None else str(Fraction(s) + 4 * Fraction(w))
            for s, w in zip(starts, wholes)
        ]
    )


class BeethovenNotesLoader(TsvLoader):
    """Generic TSV configuration for the coordinate and pitch columns used here."""

    start_column = "quarterbeats"
    end_column = ComputedField("end", expr=end_in_quarters)
    _default_unit = TimeUnit.quarters
    coordinate_type = NumberType.fraction
    # The whole-note column keeps its exact values under its own name, so it
    # is not read as the events' duration, which is in quarters.
    column_specs = {
        "duration": RedundantNumberField(name="duration_whole"),
        "midi": int,
        "staff": int,
        "voice": int,
    }


loader = BeethovenNotesLoader()
loader.load(BEETHOVEN / "WoO71.notes.tsv")

f"{len(loader.events):,} notes loaded"

# %% [markdown]
# ## Converting to pandas
#
# Use `to_dataframe()` to get a DataFrame with clean coordinate values:

# %%
loader.events.to_dataframe().head()

# %% [markdown]
# ## Quick Statistics
#
# The loader provides immediate access to summary information:

# %%
{
    "event_count": len(loader.events),
    "coordinate_range": loader.events.coordinate_range(),
    "unit": str(loader.unit),
    "number_type": str(loader.number_type),
}

# %% [markdown]
# ## Creating Timelines
#
# TimeToAlign! represents temporal data as **Timelines**:

# %%
timeline = loader.create_timeline(uid="beethoven_notes")
timeline

# %% [markdown]
# ## Custom Loaders for Non-Standard Formats
#
# For files that don't match the ms3 format, create a custom loader by
# subclassing `TsvLoader` or `CsvLoader`. You point the canonical column
# attributes (`start_column`, `duration_column`, …) at the names in your
# file.
#
# Let's load the Thoresen annotations file, which has a different column
# structure:

# %%
import pandas as pd  # noqa: E402

pd.read_csv(THORESEN / "thoresen_test.tsv", sep="\t", nrows=3)

# %% [markdown]
# ### The Simplest Custom Loader
#
# Map the core coordinate columns and nothing else. Every source column
# that you do *not* name survives as an opaque **property column** —
# carried alongside the event so you never lose data, but left untyped:

# %%
from timetoalign.loader.tabular import TsvLoader  # noqa: E402


class ThoresenLoader(TsvLoader):
    """Minimal loader — maps the core event fields."""

    id_column = "event_id"
    start_column = "start_time_sec"
    duration_column = "duration_sec"
    event_type_column = "event_type"
    name_column = "description"

    _default_unit = TimeUnit.seconds
    coordinate_type = NumberType.float


thoresen = ThoresenLoader()
thoresen.load(THORESEN / "thoresen_test.tsv")
thoresen.events.to_dataframe()

# %% [markdown]
# ### Promoting Columns to Typed Fields with `column_specs`
#
# When a property column should become a typed field, name it in
# `column_specs`. The keys are source-column names; the values are
# anything the loader can resolve to a field — a bare Python type
# (`int` / `float` / `str`) is the simplest form. Here we give two
# extra columns explicit types:


# %%
class ThoresenTypedLoader(TsvLoader):
    """Selected columns promoted to typed fields."""

    id_column = "event_id"
    start_column = "start_time_sec"
    duration_column = "duration_sec"
    event_type_column = "event_type"
    name_column = "description"

    _default_unit = TimeUnit.seconds
    coordinate_type = NumberType.float

    # Promote these source columns to typed fields.
    column_specs = {
        "image_filename": str,
        "graphical_element_id": str,
    }


typed = ThoresenTypedLoader()
typed.load(THORESEN / "thoresen_test.tsv")
typed.events.to_dataframe()

# %% [markdown]
# That is the entire idea: **columns are a source artefact; fields are a
# TimeToAlign! artefact.** `column_specs` is the bridge. A bare type is
# the gentlest entry — the in-depth CSV/TSV how-to covers the full
# resolution chain (composite columns, semantic pitch/id fields, and the
# Step-2 `field_specs` promotion stage) for richer formats.

# %% [markdown]
# ## Nested JSON Column Access with `Field`
#
# The Thoresen data has a `rect_coords_json` column containing pixel
# coordinates as JSON:
# ```json
# {"x": 10, "y": 90, "width": 148, "height": 55}
# ```
#
# Use `Field("column", "nested_field")` to point a coordinate attribute
# straight at a nested value. TimeToAlign! parses the JSON
# automatically. `ComputedField` lets you derive a coordinate from a
# small formula over those nested values:

# %%
from timetoalign.loader import Field  # noqa: E402


class ThoresenGraphicalLoader(TsvLoader):
    """Loader using PIXEL coordinates from a nested JSON column."""

    # Nested fields are addressed directly; JSON is parsed automatically.
    start_column = Field("rect_coords_json", "x")
    end_column = ComputedField(
        "end", formula="rect_coords_json.x + rect_coords_json.width"
    )

    _default_unit = TimeUnit.pixels
    coordinate_type = NumberType.float
    default_event_type = "Rectangle"


graphical = ThoresenGraphicalLoader()
graphical.load(THORESEN / "thoresen_test.tsv")

{
    "unit": str(graphical.unit),
    "coordinate_range": graphical.events.coordinate_range(),
}

# %% [markdown]
# ### Two Coordinate Systems from One File
#
# The same TSV file backs timelines in different coordinate systems —
# seconds from the time columns, pixels from the JSON column:

# %%
# Physical timeline (seconds)
typed.events.to_dataframe()

# %%
# Graphical timeline (pixels)
graphical.events.to_dataframe()

# %% [markdown]
# ### Creating Timelines
#
# Use `create_timeline()` to turn loaded events into a Timeline object:

# %%
# Physical timeline (seconds)
physical_tl = typed.create_timeline(uid="thoresen_physical")
physical_tl

# %%
physical_tl.get_timestamp_table()

# %%
# Graphical timeline (pixels)
graphical_tl = graphical.create_timeline(uid="thoresen_graphical")
graphical_tl

# %% [markdown]
# **Note:** Both timelines represent the same 11 events in different
# coordinate systems:
# - **Physical:** `0 - 142.5 seconds` (audio time)
# - **Graphical:** `10 - 760 pixels` (image coordinates)
#
# TimeToAlign! uses these dual representations to align graphical
# annotations with audio.

# %% [markdown]
# ### Child Timelines from Column Values with `group_by`
#
# When your data carries events from **multiple sources** (images, pages,
# tracks), pass `group_by` to `create_timeline()` to split the events into
# one child timeline per unique value. The Thoresen data has events from
# five different image files:

# %%
from timetoalign.timelines import create_timeline  # noqa: E402

grouped_tl = create_timeline(typed, group_by="image_filename")
grouped_tl

# %% [markdown]
# Each child gets a minted ID (`cpt_…`) and is named after its
# column value, so the image a child holds is its `name`:

# %%
# Each child timeline represents events from one image
{
    "parent_id": grouped_tl.id,
    "n_children": grouped_tl.n_children,
    "children": {
        child.id: (child.name, child.n_events)
        for _, child in grouped_tl.iter_children()
    },
}

# %% [markdown]
# ---
#
# ## Summary
#
# | Goal | How |
# |------|-----|
# | Load an ms3 score TSV | `Ms3Loader()` + `loader.load(path)` |
# | Map your own columns | Subclass `TsvLoader` / `CsvLoader`, set `start_column` etc. |
# | Promote a column to a typed field | Name it in `column_specs` (`{"col": int}`) |
# | Reach a nested JSON value | `Field("column", "nested")` as a coordinate attribute |
# | Derive a coordinate | `ComputedField("end", formula="...")` |
# | Split into child timelines | `create_timeline(loader, group_by="column")` |
#
# ```python
# from timetoalign.loader.tabular import TsvLoader
#
# class MyLoader(TsvLoader):
#     start_column = "onset"
#     duration_column = "dur"
#     column_specs = {"pitch": int, "velocity": int}
# ```
#
# > **Key Takeaway:** Tabular loaders map CSV/TSV columns to TimeToAlign!
# > events declaratively. Unnamed columns ride along as property columns;
# > `column_specs` promotes the ones you want as typed fields; `Field`
# > reaches nested JSON. For the full column-to-field mechanism — composite
# > columns, semantic field types, and Step-2 promotion — continue to the
# > in-depth CSV/TSV how-to.
