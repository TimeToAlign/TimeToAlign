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
# # How to Load CSV/TSV with `column_specs` and `field_specs`
#
# Tabular loaders in TimeToAlign! turn **columns** into **fields** in two
# explicit steps:
#
# | Term | Meaning |
# |---|---|
# | **column** | A column in the original source (CSV / TSV / DataFrame / pa.Table). |
# | **field** | A raw or semantic `DataField` (TTA-internal). May be nested. |
#
# The loader's job is the column to field mapping:
#
# - **Step 1 — `column_specs`** processes columns one-by-one. Mandatory for
#   header-based sources; trivial for sources that already carry types.
# - **Step 2 — `field_specs`** combines or promotes the resulting fields.
#   Fully optional — many loaders are complete with Step 1 alone.
#
# This guide walks through the Step-1 resolution chain (every entry ends
# at a `DataField`), the `CompositeFieldParser` mechanism for
# one-column-to-many-fields, and a worked example in which we build a
# loader from scratch for the header-less `.solo` performance-analysis
# format.

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pyarrow as pa

from timetoalign.core import (
    DenominateNumberField,
    EnharmonicPitch,
    EnharmonicPitchField,
    Id,
    IdField,
    IntField,
    MeasureNumberField,
    NumberType,
    RedundantNumberField,
    StringField,
    TimeUnit,
)
from timetoalign.loader.tabular import (
    CompositeFieldParser,
    CsvLoader,
    resolve_field_parser,
)
from timetoalign.testdata import ensure_data

PERF_PRECISION = ensure_data("performance_precision")
CHOPIN_SOLO = PERF_PRECISION / "Chopin Nocturne Op. 9 No. 2.solo"

# A small temp directory whose lifetime spans the whole notebook.  Synthetic
# CSV / TSV examples in subsequent cells write through here so the standard
# ``Loader.load(path)`` API can pick them up.
TMP = Path(tempfile.mkdtemp(prefix="how04_"))


def write_csv(name: str, body: bytes) -> Path:
    """Write a synthetic CSV body to ``TMP / name`` and return its path."""
    p = TMP / name
    p.write_bytes(body)
    return p


def head_table(table: pa.Table, n: int = 5) -> pd.DataFrame:
    """Return the first ``n`` rows of ``table`` as a DataFrame.

    DataFrames render as proper HTML tables in Jupyter, which is far more
    legible than dumping a column-keyed dict.  The helper sits at the top
    of the notebook so every preview cell uses the same display path.
    """
    return table.slice(0, n).to_pandas()


# %% [markdown]
# ## 1. From columns to fields: the two-step pipeline
#
# We start with the simplest case — a CSV with a header line and one
# Python type per column. The keys of `column_specs` are source-column
# names; the values are anything that resolves through the **Step-1
# resolution chain** (Section 2). Here, a bare `int` / `str` is enough
# for the columns that aren't already canonical (the loader still
# resolves `start` / `end` from the underlying source on its own):

# %%
mini_csv = (
    b"id,start,end,pitch,label\n"
    b"e0,0.0,0.5,60,intro\n"
    b"e1,0.5,1.0,64,intro\n"
    b"e2,1.0,1.5,67,verse\n"
    b"e3,1.5,2.0,72,verse\n"
)
mini_path = write_csv("mini.csv", mini_csv)


class MiniLoader(CsvLoader):
    column_specs = {"pitch": int, "label": str}


mini = MiniLoader()
mini.load(mini_path)
events = mini.get_events()
head_table(events.table)

# %% [markdown]
# Step 1 has produced one raw field per declared source column. The raw
# fields are reachable through `get_raw()`; `get_field()` is reserved for
# **semantic** fields with paired-class metadata (we meet that path in
# Section 4):

# %%
{
    "pitch": events.get_raw("pitch"),
    "label": events.get_raw("label"),
}

# %% [markdown]
# Note the asymmetry: **columns are a source artefact, fields are a TTA
# artefact**. A loader without `column_specs` leaves the non-canonical
# source columns alone as opaque property columns; with `column_specs`,
# each named column is processed into a typed field whose semantics
# travel with it.

# %% [markdown]
# ## 2. The Step-1 resolution chain
#
# Every entry in `column_specs` (and every entry in a `CompositeFieldParser`'s
# `parts`) goes through a single resolver, `resolve_field_parser`. The
# resolver is best read as a **chain**, not a lookup table: whatever the
# user passes, the chain ends at one of two producers:
#
# 1. a `DataField` blueprint — an empty instance carrying just a name and
#    (for semantic fields) the paired scalar class. The blueprint's
#    `emit(source, name=...)` consumes the raw `pa.Array` from the source
#    table and returns a fully materialised field.
# 2. a `FieldParser` instance — a genuine processor with its own
#    `emit(source, name=...)`. Parsers are reserved for the cases that
#    don't reduce to a typed cast: `CompositeFieldParser` for one column
#    to many fields, `CallableFieldParser` for arbitrary user callables.
#
# Both terminals expose the same `emit()` signature, so the loader does
# not care which one came out of the chain.
#
# The chain, top-to-bottom:
#
# | User passes | Resolves to |
# |---|---|
# | a `FieldParser` instance | itself (genuine processor) |
# | a `DataField` blueprint instance (e.g. `IntField(name="x")`) | itself |
# | a `DataField` **subclass** (raw or paired `SemanticField`) | a blueprint with `name=<dict key>` |
# | a Python type (`int` / `float` / `str` / `Fraction`) | the matching raw `DataField` blueprint |
# | a `pa.DataType` (`pa.int64()`, `pa.float64()`, `pa.string()`) | the matching raw `DataField` blueprint |
# | any callable `(pa.Array) -> DataField` | wrapped in a `CallableFieldParser` |
#
# The next cells run `resolve_field_parser` directly so the chain stops
# being abstract — every input ends at a concrete `DataField` blueprint
# or a `FieldParser`:

# %%
# Python-type input -> matching raw DataField blueprint.  ``default_name``
# stands in for the dict key that a real ``column_specs`` entry would
# carry; here we just supply a placeholder to satisfy the blueprint's
# name requirement.
{
    "int": type(resolve_field_parser(int, default_name="x")).__name__,
    "float": type(resolve_field_parser(float, default_name="x")).__name__,
    "str": type(resolve_field_parser(str, default_name="x")).__name__,
    "Fraction": type(resolve_field_parser(Fraction, default_name="x")).__name__,
}

# %%
# pa.DataType input -> matched by-type to the equivalent raw blueprint.
{
    "pa.int64()": type(resolve_field_parser(pa.int64(), default_name="x")).__name__,
    "pa.float64()": type(resolve_field_parser(pa.float64(), default_name="x")).__name__,
    "pa.string()": type(resolve_field_parser(pa.string(), default_name="x")).__name__,
}

# %%
# A pre-built blueprint passes through unchanged (identity check).
bp = IntField(name="channel")
resolve_field_parser(bp) is bp

# %% [markdown]
# A blueprint is also accepted directly as a `column_specs` value — the
# chain sees that the value is already a `DataField` and returns it as-is.
# The blueprint's `name=` overrides the dict key:

# %%
typed_csv = b"id,start,end,x\n" b"e0,0.0,0.5,1\n" b"e1,0.5,1.0,2\n" b"e2,1.0,1.5,3\n"
typed_path = write_csv("typed.csv", typed_csv)


class TypedLoader(CsvLoader):
    column_specs = {"x": IntField(name="x_renamed")}


typed = TypedLoader()
typed.load(typed_path)
head_table(typed.get_events().table)

# %% [markdown]
# Two terminal categories are worth restating, because they cover the
# overwhelming majority of real loaders:
#
# - **DataField blueprints** are what you reach for whenever the source
#   column maps to a single typed value — possibly with semantic
#   metadata. `IntField`, `FloatField`, `StringField`,
#   `RedundantNumberField` cover the raw side; every paired
#   `SemanticField` subclass
#   (`DenominateNumberField`, `EnharmonicPitchField`, `IdField`, …) is
#   blueprint-constructible in exactly the same way: `name=` plus
#   whatever extra kwargs the scalar requires.
# - **FieldParsers** show up when the column-to-field map is genuinely
#   procedural — when a typed cast is not enough.
#
# The next section meets the only `FieldParser` you'll need in practice.

# %% [markdown]
# ## 3. `CompositeFieldParser`: one column, multiple fields
#
# Some source columns pack several values into one string. The classic
# musicological example is `1+3/8` — a measure number, a literal `+`,
# and a fractional offset inside the measure. `CompositeFieldParser` is
# the universal splitter: a separator (or regex pattern) plus a `parts`
# declaration that follows the same resolution chain as `column_specs`:

# %%
composite_csv = (
    b"id,start,end,position\n"
    b"e0,0.0,0.5,1+0/8\n"
    b"e1,0.5,1.0,1+3/8\n"
    b"e2,1.0,1.5,2+1/4\n"
    b"e3,1.5,2.0,2+5/8\n"
)
composite_path = write_csv("composite.csv", composite_csv)


class CompositeLoader(CsvLoader):
    column_specs = {
        "position": CompositeFieldParser(
            separator="+",
            parts={
                "measure": int,
                "offset": Fraction,
            },
        ),
    }


comp = CompositeLoader()
comp.load(composite_path)
head_table(comp.get_events().table)

# %% [markdown]
# Each entry of `parts` is resolved through the same chain — so `int`
# becomes an `IntField` blueprint and `Fraction` becomes a
# `RedundantNumberField` blueprint. Composites also nest: a `parts`
# entry may itself be a `CompositeFieldParser`.
#
# `RedundantNumberField` parses string entries like `"3/8"` into a
# Fraction, storing the exact numerator/denominator pair alongside a
# float mirror;
# `DenominateNumberField` does the same and binds the result to a
# `TimeUnit`, so the field arrives at the next pipeline stage already
# carrying time-unit semantics:

# %%
denom_csv = b"id,start,end,duration\n" b"e0,0.0,0.5,1/2\n" b"e1,0.5,0.75,1/4\n"
denom_path = write_csv("denom.csv", denom_csv)


class DenomLoader(CsvLoader):
    column_specs = {
        "duration": DenominateNumberField(name="duration", unit=TimeUnit.quarters),
    }


dl = DenomLoader()
dl.load(denom_path)
head_table(dl.get_events().table)

# %% [markdown]
# ## 4. Worked example: building a `.solo` loader from scratch
#
# Performance-analysis pipelines emit `.solo` files: a header-less,
# tab-separated format in which each row carries six columns:
#
# 1. Measure-plus-offset (`1+3/8`) — measure number, a literal `+`, then
#    a fractional onset within the measure (in quarters).
# 2. Duration as a fraction of a quarter note (`1/4`).
# 3. Channel (integer).
# 4. MIDI pitch (integer, 0–127).
# 5. Velocity (integer, 0–127; 0 marks a note-off).
# 6. Opaque alphanumeric note identifier (`"n1b8xktz"`).
#
# Here is the head of the Chopin specimen:

# %%
{"first lines": CHOPIN_SOLO.read_text().splitlines()[:5]}

# %% [markdown]
# We're going to write a `CsvLoader` subclass for this format from
# scratch — exercising every mechanism in this guide.  Two things make
# `.solo` non-trivial:
#
# - There is no header line, so the dataframe-reading step needs to be
#   overridden to supply positional column names.
# - The first column packs measure number + onset together — exactly
#   the case `CompositeFieldParser` is for.
#
# The pattern for overriding the dataframe read is to set
# `header_row = -1` (the sentinel for "no header") and override
# `_read_dataframe` to call `pd.read_csv(..., header=None, names=[...])`
# with positional names matching the `column_specs` declaration order.
# Everything else — Step 1 emissions, Step 2 promotions, property
# columns — works exactly as in the header-based examples above.


# %%
class ChopinSoloLoader(CsvLoader):
    """Loader for the header-less ``.solo`` performance-analysis format."""

    # Header-less, tab-delimited source.
    delimiter: ClassVar[str] = "\t"
    header_row: ClassVar[int] = -1  # sentinel for "no header"

    # Canonical column wiring.  ``mn_onset`` is the fractional onset within
    # the measure, produced by Step 1's composite split; measure-number
    # resolution against a MetricMap is left to downstream consumers.
    id_column: ClassVar[str | None] = None  # auto-generated
    name_column: ClassVar[str | None] = None
    start_column: ClassVar[str] = "mn_onset"
    end_column: ClassVar[str | None] = None
    duration_column: ClassVar[str | None] = "duration"
    event_type_column: ClassVar[str | None] = None
    default_event_type: ClassVar[str] = "Note"

    _default_unit: ClassVar[TimeUnit] = TimeUnit.quarters
    coordinate_type: ClassVar[NumberType] = NumberType.fraction

    # Step 1: one entry per source column, in declaration order.  An
    # iterable form is used because the source has no header to key on.
    column_specs: ClassVar[list[Any]] = [
        # col 0: measure_number + fractional onset within the measure.
        CompositeFieldParser(
            separator="+",
            parts=[
                MeasureNumberField,  # default name 'measure_number'
                RedundantNumberField(name="mn_onset"),
            ],
            name="position",
        ),
        # col 1: duration as a fraction of a quarter note (semantic).
        DenominateNumberField(name="duration", unit=TimeUnit.quarters),
        # cols 2-5: simple typed blueprints.
        IntField(name="channel"),
        IntField(name="pitch"),
        IntField(name="velocity"),
        StringField(name="note_id"),
    ]

    # Step 2: promote two raw Step-1 fields to semantic ones using the
    # ``source_fields=<name>`` blueprint shorthand — "one raw field by
    # this name, fed into the target's canonical 'value'".
    field_specs: ClassVar[list[Any]] = [
        EnharmonicPitchField(source_fields="pitch"),
        IdField(source_fields="note_id"),
    ]

    def _read_dataframe(self, source):  # type: ignore[override]
        """Read a header-less ``.solo`` file with positional column names."""
        names = ["position", "duration", "channel", "pitch", "velocity", "note_id"]
        return pd.read_csv(
            source,
            sep=self.delimiter,
            header=None,
            names=names,
            encoding=self.encoding,
            dtype=str,
        )


loader = ChopinSoloLoader()
loader.load(CHOPIN_SOLO)
events = loader.get_events()
head_table(events.table)

# %% [markdown]
# After Step 1, the working table holds typed fields — measure number
# and fractional offset broken out into the `position` struct, duration
# bound to quarters, the three integer columns, and the string ID.
# Step 2 then promotes the raw `pitch` int into a semantic
# `EnharmonicPitchField` and the raw `note_id` string into a paired
# `IdField`:

# %%
events.table.column_names

# %% [markdown]
# `EnharmonicPitchField` is now reachable by type, and indexing it
# returns `EnharmonicPitch` scalars. The faithfulness rule applies: the
# field records the MIDI pitch number that the source data carried, not
# anything inferred about spelling — both `G♯3` and `A♭3` show as MIDI
# 56 because `.solo` does not carry an accidental:

# %%
ep_field = events.get_field(EnharmonicPitch)
{"first 5": [ep_field[i] for i in range(5)]}

# %% [markdown]
# `IdField` holds the original note identifiers. The Chopin specimen
# re-uses some IDs across event onsets/offsets, which the shallow form
# preserves verbatim:

# %%
id_field = events.get_field(Id)
{"first 5": [id_field[i] for i in range(5)]}

# %% [markdown]
# That is the entire loader. No subclassing of internal types, no struct
# dicts, no special-casing — every column flowed through the same
# Step 1 resolution chain, and two raw fields graduated to semantic via
# Step 2 blueprints.

# %% [markdown]
# ## 5. Property columns
#
# By default, source columns that are NOT consumed by any `column_specs`
# entry survive in the EventData as **property columns** — convenient
# for ad-hoc inspection but excluded from the field machinery. The
# `properties=` kwarg on `get_events()` controls which ones make it
# through:
#
# | Value | Meaning |
# |---|---|
# | `True` (default) | all remaining source columns survive as properties |
# | `False` | no property columns; the EventData carries only fields |
# | `"col"` | shorthand for a one-element tuple |
# | `(`col1`, `col2`, …)` | explicit named subset |
#
# To exercise this we need a CSV that genuinely carries an unconsumed
# column. Here `page` and `scribbled_note` are present in the source
# but absent from `column_specs`, so they default to property columns:

# %%
extras_csv = (
    b"id,start,end,pitch,label,scribbled_note,page\n"
    b"e0,0.0,0.5,60,intro,one,1\n"
    b"e1,0.5,1.0,64,intro,two,1\n"
    b"e2,1.0,1.5,67,verse,three,2\n"
)
extras_path = write_csv("extras.csv", extras_csv)


class ExtrasLoader(CsvLoader):
    column_specs = {"pitch": int, "label": str}


ex = ExtrasLoader()
ex.load(extras_path)

# %% [markdown]
# `properties=True` (the default) keeps every unconsumed column;
# `properties=False` drops them all. The string shorthand
# `properties="page"` and the tuple form `properties=("page",)` are
# equivalent — both keep just `page` as a property and drop
# `scribbled_note`:

# %%
{
    "properties=True  (default)": ex.get_events(properties=True).table.column_names,
    "properties=False": ex.get_events(properties=False).table.column_names,
    "properties='page'": ex.get_events(properties="page").table.column_names,
    "properties=('page',)": ex.get_events(properties=("page",)).table.column_names,
}

# %% [markdown]
# Property columns let users keep source-side metadata alongside the
# fields without forcing every loose annotation through `column_specs`.
# When a property column needs to gain semantic identity — to become
# a `DataField` proper — that's exactly when `field_specs` enters: name
# the column in Step 1 with the appropriate spec, and Step 2 promotes it
# (as it did with `pitch` and `note_id` in the `.solo` loader above).
