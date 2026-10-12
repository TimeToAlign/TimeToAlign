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
# # Quickstart
#
# One piece of music can exist at once as a recording, as notation, and as
# images on a page. A {{< glossary Domain >}} says which kind of representation
# an axis measures: physical, logical, or graphical. A continuous axis permits
# positions between its marks; a {{< glossary Discrete >}} axis uses numbered
# steps such as samples, ticks, or pixels.
#
# | Domain | Continuous | Discrete |
# |---|---|---|
# | Physical: recording | seconds | samples |
# | Logical: notation | quarters | ticks |
# | Graphical: page image | centimetres | pixels |
#
# TimeToAlign! represents these axes and the relationships between them. A
# {{< glossary Coordinate >}} keeps a position together with the unit of its
# axis, so the library can distinguish 12.5 seconds from 12.5 quarters.
#
# ## What you will build
#
# In five minutes, you will connect a 30-second recording axis to a small loaded
# score and inspect one exact cross-representation position. Each section gives
# a short preview and points to the tutorial that develops that idea.

# %% [markdown]
# ## Before you start
#
# Use Python 3.11 or later in an environment with TimeToAlign! and its tutorial
# dependencies installed; see [Getting Started](../index.qmd#getting-started).

# %%
from fractions import Fraction

from timetoalign import MatchClaim, PartituraLoader
from timetoalign.core import EnharmonicPitch, SpecificPitch
from timetoalign.testdata import ensure_data
from timetoalign.timelines import (
    ContinuousPhysicalTimeline,
    TimelineGroup,
)

vienna = ensure_data("vienna_1x22")

# %% [markdown]
# ## Timelines and Coordinates
#
# A {{< glossary Timeline >}} is a measured axis for one representation of the
# music. A `Coordinate` places a value on that axis and keeps its unit attached.

# %%
audio = ContinuousPhysicalTimeline(length=Fraction(30), uid="audio")
audio_coordinate = audio.make_coordinate(Fraction(25, 2))
audio_coordinate

# %% [markdown]
# The result is the exact rational position 25/2 seconds on a 30-second audio
# axis, displayed with its unit. Open
# [Timelines and Coordinates](tut01_timelines_and_coordinates.ipynb) to build
# and inspect timelines carefully.

# %% [markdown]
# ## Nesting and Timestamps
#
# Nested timelines are {{< glossary Child >}} objects. A
# {{< glossary TimeStamp >}} reads one position across the parent and every
# child present there.

# %%
left_channel = ContinuousPhysicalTimeline(length=Fraction(30), uid="left-channel")
right_channel = ContinuousPhysicalTimeline(length=Fraction(30), uid="right-channel")
audio.add_child(left_channel, offset=Fraction(0))
audio.add_child(right_channel, offset=Fraction(0))
audio_timestamp = audio.get_timestamp(audio_coordinate)
timestamp_coordinates = {
    "audio parent": audio_timestamp.get_coordinate(audio.id),
    "left channel": audio_timestamp.get_coordinate(left_channel.id),
    "right channel": audio_timestamp.get_coordinate(right_channel.id),
}
timestamp_coordinates

# %% [markdown]
# The three named values show the parent and both children at the same exact
# 25/2-second position. The timestamp itself is the cross-section used to
# retrieve them. Open
# [Nesting and Timestamps](tut02_nesting_and_timestamps.ipynb) to learn how
# nested coordinate systems work.

# %% [markdown]
# ## Events on a Timeline
#
# An {{< glossary Event >}} records something that happens at an instant or
# over an interval. Here three labelled cues become part of the audio timeline.

# %%
audio.add_events(
    [
        {
            "id": "opening",
            "event_type": "Cue",
            "temporal_type": "interval",
            "start": 0.0,
            "end": 2.0,
        },
        {
            "id": "theme",
            "event_type": "Cue",
            "temporal_type": "interval",
            "start": 8.0,
            "end": 14.0,
        },
        {
            "id": "coda",
            "event_type": "Cue",
            "temporal_type": "interval",
            "start": 25.0,
            "end": 30.0,
        },
    ]
)
audio_events = audio.get_events(event_type="Cue")
audio_events

# %% [markdown]
# The event store contains the three non-empty cue intervals in seconds.
# Open [Events on a Timeline](tut03_events.ipynb) to add, filter, and retrieve
# musical events.

# %% [markdown]
# ## Loading Real Data
#
# Loaders turn common research formats into the same kind of timeline. This
# preview reads a small MusicXML score prepared in the setup, then shows what
# the loader populated.

# %%
score_loader = PartituraLoader.from_file(vienna / "Chopin_op10_no3.musicxml")
score = score_loader.create_timeline(uid="score")
score_children = [score.get_child(child_id).name for child_id in score.list_children()]
score_notes = score.get_events(event_type="Note")
score_measures = score.get_events(event_type="Measure")
note_frame = score_notes.to_dataframe(coordinates=True)
measure_frame = score_measures.to_dataframe(coordinates=True)
preview_columns = ["name", "start", "end"]
note_preview = note_frame.loc[0, preview_columns].to_dict()
measure_preview = measure_frame.loc[0, preview_columns].to_dict()
loaded_score_preview = {
    "children": score_children,
    "first note": note_preview,
    "first measure": measure_preview,
}
loaded_score_preview

# %% [markdown]
# The named children include notes and measures. The non-empty samples show one
# event from each of those children; their positions are exact, unit-bearing
# coordinates. Open
# [Loading Real Data](tut04_loading_data.ipynb) to choose a loader and
# understand what it creates.

# %% [markdown]
# ## Timeline Groups
#
# A {{< glossary TimelineGroup >}} relates commensurable timelines so that a
# position on one can be transferred to another. Here their full extents are
# treated as corresponding.

# %%
group = TimelineGroup(id="quickstart", timelines=[audio, score])
score_coordinate = group.get_coordinate_at(
    timestamp_coordinates["audio parent"],
    timeline_id=score.id,
)
coordinate_pair = {
    "audio": timestamp_coordinates["audio parent"],
    "score": score_coordinate,
}
coordinate_pair

# %% [markdown]
# The dictionary shows what `convert` returns directly: the exact 25/2 seconds
# maps to `Fraction(415, 24)` quarters, and both values remain unit-bearing
# coordinates. Open
# [Timeline Groups](tut05_timeline_groups.ipynb) to control these relationships.

# %% [markdown]
# ## Alignment Bundles
#
# An {{< glossary AlignmentBundle >}} collects broader alignment evidence; its
# basic link is a {{< glossary MatchClaim >}}. A synchronous claim asserts that
# two unit-bearing positions correspond.

# %%
claim = MatchClaim.from_events(
    {"start": coordinate_pair["audio"].value},
    audio.id,
    {"start": coordinate_pair["score"].value},
    score.id,
    unit_a=audio.unit,
    unit_b=score.unit,
)
claim

# %% [markdown]
# The rendered claim shows the exact, unit-bearing positions `@25/2 seconds`
# and `@415/24 quarters`, with the `ANCHOR` tag marking a claim built from bare
# coordinates, as explained with other claim types in the alignment tutorial. A
# bundle can combine this claim with wider alignment evidence. Open
# [Alignment Bundles](tut06_alignment_bundles.ipynb) to assemble and query
# larger alignments.

# %% [markdown]
# ## Repeat Unfolding
#
# A {{< glossary FlowMap >}} can unfold a printed passage that is played more
# than once. Here the written opening from quarter 0 up to, but not including,
# quarter 4 is played twice.

# %%
opening_span = (Fraction(0), Fraction(4))
played_spans = [opening_span, opening_span]
repeat_map = score.create_flow_map(played_spans, id="quickstart-repeat")
folded_coordinate = score.make_coordinate(Fraction(2))
unfolded_values = repeat_map.unfold_coordinate(folded_coordinate.value)
unfolded_coordinates = [score.make_coordinate(value) for value in unfolded_values]
flow_preview = {"written": folded_coordinate, "played": unfolded_coordinates}
flow_preview

# %% [markdown]
# The written position at two quarters appears at two and six quarters in the
# played order. This cell previews only repeat unfolding; beat grids are a
# separate subject. Open
# [Flow and Grids](tut07_flow_and_grids.ipynb) to model complete playthroughs
# and to ask where measures and beats fall in a recording.

# %% [markdown]
# ## The Data Model
#
# Arrow is the column-oriented table storage used underneath the library: it
# keeps many event values compact and consistent. Asking a typed column for its
# `start` field reconstructs the library object represented in that storage.

# %%
first_start = score_notes.get_field("start")[0]
first_start

# %% [markdown]
# The loaded note's first start value is an exact `Coordinate` in quarters, not
# a bare number. Open [The Data Model](tut08_data_model.ipynb) to understand
# fields, scalars, and their tabular representation.

# %% [markdown]
# ## Pitch and Harmony
#
# MIDI pitch can leave a black key's spelling open, while notation can preserve
# the written accidental. TimeToAlign! keeps those meanings distinct.

# %%
enharmonic_pitch = EnharmonicPitch(61)
spelled_pitch = SpecificPitch.from_string("C♯4")
pitch_pair = {"enharmonic": enharmonic_pitch, "spelled": spelled_pitch}
pitch_pair

# %% [markdown]
# The first pitch allows C♯ or D♭; the second records C♯ specifically. Open
# [Pitch and Harmony](tut09_pitch_and_harmony.ipynb) to work with pitch
# representations, spelling, and harmonic labels.

# %% [markdown]
# ## Series Map
#
# | Tutorial | Promise |
# |---|---|
# | 1. Timelines and Coordinates | Build a measured axis and place a coordinate on it. |
# | 2. Nesting and Timestamps | Read one position across nested timelines. |
# | 3. Events on a Timeline | Store and retrieve musical observations. |
# | 4. Loading Real Data | Turn a research file into timelines and events. |
# | 5. Timeline Groups | Transfer positions between commensurable timelines. |
# | 6. Alignment Bundles | Collect claims and query cross-timeline matches. |
# | 7. Flow and Grids | Unfold score order and ask metrical questions. |
# | 8. The Data Model | Recover typed library objects from columns. |
# | 9. Pitch and Harmony | Preserve pitch meaning and written spelling. |

# %% [markdown]
# ## What you learned
#
# - You can build a timeline and make a unit-bearing coordinate.
# - You can nest child timelines and read across them with a timestamp.
# - You can add events to a timeline and retrieve a non-empty selection.
# - You can load a score file into the same timeline model.
# - You can group timelines and transfer a coordinate between them.
# - You can express one match claim between exact, unit-bearing positions.
# - You can unfold one written position into repeated played positions.
# - You can retrieve a typed coordinate from an event-data field.
# - You can distinguish an enharmonic pitch from a specifically spelled pitch.
#
# ## Next
#
# Continue with [Timelines and Coordinates](tut01_timelines_and_coordinates.ipynb).
#
# ## Go deeper
#
# - [Coordinate arithmetic](../howto/how01_coordinate_math.ipynb)
# - [Advanced timestamps](../howto/how01_advanced_timestamps.ipynb)
# - [Loading data](../howto/how01_loading_data.ipynb)
# - [Flow control](../howto/how01_flow_control.ipynb)
# - [Creating a note alignment](../howto/how03_create_note_alignment.ipynb)
