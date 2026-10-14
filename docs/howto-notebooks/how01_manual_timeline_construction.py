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
# # How to Construct Timelines Manually
#
# Building timelines by hand, adding events via dictionaries, creating
# parent/child hierarchies, and serialisation.

# %%
from fractions import Fraction

from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteLogicalTimeline,
)

# %% [markdown]
# ## Creating Timelines

# %%
audio_tl = ContinuousPhysicalTimeline(length=10.0, uid="audio")
audio_tl

# %%
score_tl = ContinuousLogicalTimeline(length=16, uid="score")
score_tl_frac = ContinuousLogicalTimeline(length=Fraction(33, 2), uid="score_frac")

{"integer length": score_tl.length, "fraction length": score_tl_frac.length}

# %%
midi_tl = DiscreteLogicalTimeline(length=1920, uid="midi")
{"unit": midi_tl.unit, "domain": midi_tl.domain, "length": midi_tl.length}

# %% [markdown]
# ## Adding Events
#
# | Field | Required | Description |
# |-------|----------|-------------|
# | `id` | Yes | Unique identifier |
# | `temporal_type` | Yes | `"instant"` or `"interval"` |
# | `event_type` | Yes | e.g. `"Note"`, `"Beat"` |
# | `instant` | For instant | Coordinate value |
# | `start`, `end` | For interval | Coordinate values |

# %%
audio_tl.add_events(
    [
        {"id": "b1", "temporal_type": "instant", "event_type": "Beat", "instant": 0.0},
        {"id": "b2", "temporal_type": "instant", "event_type": "Beat", "instant": 0.5},
        {"id": "b3", "temporal_type": "instant", "event_type": "Beat", "instant": 1.0},
    ]
)

audio_tl.add_events(
    [
        {
            "id": "n1",
            "temporal_type": "interval",
            "event_type": "Note",
            "start": 0.0,
            "end": 0.4,
        },
        {
            "id": "n2",
            "temporal_type": "interval",
            "event_type": "Note",
            "start": 0.5,
            "end": 0.9,
        },
    ]
)

audio_tl.events.to_dataframe()[["id", "temporal_type", "event_type"]]

# %% [markdown]
# ### Event Validation

# %%
try:
    audio_tl.add_events(
        [
            {
                "id": "oob",
                "temporal_type": "instant",
                "event_type": "Beat",
                "instant": 15.0,
            }
        ]
    )
except ValueError as e:
    print(f"ValueError: {e}")

# %%
audio_tl.add_events(
    [{"id": "oob", "temporal_type": "instant", "event_type": "Beat", "instant": 15.0}],
    allow_expansion=True,
)
{"n_events": audio_tl.n_events, "length": audio_tl.length}

# %% [markdown]
# ### Filtering Events

# %%
beat_store = audio_tl.get_events(event_type="Beat")
interval_store = audio_tl.get_events(temporal_type="interval")
{"beats": len(beat_store), "intervals": len(interval_store)}

# %% [markdown]
# ## Child Timelines (Hierarchies)
#
# Children share the parent's unit, are placed at an offset, and are
# locked after being added.

# %%
parent = ContinuousPhysicalTimeline(length=100, uid="parent")

child1 = ContinuousPhysicalTimeline(length=20, uid="verse1")
child1.add_events(
    [
        {
            "id": "v1_start",
            "temporal_type": "instant",
            "event_type": "Marker",
            "instant": 0.0,
        },
        {
            "id": "v1_mid",
            "temporal_type": "instant",
            "event_type": "Marker",
            "instant": 10.0,
        },
    ]
)

child2 = ContinuousPhysicalTimeline(length=15, uid="chorus")
child2.add_events(
    [
        {
            "id": "ch_start",
            "temporal_type": "instant",
            "event_type": "Marker",
            "instant": 0.0,
        },
    ]
)

parent.add_child(child1, offset=10)
parent.add_child(child2, offset=50)

parent

# %%
for offset, child in parent.iter_children():
    print(f"  {child.id}: offset={offset.value}, length={child.length.value}")

# %% [markdown]
# A root timeline is unlocked unless it is built with `locked=True`, so it
# grows to hold whatever you add. `lock()` locks it after construction, the
# state `add_child` puts every child in: the timeline no longer grows, so
# events and children must fit inside its length. A child that would overrun
# it raises `ValueError`, and `append_child` refuses outright. A lock cannot
# be lifted.

# %%
locked_before = parent.is_locked
parent.lock()
try:
    parent.add_child(ContinuousPhysicalTimeline(length=40, uid="coda"), offset=80)
except ValueError as e:
    print(f"ValueError: {e}")
{"locked before": locked_before, "locked after": parent.is_locked}

# %% [markdown]
# ### Nested Hierarchies

# %%
piece = ContinuousPhysicalTimeline(length=600, uid="symphony")
movement1 = ContinuousPhysicalTimeline(length=300, uid="mov1")
section1a = ContinuousPhysicalTimeline(length=60, uid="exposition")
section1b = ContinuousPhysicalTimeline(length=90, uid="development")

movement1.add_child(section1a, offset=0)
movement1.add_child(section1b, offset=60)
piece.add_child(movement1, offset=0)

piece

# %% [markdown]
# ## Serialisation

# %%
tl = ContinuousPhysicalTimeline(length=10, uid="test")
tl.add_events(
    [
        {"id": "e1", "temporal_type": "instant", "event_type": "Beat", "instant": 0.0},
        {"id": "e2", "temporal_type": "instant", "event_type": "Beat", "instant": 5.0},
    ]
)

data = tl.to_dict(events=True)
{"keys": list(data.keys()), "events": len(data["events"])}

# %%
restored = ContinuousPhysicalTimeline.from_dict(data)
{"id": restored.id, "n_events": restored.n_events}
