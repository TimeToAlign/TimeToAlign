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
# # How to Save and Load Groups and Bundles
#
# A `TimelineGroup` and an `AlignmentBundle` each turn into one plain
# dictionary with `to_dict()` and come back with `from_dict()`. This guide
# builds a group and a bundle, inspects the dictionaries they write, rebuilds
# them, checks that the rebuilt objects answer every query exactly as the
# originals do, and saves both as JSON files.
#
# The dictionaries hold structure only: timelines, their children, the groups
# and their members. MatchClaims are not part of a bundle's dictionary, and
# events stay out unless you ask for them.

# %% [markdown]
# ## Setup
#
# Three small helpers keep the cells below short: `answers` lists what a
# stamp says about every timeline it reaches, `outline` shows a dictionary
# with each timeline reduced to its id, and `save` / `load` write and read a
# file with its `meta`.

# %%
import json
import tempfile
from fractions import Fraction
from pathlib import Path

import timetoalign
from timetoalign import (
    AlignmentBundle,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscretePhysicalTimeline,
    TimelineGroup,
)

FILES = Path(tempfile.mkdtemp())


def answers(stamps):
    """Map each stamp to its coordinates, one per timeline it reaches."""
    return [
        {
            timeline_id: stamp.get_coordinate_for(timeline_id)
            for timeline_id in stamp.present_timelines
        }
        for stamp in stamps
    ]


def outline(data):
    """Show a dictionary with each root timeline reduced to its id."""
    return {**data, "timelines": [timeline["id"] for timeline in data["timelines"]]}


def save(name, data):
    """Write a dictionary as a JSON file with its meta, and return the path."""
    path = FILES / f"{name}.json"
    meta = {
        "schema_version": "0.7.0",
        "generator": f"timetoalign {timetoalign.__version__}",
    }
    path.write_text(json.dumps({"meta": meta, **data}, indent=2), encoding="utf-8")
    return path


def load(path):
    """Read a JSON file back into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


# %% [markdown]
# ## Build a group
#
# A group's members share one start and one end: every member's 0 coincides
# with every other member's 0, and every member's length with every other
# member's length. Positions in between convert linearly and exactly.
#
# Here only the second half of a score takes part, alongside a recording and
# a scanned page. The second half is a child of the score, placed at its
# exact offset, and the child is the member. The three members carry three
# number types: quarters as fractions, seconds as floats, pixels as integers.
# They are added page first, so their display order differs from their id
# order.

# %%
score = ContinuousLogicalTimeline(length=Fraction(16), name="Score")
second_half = ContinuousLogicalTimeline(length=Fraction(8), name="Second half")
score.add_child(second_half, offset=Fraction(8))
recording = ContinuousPhysicalTimeline(length=10.0, name="Recording")
page = DiscreteGraphicalTimeline(length=1920, name="Page")

group = TimelineGroup(name="Second half", timelines=[page, second_half, recording])
group.id, group.timeline_ids

# %% [markdown]
# One third of a quarter into the second half is
# `1/3 · 10/8 = 5/12` seconds into the recording and `1/3 · 1920/8 = 80`
# pixels into the page. The conversion is computed exactly, and each answer
# is then written in its member's own number type: a Fraction of quarters,
# an integer of pixels, and for the recording the float nearest to 5/12,
# `0.4166666666666667`.

# %%
positions = [Fraction(0), Fraction(1, 3), Fraction(3), Fraction(8)]
before = answers(group.get_timestamps_at(positions, second_half.id))
before

# %% [markdown]
# ## Write the group as a dictionary
#
# `to_dict()` returns a dictionary with three keys:
#
# - `timelines`: the root of every member's timeline tree, each exactly what
#   `Timeline.to_dict()` writes, sorted by id. The child member arrives inside
#   its root, the score.
# - `group`: the group's `id`, `name`, `meta`, and its `members`, the member
#   ids sorted.
# - `graphical`: `order`, the members in the order they were added. This is
#   the display order; `members` is sorted so that two equal groups write
#   equal text.

# %%
group_data = group.to_dict()
outline(group_data)

# %% [markdown]
# Every root timeline is written in full; the score's entry holds the
# second half among its `children`, keyed by the child's id. Its offset is a
# coordinate dict whose value is the exact Fraction `[8, 1]`, written as
# `[numerator, denominator]`.

# %%
group_data["timelines"][0]["children"][second_half.id]["offset"]

# %% [markdown]
# ## Read the group back
#
# `TimelineGroup.from_dict()` rebuilds the timelines, resolves each member id
# among the roots and their descendants, and adds the members in their
# display order. Writing the rebuilt group gives the same dictionary.

# %%
rebuilt = TimelineGroup.from_dict(group_data)
rebuilt.timeline_ids, rebuilt.to_dict() == group_data

# %% [markdown]
# The rebuilt group answers the same queries with the same coordinates, in
# the same number types.

# %%
answers(rebuilt.get_timestamps_at(positions, second_half.id)) == before

# %% [markdown]
# ## Build a bundle
#
# A bundle holds timelines and the groups among them. This one has two
# groups and timelines that belong to none:
#
# - **Performance**: a score and a performance recording;
# - **Facsimile**: a page and an excerpt of a long recording, the excerpt
#   being a child of that recording;
# - ungrouped: an annotation layer, and the long recording itself. When a
#   child joins a group, its root joins the bundle too, ungrouped unless you
#   group it.
#
# `as_group=` names a new group; its id is minted (`group_1`, `group_2`, ...).

# %%
score = ContinuousLogicalTimeline(length=Fraction(12), name="Score")
performance = ContinuousPhysicalTimeline(length=12.5, name="Performance")
page = DiscreteGraphicalTimeline(length=1920, name="Page")
long_recording = DiscretePhysicalTimeline(length=441000, name="Recording")
excerpt = DiscretePhysicalTimeline(length=88200, name="Excerpt")
long_recording.add_child(excerpt, offset=44100)
annotations = ContinuousLogicalTimeline(length=Fraction(12), name="Annotations")

bundle = AlignmentBundle(name="Edition")
bundle.add_timeline(performance, as_group="Performance")
bundle.add_timeline(score, grouped_with=performance.id)
bundle.add_timeline(page, as_group="Facsimile")
bundle.add_timeline(excerpt, grouped_with=page.id)
bundle.add_timeline(annotations)

{group_id: bundle.get_group(group_id).timeline_ids for group_id in bundle.group_ids}

# %% [markdown]
# Within a group, a MatchStamp converts linearly, as the group does. A
# position on the excerpt also reaches its root through the child's offset.

# %%
bundle_queries = {
    score.id: [Fraction(0), Fraction(3), Fraction(12)],
    excerpt.id: [0, 22050, 88200],
}
bundle_before = {
    timeline_id: answers(bundle.get_matchstamps_at(at, timeline_id))
    for timeline_id, at in bundle_queries.items()
}
bundle_before

# %% [markdown]
# ## Write the bundle as a dictionary
#
# A bundle's dictionary has one `timelines` array holding every root
# timeline of the bundle, so each timeline is written once. `bundle` carries
# the bundle's `id`, `name`, `meta`, its `support_policy`, and its `groups`,
# sorted by id, whose `members` point into that one array. The roots that no
# group names, here the annotation layer and the long recording, are the
# ungrouped timelines. `graphical.order` holds each group's display order.

# %%
bundle_data = bundle.to_dict()
outline(bundle_data)

# %% [markdown]
# ## Read the bundle back
#
# `AlignmentBundle.from_dict()` rebuilds the timelines and the groups, each
# group in its display order, and lets every root that no group names join
# ungrouped. The rebuilt bundle writes the same dictionary and answers the
# same queries.

# %%
rebuilt_bundle = AlignmentBundle.from_dict(bundle_data)
rebuilt_bundle.to_dict() == bundle_data

# %%
{
    timeline_id: answers(rebuilt_bundle.get_matchstamps_at(at, timeline_id))
    for timeline_id, at in bundle_queries.items()
} == bundle_before

# %% [markdown]
# ## Save the files
#
# The dictionaries contain only JSON values (a Fraction is a
# `[numerator, denominator]` pair, a coordinate a coordinate dict
# `{unit, number_type, value}`), so `json` writes them as they are. The file
# adds a `meta` object naming the TimeSkeleton schema version, which the
# writer supplies. With it, the files are a TimeSkeleton group file and a
# TimeSkeleton bundle file.

# %%
group_path = save("second_half", group_data)
bundle_path = save("edition", bundle_data)
load(group_path)["meta"]

# %% [markdown]
# `from_dict()` accepts the `meta` and ignores it, so a file read back with
# `json` goes straight in.

# %%
(
    TimelineGroup.from_dict(load(group_path)).to_dict() == group_data,
    AlignmentBundle.from_dict(load(bundle_path)).to_dict() == bundle_data,
)

# %% [markdown]
# Floats survive the trip as well: `json` writes `10.0` as `10.0`, and a
# float coordinate comes back as the same float.

# %%
(
    answers(
        TimelineGroup.from_dict(load(group_path)).get_timestamps_at(
            positions, second_half.id
        )
    )
    == before
)

# %% [markdown]
# ## Next
#
# - Events: `to_dict(events=True)` passes on to every root's
#   `Timeline.to_dict()`; see
#   [How to Construct Timelines Manually](how01_manual_timeline_construction.ipynb).
# - Groups in depth: the [Timeline Groups](../tutorials/tut05_timeline_groups.ipynb)
#   tutorial.
# - Alignments between groups: the
#   [Alignment Bundles and MatchClaims](../tutorials/tut06_alignment_bundles.ipynb)
#   tutorial.
