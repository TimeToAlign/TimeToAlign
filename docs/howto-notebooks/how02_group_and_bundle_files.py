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
# # How to Save and Load Timelines, Groups and Bundles
#
# A `Timeline`, a `TimelineGroup` and an `AlignmentBundle` each write one
# TimeToShare manifest with `to_dict()` and read one with `from_dict()`.
# Every manifest has the same shape, whatever it holds, and every manifest
# is an alignment bundle:
#
# - a single timeline writes a bundle of one root timeline;
# - a group writes a bundle of one group;
# - a bundle writes any number of timelines and groups.
#
# A manifest that carries an `id` or a `name` is a named bundle; one with
# neither is an anonymous bundle.
#
# This guide writes each of the three, reads them back, checks that the
# rebuilt objects answer every query exactly as the originals do, selects one
# timeline or group out of a larger manifest, and saves the manifests as JSON
# files. It ends with a timeline whose length is not known.
#
# A manifest holds structure: timelines, their children, conversion maps,
# regions and flow control, the measure maps the conversion maps read, and
# the groups and their members, with the members' display order. MatchClaims
# are not part of it, nor is a bundle's support policy, which is a setting of
# the bundle. Events stay out unless you ask for them.

# %% [markdown]
# ## Setup
#
# Install the library with `pip install timetoalign`. Four small helpers keep
# the cells below short: `answers` lists what a stamp says about every
# timeline it reaches, `show` prints a manifest as the JSON text a file would
# hold, `outline` shows a manifest with each root timeline reduced to its id,
# and `save` / `load` write and read a JSON file.

# %%
import json
import tempfile
from fractions import Fraction
from pathlib import Path

from timetoalign import (
    AlignmentBundle,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscretePhysicalTimeline,
    Timeline,
    TimelineGroup,
    UnknownLengthError,
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


def show(data):
    """Print a manifest as JSON text."""
    print(json.dumps(data, indent=2))


def outline(data):
    """Show a manifest with each root timeline reduced to its id."""
    return {**data, "timelines": [timeline["id"] for timeline in data["timelines"]]}


def save(name, data):
    """Write a manifest as a JSON file, and return the path."""
    path = FILES / f"{name}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load(path):
    """Read a JSON file back into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


# %% [markdown]
# ## Write a timeline
#
# A score of sixteen quarters with one region.

# %%
score = ContinuousLogicalTimeline(length=Fraction(16), name="Score")
score.create_region(0, 8, name="First half")
score

# %% [markdown]
# `to_dict()` writes the **complete form**: every key the format defines,
# each written even when it holds its default (`null`, `{}`, `[]`). The
# manifest opens with the version marker `"timetoshare": "0.1.0"`; its
# `timelines` array holds the score as its one root.
#
# A value on the timeline's own axis (its `length`, a region's `start` and
# `end`) is written bare, in the axis's number type: quarters count in exact
# fractions, so sixteen quarters are the pair `[16, 1]`. The library's class
# is not written: the unit decides it.

# %%
score_data = score.to_dict()
show(score_data)

# %% [markdown]
# `to_dict(minimal=True)` writes the **minimal form**: every key holding its
# default is left out. `length` is the one exception, written in both forms.

# %%
score_minimal = score.to_dict(minimal=True)
show(score_minimal)

# %% [markdown]
# ## Read a timeline
#
# `Timeline.from_dict()` reads either form and supplies every default that
# is left out. It returns the class the unit decides, here a
# `ContinuousLogicalTimeline` for quarters. Both forms read to the same
# timeline: each writes back the complete form of the original, and its
# minimal form.

# %%
from_complete = Timeline.from_dict(score_data)
from_minimal = Timeline.from_dict(score_minimal)
(
    type(from_minimal).__name__,
    from_complete.to_dict() == from_minimal.to_dict() == score_data,
    from_minimal.to_dict(minimal=True) == score_minimal,
)

# %% [markdown]
# Because every default may be left out, a manifest is easy to write by
# hand. The smallest one names the version and one timeline with its `id`,
# `unit` and `length`. In a manifest of one timeline and no group,
# `timeline_id=` and `name=` override that timeline's id and name as it is
# read, unless a timeline of the manifest already has that id or name: such
# a timeline is selected instead, and an id and a name given together must
# then name the same timeline.

# %%
smallest = {
    "timetoshare": "0.1.0",
    "timelines": [{"id": "piece", "unit": "quarters", "length": [32, 1]}],
}
piece = Timeline.from_dict(smallest, name="Full score")
piece.id, piece.name, piece.length

# %% [markdown]
# A bare value must be written in its axis's number type; reading never
# coerces it. A length of `32` on a quarters axis is an integer where the
# axis counts fractions, and the error says what to write instead.

# %%
try:
    Timeline.from_dict(
        {
            "timetoshare": "0.1.0",
            "timelines": [{"id": "piece", "unit": "quarters", "length": 32}],
        }
    )
except ValueError as error:
    print(f"ValueError: {error}")

# %% [markdown]
# ## Write a group
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
# A group writes an anonymous bundle holding its group:
#
# - `timelines`: the root of every member's tree, sorted by id. The child
#   member arrives inside its root, the score, among the score's
#   `children`.
# - `groups`: the one group, with its `id`, its `name` and its `members`,
#   the member ids sorted so that two equal groups write equal text.
# - `graphical.order`: the members in the order they were added, keyed by
#   the group's id. This is the display order.
#
# The minimal form shows the structure without the defaults.

# %%
group_data = group.to_dict()
show(group.to_dict(minimal=True))

# %% [markdown]
# A child's `offset` is a bare value on its parent's axis, in its parent's
# number type: the second half starts at `[8, 1]` quarters into the score.
# `add_child()` locks a child unless it is given `locked=False`, so this one
# writes `"locked": true`. An unlocked child writes `false`, and reading keeps
# whichever the manifest states.

# %%
group_data["timelines"][0]["children"][0]["offset"]

# %% [markdown]
# `graphical` may also carry `styles` and `statements`, the TimeToShare
# styling. The library does not interpret them: `from_dict()` keeps them as
# the group's `styling` and `to_dict()` writes them back. In the complete
# form they appear at their defaults.

# %%
group_data["graphical"]

# %% [markdown]
# ## Read a group
#
# `TimelineGroup.from_dict()` rebuilds the timelines, resolves each member id
# among the roots and their descendants, and adds the members in their
# display order. The rebuilt group writes the same manifest and answers the
# same queries with the same coordinates, in the same number types.

# %%
rebuilt = TimelineGroup.from_dict(group_data)
(
    rebuilt.timeline_ids,
    rebuilt.to_dict() == group_data,
    answers(rebuilt.get_timestamps_at(positions, second_half.id)) == before,
)

# %% [markdown]
# ## Write a bundle
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
# A bundle writes the same shape. Its `timelines` array holds every root
# timeline of the bundle, so each timeline is written once; its `groups`,
# sorted by id, point into that one array; `graphical.order` holds each
# group's display order. The roots that no group names, here the annotation
# layer and the long recording, are the ungrouped timelines.
#
# This bundle has a `name`, so it writes a named bundle. It has no `id`: the
# bundle was given none, and the id it minted for itself is never written.
# `id` and `name` are independent; a manifest may carry either, both or
# neither.

# %%
bundle_data = bundle.to_dict()
outline(bundle.to_dict(minimal=True))

# %% [markdown]
# ## Read a bundle
#
# `AlignmentBundle.from_dict()` reads any manifest. It rebuilds the timelines
# and the groups, each group in its display order, and lets every root that
# no group names join ungrouped. The rebuilt bundle writes the same manifest
# and answers the same queries.

# %%
rebuilt_bundle = AlignmentBundle.from_dict(bundle_data)
(
    rebuilt_bundle.to_dict() == bundle_data,
    {
        timeline_id: answers(rebuilt_bundle.get_matchstamps_at(at, timeline_id))
        for timeline_id, at in bundle_queries.items()
    }
    == bundle_before,
)

# %% [markdown]
# ## Select one timeline or group from a larger manifest
#
# From a manifest that holds several timelines or any group,
# `Timeline.from_dict()` reads one timeline, which `timeline_id=` or
# `name=` (or both) select. The selected timeline may be a root or any
# descendant, here the excerpt, a child of the long recording.

# %%
excerpt_read = Timeline.from_dict(bundle_data, name="Excerpt")
excerpt_read

# %% [markdown]
# `TimelineGroup.from_dict()` selects one group the same way, by
# `group_id=` or `name=`, and reads it with the timelines its members need.

# %%
facsimile = TimelineGroup.from_dict(bundle_data, name="Facsimile")
facsimile.id, facsimile.timeline_ids

# %% [markdown]
# A selection must match exactly one timeline. Without a selection the read
# is refused; a selection that matches none lists the manifest's timelines to
# choose from.

# %%
for selection in ({}, {"name": "Coda"}):
    try:
        Timeline.from_dict(bundle_data, **selection)
    except ValueError as error:
        print(f"ValueError: {error}")

# %% [markdown]
# The other way round, `TimelineGroup.from_dict()` makes a group of the one
# timeline of a manifest when you give the group an id or a name.

# %%
TimelineGroup.from_dict(smallest, name="Piece alone").timeline_ids

# %% [markdown]
# ## Keep keys the library does not know
#
# A manifest may carry keys the format does not define, written by another
# application for its own use. The library keeps every such key on the object
# read from the JSON object it was found in, and writes it back in place.
# Here an editor stored its zoom on the manifest, a colour on the timeline
# and a note on a region.

# %%
annotated = {
    "timetoshare": "0.1.0",
    "editor": {"zoom": 2.5},
    "timelines": [
        {
            "id": "piece",
            "unit": "quarters",
            "length": [32, 1],
            "colour": "#c03030",
            "regions": [
                {"id": "intro", "start": [0, 1], "end": [4, 1], "source": "manual"}
            ],
        }
    ],
}
Timeline.from_dict(annotated).to_dict(minimal=True) == annotated

# %% [markdown]
# ## Save the files
#
# A manifest contains only JSON values, so `json` writes it as it is, and a
# file read back with `json` goes straight into `from_dict()`. Floats survive
# the trip as well: `json` writes `10.0` as `10.0`, and the recording's
# coordinates come back as the same floats.

# %%
group_path = save("second_half", group_data)
bundle_path = save("edition", bundle_data)
(
    TimelineGroup.from_dict(load(group_path)).to_dict() == group_data,
    AlignmentBundle.from_dict(load(bundle_path)).to_dict() == bundle_data,
    answers(
        TimelineGroup.from_dict(load(group_path)).get_timestamps_at(
            positions, second_half.id
        )
    )
    == before,
)

# %% [markdown]
# ## A timeline of unknown length
#
# Some timelines have no known end: a live stream still running, a recording
# whose duration has not been measured. `length=None` states that the length
# is not known. It is a fact about the timeline, never inferred from its
# content: adding content leaves it unknown.

# %%
stream = ContinuousPhysicalTimeline(length=None, name="Live stream")
stream.create_region(30.0, 45.0, name="Interview")
stream.add_child(ContinuousPhysicalTimeline(length=20.0, name="Clip"), offset=100.0)
stream.length, stream.end, stream.is_length_known

# %% [markdown]
# Its positions are not bounded above, so a query far beyond all content
# still resolves. The `reach` is the farthest coordinate the content
# reaches, here the clip's end at 120 seconds; the diagram is drawn to it
# and marks the unknown end `?`.

# %%
stream.get_timestamp_at(1000.0), stream.reach

# %%
stream

# %% [markdown]
# A manifest writes the unknown length as `null`, in both forms, and reading
# it gives back a timeline whose length is not known.

# %%
stream_data = stream.to_dict(minimal=True)
show(stream_data)
Timeline.from_dict(stream_data).is_length_known

# %% [markdown]
# Operations that need the end raise `UnknownLengthError`, a `ValueError`
# that names the operation and the timeline, before they change anything.
# A group converts positions in proportion to its members' lengths, so a
# conversion to or from a member of unknown length is one of them; appending
# a child at a parent's end is another.

# %%
programme = ContinuousLogicalTimeline(length=Fraction(16), name="Programme")
broadcast = TimelineGroup(name="Broadcast", timelines=[programme, stream])
for operation in (
    lambda: broadcast.get_timestamp_at(Fraction(4), programme.id),
    lambda: stream.append_child(ContinuousPhysicalTimeline(length=10.0)),
):
    try:
        operation()
    except UnknownLengthError as error:
        print(f"UnknownLengthError: {error}")

# %% [markdown]
# Once the length is known, setting it is an ordinary edit, and every
# operation is available again.

# %%
stream.length = 3600.0
stream.is_length_known, answers([broadcast.get_timestamp_at(Fraction(4), programme.id)])

# %% [markdown]
# ## Next
#
# - Events: `to_dict(events=True)` writes every timeline's event rows; see
#   [How to Construct Timelines Manually](how01_manual_timeline_construction.ipynb).
# - Groups in depth: the [Timeline Groups](../tutorials/tut05_timeline_groups.ipynb)
#   tutorial.
# - Alignments between groups: the
#   [Alignment Bundles and MatchClaims](../tutorials/tut06_alignment_bundles.ipynb)
#   tutorial.
