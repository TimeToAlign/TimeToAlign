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
# # How to Encode and Play Flow Control
#
# A score prints a passage once and has it played several times: repeat
# signs, alternative endings, a *Dal Segno* and a *Fine* say in which order.
# A {{< glossary Timeline >}} holds these signs as its **flow control**, a list
# of {{< glossary "Flow control element" >}}s of five classes:
#
# | Class | When the flow plays up to it |
# |---|---|
# | {{< glossary Jump >}} | one of its destinations that holds is taken |
# | {{< glossary JumpTarget >}} | nothing: it is a place a destination names |
# | {{< glossary Break >}} | the flow does not go on across it |
# | {{< glossary SectionBreak >}} | the flow goes on; a section ends |
# | {{< glossary Barline >}} | nothing: a barline the source prints |
#
# The library plays the flow control into a **flow**, the stretches of the
# axis in the order played, and reports what it finds on the way. It writes
# flows as {{< glossary FlowMap >}}s, which name stretches of the timeline in
# the order played, and converts positions between the timeline and a
# {{< glossary "Virtual timeline" >}}, the axis of a FlowMap.
#
# This guide encodes a short piece by hand, plays it, breaks it and repairs
# it, plays a work whose movement states its own repeat, loads a real score,
# writes its flows, writes a FlowMap of its own, and saves everything in a
# manifest.
#
# **Prerequisites:** [How to Construct Timelines Manually](how01_manual_timeline_construction.ipynb)
# and [How to Save and Load Timelines, Groups and Bundles](how02_group_and_bundle_files.ipynb).

# %% [markdown]
# ## Setup
#
# `q()` writes a position in quarters. `elements()` shows a timeline's flow
# control as a table: each element's ID, class (`role`), coordinate and what
# else it states, as a manifest writes it (a Fraction as `[numerator,
# denominator]`), and whether it is rejected. `in_measures()` states a flow as
# right-open ranges of measure counts, `placed()` shows where each stretch of a
# FlowMap lies on its virtual axis.

# %%
import tempfile
from pathlib import Path

import pandas as pd
from timetoshare import dumps

from timetoalign import Coordinate, Gap, Ms3Loader, Timeline
from timetoalign.testdata import ensure_data
from timetoalign.timelines import ContinuousLogicalTimeline
from timetoalign.timelines.flow import (
    Barline,
    Break,
    DestinationReference,
    FlowCondition,
    Jump,
    JumpDestination,
    JumpTarget,
    PlacedChunk,
    SectionBreak,
    atomic_sections,
    measure_ranges,
    play,
)

pd.set_option("display.max_colwidth", None)

EDITOR = "https://example.org/people/editor"
READER = "https://example.org/people/reader"
FILES = Path(tempfile.mkdtemp())


def q(value):
    """A position in quarters."""
    return Coordinate(value, "quarters")


def elements(registry):
    """One row per flow-control element, as a manifest writes it."""
    rows = []
    for element in registry.elements:
        written = element.to_dict()
        states = {
            key: value
            for key, value in written.items()
            if key not in ("id", "role", "coordinate", "provenance")
        }
        rows.append(
            {
                "id": element.id,
                "role": element.role,
                "coordinate": str(element.coordinate),
                "states": states,
                "rejected": element.is_rejected,
            }
        )
    return pd.DataFrame(rows).set_index("id")


def in_measures(flow, timeline):
    """The stretches of a flow as right-open ranges of measure counts."""
    sources = [chunk.source for chunk in flow.chunks]
    return [
        f"[{row.start}, {row.end})"
        for row in measure_ranges(sources, timeline.measure_map)
    ]


def placed(resolved):
    """What each stretch of a resolved FlowMap plays and where it lies."""
    return pd.DataFrame(
        {
            "plays": (
                str(chunk.source)
                if isinstance(chunk, PlacedChunk)
                else f"a gap of {chunk.length}"
            ),
            "from": str(chunk.start),
            "to": str(chunk.end),
        }
        for chunk in resolved.chunks
    )


# %% [markdown]
# ## The piece
#
# Twelve measures of 4/4, 48 quarters, in two sections. Measure *n* lies
# from quarter 4(*n* − 1) to 4*n*.
#
# | Where | Sign |
# |---|---|
# | start of m. 2 (quarter 4) | segno |
# | start of m. 4 (12) | repeat start |
# | m. 5 (16 to 20) | first bracket, with a repeat end at its end |
# | m. 6 (20 to 24) | second bracket |
# | end of m. 6 (24) | *Fine*; a repeat start at the start of m. 7 |
# | end of m. 8 (32) | repeat end and *D.S. al Fine* on one barline; the first section ends |
# | end of m. 12 (48) | repeat end without a repeat start; final barline |

# %%
score = ContinuousLogicalTimeline(length=48, uid="piece", name="Piece")
score

# %% [markdown]
# ## The places a destination names
#
# A {{< glossary JumpTarget >}} is a place: nothing happens when the flow
# plays up to it. Its `manifestation` says what it looks like in the source
# and decides nothing. A given `id` is kept; the IDs here make the
# destinations below readable.

# %%
places = [
    JumpTarget(q(4), id="segno", manifestation="segno"),
    JumpTarget(q(12), id="repeat_start", manifestation="repeat_start"),
    JumpTarget(q(20), id="second_bracket", manifestation="volta"),
    JumpTarget(q(24), id="coda_repeat_start", manifestation="repeat_start"),
]

# %% [markdown]
# ## Alternative endings
#
# A {{< glossary Jump >}} lies where the flow may leave the axis and holds one
# {{< glossary Destination >}} for every place it may go on at. The brackets
# are one Jump before them, at the start of measure 5:
#
# - `by=0, passes=[1]`: on the first arrival the flow goes on into the first
#   bracket. `by: 0` is never taken; it is written where the source prints
#   the bracket's number.
# - `to="second_bracket", passes=[2]`: on the second arrival it goes on at the
#   second bracket.
#
# A {{< glossary Break >}} between the brackets keeps the flow from running
# from the first bracket into the second. At the end of the first bracket a
# second Jump holds the repeat end: back to the repeat start, once
# (`repeat_count=1`).

# %%
brackets = [
    Jump(
        q(16),
        [
            JumpDestination(by=0, passes=[1], manifestation="volta"),
            JumpDestination(to="second_bracket", passes=[2], manifestation="volta"),
        ],
        id="brackets",
    ),
    Break(q(20), id="between_brackets", manifestation="volta"),
    Jump(
        q(20),
        [
            JumpDestination(
                to="repeat_start", repeat_count=1, manifestation="repeat_end"
            )
        ],
        id="first_bracket_end",
    ),
]

# %% [markdown]
# ## Several signs on one barline, and a *Fine*
#
# The barline at the end of measure 8 holds a repeat end and a *D.S. al
# Fine*: one Jump with two destinations. Both lie before the Jump. A
# destination that goes back with `reset=True` (the default, a repeat) is
# taken before one that goes back without it (a leap), so the repeat is taken
# first; once it is used up the *D.S.* holds. `reset=False` says that the
# leap restores nothing it goes back over: the repeats inside are not played
# again.
#
# The *Fine* is a Break with `ends`: it ends the flow only after the *D.S.*
# has been taken. `after` names that destination by its Jump and its place.

# %%
end_of_first_section = Jump(
    q(32),
    [
        JumpDestination(
            to="coda_repeat_start", repeat_count=1, manifestation="repeat_end"
        ),
        JumpDestination(
            to="segno",
            repeat_count=1,
            reset=False,
            manifestation="dal_segno_al_fine",
        ),
    ],
    id="end_of_first_section",
)
fine = Break(
    q(24),
    id="fine",
    manifestation="fine",
    ends=FlowCondition(
        after=DestinationReference(jump="end_of_first_section", to="segno")
    ),
)
{
    "destinations": [
        destination.to_dict() for destination in end_of_first_section.targets
    ],
    "fine ends": fine.ends.to_dict(),
}

# %% [markdown]
# ## Sections
#
# A {{< glossary SectionBreak >}} ends a section and lets the flow go on. A
# destination without a place goes on at the nearest section break before its
# Jump, never before the start of the timeline stating the Jump (here the
# score's 0), so the repeat end at measure 12 repeats the second section only. A
# flow ended by a Break begins again at the next section break: after the
# *Fine*, the second section is played. A {{< glossary Barline >}} records the
# final barline and changes nothing.

# %%
second_section = [
    SectionBreak(q(32), id="first_section_end"),
    Jump(
        q(48),
        [JumpDestination(repeat_count=1, manifestation="repeat_end")],
        id="second_section_end",
    ),
    Barline(q(48), manifestation="final_barline"),
]

# %% [markdown]
# ## Adding the elements
#
# `flow_control.add_all()` adds elements that name one another together, or
# none of them. The registry refuses what cannot be played: a `to` naming
# nothing, a second Jump at one coordinate (of the timeline, and of its
# root's axis over the root's whole tree), an element outside the axis. It
# holds the elements in canonical order, by coordinate and then by ID; the
# Barline, given no ID, gets a minted one.

# %%
score.flow_control.add_all(
    [*places, *brackets, end_of_first_section, fine, *second_section]
)
elements(score.flow_control)

# %% [markdown]
# ## Playing the flow control
#
# `get_flow()` plays the flow control and writes nothing. It returns the
# stretches of the axis in the order played, placed one after the other. Three
# conventions name three flows:
#
# - `default`: every Jump and Break acts as its conditions say;
# - `single`: once through, nothing goes back, each Jump arrived at on the
#   pass it would have after the repeats around it;
# - `printed`: the axis from 0 to its end.

# %%
flows = {
    convention: score.get_flow(convention)
    for convention in ("default", "single", "printed")
}
{
    convention: {
        "stretches": [str(chunk.source) for chunk in flow.chunks],
        "length": flow.length,
    }
    for convention, flow in flows.items()
}

# %% [markdown]
# Read the `default` flow sign by sign. The flow plays up to the brackets'
# Jump, goes on into the first bracket and at its end takes the repeat end
# back to quarter 12. On the second arrival at the brackets it goes on at the
# second bracket, then through measures 7 and 8. At quarter 32 the repeat end
# is taken back to 24, then the *D.S.* back to the segno. The leap restored
# nothing, so the brackets' Jump is arrived at a third time: beyond the
# highest pass any of its destinations lists, the destination listing it
# holds, and the second bracket is played again. At the *Fine* the flow ends,
# and begins again at the section break at 32, where the second section is
# played twice.
#
# `single` plays measures 1 to 4, skips the first bracket, and plays the rest
# once. The default flow ends only where the *Fine* says so and plays every
# stretch of the axis, so it has no findings. Each flow carries its own:
# `single` reports the first bracket it leaves out, with the cause
# `single_playback` (going through once took the last bracket, while the
# default flow plays it); `printed` plays everything and finds nothing.

# %%
{
    convention: [
        (
            str(finding.kind),
            str(finding.coordinate),
            str(finding.end),
            str(finding.cause),
        )
        for finding in flow.findings
    ]
    for convention, flow in flows.items()
}

# %% [markdown]
# A flow converts positions both ways. `unfold()` gives every position of
# the flow at which a position of the score is played; quarter 14 (measure 4)
# is played three times. `fold()` gives the position of the score played at a
# position of the flow.

# %%
default = flows["default"]
{"14 is played at": default.unfold(14), "54 plays": default.fold(54)}

# %% [markdown]
# ## Atomic sections
#
# The {{< glossary "Atomic section" >}}s are the stretches flow control never
# divides: between consecutive coordinates of Jumps, JumpTargets, Breaks and
# SectionBreaks and the places their destinations go on at, named by letters
# in the order of the axis. The engine functions `atomic_sections()`
# and `play()` take a registry and a length and need no timeline.

# %%
[
    (section.name, str(section.start), str(section.end))
    for section in atomic_sections(score.flow_control, score.length)
]

# %% [markdown]
# ## Conditions nobody can check
#
# `determined=False` marks a destination nobody can check, such as an ossia:
# it is never taken, and passing it is a {{< glossary Finding >}}. Here an
# ossia is added to a copy of the registry; the copy belongs to no timeline,
# so `play()` gets the length.

# %%
with_ossia = score.flow_control.copy()
with_ossia.add(
    Jump(q(40), [JumpDestination(by=4, determined=False, manifestation="ossia")])
)
played_with_ossia = play(with_ossia, score.length)
{
    "same runs as the score": played_with_ossia.runs
    == play(score.flow_control, score.length).runs,
    "findings": [finding.message for finding in played_with_ossia.findings],
}

# %% [markdown]
# ## A broken encoding
#
# An encoder put the repeat end of the first bracket at the end of the second
# bracket (quarter 24). The first bracket now ends at the Break between the
# brackets without a sign that leads anywhere.

# %%
draft = ContinuousLogicalTimeline(length=48, uid="draft", name="Draft")
misplaced = Jump(
    q(24),
    [JumpDestination(to="repeat_start", repeat_count=1, manifestation="repeat_end")],
)
draft.flow_control.add_all(
    [
        *[
            element
            for element in score.flow_control.elements
            if element.id != "first_bracket_end"
        ],
        misplaced,
    ]
)
broken = draft.get_flow()
{
    "stretches": [str(chunk.source) for chunk in broken.chunks],
    "findings": [finding.message for finding in broken.findings],
}

# %% [markdown]
# The flow ends at the Break at 20, begins again at the section break, and
# never plays quarters 20 to 32. Both are findings of the `default` flow and
# have no cause: the encoding, not a convention, leaves the stretch out.

# %% [markdown]
# ## Repairing it
#
# An element is never changed or deleted to repair flow control: the wrong
# one is rejected by an actor's belief `"false"`, and the right one is added.
# The rejected Jump stays in the registry and is written, but is not played.
# The added Jump states who made it in its provenance.

# %%
draft.flow_control.with_belief(
    misplaced.id,
    EDITOR,
    "false",
    date="2026-11-25",
    note="The repeat end belongs to the end of the first bracket.",
)
draft.flow_control.add(
    Jump(
        q(20),
        [
            JumpDestination(
                to="repeat_start", repeat_count=1, manifestation="repeat_end"
            )
        ],
        provenance={"method": "manual", "stated_by": [EDITOR]},
    )
)
elements(draft.flow_control)

# %% [markdown]
# The rejected Jump keeps its belief in its provenance:

# %%
draft.flow_control.jump_at(24).to_dict()["provenance"]

# %% [markdown]
# The repaired draft plays the score's flow:

# %%
repaired = draft.get_flow()
{
    "same stretches as the score": [chunk.source for chunk in repaired.chunks]
    == [chunk.source for chunk in default.chunks],
    "findings": repaired.findings,
}

# %% [markdown]
# A rejection is the belief of the actor who records it. Without a believer,
# every element some actor rejects is left out. Named as `believer`, one
# actor's beliefs alone decide: a reader who has not decided on the editor's
# repair reads both Jumps and gets another flow.

# %%
{
    "no believer": draft.get_flow().length,
    "the editor": draft.get_flow(believer=EDITOR).length,
    "another reader": draft.get_flow(believer=READER).length,
}

# %% [markdown]
# ## A work and its movements
#
# Flow control holds across descendants: a movement states its own repeat
# signs, the work states what concerns the whole, and the work is played
# with both. A sonata of two movements of 16 quarters each; the second, at
# offset 16, states a repeat end without a place at its quarter 8:

# %%
sonata = ContinuousLogicalTimeline(length=32, uid="sonata", name="Sonata")
first = sonata.create_child(16, 0, uid="first", name="First movement")
second = sonata.create_child(16, 16, uid="second", name="Second movement")
second.flow_control.add(
    Jump(
        q(8),
        [JumpDestination(repeat_count=1, manifestation="repeat_end")],
        id="second_repeat",
    )
)
{
    "stated by the sonata": sonata.flow_control.elements,
    "played on the sonata": sonata.tree_flow_control.elements,
}

# %% [markdown]
# `tree_flow_control` is a `FlowControlView`: the elements of
# the timeline and of every descendant, each at its position on the
# timeline's axis (its coordinate plus the offsets down to it), read only and
# under their IDs. The elements stay stored on the timeline stating them.
# `get_flow()` plays this view: the sonata its whole tree, a movement its own
# subtree.
#
# The repeat end has no place, so it goes back to the nearest section break
# before it on the played axis, stated by any timeline of the tree, but never
# before the start of the timeline stating it. No section break lies between
# 16 and 24, so the sonata goes back to the start of the second movement,
# not to its own 0, as the movement played alone goes back to its own 0:

# %%
{
    "the sonata": [str(chunk.source) for chunk in sonata.get_flow().chunks],
    "the second movement": [str(chunk.source) for chunk in second.get_flow().chunks],
}

# %% [markdown]
# One Jump lies at a position of the root's axis over the root's whole tree.
# The sonata cannot state a second Jump at 24, where the movement's repeat
# end lies; several signs at one place are destinations of one Jump:

# %%
try:
    sonata.flow_control.add(Jump(q(24), [JumpDestination(by=4)], id="skip"))
except ValueError as e:
    print(f"ValueError: {e}")

# %% [markdown]
# ## A loaded score
#
# A score loader writes the signs of its source as flow control. Beethoven's
# String Quartet Op. 18 no. 4, fourth movement, from its measure table:

# %%
op18_dir = ensure_data("score") / "beethoven_op18-4iv_multimodal" / "op18_no4_mov4_flow"
loader = Ms3Loader.from_file(op18_dir / "op18_no4_mov4_flow.measures.tsv")
rondo = loader.create_timeline(uid="op18_4_iv")
rondo.flow_control

# %% [markdown]
# The loader mints the IDs, `jump_N`, `jump_target_N`, `break_N` and so on.
# The movement's first brackets, with the repeat start their repeat end goes
# back to:

# %%
elements(rondo.flow_control).loc[
    ["jump_target_3", "jump_4", "break_1", "jump_5", "jump_target_4"]
]

# %% [markdown]
# The default flow, in measure counts, and its findings:

# %%
rondo_flow = rondo.get_flow()
{
    "measures played": in_measures(rondo_flow, rondo),
    "findings": rondo_flow.findings,
}

# %% [markdown]
# The measure table also states the order in which its measures are played
# (its `next` column). `create_stated_flow_map()` writes that order as a
# FlowMap over Regions of the timeline and compares it with the default flow:
# `differs_at` is None where the two agree throughout.

# %%
stated = loader.create_stated_flow_map(rondo)
{
    "FlowMap": stated.flow_map,
    "Regions": len(stated.regions),
    "differs_at": stated.differs_at,
}

# %% [markdown]
# ## Writing the flows
#
# `create_flow_maps_from_flow_control()` writes the atomic sections as
# Regions of the timeline, each named by its letter, and the flows as
# FlowMaps listing them: `default`, then `single` and `printed` where they
# differ from it. Each FlowMap names its convention and is computed: written
# again from the flow control whenever this is called.

# %%
written = score.create_flow_maps_from_flow_control()
section_names = {section.id: section.name for section in written.sections}
pd.DataFrame(
    {
        "FlowMap": flow_map.id,
        "convention": str(flow_map.convention),
        "sections played": "".join(section_names[chunk] for chunk in flow_map.chunks),
        "computed": flow_map.is_computed,
    }
    for flow_map in written.flow_maps
)

# %% [markdown]
# The Regions are ordinary Regions, which carry the statement that they are
# atomic sections:

# %%
[
    (section.id, section.name, str(section.as_interval), section.is_atomic_section)
    for section in written.sections
]

# %% [markdown]
# The loaded movement has thirteen atomic sections:

# %%
rondo_written = rondo.create_flow_maps_from_flow_control()
rondo_names = {section.id: section.name for section in rondo_written.sections}
{
    str(flow_map.convention): "".join(rondo_names[chunk] for chunk in flow_map.chunks)
    for flow_map in rondo_written.flow_maps
}

# %% [markdown]
# ## A FlowMap written by hand
#
# A FlowMap names stretches of a root timeline in the order played: Regions,
# child timelines and virtual timelines by ID or as objects, and
# {{< glossary Gap >}}s, stretches in which nothing of the timeline is played.
# An encore of the second section, four quarters of silence, and the first
# four measures:

# %%
score.create_region(32, 48, id="second_section", name="Second section")
score.create_region(0, 16, id="opening", name="Opening")
encore = score.create_flow_map(
    ["second_section", Gap(4), "opening"], id="encore", name="Encore"
)
encore

# %% [markdown]
# `resolve()` reads the FlowMap against its timeline and places each stretch
# where the one before it ends:

# %%
resolved = encore.resolve(score)
placed(resolved)

# %% [markdown]
# A position of the score that the FlowMap does not play unfolds to no
# position; a position inside the gap folds to None.

# %%
{
    "2 is played at": resolved.unfold(2),
    "20 is played at": resolved.unfold(20),
    "18 plays": resolved.fold(18),
    "22 plays": resolved.fold(22),
}

# %% [markdown]
# ## Virtual timelines
#
# A {{< glossary "Virtual timeline" >}} states that a FlowMap was applied to
# its timeline. It has an ID, the timeline's unit and number type, and the
# FlowMap's length, and holds nothing else: no events, no children. Positions
# convert through it in both directions, given the timeline it comes from.

# %%
as_played = score.create_virtual_timeline(
    written.flow_maps[0], id="as_played", name="As played"
)
encore_as_played = score.create_virtual_timeline(encore, id="encore_as_played")
{
    "as played": as_played,
    "14 is played at": as_played.unfold(14, score),
    "60 plays": as_played.fold(60, score),
    "the encore": encore_as_played,
}

# %% [markdown]
# ## Saving and reading the manifest
#
# A manifest writes the flow control in the timeline's `flow_control`, and
# the FlowMaps and virtual timelines at its top level. The first section's
# last barline, as written:

# %%
path = score.to_file(FILES / "piece.yaml")
manifest = score.to_dict()
timeline_entry = manifest["timelines"][0]
print(
    dumps(
        [
            element
            for element in timeline_entry["flow_control"]
            if element["coordinate"] == [32, 1]
        ]
    ),
    end="",
)

# %% [markdown]
# A FlowMap lists its chunks; the IDs in a file begin with `#`.

# %%
print(
    dumps(
        {
            "flow_maps": [m for m in manifest["flow_maps"] if m["id"] == "#encore"],
            "virtual_timelines": manifest["virtual_timelines"],
        }
    ),
    end="",
)

# %% [markdown]
# The timeline read from the file holds the same flow control, FlowMaps and
# virtual timelines, and answers the same:

# %%
read = Timeline.from_file(path)
{
    "same flow control": read.flow_control == score.flow_control,
    "FlowMaps": read.list_flow_maps(),
    "virtual timelines": read.list_virtual_timelines(),
    "14 is played at": read.get_virtual_timeline("as_played").unfold(14, read),
}

# %% [markdown]
# ## Summary
#
# | Task | API |
# |---|---|
# | Add flow control | `timeline.flow_control.add_all([...])`, `.add(element)` |
# | The five classes | `Jump`, `JumpTarget`, `Break`, `SectionBreak`, `Barline` |
# | A destination | `JumpDestination(to= or by=, passes=, repeat_count=, after=, determined=, reset=)` |
# | A Break that ends the flow only sometimes | `Break(..., ends=FlowCondition(passes=, after=))` |
# | Play, writing nothing | `timeline.get_flow(convention, believer=)` → stretches, `findings`, `unfold()`, `fold()` |
# | Flow control with the descendants' | `timeline.tree_flow_control`, which `get_flow()` plays |
# | Play a registry | `play(registry, length)`, `atomic_sections(registry, length)` |
# | Reject an element | `flow_control.with_belief(id, actor, "false")`, several at once with `with_beliefs()` |
# | The order a source states | `loader.create_stated_flow_map(timeline)` |
# | Write the flows | `timeline.create_flow_maps_from_flow_control()` |
# | Write a FlowMap by hand | `timeline.create_flow_map([region, Gap(length), ...])`, then `resolve(timeline)` |
# | A virtual timeline | `timeline.create_virtual_timeline(flow_map)`, `unfold(x, timeline)`, `fold(x, timeline)` |
