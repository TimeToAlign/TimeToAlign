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
# # How to Align an Engraved Score with a Recording and a Drawn Time Axis
#
# Three files often travel together when a score is studied against a
# performance:
#
# - the **engraved score**, here an MEI file, read through
#   [Verovio](https://www.verovio.org) by `VerovioLoader`;
# - a **note alignment** between that score and a recording, in the MAPS
#   format that the TROMPA project's
#   [trompa-align](https://github.com/trompamusic/trompa-align) writes, read
#   by `MapsLoader` as {{< glossary MatchClaim >}}s;
# - a **drawing** of the score shifted to the recording's time, with a time
#   axis below it, as [ScoreWarp](https://github.com/iwk-digital/scorewarp)
#   writes it into an SVG, read by `ScoreWarpLoader`.
#
# Each file states its time in its own unit: quarters for the score, seconds
# for the recording, CSS pixels for the drawing. This guide loads all three
# and answers one question across them: which score position lies under a
# point of the drawing?
#
# The score comes from the library's test corpus. The alignment and the
# drawing are **illustrative**: this guide writes both itself, a few rows and
# a few ticks, with invented times. They have the shape the two tools write,
# so the same calls read the real files.
#
# The arc:
#
# 1. Load the score: exact quarters, notes by `xml:id`, Verovio's own
#    seconds.
# 2. Write a MAPS file, state it as MatchClaims, and ask a bundle for a
#    {{< glossary MatchStamp >}} between two aligned notes.
# 3. Write a ScoreWarp SVG, read its time axis, and take a point of the
#    drawing to seconds and on to the score.
# 4. Calibrate a drawn axis by hand with `create_calibrated_child`.
# 5. Save the bundle as a package and read it back.

# %% [markdown]
# ## Setup
#
# `VerovioLoader` needs the `verovio` extra: install the library with
# `pip install "timetoalign[verovio]"`. The files this guide writes go to a
# temporary directory. Three helpers keep the cells short: `notes_at` lists
# the score's notes in a span of quarters, `ids_at` names the notes starting
# at one position, and `claims_table` shows a list of claims one row each,
# a NOMATCH claim at the start of the one event it names.

# %%
from __future__ import annotations

import json
import tempfile
from fractions import Fraction
from pathlib import Path

import pandas as pd

from timetoalign import AlignmentBundle, MapsLoader, ScoreWarpLoader, VerovioLoader
from timetoalign.testdata import ensure_data
from timetoalign.timelines import (
    ContinuousGraphicalTimeline,
    ContinuousPhysicalTimeline,
)

SCORE_PATH = (
    ensure_data("score")
    / "flow_control"
    / "out_of_the_flow_experience-polyrhythm_only.mei"
)
FILES = Path(tempfile.mkdtemp())

NOTE_COLUMNS = ["id", "start", "end", "mc", "staff", "midi"]


def notes_at(timeline, start, stop) -> pd.DataFrame:
    """The notes of *timeline* starting in ``[start, stop)`` quarters."""
    notes = timeline.get_events(include_children=True, event_type="Note").to_dataframe()
    span = notes[(notes["start"] >= start) & (notes["start"] < stop)]
    return span[NOTE_COLUMNS].reset_index(drop=True)


def ids_at(timeline, position) -> list[str]:
    """The ``xml:id`` of every note of *timeline* starting at *position*."""
    return list(notes_at(timeline, position, position + Fraction(1, 1000))["id"])


def claims_table(claims, *timelines) -> pd.DataFrame:
    """One row per claim: its type, the events it names, and both sides.

    A synchronous claim gives the two coordinates of its anchor. A NOMATCH
    claim holds no coordinate: it names one event and the sentinel. MAPS
    states that event on timeline A and the sentinel on timeline B, so its
    ``at`` is the event's start, read from the event on timeline A among
    *timelines*; its ``there`` is empty, and its ``event there`` is the
    sentinel as the MatchClaim table writes it, ``NOMATCH_SENTINEL``.
    """
    by_id = {timeline.id: timeline for timeline in timelines}
    rows = []
    for claim in claims:
        if claim.is_synchronous:
            here, there = (
                claim.start_anchor.coordinate_a,
                claim.start_anchor.coordinate_b,
            )
        else:
            here = by_id[claim.timeline_a_id].get_coordinate_for(
                claim.event_a_id, format="coordinate"
            )
            there = None
        rows.append(
            {
                "claim_type": claim.claim_type.value,
                "event": claim.event_a_id,
                "from": claim.timeline_a_id,
                "at": str(here),
                "to": claim.timeline_b_id,
                "there": "" if there is None else str(there),
                "event there": claim.event_b_id,
            }
        )
    return pd.DataFrame(rows)


# %% [markdown]
# ## 1. Load the score
#
# `VerovioLoader` reads any format Verovio reads (MEI, MusicXML, Humdrum, …)
# and takes the score's time from Verovio's *timemap*, which states every
# onset as an exact fraction of a quarter. The test score sets tuplets
# against plain sixteenths: seven eighths in the time of three, each 3/14 of
# a quarter long, and five sixteenths in the time of three, each 3/20. Its
# positions are therefore multiples of 3/14 and of 3/20 of a quarter, such as
# 3/20, 3/14 and 3/10 in the first quarter: exact fractions keep them apart.
#
# `create_timeline()` builds the logical timeline in quarters. A movement
# (`<mdiv>`) of the file that is named, by its `@label`, becomes a child
# timeline holding that movement's events. This score's one `<mdiv>` has no
# label, so the score is a single timeline holding every event.

# %%
loader = VerovioLoader.from_file(SCORE_PATH)
score = loader.create_timeline(name="Out-of-the-flow Experience")
score

# %% [markdown]
# Every note keeps the `xml:id` it has in the MEI file; these ids were
# written by the notation program that exported the score. The first
# quarter, all four staves:

# %%
notes_at(score, 0, 1)

# %% [markdown]
# An id is a key: `get_coordinate_for` gives a note's position, and
# `get_timestamp_for` its span on the score. The
# timestamp of a position also shows the score's
# {{< glossary ConversionMap display="conversion map" >}} to seconds, which
# the loader builds from Verovio's own timing: the time Verovio renders from
# the score's tempo, not a performance.

# %%
first_beat = ids_at(score, 1)[0]
score.get_coordinate_for(first_beat)

# %%
print(score.get_timestamp_for(first_beat))

# %%
score.get_timestamp_at(1)

# %% [markdown]
# The same rendered time is available as a timeline of its own, in seconds,
# with the same events and ids:

# %%
rendered = loader.create_physical_timeline()
rendered.get_coordinate_for(first_beat)

# %% [markdown]
# ## 2. A recording, aligned note by note
#
# A MAPS file is a JSON list of rows. Each row pairs the score notes heard
# together, named by `xml:id`, with one time of the recording in seconds,
# `obs_mean_onset`; `obs_num` numbers the rows from 1. trompa-align writes two
# more kinds of rows: a score note that was not played, at onset `-1`, and a
# played note the score does not have, under an id beginning
# `trompa-align_inserted_`.
#
# The file below aligns the notes on the sixteenth-note grid of the first
# measures, one unplayed note and one inserted note. Its times are invented.

# %%
onsets = {
    Fraction(0): 0.5,
    Fraction(1, 4): 0.88,
    Fraction(1, 2): 1.27,
    Fraction(3, 4): 1.69,
    Fraction(1): 2.08,
    Fraction(3, 2): 2.9,
}
rows = [{"obs_mean_onset": -1, "xml_id": ids_at(score, Fraction(3, 20))}]
rows += [
    {"obs_mean_onset": seconds, "xml_id": ids_at(score, quarter)}
    for quarter, seconds in onsets.items()
]
rows.insert(5, {"obs_mean_onset": 1.85, "xml_id": "trompa-align_inserted_Cs5"})
for number, row in enumerate(rows, start=1):
    row["obs_num"] = number

maps_path = FILES / "alignment.json"
maps_path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
maps = MapsLoader.from_file(maps_path)
maps

# %% [markdown]
# `create_match_claims(score, performance)` states every listed id as a
# claim of its own. A MatchClaim joins two events, and the recording has no
# events of its own: for a note at a time of the recording, a
# {{< glossary Projection >}} of that note is added to the recording at that
# time, and the claim is an `event_match` between the score note and the
# Projection. The unplayed note becomes a {{< glossary NOMATCH >}} claim from
# the score. The inserted note, which the score lacks, is added to the
# recording as a Projection of no event under its MAPS id, and becomes a
# NOMATCH claim from the recording.
#
# The recording is a timeline in seconds; any physical timeline works, and
# one in samples reaches seconds through its C-Map.

# %%
recording = ContinuousPhysicalTimeline(length=4.0, unit="seconds", name="Recording")
claims = maps.create_match_claims(score, recording)
claims_table(claims, score, recording)

# %% [markdown]
# A bundle relates the two timelines through the claims. Added with
# `add_timeline()` alone, each timeline stands in no group, and the bundle
# still builds its {{< glossary WarpMap >}} between them. A matchstamp at an
# aligned note gives the recording's time; one at a note between two aligned
# notes, such as the septuplet note at 3/7 of a quarter, gives a time
# interpolated by the warp:

# %%
bundle = AlignmentBundle(name="Out-of-the-flow Experience")
bundle.add_timeline(score)
bundle.add_timeline(recording)
bundle.add_match_claims(claims)
bundle

# %%
bundle.get_matchstamp_at(Fraction(1, 4), score.id), bundle.get_matchstamp_at(
    Fraction(3, 7), score.id
)

# %% [markdown]
# `create_bundle(score, performance)` is the one-call form: it puts each
# timeline in a group of its own, named `score` and `performance`, and gives
# the same answers. On this recording it adds no Projection: the recording
# already holds one for every note and time, and the claims name those.

# %%
maps.create_bundle(score, recording).get_matchstamp_at(Fraction(3, 7), score.id)

# %% [markdown]
# ## 3. A drawing with a time axis
#
# ScoreWarp engraves the score with Verovio, shifts every note to its time in
# the recording, and appends a time axis to the SVG: a `<g class="timeAxis">`
# holding a horizontal `line.horizontalAxis`, a vertical line per tick, a
# number under every labelled tick and one title, such as `Time (s)`, naming
# the unit. The helper below writes such an axis, from 0 to 4 seconds at 100
# pixels per second, starting 50 pixels from the left edge.


# %%
def scorewarp_svg(path: Path, *, left: float, per_second: float, seconds: int) -> Path:
    """Write an SVG holding a ScoreWarp-shaped time axis, ticks every half second."""
    right = left + per_second * seconds
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{right + 20}" height="120" '
        f'viewBox="0 0 {right + 20} 120" overflow="visible">',
        '<g class="timeAxis">',
        f'<line class="horizontalAxis" x1="{left}" x2="{right}" y1="100" y2="100"/>',
        '<text x="2" y="96">Time (s)</text>',
    ]
    for half in range(2 * seconds + 1):
        x = left + per_second * half / 2
        parts.append(
            f'<line x1="{x}" x2="{x}" y1="100" y2="{106 if half % 2 == 0 else 103}"/>'
        )
    for second in range(seconds + 1):
        parts.append(f'<text x="{left + per_second * second}" y="116">{second}</text>')
    parts.append("</g></svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


svg_path = scorewarp_svg(FILES / "warped.svg", left=50.0, per_second=100.0, seconds=4)
warped = ScoreWarpLoader.from_file(svg_path)
warped.axis_segment, warped.calibration

# %% [markdown]
# The drawing becomes a graphical timeline in `css_pixels`, the unit of an
# SVG's outer viewport (1/96 inch, a continuous unit unlike the discrete
# `pixels` of an image). Its time axis is a child over the drawn line, named
# by the title, carrying the C-Map from `css_pixels` to the title's unit,
# fitted through the labelled ticks:

# %%
drawing = warped.create_timeline()
axis = drawing.get_child(drawing.list_children()[0])
drawing, axis.get_conversion_map("seconds")

# %% [markdown]
# A point of the drawing, say a notehead ScoreWarp drew at x = 258, is a
# position on the drawing; its timestamp reaches the axis child, 50 pixels
# further left, and the child's C-Map gives the time:

# %%
drawing.get_timestamp_at(258)

# %% [markdown]
# That time is a position on the recording, and the bundle takes it to the
# score: the notes there are the ones the drawing shows at x = 258.

# %%
seconds = drawing.get_timestamp_at(258).get_conversion_for("seconds")
stamp = bundle.get_matchstamp_at(seconds, recording.id)
stamp

# %%
quarter = stamp.get_coordinate_for(score.id, format="fraction")
notes_at(score, quarter, quarter + Fraction(1, 1000))

# %% [markdown]
# ## 4. Calibrating an axis by hand
#
# `ScoreWarpLoader` reads the axis it finds in CSS pixels as drawn; an axis in
# another drawing, or one placed through a transform, is calibrated by hand
# with `Timeline.create_calibrated_child`. It takes the child's length and
# offset and a list of `(position, value)` pairs, such as the labelled ticks
# of an axis, and fits the conversion map through them.
#
# Here the drawing is in `user_units`, the coordinate system of an SVG
# `viewBox`, whose size in inches only the document knows. The axis runs from
# 120 to 1920 user units, labelled every second:

# %%
page = ContinuousGraphicalTimeline(length=2000, unit="user_units", name="Page")
ticks = [(120, 0), (720, 1), (1320, 2), (1920, 3)]
page_axis = page.create_calibrated_child(
    1800, 120, ticks, value_unit="seconds", name="Time (s)"
)
page.get_timestamp_at(594)

# %% [markdown]
# The pairs are checked against the map's law, a straight line by default,
# drawn through the first and the last pair. Below, the tick at 720 is
# mislabelled 1.5 seconds: the line from `(120, 0)` to `(1920, 3)` reaches 1.5
# only at 1020, so that pair is refused before the child is created, and the
# page keeps only the axis made above:

# %%
try:
    page.create_calibrated_child(
        1800, 120, [(120, 0), (720, 1.5), (1320, 2), (1920, 3)], value_unit="seconds"
    )
except ValueError as error:
    print(error)
page.list_children()

# %% [markdown]
# ## 5. Saving the bundle
#
# The drawing joins the bundle like the other two timelines. `to_file()`
# writes the bundle as a package: a
# {{< glossary "TimeToShare manifest" >}} holding the timelines, their
# children and conversion maps and the score's measures, and beside it the
# MatchClaim table and the recording's Projections table, which the
# manifest names in its `resources`. A manifest names no event: the claims
# name the score's notes by their `xml:id`, which the MEI file holds. The
# file opens with the manifest's inline JSON-LD `@context`; the cell below
# lists the directory, where the MAPS file and the SVG are this guide's
# own, and prints the manifest from `timelines` on.

# %%
bundle.add_timeline(drawing)
bundle_path = bundle.to_file(FILES / "out_of_the_flow.yaml")
text = bundle_path.read_text(encoding="utf-8").splitlines()
start = text.index("timelines:")
print(sorted(path.name for path in FILES.iterdir()))
print("\n".join(text[start:][:24]))

# %% [markdown]
# `from_file()` reads the package back, the claims and the recording's
# Projections with it, and the bundle answers as before: the drawing's
# point at x = 258 is still 2.08 seconds, and the septuplet note still
# warps to the same time.

# %%
read = AlignmentBundle.from_file(bundle_path)
read.get_timeline(drawing.id).get_timestamp_at(258), read.get_matchstamp_at(
    Fraction(3, 7), score.id
)

# %% [markdown]
# ## Next
#
# - Claims, claim types and WarpMaps in depth: the
#   [Alignment Bundles and MatchClaims](../tutorials/tut06_alignment_bundles.ipynb)
#   tutorial.
# - Drawn and scanned time axes in pixels:
#   [How to Work with Graphical Timelines](how01_graphical_timelines.ipynb).
# - Manifests and files:
#   [How to Save and Load Timelines, Groups and Bundles](how02_group_and_bundle_files.ipynb).
