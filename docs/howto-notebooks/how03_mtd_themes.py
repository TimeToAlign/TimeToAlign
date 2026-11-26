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
# # How to Load Musical Themes Aligned to Recordings (MTD)
#
# The Musical Theme Dataset (MTD; Zalkow, Balke, Arifi-Müller and Müller,
# TISMIR 2020, [doi:10.5334/tismir.68](https://doi.org/10.5334/tismir.68))
# takes 2,067 themes from Barlow and Morgenstern's *A Dictionary of Musical
# Themes*. Each theme is sixteen files: its metadata, an excerpt of a
# commercial recording of its movement, an alignment of the excerpt with an
# encoding of the theme, and four encodings of the theme as MIDI files, note
# lists, PDFs, a Sibelius file and a MusicXML file. This guide reads one
# movement with `MtdLoader`:
#
# 1. the CD track, a seconds timeline holding each theme's excerpt as a
#    child;
# 2. per theme, one timeline per file, each naming its file and what it was
#    made from;
# 3. per theme, four {{< glossary TimelineGroup >}}s, one per encoding;
# 4. MTD's alignment as {{< glossary MatchClaim >}}s;
#
# and writes the {{< glossary AlignmentBundle >}} as a TimeToShare manifest.

# %% [markdown]
# ## Setup
#
# Install the library with `pip install timetoalign`. The example data is a
# sample of four themes in the layout of the MTD archive, `MTD_sample.zip`,
# fetched by the test-data helper; the full `MTD_1.0.0.zip` is read the same
# way, in place, without extracting it. The files that MTD's licence does not
# cover (the excerpts, the EDM encodings, the MusicXML files) are stand-ins
# shaped like MTD's own; an excerpt is silence as long as MTD's.
#
# Three helpers show what the loader made: `files` one row per timeline of a
# theme, `members` one row per member of a theme's groups, and `claims` one
# row per MatchClaim from a timeline.

# %%
import tempfile
from fractions import Fraction
from pathlib import Path

import pandas as pd
from timetoshare import loads, validate

from timetoalign import AlignmentBundle, MatchClaim, MtdLoader
from timetoalign.testdata import ensure_data

ARCHIVE = ensure_data("mtd") / "MTD_sample.zip"


def label(timeline, theme):
    """A timeline's name without its theme: ``SCORE MIDI``, the excerpt ``excerpt``."""
    if timeline.name == f"Theme {theme}":
        return "excerpt"
    return timeline.name.removeprefix(f"{theme}, ")


def theme_timelines(bundle, theme):
    """The excerpt and the file timelines of *theme*, as the bundle holds them."""
    timelines = [bundle.get_timeline(uid) for uid in bundle.timeline_ids]
    children = {uid for timeline in timelines for uid in timeline.list_children()}
    excerpt = next(t for t in timelines if t.name == f"Theme {theme}")
    roots = [
        t for t in timelines if t.id not in children and t.name.startswith(f"{theme}, ")
    ]
    return [excerpt, *roots]


def files(bundle, theme):
    """One row per timeline of *theme*: its axis, length, file and source."""
    timelines = theme_timelines(bundle, theme)
    labels = {f"#{t.id}": label(t, theme) for t in timelines}
    rows = []
    for timeline in timelines:
        statements = timeline.statements
        (item,) = statements["lrmoo:R7i_is_exemplified_by"]
        sources = statements.get("dcterms:source", [])
        rows.append(
            {
                "timeline": label(timeline, theme),
                "unit": timeline.unit.value,
                "length": (
                    "unknown" if timeline.length is None else str(timeline.length)
                ),
                "notes": len(timeline.get_events()),
                "file": item["label"],
                "derived from": ", ".join(labels[s["id"]] for s in sources),
                "children": ", ".join(
                    label(timeline.get_child(uid), theme)
                    for uid in timeline.list_children()
                ),
            }
        )
    return pd.DataFrame(rows)


def members(bundle, theme):
    """One row per member of *theme*'s groups: the group, the member, its length."""
    rows = []
    for group_id in bundle.group_ids:
        group = bundle.get_group(group_id)
        if not group.name.startswith(f"{theme}, "):
            continue
        for uid in group.timeline_ids:
            member = group.get_timeline(uid)
            rows.append(
                {
                    "group": group.name.removeprefix(f"{theme}, "),
                    "member": label(member, theme),
                    "length": (
                        "unknown" if member.length is None else str(member.length)
                    ),
                }
            )
    return pd.DataFrame(rows)


def claims(field, timeline):
    """One row per MatchClaim from *timeline*: the note, the excerpt's side."""
    events = timeline.get_events().table
    pitches = dict(
        zip(
            events.column("id").to_pylist(),
            [pitch["midi_number"] for pitch in events.column("pitch").to_pylist()],
        )
    )
    return pd.DataFrame(
        [
            {
                "onset (quarters)": str(
                    Fraction(claim.start_anchor.coordinate_a.value)
                ),
                "MIDI pitch": pitches[claim.event_a_id],
                "excerpt side": claim.event_b_id,
                "excerpt (seconds)": claim.start_anchor.coordinate_b.value,
                "claim type": claim.claim_type.value,
            }
            for claim in field
            if claim.timeline_a_id == timeline.id
        ]
    )


# %% [markdown]
# ## Load one movement
#
# A movement is named as MTD names it, `<ComposerID>_<WorkID>`; the first
# movement of Beethoven's Piano Sonata Op. 2 No. 1 is
# `Beethoven_Op002No1-01`. The loader reads every theme's metadata to find
# the movement's themes, then only their files. Themes can also be named by
# their MTD IDs, `themes=["MTD0807"]`; without either, every theme is read.

# %%
loader = MtdLoader.from_file(ARCHIVE, movements=["Beethoven_Op002No1-01"])
loader

# %% [markdown]
# Two themes of the movement lie in one CD track of Alfred Brendel's
# recording. A track is named as MTD names it: the MusicBrainz release, the
# disc and the track.

# %%
loader.themes, loader.tracks

# %% [markdown]
# ## The CD track and its excerpts
#
# `create_bundle()` assembles the bundle. Its first timeline is the CD
# track, named after its movement, its performer, the disc and the track.
# MTD does not state how long the track is, so its length is unknown,
# `None`.

# %%
bundle = loader.create_bundle()
track = bundle.get_timeline(bundle.timeline_ids[0])
track.name, track.length

# %% [markdown]
# Each theme's excerpt is a child of the track, placed at the theme's
# `StartTime` and as long as `EndTime − StartTime`, both read as the
# decimals MTD writes.

# %%
excerpts = [track.get_child(uid) for uid in track.list_children()]
pd.DataFrame(
    {
        "excerpt": [excerpt.name for excerpt in excerpts],
        "offset (seconds)": [track.get_child_offset(e.id).value for e in excerpts],
        "length (seconds)": [excerpt.length.value for excerpt in excerpts],
    }
)

# %% [markdown]
# The track states what MTD says of the recording: that it records the
# movement, its release on MusicBrainz, the collection, disc and track as
# MTD names them, the performer, and that the recording is in copyright and
# identified here, not shared.

# %%
track.statements

# %% [markdown]
# ## The files of one theme
#
# The first theme, B689 in Barlow and Morgenstern, is fourteen timelines:
# the excerpt and one per file of its four encodings. A MIDI file, the
# MusicXML file and the Sibelius file are quarters timelines, a note list
# (CSV) and the excerpt seconds timelines, a PDF a points timeline. Each
# names its file in the archive (`lrmoo:R7i_is_exemplified_by`) and, where
# MTD's paper states that the file was made from another, that file's
# timeline (`dcterms:source`): the SCORE exports derive from the Sibelius
# file, EDM-corr's MIDI file from EDM-orig's, each EDM encoding's note list
# and PDF from its MIDI file, and EDM-alig's files from EDM-corr's MIDI file.
#
# A timeline is as long as its file states: a MIDI file its end of track,
# the MusicXML file the end of its last bar, the excerpt its seconds. A note
# list, a PDF and the Sibelius file state no end: their length is unknown
# (`length` is `None`). Only the MIDI files' timelines hold notes.

# %%
theme = excerpts[0].name.removeprefix("Theme ")
files(bundle, theme)

# %% [markdown]
# The theme's identifiers and its movement are stated on the two timelines
# nothing is derived from, EDM-orig's MIDI file and the Sibelius file.

# %%
edm_orig = next(
    t for t in theme_timelines(bundle, theme) if label(t, theme) == "EDM-orig MIDI"
)
edm_orig.statements

# %% [markdown]
# ## Four groups, one per encoding
#
# Each encoding of the theme is a group: EDM-orig, EDM-corr and SCORE of
# their files, EDM-alig of its two files and the excerpt, to which
# EDM-alig is aligned (its quarter notes are the excerpt's seconds). The
# members of a group share its start and end. Two pairs of members state
# different lengths: EDM-alig's MIDI file ends at 1369/480 quarters, 25/480
# past the excerpt's 2.8 seconds, since the aligned notes run past the
# excerpt's cut; the MusicXML file's last bar ends at 9 quarters, past the
# SCORE MIDI file's end of track at 8. Nothing is stretched to fit: the
# longer member takes part through a child of it, starting at 0 and as long
# as the shorter one, named after the span it covers (the children of the
# table above).

# %%
members(bundle, theme)

# %% [markdown]
# Every group holds a member of unknown length (a note list, a PDF or the
# Sibelius file), so a group converts no position until those lengths are
# known.

# %% [markdown]
# ## The recording's transposition
#
# The notes keep their encoding's pitches. META's `MidiTransposition`, the
# semitones by which the recording sounds transposed against the encodings,
# is a comment on the excerpt: 0 for this theme. The finale of the Fifth
# Symphony, read by its MTD ID, states 5.

# %%
excerpts[0].statements["rdfs:comment"]

# %%
fifth = MtdLoader.from_file(ARCHIVE, themes=["MTD1080"]).create_bundle()
fifth_track = fifth.get_timeline(fifth.timeline_ids[0])
fifth_track.get_child(fifth_track.list_children()[0]).statements["rdfs:comment"]

# %% [markdown]
# ## MTD's alignment as MatchClaims
#
# The alignment file has rows of quarters in EDM-corr's MIDI file and
# seconds in the excerpt. The loader reads each row but the last as
# synchronous MatchClaims from each note at that onset of the EDM-corr MIDI
# timeline to the sentinel `CHUNK` on the excerpt (written `CHUNK_SENTINEL`
# in the MatchClaim table): the excerpt holds no event, the claim states a
# position on it. The file writes the onset with five decimals (`6.66667`);
# the claim holds the exact onset of the note it meets, `20/3`. The last
# row, the end of the last note, is checked and not kept. The loader hands
# the claims over as one field.

# %%
field = loader.get_field(MatchClaim)
edm_corr = next(
    t for t in theme_timelines(bundle, theme) if label(t, theme) == "EDM-corr MIDI"
)
claims(field, edm_corr)

# %% [markdown]
# The bundle holds the same claims (`bundle.match_claim_table`) beside its
# groups, whose members of unknown length leave its matchstamps without an
# answer. A bundle of the timelines and the claims without the groups
# answers one: from the triplet's onset, the excerpt's second and, through
# the excerpt's offset, the track's.

# %%
plain = AlignmentBundle(name="timelines and claims")
for timeline in loader.create_timelines():
    plain.add_timeline(timeline)
plain.add_match_claim_field(field)
plain.get_matchstamp_at(Fraction(20, 3), edm_corr.id)

# %% [markdown]
# ## Track lengths from elsewhere
#
# A track length known from elsewhere, MusicBrainz for instance, is given to
# `create_bundle` by the track's key (431 seconds here, for illustration).

# %%
(key,) = loader.tracks
loader.create_bundle(track_lengths={key: 431.0}).get_timeline(track.id).length

# %% [markdown]
# ## Write the manifest
#
# `to_file()` writes the manifest and, beside it, the MatchClaim table
# (`match_claims.parquet`), which the manifest names among its resources.
# With `events=False` the notes are not written; the claims keep their IDs.

# %%
package = Path(tempfile.mkdtemp()) / "beethoven_op002no1-01"
package.mkdir()
manifest = bundle.to_file(package / "manifest.yaml", events=False)
sorted(path.name for path in package.iterdir())

# %% [markdown]
# `to_file()` validates the manifest before writing it; TimeToShare's
# `validate` repeats the check on the file and raises at the first
# violation. `from_file()` reads the manifest and its table back: the
# timelines (children included), the groups and the claims.

# %%
validate(loads(manifest.read_text(encoding="utf-8")))
read = AlignmentBundle.from_file(manifest)
len(read.timeline_ids), len(read.group_ids), len(read.match_claim_table)
