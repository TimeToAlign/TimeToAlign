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
# Themes* and gives each one an encoding, an excerpt of a commercial
# recording of its movement, and an alignment of the two: per note onset of
# the encoding, the second of the excerpt where it sounds. This guide loads
# one movement with `MtdLoader` and walks through what it makes:
#
# 1. per CD track, a seconds timeline, its length unknown unless you give it,
#    with what MTD says of the recording as statements;
# 2. per theme, the excerpt, a child of the track at its start time;
# 3. per theme, the theme's encoding, a quarters timeline of its notes, in
#    exact fractions of a quarter;
# 4. per aligned onset, a {{< glossary MatchClaim >}} from the note to
#    `CHUNK` on the excerpt: the excerpt holds no event, the claim states a
#    position on it.
#
# Finally the {{< glossary AlignmentBundle >}} is written as a package, a
# TimeToShare manifest with its MatchClaim table beside it.

# %% [markdown]
# ## Setup
#
# Install the library with `pip install timetoalign`. The example data is a
# sample of the MTD archive in its own layout, `MTD_sample.zip`, fetched by
# the test-data helper; the full `MTD_1.0.0.zip` is read the same way, in
# place, without extracting it. The helper `notes` shows a timeline's notes
# one per row, and `claims` shows MatchClaims one per row, each side as its
# onset and the note's pitch, or as the position the claim states.

# %%
import tempfile
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc

from timetoalign import AlignmentBundle, MatchClaim, MtdLoader
from timetoalign.testdata import ensure_data

ARCHIVE = ensure_data("mtd") / "MTD_sample.zip"


def exact(column):
    """The exact values of a coordinate column."""
    numerators = pc.struct_field(column, "numerator").to_pylist()
    denominators = pc.struct_field(column, "denominator").to_pylist()
    return [Fraction(n, d) for n, d in zip(numerators, denominators)]


def notes(timeline):
    """A timeline's notes: onset and end in quarters, MIDI pitch."""
    table = timeline.get_events().table
    return pd.DataFrame(
        {
            "start": [str(value) for value in exact(table.column("start"))],
            "end": [str(value) for value in exact(table.column("end"))],
            "MIDI pitch": pc.struct_field(
                table.column("pitch"), "midi_number"
            ).to_pylist(),
        }
    )


def claims(field, timeline):
    """MatchClaims one per row: the note's onset and pitch, the excerpt's second."""
    events = timeline.get_events().table
    pitch = dict(
        zip(
            events.column("id").to_pylist(),
            pc.struct_field(events.column("pitch"), "midi_number").to_pylist(),
        )
    )
    rows = []
    for claim in field.to_claims():
        if claim.timeline_a_id != timeline.id:
            continue
        rows.append(
            {
                "onset (quarters)": str(
                    Fraction(claim.start_anchor.coordinate_a.value)
                ),
                "MIDI pitch": pitch[claim.event_a_id],
                "excerpt side": "CHUNK" if claim.chunk_sides else claim.event_b_id,
                "excerpt (seconds)": claim.start_anchor.coordinate_b.value,
            }
        )
    return pd.DataFrame(rows)


# %% [markdown]
# ## 1. Load one movement
#
# A movement is named as MTD names it, `<ComposerID>_<WorkID>`; the first
# movement of Beethoven's Piano Sonata Op. 2 No. 1 is
# `Beethoven_Op002No1-01`. The loader reads every theme's metadata to find
# the movement's themes, then only their alignments and encodings. Themes
# can also be named by their MTD IDs, `themes=["MTD0807"]`; without either,
# every theme is read.

# %%
loader = MtdLoader.from_file(ARCHIVE, movements=["Beethoven_Op002No1-01"])
loader

# %% [markdown]
# Two themes of the movement, both in one CD track of Alfred Brendel's
# recording. A track is named as MTD names it: the MusicBrainz release, the
# disc and the track.

# %%
loader.themes, loader.tracks

# %% [markdown]
# The alignment counts the quarter notes of MTD's corrected encoding of the
# theme from the Electronic Dictionary of Musical Themes (folder
# `data_EDM-corr_MID`), so that is the encoding the loader reads by default.
# The encoding of the engraved theme (`data_SCORE_MID`) can be chosen with
# `theme_midi="data_SCORE_MID"`; for about a quarter of the themes its notes
# differ, and the loader then refuses an alignment row that meets no onset.

# %%
loader.theme_midi

# %% [markdown]
# ## 2. The recording and its excerpts
#
# `create_bundle()` assembles the bundle: the track's timeline, holding the
# excerpts as children, and each theme's encoding as a timeline of its own.
# MTD does not state how long the track is, so its length is unknown (`?`).

# %%
bundle = loader.create_bundle()
bundle

# %% [markdown]
# Each excerpt is a child of the track, placed at the theme's `StartTime`
# and as long as `EndTime − StartTime`, both read as the decimals MTD
# writes.

# %%
track = bundle.get_timeline(loader.create_timelines()[0].id)
excerpts = [track.get_child(child_id) for child_id in track.list_children()]
pd.DataFrame(
    {
        "excerpt": [excerpt.name for excerpt in excerpts],
        "offset (seconds)": [track.get_child_offset(e.id).value for e in excerpts],
        "length (seconds)": [excerpt.length.value for excerpt in excerpts],
    }
)

# %% [markdown]
# The track states what MTD says of the recording: that it is a recording of
# the movement, its release on MusicBrainz, the collection, disc and track as
# MTD names them, the performer, and that the recording is in copyright and
# identified here, not shared. No recording is read and no media file is
# named. Each statement records how it was made: imported from MTD.

# %%
track.statements

# %%
track.provenance[0]

# %% [markdown]
# A track length known from elsewhere, MusicBrainz for instance, is given to
# `create_bundle` by the track's key (431 seconds here, for illustration):

# %%
(key,) = loader.tracks
loader.create_bundle(track_lengths={key: 431.0}).get_timeline(track.id).length

# %% [markdown]
# ## 3. The theme's encoding
#
# The first theme's encoding is a quarters timeline of its own: never a
# child of the excerpt, since the two are different manifestations of the
# theme, related by the alignment. Each tick of the MIDI file (480 per
# quarter) is an exact fraction: the triplet near the theme's end starts at
# `20/3` quarters.

# %%
theme = next(t for t in loader.create_timelines() if t.name == "B689 (MTD 0807)")
theme = bundle.get_timeline(theme.id)
notes(theme)

# %% [markdown]
# Its statements name the movement it is part of, its MTD and Barlow and
# Morgenstern numbers, and which encoding it is. A theme is not an incipit:
# it is a part of the movement, wherever it stands.

# %%
theme.statements

# %% [markdown]
# ## 4. The alignment as MatchClaims
#
# Each row of the alignment file but the last is one MatchClaim: from the
# note at that onset to `CHUNK` on the excerpt, anchored at the onset in
# quarters and the excerpt's second. The file writes the onset with five
# decimals (`6.66667`); the claim holds the exact onset of the note it
# meets, `20/3`. The last row, the end of the last note, is checked and not
# kept. The loader reaches the claims through `get_field`.

# %%
field = loader.get_field(MatchClaim)
claims(field, theme)

# %% [markdown]
# From the triplet's onset, the bundle answers the excerpt's second and,
# through the excerpt's offset, the track's:

# %%
stamp = bundle.get_matchstamp_at(Fraction(20, 3), theme.id)
stamp

# %% [markdown]
# ## 5. Write the package
#
# `to_file()` writes the manifest and, beside it, the MatchClaim table
# (`match_claims.parquet`), which the manifest names among its resources;
# `from_file()` reads both back.

# %%
package = Path(tempfile.mkdtemp()) / "beethoven_op002no1-01"
package.mkdir()
manifest = bundle.to_file(package / "manifest.json", serialization="json")
sorted(path.name for path in package.iterdir())

# %%
read = AlignmentBundle.from_file(manifest)
len(read.match_claim_table), read.get_timeline(theme.id).statements == theme.statements
