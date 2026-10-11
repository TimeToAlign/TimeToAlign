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
# # How to Load an Audio-to-Score Alignment Corpus and Compare Performances
#
# A *Performance Precision* specimen bundles one symbolic score with several
# recorded performances of it, plus the audio-to-score alignments that connect
# them. This guide loads such a specimen in a single call and arrives at the
# question the corpus was built to answer: **how do different pianists shape the
# same piece in time?**
#
# The specimen here is Chopin's Nocturne in E♭ major, Op. 9 No. 2 — a `.solo`
# score, a Verovio timemap giving each measure's position in quarter notes, and
# an `Alignments/` directory holding three alignment files (note, bar, beat) for
# each of seven recordings. By the end we will read, for one shared score
# position, the seven different moments at which each pianist played it.
#
# The arc:
#
# 1. Load the whole specimen in one call.
# 2. Inspect the measure structure recovered from the timemap.
# 3. Read the score {{< glossary Timeline >}} (in quarters) and see how a
#    `measure+offset` label resolves.
# 4. Read the per-performer {{< glossary Timeline >}}s (in seconds).
# 5. Inspect the {{< glossary MatchClaim >}}s, tagged by granularity.
# 6. Compare the performers' timing through the {{< glossary AlignmentBundle >}}.

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

from collections import Counter
from fractions import Fraction

import pandas as pd

from timetoalign.core import EnharmonicPitch
from timetoalign.loader.alignment import PerformancePrecisionLoader
from timetoalign.loader.tabular.solo import SoloLoader
from timetoalign.testdata import ensure_data

SPECIMEN_DIR = ensure_data("performance_precision")
SOLO_FILE = SPECIMEN_DIR / "Chopin Nocturne Op. 9 No. 2.solo"

# %% [markdown]
# ## 1. Load the specimen in one call
#
# {{< glossary PerformancePrecisionLoader >}} ingests the whole directory — the
# `.solo` score, the Verovio timemap, and every per-performer alignment file —
# through the standard two-phase loader pattern. `from_file()` is the one-line
# form.

# %%
loader = PerformancePrecisionLoader.from_file(SPECIMEN_DIR)
loader

# %% [markdown]
# Seven recordings, and a few thousand {{< glossary MatchClaim >}}s linking the
# score to them. Everything below reads from this single loaded object.

# %% [markdown]
# ---
#
# ## 2. The measure structure
#
# The Verovio timemap records each measure's absolute position in quarter notes.
# The loader walks it into a {{< glossary MetricMap >}} — a
# {{< glossary ConversionMap >}} from a quarter position to its measure count —
# and exposes it directly:

# %%
metric_map = loader.metric_map
metric_map

# %% [markdown]
# Thirty-eight measures spanning 212.5 quarters. The companion `.meter` file is
# deliberately left unread: it encodes only meter *changes* (four rows across
# the whole piece), so it cannot by itself bound the final measure. The
# timemap's terminal position supplies that bound, which is why it is preferred
# here.
#
# The paired {{< glossary MetricalPositionMap >}} is a
# {{< glossary CombinationMap >}} carrying both directions of the
# measure↔quarter relationship. `quarters_at(mc, beat)` goes from a metrical
# position to a quarter coordinate; `mn_at(quarters)` returns the measure-number
# label at a quarter position:

# %%
metrical_position_map = loader.metrical_position_map

{
    "downbeat of MC 2 (quarters)": metrical_position_map.quarters_at(2),
    "downbeat of MC 3 (quarters)": metrical_position_map.quarters_at(3),
    "measure label at quarter 0.0": metrical_position_map.mn_at(0.0),
    "measure label at quarter 7.0": metrical_position_map.mn_at(7.0),
}

# %% [markdown]
# ---
#
# ## 3. The score timeline
#
# The score is a logical {{< glossary Timeline >}} measured in quarters. It holds
# every note of the `.solo` score, each placed at its absolute quarter position.
# Its stored UID is `score:clt_1`.

# %%
score_tl = loader.create_timeline(uid="score:clt_1")
score_tl

# %% [markdown]
# ### How a `measure+offset` label becomes a quarter coordinate
#
# Both the `.solo` score and the alignment files write score positions as
# `"<measure>+<offset>"`, where the offset is given in **whole notes**. The
# resolver is one line of arithmetic — `measure_start + offset × 4` (the `× 4`
# converts whole notes to quarters) — with one wrinkle: **measure 0 is the
# anacrusis** (the pickup), whose offsets are measured back from a virtual
# full-bar downbeat preceding the first sounding note. The very first note of
# the piece, labelled `0+11/8`, therefore lands exactly on quarter 0:

# %%
first_three = score_tl.get_events().table.slice(0, 3).to_pandas()
first_three[["id", "start"]]

# %% [markdown]
# A {{< glossary TimeStamp >}} is the primary way to query the score timeline at
# a coordinate — here the downbeat of the first full measure, half a quarter in:

# %%
score_tl.get_timestamp(Fraction(1, 2))

# %% [markdown]
# ### A note on pitch
#
# The score timeline carries pitch as the raw MIDI pitch integer that the
# `.solo` file recorded — faithful to the source, which notes no accidental
# spelling. When a *typed* pitch field is wanted, it lives on a freshly composed
# `SoloLoader` reading the same `.solo` file. There the field materialises as
# `EnharmonicPitch` scalars, and the canonical glyphs ♯/♭ render in their
# labels:

# %%
solo = SoloLoader.from_file(SOLO_FILE)
pitch_field = solo.events.get_field(EnharmonicPitch)
{"first five pitches": [pitch_field[i] for i in range(5)]}

# %% [markdown]
# Because `.solo` records only the MIDI pitch number, an enharmonic pair such as
# `G♯3` and `A♭3` both surface as the same number — the field reports what was
# represented, not what could be inferred.

# %% [markdown]
# ---
#
# ## 4. The performance timelines
#
# Each recording is a physical {{< glossary Timeline >}} measured in seconds, one
# event per aligned note onset. Their stored UIDs follow
# `perf:<performer>:cpt_1`, so the Ashkenazy recording is
# `perf:Chopin_Ashkenazy:cpt_1`:

# %%
performer_keys = [tl.name for tl in loader.create_timelines() if tl is not score_tl]
performer_keys

# %%
ashkenazy = loader.create_timeline(uid="perf:Chopin_Ashkenazy:cpt_1")
ashkenazy

# %% [markdown]
# A score position is in quarters; a performance position is in seconds. The
# alignment is what relates the two, and that is carried by the
# {{< glossary MatchClaim >}}s.

# %% [markdown]
# ---
#
# ## 5. The granularity-tagged MatchClaims
#
# The {{< glossary AlignmentBundle >}} assembles the loaded data: the score in
# its own group, each performance standalone, and every alignment row as a
# cross-group {{< glossary MatchClaim >}}.

# %%
bundle = loader.create_bundle()
bundle

# %% [markdown]
# Each recording was aligned at three granularities — note, bar, and beat — and
# the loader records which one produced each claim in its provenance metadata:
# the aligning agent's `identifier` carries the granularity. Counting the
# claims for one performer shows the shape of a single recording's alignment.
# A {{< glossary NOMATCH >}} claim prints its unmatched score coordinate
# (`score:clt_1@…`) so the dangling position stays legible at a glance:

# %%
bundle.cross_group_claims

# %%
ashkenazy_claims = [c for c in bundle.cross_group_claims if c.connects(ashkenazy.id)]
by_granularity = Counter(c.metadata.agent.identifier for c in ashkenazy_claims)
dict(by_granularity)

# %% [markdown]
# At the note level, not every score note is found in the recording: where the
# aligner located no onset, the row is recorded as a {{< glossary NOMATCH >}}
# rather than discarded, so the dangling score position is preserved. Splitting
# the note-level claims into located and {{< glossary NOMATCH >}} makes the
# distinction explicit:

# %%
note_claims = [c for c in ashkenazy_claims if c.metadata.agent.identifier == "note"]

{
    "note claims (total)": len(note_claims),
    "located (synchronous)": sum(1 for c in note_claims if c.is_synchronous),
    "NOMATCH (no onset found)": sum(1 for c in note_claims if not c.is_synchronous),
}

# %% [markdown]
# A single bar-level claim links a score quarter to a performed second. This one
# anchors the downbeat of the first full measure (quarter 0.5) to the moment
# Ashkenazy played it:

# %%
ashkenazy_bar_claims = [
    c for c in ashkenazy_claims if c.metadata.agent.identifier == "bar"
]
ashkenazy_bar_claims[0]

# %% [markdown]
# ---
#
# ## 6. Comparing the performances
#
# The corpus exists to compare how performers shape the same music in time. The
# {{< glossary AlignmentBundle >}} answers this directly: given one score
# coordinate, `get_matchstamp_at` returns the corresponding coordinate on every
# connected timeline. A {{< glossary MatchStamp >}} at the first full downbeat is
# therefore one score quarter mapped to seven performed seconds:

# %%
downbeat = float(metrical_position_map.quarters_at(2))
stamp = bundle.get_matchstamp_at(downbeat, score_tl.id)
stamp

# %% [markdown]
# Reading that across a sequence of downbeats gives each performer's arrival time
# at each measure — the raw material of a tempo comparison. We take the first
# several measure downbeats, ask the bundle for each performer's onset there, and
# tabulate the seconds against the score quarters:

# %%
downbeat_mcs = range(2, 10)
rows = []
for mc in downbeat_mcs:
    quarters = float(metrical_position_map.quarters_at(mc))
    onset_stamp = bundle.get_matchstamp_at(quarters, score_tl.id)
    row = {"score (quarters)": quarters}
    for key in performer_keys:
        perf_id = loader.create_timeline(uid=key).id
        coord = onset_stamp.coordinates.get(perf_id)
        row[key.replace("Chopin_", "")] = (
            round(float(coord), 3) if coord is not None else None
        )
    rows.append(row)

tempo_table = pd.DataFrame(rows).set_index("score (quarters)")
tempo_table

# %% [markdown]
# Each column is one pianist's arrival times in seconds; each row is a shared
# score position. The gaps between successive rows within a column are the
# inter-downbeat durations — read down a column and the performer's local tempo
# is visible, read across a row and the performers' differing placements of the
# same beat stand side by side. The same procedure at the beat or note
# granularity yields a finer-grained timing profile, all from the one
# {{< glossary AlignmentBundle >}}.
