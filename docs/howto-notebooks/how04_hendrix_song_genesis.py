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
# # How to Encode Song Genesis Relationships (Hendrix)
#
# This notebook demonstrates how to encode **conceptual and temporal
# relationships** between multiple versions of a work using
# {{< glossary MatchClaim >}} objects, {{< glossary MatchGraph >}} structures,
# and the **NOMATCH** sentinel.
#
# The use case comes from a genesis study of Jimi Hendrix's
# *1983... (A Merman I Should Turn to Be)*, comparing three versions:
#
# - **Studio** -- the studio recording from *Electric Ladyland* (CPT1)
# - **Demo2** -- a band demo with Mitch Mitchell on drums (CPT2)
# - **Demo1** -- a solo demo (CPT3)
#
# Form analyses for each version are encoded as TiLiA hierarchy timelines.
# A CSV file records which sections correspond across versions, whether
# those correspondences are **synchronous** (temporal alignment possible)
# or merely **conceptual** (structural equivalence only), and where a
# section is explicitly absent from a version (**NOMATCH**).
#
# ## Key Concepts Demonstrated
#
# - Loading TiLiA JSON files via `TiliaJsonLoader`
# - Creating an {{< glossary AlignmentBundle >}} from independent timelines
# - Parsing a match table with **synchronous** and **NOMATCH** columns
# - Creating synchronous vs. conceptual {{< glossary MatchClaim >}} objects
# - Using `MatchClaim.nomatch()` to name a section present on one version and
#   absent from another
# - Reading each claim's `claim_type` to tell event matches, conceptual links,
#   and NOMATCH claims apart
# - Querying events by name on hierarchy timelines

# %% [markdown]
# ## Setup

# %%

import pandas as pd

from timetoalign import AlignmentBundle
from timetoalign.alignment.claims import Agent, MatchClaim, MatchMetadata
from timetoalign.core.enums import AgentType
from timetoalign.loader.alignment import TiliaJsonLoader
from timetoalign.testdata import ensure_data  # noqa: E402

DATA_DIR = ensure_data("hendrix")

# %% [markdown]
# ## 1. Load the Three Versions
#
# Each version of the song has been annotated in TiLiA, producing a JSON
# file with hierarchy timelines encoding the form analysis.  We load each
# file and extract the first hierarchy timeline (`HIERARCHY_TIMELINE_0`),
# which contains the section-level annotations (Intro, Verse, Bridge, etc.).
#
# Each file is loaded on its own, so each loader mints the same id, `cpt_1`,
# for its timeline. The version name becomes the timeline's `name`; the ids are
# told apart when the timelines join one bundle.

# %%
names = ["Studio", "Demo2", "Demo1"]

timelines = {}
for name in names:
    loader = TiliaJsonLoader.from_file(DATA_DIR / f"Hendrix_Merman_{name}.json")
    tl = loader.create_timeline(uid="HIERARCHY_TIMELINE_0")
    tl.name = name
    timelines[name] = tl

timelines

# %% [markdown]
# ## 2. Load the Match Data
#
# The file `match_data.csv` is a tab-separated table recording which
# sections correspond across the three versions.  Each row represents a
# {{< glossary MatchGraph >}} (M1--M15).
#
# - Cells contain the **event name** (section label) on the respective
#   timeline.
# - The value `NOMATCH` explicitly records that a section has no equivalent
#   in that version.
# - The `synchronous` column indicates whether the correspondence is
#   temporal (`TRUE`) or merely conceptual (`FALSE`).

# %%
df = pd.read_csv(DATA_DIR / "match_data.csv", sep="\t")
df

# %% [markdown]
# ## 3. Create the AlignmentBundle
#
# We add each hierarchy timeline to a single {{< glossary AlignmentBundle >}}.
# A bundle holds a timeline id once, so the loader-minted `cpt_1` held by the
# first version is minted again for the others: the versions become `cpt_1`,
# `cpt_2` and `cpt_3`. Their names still match the column names in
# `match_data.csv`, and the claims below name each version by its id.

# %%
bundle = AlignmentBundle(name="Hendrix Song Genesis")
for name, tl in timelines.items():
    bundle.add_timeline(tl)

{name: tl.id for name, tl in timelines.items()}

# %% [markdown]
# ## 4. Build MatchClaims from the Match Table
#
# For each row in the CSV we:
#
# 1. Split the row into the sections that are **present** (named on a
#    timeline) and the timelines from which the section is **absent** (the
#    cell reads `NOMATCH`).
# 2. Emit a **NOMATCH** claim from every present timeline to every absent one.
#    The orphaned section lives on the versions that *have* it, so a genuine
#    NOMATCH names that section and is oriented from the naming timeline to the
#    one that lacks it -- not the reverse. A NOMATCH that named no event at all
#    would merely be conceptual.
# 3. For the present timelines, create **pairwise MatchClaims** -- synchronous
#    or conceptual according to the `synchronous` column.
#
# The pairwise strategy fans out from the first present timeline to each of the
# others (a "star" topology).

# %%
metadata = MatchMetadata(
    agent=Agent(
        name="Hendrix analysis",
        type=AgentType.human,
        identifier="match_data.csv",
    )
)
tl_columns = [c for c in df.columns if c not in ("match", "synchronous")]

sync_claims = []
conceptual_claims = []
nomatch_claims = []

for _, row in df.iterrows():
    is_synchronous = str(row["synchronous"]).strip().upper() == "TRUE"

    # Split the row into sections that ARE present (named on a timeline) and
    # timelines from which the section is absent (the cell reads NOMATCH).
    present = []  # (timeline_name, event_dict)
    absent = []  # timeline_name
    for tl_name in tl_columns:
        val = str(row[tl_name]).strip()

        if val.upper() == "NOMATCH":
            absent.append(tl_name)
            continue

        # Look up the event by name
        tl = timelines[tl_name]
        evs = tl.get_events(name=val)
        if len(evs) != 1:
            continue
        event_id = str(evs.table[0][0])
        event = tl.get_event(event_id)
        present.append((tl_name, event))

    # A NOMATCH is oriented FROM the version that has the section (naming its
    # orphaned event) TO the version that lacks it. Naming that event is what
    # makes the claim a genuine NOMATCH rather than a bare conceptual link.
    for absent_tl in absent:
        for present_tl, present_ev in present:
            nomatch_claims.append(
                MatchClaim.nomatch(
                    event=present_ev,
                    source_tl_id=timelines[present_tl].id,
                    target_tl_id=timelines[absent_tl].id,
                    unit=timelines[present_tl].unit,
                    metadata=metadata,
                )
            )

    # Create pairwise claims from the first present timeline to the others
    if len(present) < 2:
        continue

    tl_a_name, ev_a = present[0]
    for tl_b_name, ev_b in present[1:]:
        if is_synchronous:
            claim = MatchClaim.from_events(
                event_a=ev_a,
                tl_a_id=timelines[tl_a_name].id,
                event_b=ev_b,
                tl_b_id=timelines[tl_b_name].id,
                unit_a=timelines[tl_a_name].unit,
                unit_b=timelines[tl_b_name].unit,
                end_coord_key="end",
                is_synchronous=True,
                metadata=metadata,
            )
            sync_claims.append(claim)
        else:
            claim = MatchClaim(
                timeline_a_id=timelines[tl_a_name].id,
                timeline_b_id=timelines[tl_b_name].id,
                is_synchronous=False,
                metadata=metadata,
            )
            conceptual_claims.append(claim)

all_claims = sync_claims + conceptual_claims + nomatch_claims
bundle.add_match_claims(all_claims)

{
    "synchronous": len(sync_claims),
    "conceptual": len(conceptual_claims),
    "nomatch": len(nomatch_claims),
    "total": len(all_claims),
}

# %% [markdown]
# ### The claim kind is derived, not stored
#
# Every `MatchClaim` reports a `claim_type` computed from its structure --
# whether it is synchronous and how many of its two sides name an event. A
# synchronous event-to-event match is `ClaimType.event_match`, a structural
# link with no temporal commitment is `ClaimType.conceptual`, and a
# named-but-absent section is `ClaimType.nomatch`. This discriminator is what
# keeps conceptual links and NOMATCH claims -- both non-synchronous -- visibly
# distinct.

# %%
{
    "sync[0]": sync_claims[0].claim_type,
    "conceptual[0]": conceptual_claims[0].claim_type,
    "nomatch[0]": nomatch_claims[0].claim_type,
}

# %% [markdown]
# ## 5. Inspect the Results
#
# ### Synchronous claims (with AlignmentAnchors)
#
# These claims carry coordinate pairs (start and end) that enable temporal
# alignment between versions.  Each interval match corresponds to a pair
# of section boundaries in seconds.
#
# The MatchClaim's rich display shows timelines, coordinates, events, and
# metadata — no need to compile info-dicts manually.

# %%
# Display an example synchronous claim (shows timeline IDs, coordinates, events)
sync_claims[0]

# %%
# Summary of all synchronous claims
{
    "synchronous_claims": len(sync_claims),
    "is_interval": all(c.is_interval for c in sync_claims),
}

# %% [markdown]
# ### Conceptual claims (no anchors)
#
# These record structural equivalence without temporal commitment — for
# instance, "both versions have an Intro" without asserting that the intros
# can be aligned beat-by-beat. A conceptual claim names no orphaned event, so
# it badges as `[CONCEPTUAL]` — visibly distinct from the `[NOMATCH]` claims
# below, even though both are non-synchronous.

# %%
# Display an example conceptual claim (no coordinates, just timeline connection)
conceptual_claims[0] if conceptual_claims else "No conceptual claims"

# %%
{"conceptual_claims": len(conceptual_claims)}

# %% [markdown]
# ### NOMATCH claims
#
# These explicitly record that a section present on one version has no
# equivalent on another — a positive assertion of absence, not a mere gap in
# the data. A NOMATCH **names the orphaned section** and is oriented from the
# version that has it to the version that lacks it. For instance, the
# "Instrumental Part" in the studio recording has no equivalent in Demo1, so
# the claim names that studio section and points at Demo1. Naming the orphaned
# event is precisely what distinguishes a NOMATCH from a conceptual link.

# %%
# Display an example NOMATCH claim (names the present section, points at the
# version that lacks it)
nomatch_claims[0] if nomatch_claims else "No NOMATCH claims"

# %%
{"nomatch_claims": len(nomatch_claims)}

# %% [markdown]
# ## Summary
#
# This notebook demonstrated how to encode heterogeneous musicological
# relationships in a single, queryable structure:
#
# | Pattern | API |
# |---------|-----|
# | Load TiLiA annotations | `TiliaJsonLoader.from_file()` |
# | Look up sections by name | `tl.get_events(name=...)` |
# | Synchronous alignment | `MatchClaim.from_events(..., is_synchronous=True)` |
# | Conceptual correspondence | `MatchClaim(..., is_synchronous=False)` |
# | Explicit absence (names orphaned section) | `MatchClaim.nomatch(event=ev, source_tl_id=has, target_tl_id=lacks)` |
# | Discriminate claim kinds | `claim.claim_type` |
# | Collect in bundle | `bundle.add_match_claims(claims)` |

# %%
