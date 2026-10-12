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
# # How to Align Multimodal Data (Beethoven)
#
# **Figure 3 acid test** for TimeToAlign! — 16+ timelines across all 3 domains
# (Physical, Logical, Graphical) in 5 `TimelineGroups` within one `AlignmentBundle`.
#
# **Structure:**
# 1. **Part I**: Build 3 recording groups (Groups 1-3) — 15 DPTs
# 2. **Part II**: Build Score group (Group 4) + align with recordings
# 3. **Part III**: Build Emerson group (Group 5) + cross-group coordinate transfer

# %% [markdown]
# ## 0. Gold Standard Reference Values
#
# | ID | Description | Samples | Rate | Grp |
# |----|-------------|---------|------|-----|
# | DPT1-5 | Normal | 11,753,638 / 11,195 / 22,389 / 45,844 / 63,965 | 44.1k / 42 / 84 / 172 / 240 | 1 |
# | DPT6-10 | Mechanical | 12,426,696 / 11,836 / 23,671 / 48,469 / 67,628 | same rates | 2 |
# | DPT11-15 | Exaggerated | 8,197,748 / 7,808 / 15,616 / 31,975 / 44,614 | same rates | 3 |
#
# | Recording | Notes | Matched | Unmatched EEP | Unmatched ABC |
# |-----------|-------|---------|---------------|---------------|
# | Normal | 4,026 | 3,740 | 16 | 23 |
# | Mechanical | 4,026 | 3,743 | 13 | 20 |
# | Exaggerated | 2,820 | 2,650 | 4 | 1,113 |

# %% [markdown]
# ## 1. Setup

# %%

import numpy as np
import pandas as pd
from PIL import Image

from timetoalign import (
    ContinuousPhysicalTimeline,
    Coordinate,
    DiscreteGraphicalTimeline,
    Ms3Loader,
    NumberType,
    RepoVizzLoader,
    TableMap,
    TimelineGroup,
    TimeUnit,
)
from timetoalign.alignment import (
    Agent,
    AlignmentAnchor,
    AlignmentBundle,
    MatchClaim,
    MatchLine,
    MatchMetadata,
    WarpMap,
)
from timetoalign.alignment.matching import (
    match_notes_by_attributes,
    prepare_abc_notes_for_matching,
    prepare_eep_notes_for_matching,
)
from timetoalign.core import AgentType
from timetoalign.core.enums import FlowMode
from timetoalign.testdata import ensure_data
from timetoalign.timelines.flow import create_unfolded_timeline
from timetoalign.timelines.types import SegmentLine

DATA_DIR = ensure_data("score") / "beethoven_op18-4iv_multimodal"

# XML manifest paths — the loader reads metadata from these files
NORMAL_XML = DATA_DIR / "StringQuartetEEP_I_Normal" / "StringQuartetEEP_I_Normal.xml"
MECHANICAL_XML = (
    DATA_DIR / "StringQuartetEEP_I_Mechanical" / "StringQuartetEEP_I_Mechanical.xml"
)
EXAGGERATED_XML = (
    DATA_DIR / "StringQuartetEEP_I_Exaggerated" / "StringQuartetEEP_I_Exaggerated.xml"
)

# Audio sources and instruments
AUDIO_SOURCES = [
    "mono",
    "binaural",
    "pickup_vln1",
    "pickup_vln2",
    "pickup_vla",
    "pickup_cello",
]
INSTRUMENTS = ["vln1", "vln2", "vla", "cello"]


# %% [markdown]
# Each EEP recording directory contains 5 modalities (audio, 3 feature types,
# MoCap) plus `.notes` files with annotated note events. The function below
# builds a `TimelineGroup` from one such directory via the XML manifest.
#
# **Structure (per manuscript):**
# - 5 parent physical timelines, each with a `SamplesToSeconds` c-map
# - Audio, Tonal, LowLevel, Rhythm parents: 6 children each (mono, binaural, 4 pickups)
# - MoCap parent: 4 children (one per instrument: vln1, vln2, vla, cello)


# %%
def build_recording_group(xml_path, group_id, group_name, dpt_base):
    """Build a TimelineGroup from one EEP recording directory via XML manifest.

    Args:
        xml_path: Path to the recording's XML manifest file.
        group_id: ID for the TimelineGroup.
        group_name: Human-readable name for the group.
        dpt_base: Starting DPT number (e.g. 1 for dpt_1-dpt_5).

    Returns:
        TimelineGroup with 5 hierarchical DPTs (parent + children).
    """
    rv = RepoVizzLoader.from_file(xml_path)
    n = dpt_base

    # 1. Audio (mono as parent, 6 sources as children). A child's id is led by
    # its parent's id: the three recordings share one bundle, and a timeline
    # id names one timeline there. The source label stays its name.
    audio = rv.create_timeline("mono", uid=f"dpt_{n}", name="Audio")
    for src in AUDIO_SOURCES:
        audio.add_child(
            rv.create_timeline(src, uid=f"{audio.id}:{src}", name=src), offset=0
        )

    # 2-4. Essentia descriptors (tonal, lowlevel, rhythm)
    desc_cfgs = [
        ("tonal", "ChordsStrength", 1),
        ("lowlevel", "Dissonance", 2),
        ("rhythm", "BeatsLoudness", 3),
    ]
    descriptors = []
    for desc_type, desc_name, offset in desc_cfgs:
        parent = rv.create_timeline(
            f"{desc_type}.{desc_name}.mono",
            uid=f"dpt_{n + offset}",
            name=desc_type.title(),
        )
        for src in AUDIO_SOURCES:
            parent.add_child(
                rv.create_timeline(
                    f"{desc_type}.{desc_name}.{src}",
                    uid=f"{parent.id}:{src}_{desc_type}",
                    name=f"{src}_{desc_type}",
                ),
                offset=0,
            )
        descriptors.append(parent)

    # 5. MoCap bb_angle (from the DescriptorGroup section of the XML)
    mocap = rv.create_timeline(
        rv.find_descriptor("bb_angle", "vln1"),
        uid=f"dpt_{n + 4}",
        name="MoCap",
    )
    for inst in INSTRUMENTS:
        child = rv.create_timeline(
            rv.find_descriptor("bb_angle", inst),
            uid=f"{mocap.id}:{inst}_mocap",
            name=f"{inst}_mocap",
        )
        mocap.add_child(child, offset=0)

    # Add notes to pickup children
    for inst in INSTRUMENTS:
        notes = rv.store.notes_for_instrument(inst)
        if notes and (pickup := audio.get_child(f"{audio.id}:pickup_{inst}")):
            pickup.add_events(notes.to_dataframe().to_dict("records"))

    return TimelineGroup(
        id=group_id,
        name=group_name,
        timelines=[audio, *descriptors, mocap],
    )


# %% [markdown]
# ---
# # Part I: Three Recording Groups (Groups 1-3)
#
# Each EEP recording = 5 DPTs (audio + 3 feature types + MoCap) at different
# sampling rates, all sharing the same physical duration. Note events live
# as a child of the audio DPT.

# %% [markdown]
# ## 2. Group 1: Normal Recording (DPT1-DPT5)

# %%
normal_group = build_recording_group(
    NORMAL_XML, "normal", "Normal Recording", dpt_base=1
)
normal_group

# %% [markdown]
# The audio timeline now carries the note annotations as a child:

# %%
normal_group.get_timeline("dpt_1")

# %% [markdown]
# ## 3. Group 2: Mechanical Recording (DPT6-DPT10)

# %%
mechanical_group = build_recording_group(
    MECHANICAL_XML, "mechanical", "Mechanical Recording", dpt_base=6
)
mechanical_group

# %% [markdown]
# ## 4. Group 3: Exaggerated Recording (DPT11-DPT15)
#
# Shorter recording (~186s) — stops after measure 131.

# %%
exaggerated_group = build_recording_group(
    EXAGGERATED_XML,
    "exaggerated",
    "Exaggerated Recording",
    dpt_base=11,
)
exaggerated_group

# %% [markdown]
# ## 5. Part I Summary
#
# 3 groups, 15 timelines. Each audio DPT carries note events as a child
# timeline, making them accessible for matching in Part II.
#
# **Next:** Part II builds the Score group and aligns each recording via note matching.

# %% [markdown]
# ---
# # Part II: Score Group + Alignment to Recordings (Group 4)
#
# The score group brings together three representations of the same music:
#
# - **CLT1**: ABC v2.6 score (notes, measures, harmonies) — `ContinuousLogicalTimeline`
# - **DGT1**: OMR ground truth (3,190 note heads across 22 pages) — `DiscreteGraphicalTimeline`
# - **OpenScore**: OpenScore String Quartet edition (4th movement) — `ContinuousLogicalTimeline`
#
# All three go into one `TimelineGroup`. Cross-domain coordinate transfer
# (pixels ↔ quarters ↔ seconds) works automatically via linear interpolation.

# %% [markdown]
# ## 6. CLT1: ABC v2.6 Score

# %%
ABC_DIR = DATA_DIR / "ABC"
abc_loader = Ms3Loader.from_file(
    ABC_DIR / "n04op18-4_04.notes.tsv",
    ABC_DIR / "n04op18-4_04.measures.tsv",
    ABC_DIR / "n04op18-4_04.harmonies.tsv",
)
clt1 = abc_loader.create_timeline(uid="clt_1")
clt1

# %% [markdown]
# ### 6.1 ABC Flow Control: Repeat Structure
#
# The ABC score has repeats and volta brackets. The loader's
# `create_flow_controller()` derives the repeat structure from the
# measure data and computes the default flow (all repeats taken).
# This is **the same** flow control machinery used later for CLT2
# (the recordings edition) in Part III.

# %%
abc_controller = abc_loader.create_flow_controller()
abc_flow = abc_controller.compute_flow(FlowMode.default)
abc_flow

# %% [markdown]
# The flow controller and flow will be used in §9.2 to unfold the
# **entire** score group at once — not just CLT1, but all timelines.

# %% [markdown]
# ## 7. DGT1: OMR Ground Truth
#
# The OMR data contains 3,190 note head bounding boxes across 22 score pages.
# Each page has 2 systems (except the last which has 1), giving 43 system
# segments in reading order. Note events use `Left` (start) and `Width`
# (duration) as pixel coordinates. Each system's `onset_beats` values
# provide a c-map from pixels to quarters.
#
# **Architecture:** `SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]` →
# 22 page `SegmentLine[DiscreteGraphicalTimeline]` segments → 2 system sub-segments each.

# %%
OMR_CSV = DATA_DIR / "OMR_groundtruth" / "OMR_xml_by_score" / "omr_note_heads.csv"
OMR_IMAGES = DATA_DIR / "OMR_groundtruth" / "Images"
omr_df = pd.read_csv(OMR_CSV)
IMAGE_WIDTH = Image.open(next(OMR_IMAGES.glob("*.png"))).size[0]

# %% [markdown]
# Build the DGT1 bottom-up: system segments →
# page `SegmentLine[DiscreteGraphicalTimeline]` →
# top-level `SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]`.
# Events and c-maps must be added **before** a timeline is locked as a child.

# %%
noteheads = pd.DataFrame(
    {
        "start": omr_df["Nodes.Node.Left"].astype(int),
        "end": (omr_df["Nodes.Node.Left"] + omr_df["Nodes.Node.Width"]).astype(int),
        "onset_beats": omr_df["onset_beats"].astype(float),
        "pitch": omr_df["pitch"],
        "staff_id": omr_df["staff_id"].astype(int),
        "midi_pitch": omr_df["midi_pitch_code"].astype(int),
        "top": omr_df["Nodes.Node.Top"].astype(int),
        "page": omr_df["@pageIndex"],
        "spacing_run_id": omr_df["spacing_run_id"],
    }
)

dgt1 = SegmentLine[SegmentLine[DiscreteGraphicalTimeline]](
    length=0,
    unit=TimeUnit.pixels,
    number_type=NumberType.int,
    uid="dgt_1",
)

for page_idx, page_data in noteheads.groupby("page", sort=True):
    # Systems ordered by vertical position (top first = reading order)
    sys_top = page_data.groupby("spacing_run_id")["top"].min()
    sys_order = sys_top.sort_values().index

    page = SegmentLine[DiscreteGraphicalTimeline](
        length=0,
        unit=TimeUnit.pixels,
        number_type=NumberType.int,
    )

    for sys_rank, sys_id in enumerate(sys_order):
        sys_data = page_data[page_data["spacing_run_id"] == sys_id]

        system = DiscreteGraphicalTimeline(
            length=IMAGE_WIDTH,
            uid=f"p{page_idx}_s{sys_rank}",
            name=f"Page {page_idx + 1}, System {sys_rank + 1}",
        )

        events = sys_data.drop(columns=["page", "spacing_run_id"])
        system.add_events(events.assign(event_type="Notehead").to_dict("records"))

        # C-map: pixels → quarters (deduplicated for chords at the same x)
        pairs = (
            events[["start", "onset_beats"]]
            .drop_duplicates("start")
            .sort_values("start")
        )
        if len(pairs) >= 2:
            system.add_conversion_map(
                TableMap(
                    x_values=pairs["start"].tolist(),
                    y_values=pairs["onset_beats"].tolist(),
                    source_unit="pixels",
                    target_unit="quarters",
                    uid=f"p{page_idx}_s{sys_rank}_px_to_qb",
                )
            )

        page.append_segment(system)

    dgt1.append_segment(page, name=f"page_{page_idx}")

dgt1.diagram(depth=True)

# %% [markdown]
# ## 8. OpenScore (4th Movement Only)
#
# The OpenScore edition covers all 4 movements. We use the flow controller
# to identify section breaks (movement boundaries) and extract the 4th
# movement as a child timeline.

# %%
OPENSCORE_DIR = DATA_DIR / "OpenScoreSQ"
os_loader = Ms3Loader.from_file(
    OPENSCORE_DIR / "sq8913219.notes.tsv",
    OPENSCORE_DIR / "sq8913219.measures.tsv",
)
os_full = os_loader.create_timeline(uid="openscore_full")
os_full

# %% [markdown]
# The loader's `create_flow_controller()` derives section boundaries from
# the score's flow control markup. Splitting at those coordinates creates
# one region per movement, named `movement_1` to `movement_4`; the child is
# taken from the fourth by its id.

# %%
os_flow_controller = os_loader.create_flow_controller()
boundaries = os_flow_controller.get_section_boundary_coordinates()
movements = os_full.create_regions_from_boundaries(
    [0, *[float(b) for b in boundaries], float(os_full.length.value)], prefix="movement"
)
openscore = os_full.create_child_from_region(movements[3].id, uid="openscore")
openscore

# %% [markdown]
# The four movement regions and the extracted child timeline:

# %%
os_full.diagram(show={"regions", "children"}, depth=1)

# %% [markdown]
# ## 9. Score Group (Group 4)
#
# All three score representations in one `TimelineGroup`. Cross-domain
# coordinate transfer (pixels ↔ quarters) works via linear interpolation.

# %%
score_group = TimelineGroup(
    id="score",
    name="Score (ABC + OMR + OpenScore)",
    timelines=[clt1, dgt1, openscore],
)
score_group

# %% [markdown]
# ### 9.1 Cross-Domain Section Boundaries (Quarters → Pixels → Pages)
#
# The playthrough section boundaries (from §6.1) can now be mapped
# through the score group to DGT1 pixel coordinates. This demonstrates
# cross-domain coordinate transfer within a `TimelineGroup`: every member
# spans the whole group, so a position on CLT1 (quarters) converts to DGT1
# (pixels) linearly, by the ratio of the two lengths.

# %%
# Build a page-boundary lookup from DGT1's segment structure
_page_bounds = []
for _seg_id in dgt1.list_segments():
    _off = dgt1.get_child_offset(_seg_id)
    _seg = dgt1.get_child(_seg_id)
    _page_bounds.append(
        (float(_off.value), float(_off.value) + float(_seg.length.value))
    )

_section_rows = []
for _sid, _qb in abc_controller.get_atomic_section_coordinates(flow=abc_flow).items():
    _ts = score_group.get_timestamp_at(float(_qb), "clt_1")
    _px = (
        _ts.get_coordinate_for("dgt_1", format="int")
        if "dgt_1" in _ts.present_timelines
        else None
    )
    _page = next(
        (
            i + 1
            for i, (s, e) in enumerate(_page_bounds)
            if _px is not None and s <= _px < e
        ),
        "-",
    )
    _section_rows.append(
        {"section": _sid, "quarters": float(_qb), "dgt1_pixels": _px, "page": _page}
    )
section_boundary_table = pd.DataFrame(_section_rows).set_index("section")
section_boundary_table

# %% [markdown]
# Each atomic section's start coordinate is located precisely on a
# specific page of the OMR score image. The pixel column gives the
# linearised x-coordinate across all 22 pages; the page column tells
# which score image to open.

# %% [markdown]
# ### 9.2 Unfolding the Entire Score Group
#
# The score has repeats and volta brackets. Rather than unfolding each
# timeline individually, `TimelineGroup.apply_flow()` does it in one call:
# the flow controller's section boundaries are resolved via the group's
# linear conversion, so every timeline — regardless of domain — is
# sliced and reassembled in playthrough order.

# %%
score_group_unfolded = score_group.apply_flow(
    abc_flow, abc_controller, reference_timeline_id="clt_1"
)
score_group_unfolded

# %% [markdown]
# The unfolded CLT1 carries all note events in playthrough order.
# Extract them for note matching:

# %%
clt1_unfolded = score_group_unfolded.get_timeline("clt_1")
abc_notes_df = clt1_unfolded.get_events(
    event_type="Note", include_children=True
).to_dataframe()

# Cast types restored from string (EventData stores extra columns as strings)
abc_notes_df["staff"] = pd.to_numeric(abc_notes_df["staff"], errors="coerce").astype(
    "Int64"
)
abc_notes_df["tied"] = pd.to_numeric(abc_notes_df["tied"], errors="coerce")
abc_notes_df.loc[abc_notes_df["tied"] == 0, "tied"] = np.nan
abc_notes_df["quarterbeats_playthrough"] = abc_notes_df["start"]

abc_prepared = prepare_abc_notes_for_matching(abc_notes_df)
len(abc_prepared)  # note onsets after dropping tied notes

# %% [markdown]
# ## 10. Aligning Recordings with the Score via Note Matching
#
# Each EEP recording's note events (seconds, pitch, staff) are matched
# against the ABC **unfolded** score notes (quarterbeats, pitch, staff)
# prepared in §9.2 using greedy sequential matching. The result:
# `MatchClaim` objects that connect recording coordinates to score
# coordinates. No pre-computed TSV is needed — the unfolded CLT1 carries
# all the notes.

# %% [markdown]
# Match each recording against the score. The `source_timeline_id` and
# `target_timeline_id` are the audio DPT and CLT1 respectively — these
# appear in the resulting `MatchClaim` anchors.
#
# We use `rv.store.notes` to access the EEP notes from the XML manifest's
# score section — no direct `EepNotesLoader` import needed.

# %%
match_results = {}
for xml_path, dpt_id in [
    (NORMAL_XML, "dpt_1"),
    (MECHANICAL_XML, "dpt_6"),
    (EXAGGERATED_XML, "dpt_11"),
]:
    rv = RepoVizzLoader.from_file(xml_path)
    eep_events = rv.store.notes.to_dataframe()
    eep_prepared = prepare_eep_notes_for_matching(eep_events)
    match_results[dpt_id] = match_notes_by_attributes(
        eep_prepared,
        abc_prepared,
        match_columns=["pitch", "staff"],
        source_coord_column="start",
        target_coord_column="quarterbeats_playthrough",
        source_timeline_id=dpt_id,
        target_timeline_id="clt_1",
        source_unit=TimeUnit.seconds,
        target_unit=TimeUnit.quarters,
    )

normal_match = match_results["dpt_1"]
mechanical_match = match_results["dpt_6"]
exaggerated_match = match_results["dpt_11"]

# %%
{
    "Normal": normal_match.summary(),
    "Mechanical": mechanical_match.summary(),
    "Exaggerated": exaggerated_match.summary(),
}

# %% [markdown]
# ## Part II Summary
#
# The score group unites 3 score representations across 2 domains (Logical +
# Graphical). Note matching produced MatchClaims connecting each recording
# group's audio timeline to CLT1:
#
# | Recording | Matched | Unmatched EEP | Unmatched ABC |
# |-----------|---------|---------------|---------------|
# | Normal | 3,740 | 16 | 23 |
# | Mechanical | 3,743 | 13 | 20 |
# | Exaggerated | 2,650 | 4 | 1,113 |
#
# **Next:** Part III adds the Emerson group and demonstrates cross-group
# coordinate transfer using an `AlignmentBundle`.

# %% [markdown]
# ---
# # Part III: Emerson Recording + Cascading Alignment (Group 5)
#
# The Emerson group connects a commercial recording to a second score
# edition via segment-level alignment. Unlike the EEP groups (per-note
# alignment), the Emerson recording is aligned at the level of 10
# structural sections (alpha through kappa), derived from the score's
# repeat structure.
#
# The central payoff of this notebook is **cascading alignment**: by
# adding the recordings edition's unfolded score (CLT2) to the same
# group as CLT1, coordinate transfer chains automatically from the EEP
# recordings through both score editions to the Emerson recording.
#
# - **CLT2**: ABC v1.0 ("recordings edition") score — `ContinuousLogicalTimeline`
# - **DPT16**: Emerson String Quartet recording (DG 1997) — `ContinuousPhysicalTimeline`

# %% [markdown]
# ## 11. Building the Emerson Recording Components

# %% [markdown]
# ### 11.1 CLT2: Recordings Edition Score
#
# The recordings edition uses the same measure/repeat structure as CLT1 but
# was encoded independently (ABC v1.0). We load it via Ms3Loader and use its
# flow controller to compute the traversal map.

# %%
REC_DIR = DATA_DIR / "recordings"
rec_loader = Ms3Loader.from_file(
    REC_DIR / "Beethoven_Op018No4-04.notes.tsv",
    REC_DIR / "Beethoven_Op018No4-04.measures.tsv",
    REC_DIR / "Beethoven_Op018No4-04.harmonies.tsv",
)
clt2 = rec_loader.create_timeline(uid="clt_2")
clt2

# %% [markdown]
# ### 11.2 Flow Control: Inspect the Score's Repeat Structure
#
# The loader's `create_flow_controller()` identifies atomic sections and
# flow control events (repeats, voltas) from the measure data.

# %%
rec_controller = rec_loader.create_flow_controller()
rec_controller

# %% [markdown]
# Compute the default flow (all repeats taken) and a single-pass flow
# (no repeats, last volta only) for comparison:

# %%
default_flow = rec_controller.compute_flow(FlowMode.default)
default_flow

# %%
single_flow = rec_controller.compute_flow(FlowMode.single)
single_flow

# %% [markdown]
# ### 11.3 Unfolding CLT2
#
# The recordings edition has the same repeat structure as CLT1.
# We unfold it via the standalone `create_unfolded_timeline()` function,
# passing the default flow (all repeats taken). The result is a flat
# timeline with all sections concatenated in playthrough order —
# coordinates in quarter-beats, suitable for matching against the
# Emerson CSV's unfolded floating-measure boundaries.

# %%
clt2_unfolded = create_unfolded_timeline(
    clt2, default_flow, flow_controller=rec_controller, uid="clt2_unfolded"
)
clt2_unfolded

# %% [markdown]
# ### 11.4 DPT16: Emerson Recording
#
# The `measureMapAudio.csv` provides a 10-segment alignment between the
# unfolded score (floating measures) and the Emerson recording (seconds).
# Each segment is labelled with a Greek letter (alpha through kappa).

# %%
ema_df = pd.read_csv(
    REC_DIR / "Beethoven_Op018No4-04_EmersonStringQuartet_DG_measureMapAudio.csv",
    sep="\t",
    index_col=0,
)
ema_df

# %% [markdown]
# Create DPT16 as a `ContinuousPhysicalTimeline` in seconds. Unlike
# the EEP recordings (per-note alignment), the Emerson alignment
# operates at the level of section boundaries — the coordinates in
# `ema_df` will become MatchClaims in §11.5 rather than a C-map.

# %%
dpt16_duration = float(ema_df["seconds_end"].iloc[-1])
dpt16 = ContinuousPhysicalTimeline(length=dpt16_duration, uid="dpt_16")
dpt16

# %% [markdown]
# ### 11.5 Emerson MatchClaims (alpha through kappa)
#
# Each row in the measure-map CSV defines a section boundary: a
# correspondence between an unfolded floating-measure coordinate on
# CLT2 and a seconds coordinate on DPT16. We create one MatchClaim
# per boundary, plus the final end boundary.
#
# These cross-group claims are the key connection between the Emerson
# recording and the score group.
# `AlignmentAnchor` stores unit-bearing `Coordinate` values, so the units are
# explicit at the claim boundary even though the source data is numeric.

# %%
emerson_claims = []
for _, row in ema_df.iterrows():
    anchor = AlignmentAnchor(
        timeline_a_id="clt2_unfolded",
        coordinate_a=Coordinate(float(row["measure_unfold_start"]), TimeUnit.quarters),
        timeline_b_id="dpt_16",
        coordinate_b=Coordinate(float(row["seconds_start"]), TimeUnit.seconds),
    )
    emerson_claims.append(
        MatchClaim(
            timeline_a_id="clt2_unfolded",
            timeline_b_id="dpt_16",
            start_anchor=anchor,
            metadata=MatchMetadata(
                agent=Agent(
                    name="dataset",
                    type=AgentType.software,
                    identifier="measure_map_audio",
                ),
            ),
        )
    )

# Final end boundary
final_anchor = AlignmentAnchor(
    timeline_a_id="clt2_unfolded",
    coordinate_a=Coordinate(
        float(ema_df["measure_unfold_end"].iloc[-1]), TimeUnit.quarters
    ),
    timeline_b_id="dpt_16",
    coordinate_b=Coordinate(float(ema_df["seconds_end"].iloc[-1]), TimeUnit.seconds),
)
emerson_claims.append(
    MatchClaim(
        timeline_a_id="clt2_unfolded",
        timeline_b_id="dpt_16",
        start_anchor=final_anchor,
        metadata=MatchMetadata(
            agent=Agent(
                name="dataset",
                type=AgentType.software,
                identifier="measure_map_audio",
            ),
        ),
    )
)

len(emerson_claims)

# %% [markdown]
# ## 12. Bridging the Two AlignmentBundles

# %% [markdown]
# ### 12.1 The Key Move: Adding CLT2\_unfolded to the Unfolded Score Group
#
# The Unfolded Score Group and the Emerson Group are currently independent:
# neither shares a timeline with the other, and no MatchClaims connect them.
# The Emerson MatchClaims (§11.5) link CLT2\_unfolded to DPT16 — but
# CLT2\_unfolded is not yet in any group that the bundle's existing WarpMaps
# can reach.
#
# The insight: CLT1\_unfolded and CLT2\_unfolded encode the *same music*
# from different editions. By adding CLT2\_unfolded to the Unfolded Score
# Group, any coordinate on CLT1\_unfolded can be transferred to
# CLT2\_unfolded via within-group interpolation, and from there to DPT16
# via the Emerson MatchLine's WarpMap. The cascading path:
#
# **DPT1 -> (WarpMap) -> CLT1 -> (interpolation) -> CLT2\_unfolded -> (WarpMap) -> DPT16**
#
# A single additional group membership retroactively enriches every
# timeline in both groups.

# %%
score_group_unfolded.add_timeline(clt2_unfolded)
score_group_unfolded

# %% [markdown]
# CLT2\_unfolded now appears alongside CLT1, DGT1, and OpenScore in
# the unfolded score group. Every member spans the whole group, so the group
# links all four timelines pairwise, bridging quarter-beats and floating
# measures.

# %% [markdown]
# ### 12.2 The Emerson Group
#
# The Emerson group contains only DPT16 — the recording timeline.
# CLT2\_unfolded lives in the score group, and the Emerson MatchClaims
# connect the two groups via cross-group claims.

# %%
emerson_group = TimelineGroup(
    id="emerson",
    name="Emerson Recording (DG 1997)",
    timelines=[dpt16],
)
emerson_group

# %% [markdown]
# ## 13. The AlignmentBundle
#
# The bundle collects all 5 groups and connects them via MatchClaims.
# Within each group, coordinate transfer uses linear interpolation.
# Between groups, WarpMaps (built from MatchClaims) enable cross-domain
# transfer.

# %%
bundle = AlignmentBundle(name="Beethoven Op.18/4 — Multimodal Alignment")

bundle.add_group(score_group_unfolded)
bundle.add_group(normal_group)
bundle.add_group(mechanical_group)
bundle.add_group(exaggerated_group)
bundle.add_group(emerson_group)

# Add EEP recording <-> CLT1 match claims
for dpt_id in ["dpt_1", "dpt_6", "dpt_11"]:
    bundle.add_match_claims(match_results[dpt_id].match_claims)

# Add Emerson section boundary claims (CLT2_unfolded <-> DPT16)
bundle.add_match_claims(emerson_claims)

bundle

# %% [markdown]
# Match claims per connection:

# %%
pd.DataFrame(
    [
        {
            "recording": name,
            "source": dpt_id,
            "target": "clt_1",
            "matched": match_results[dpt_id].n_matched,
            "unmatched_source": match_results[dpt_id].n_unmatched_source,
            "unmatched_target": match_results[dpt_id].n_unmatched_target,
        }
        for name, dpt_id in [
            ("Normal", "dpt_1"),
            ("Mechanical", "dpt_6"),
            ("Exaggerated", "dpt_11"),
        ]
    ]
    + [
        {
            "recording": "Emerson",
            "source": "clt2_unfolded",
            "target": "dpt_16",
            "matched": len(emerson_claims),
            "unmatched_source": 0,
            "unmatched_target": 0,
        }
    ]
).set_index("recording")

# %% [markdown]
# ### 13.1 Explicit MatchLine and WarpMap
#
# Before demonstrating bundle-level coordinate transfer, it is
# instructive to see the intermediate MatchLine and WarpMap that the
# bundle constructs internally. The MatchLine orders the 11 Emerson
# anchors by source coordinate; the WarpMap interpolates between them.

# %%
emerson_matchline = MatchLine.from_claims(
    emerson_claims, source_timeline_id="clt2_unfolded"
)
emerson_matchline

# %%
emerson_warpmap = WarpMap.from_match_line(
    emerson_matchline, target_timeline_id="dpt_16"
)
emerson_warpmap

# %% [markdown]
# Verify the WarpMap manually: transfer a coordinate from
# CLT2\_unfolded to DPT16 and compare with a known section boundary:

# %%
# The first section boundary from ema_df
first_fm = float(ema_df["measure_unfold_start"].iloc[0])
first_sec = float(ema_df["seconds_start"].iloc[0])
transferred = emerson_warpmap.get_coordinate_at(first_fm)
{
    "CLT2_unfolded (floating measures)": first_fm,
    "DPT16 expected (seconds)": first_sec,
    "DPT16 via WarpMap": transferred,
}

# %% [markdown]
# ## 14. Cross-Group Coordinate Transfer
#
# The bundle's `get_matchstamp_at()` method is the primary interface for
# cross-domain coordinate transfer. Given a coordinate on any timeline,
# it returns a `MatchStamp` with corresponding coordinates on all connected
# timelines —
# regardless of domain. With CLT2\_unfolded bridging the score group
# and the Emerson MatchClaims, the bundle now reaches all 5 groups.

# %% [markdown]
# ### 14.1 Inspecting CLT1's Harmony Annotations
#
# Before transferring coordinates, let us see what harmonic events live
# on CLT1. The annotations child carries all harmony labels from the
# ABC score:

# %%
# Loader children are named after their category; their ids are minted.
annotations_id = next(
    child_id
    for child_id in clt1.list_children()
    if clt1.get_child(child_id).name == "annotations"
)
annotations_df = clt1.get_child(annotations_id).get_events().to_dataframe()
annotations_df[["start", "name"]].head(15)

# %% [markdown]
# ### 14.2 USE CASE A — Transfer a Harmony Across All Groups
#
# The `V7` at quarterbeat 79 (m. 20) is a dominant seventh — one of the
# most recognisable sonorities. Where does this moment land across all
# 5 groups, in every domain? The nested format groups results by
# `TimelineGroup`:

# %%
stamp = bundle.get_matchstamp_at(79.0, "clt_1")
stamp

# %% [markdown]
# `MatchStamp` belongs to the same unified stamp family as `TimeStamp`:
# `get_coordinate()` returns a unit-bearing coordinate,
# while `is_interpolated` reports whether resolution used interpolation.
# Those two are independent. This position was reached by interpolation and is
# still an exact `Fraction`, because the number type follows the quarters axis
# and never records how a value was obtained. If you want to know whether a
# reading was estimated, `is_interpolated` is the only thing that tells you.

# %%
{
    "CLT1 coordinate": stamp.get_coordinate("clt_1"),
    "interpolated": stamp.is_interpolated,
}

# %% [markdown]
# `to_dict()` is serialization rather than a shortcut to numbers: every leaf is
# the dictionary `{value, numerator, denominator, unit, number_type}`. The
# float `value` is a mirror for consumers that only speak JSON numbers, the
# ratio members carry the exact value where the axis is fraction-canonical, and
# `number_type` names which side is authoritative. When you want a plain number
# instead, ask for one with `get_coordinate_for(id, format="float")`.

# %%
bundle.get_matchstamp_at(79.0, "clt_1").to_dict(format="nested")

# %% [markdown]
# Note that the `emerson` group now appears in the output: the
# cascading path CLT1 -> CLT2\_unfolded -> DPT16 connects the Emerson
# recording to the rest of the bundle.

# %% [markdown]
# The flat format is useful for programmatic access:

# %%
bundle.get_matchstamp_at(79.0, "clt_1").to_dict(format="flat")

# %% [markdown]
# ### 14.3 USE CASE B — Reverse Transfer: Emerson to All Groups
#
# The cascading alignment is bidirectional. Starting from a seconds
# coordinate on DPT16 (the Emerson recording), we can reach every
# connected timeline — including the three EEP recording groups:

# %%
bundle.get_matchstamp_at(120.0, "dpt_16").to_dict(format="nested")

# %% [markdown]
# A coordinate at 120 seconds into the Emerson recording is mapped
# through the WarpMap to CLT2\_unfolded, then via interpolation to
# CLT1, and from there via the per-note WarpMaps to DPT1, DPT6, and
# DPT11 — all in a single call.

# %% [markdown]
# ### 14.4 USE CASE C — Section Boundaries Across All Groups
#
# The score's repeat structure defines atomic sections (A through M).
# The flow controller (from §6.1) computes each section's **unfolded**
# quarterbeat start coordinate. With the Emerson group now connected,
# the boundary table includes DPT16:

# %%
section_coords = abc_controller.get_atomic_section_coordinates(flow=abc_flow)
section_coords

# %%
boundary_rows = []
for _qb in section_coords.values():
    _stamp = bundle.get_matchstamp_at(float(_qb), "clt_1")
    _row = {}
    for _tl_id in _stamp.present_timelines:
        _coord = _stamp.get_coordinate_for(_tl_id)
        _row[f"{_tl_id} ({_coord.unit})"] = _coord.value
    boundary_rows.append(_row)
boundary_df = pd.DataFrame(
    boundary_rows,
    index=list(section_coords.keys()),
)
boundary_df.index.name = "section"
boundary_df

# %% [markdown]
# Each row gives the exact coordinate of a section boundary in every
# timeline and domain — including the Emerson recording's `dpt_16`
# column. Every column matches its own timeline's native type, and that is
# the point worth pausing on: pixel and sample counts are integers because
# those axes are discrete, seconds are floats, and quarterbeats are exact
# fractions. One cross-section can therefore hold three different number
# types at once without any of them being a compromise. Nothing here was
# converted for display — each value is the canonical one its axis declares.

# %% [markdown]
# ## 15. Summary & Key Takeaways
#
# > *"Any two events in the bundle can be related with each other —
# > regardless of whether they live on the same timeline, in the same
# > group, or even in the same domain — as long as a path of MatchClaims
# > or ConversionMaps connects them."*
#
# ### The Cascading Alignment Pattern
#
# The central demonstration of this notebook is that **a single additional
# group membership retroactively enriches every timeline already present
# in the bundle.** Adding CLT2\_unfolded to the Unfolded Score Group
# bridges two independent alignment networks:
#
# - **EEP recordings** (per-note MatchClaims) connect DPT1-DPT15 to CLT1
# - **Emerson recording** (section-boundary MatchClaims) connects DPT16 to
#   CLT2\_unfolded
# - **CLT2\_unfolded in the score group** bridges the two via within-group
#   interpolation
#
# ### Patterns Demonstrated
#
# | Pattern | Example | Section |
# |---------|---------|---------|
# | `build_recording_group()` | Reusable factory for EEP recordings | 2-4 |
# | `Ms3Loader.from_file()` | Load ABC score with notes, measures, annotations | 6 |
# | `create_flow_controller()` | Repeat structure + default flow | 6.1 |
# | `SegmentLine` nesting | OMR pages -> systems -> noteheads | 7 |
# | Region extraction | OpenScore 4-movement -> movement 4 child | 8 |
# | Cross-domain timestamps | Quarters -> pixels -> page number | 9.1 |
# | `TimelineGroup.apply_flow()` | Unfold entire group via one flow | 9.2 |
# | `match_notes_by_attributes()` | EEP <-> ABC note matching (from unfolded TL) | 10 |
# | `create_unfolded_timeline()` | Unfold a single timeline | 11.3 |
# | `MatchClaim` + `AlignmentAnchor` | Section-boundary alignment (alpha-kappa) | 11.5 |
# | `add_timeline()` on a group | Bridge independent alignment networks | 12.1 |
# | `MatchLine` + `WarpMap` | Explicit construction from MatchClaims | 13.1 |
# | `AlignmentBundle` | Multi-group cross-domain transfer | 13 |
# | `get_matchstamp_at()` | Universal coordinate resolution | 14 |
# | Reverse transfer | DPT16 -> all groups | 14.3 |
# | Cascading alignment | EEP <-> Score <-> Emerson via shared group | 12-14 |
#
# **5 groups, 18+ timelines, 3 domains, 1 bundle.**

# %%
