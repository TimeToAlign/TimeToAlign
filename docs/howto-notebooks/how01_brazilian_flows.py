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
# # Inspect, Correct, and Re-Define Flows for Brazilian Choros
#
# For the whole API, see [How to Encode and Play Flow Control](how01_flow_control.ipynb).
# The condensed primer below covers only what is needed to read the inspection
# cells in this notebook.
#
# A score is **folded**: repeated sections appear once on the page but play
# multiple times in performance. A score loader writes the signs that say how
# as the timeline's **flow control**, a list of elements of five classes:
#
# - {{< glossary Jump >}}: where the flow may leave the axis, with one
#   destination for every place it may go on at (a repeat end, a *D.C.*, a
#   *D.S.*, a *To Coda*, the brackets of alternative endings);
# - {{< glossary JumpTarget >}}: a place a destination names (a repeat start,
#   a segno, a coda, a bracket);
# - {{< glossary Break >}}: a place the flow does not go on across (a *Fine*,
#   the end of a first bracket);
# - {{< glossary SectionBreak >}}: the end of a section;
# - {{< glossary Barline >}}: a barline the source prints.
#
# `Timeline.get_flow(convention)` plays the flow control (`default`, `single`
# or `printed`) and returns the flow with its {{< glossary Finding >}}s;
# `Timeline.create_flow_maps_from_flow_control()` writes the
# {{< glossary "Atomic section" >}}s as Regions and the flows as
# {{< glossary FlowMap >}}s. A wrong element is corrected by rejecting it
# (`flow_control.with_belief(id, actor, "false")`) and adding the right one.
#
# This notebook inspects three Choros scores. Each piece's section ends by
# exporting its flows to a CSV under the corpus `flows/` directory.

# %% [markdown]
# ## Setup

# %%
from pathlib import Path

import pandas as pd

from timetoalign import Ms3Loader, __version__
from timetoalign.core.enums import FlowConvention
from timetoalign.timelines.flow import atomic_sections, measure_ranges

DATA_DIR = Path("~/git/brazilian_flows").expanduser()


def make_paths(name: str, data_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    """Return ``(notes_tsv, measures_tsv)`` paths for a Brazilian Choros score basename."""
    return (
        data_dir / "notes" / f"{name}.notes.tsv",
        data_dir / "measures" / f"{name}.measures.tsv",
    )


def load_timeline(name: str):
    """Load a score's measures TSV as a timeline with its flow control."""
    _, measures = make_paths(name)
    return Ms3Loader.from_file(measures).create_timeline(uid=name)


def sections_table(timeline) -> pd.DataFrame:
    """One row per atomic section: its letter and its stretch in QB."""
    return pd.DataFrame(
        {"section": section.name, "qb_start": section.start, "qb_end": section.end}
        for section in atomic_sections(timeline.flow_control, timeline.length)
    )


def conventions_table(timeline) -> pd.DataFrame:
    """One row per convention: played length and number of findings, or the error."""
    rows = []
    for convention in FlowConvention:
        try:
            flow = timeline.get_flow(convention)
            rows.append(
                {
                    "convention": str(convention),
                    "folded_qb": timeline.length,
                    "played_qb": flow.length,
                    "findings": len(flow.findings),
                }
            )
        except ValueError as exc:
            rows.append({"convention": str(convention), "error": str(exc)})
    return pd.DataFrame(rows)


def findings(timeline) -> list[str]:
    """What playing the default flow found, one sentence per finding."""
    return [finding.message for finding in timeline.get_flow().findings]


def flow_rows(timeline, convention, source_file: str, software_version: str):
    """The rows of one flow in ``.flow.csv`` form."""
    sections = atomic_sections(timeline.flow_control, timeline.length)
    divide_at = [brk.coordinate for brk in timeline.flow_control.section_breaks]
    if convention == "atomic":
        stretches = sections
    else:
        stretches = [chunk.source for chunk in timeline.get_flow(convention).chunks]
    return [
        {
            "flow_mode": str(convention),
            "source_file": source_file,
            "software_version": software_version,
            "mc_start": row.start,
            "mc_end": row.end,
            "atomic_segments": ";".join(row.sections),
        }
        for row in measure_ranges(
            stretches, timeline.measure_map, sections=sections, divide_at=divide_at
        )
    ]


def export_flows(timeline, name: str) -> Path:
    """Write the atomic sections and every flow to ``{DATA_DIR}/flows/{name}.csv``.

    One block of rows for the atomic sections, then one per convention, in
    the columns of a ``.flow.csv`` file: ``flow_mode``, ``source_file``,
    ``software_version``, ``mc_start``, ``mc_end``, ``atomic_segments``.
    """
    _, measures = make_paths(name)
    software_version = f"timetoalign {__version__}"
    rows = []
    for convention in ("atomic", *FlowConvention):
        rows.extend(flow_rows(timeline, convention, measures.name, software_version))
    out_path = DATA_DIR / "flows" / f"{name}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


# %% [markdown]
# ---
#
# ## score_1988-Ciume_e_brincadeira

# %%
piece_name = "score_1988-Ciume_e_brincadeira"
timeline = load_timeline(piece_name)
timeline.flow_control

# %%
sections_table(timeline)

# %%
timeline.flow_control.breaks

# %%
timeline.flow_control.jumps

# %%
timeline.flow_control.jump_targets

# %%
conventions_table(timeline)

# %%
# Export all flows for this piece to {DATA_DIR}/flows/{piece_name}.csv.
export_flows(timeline, piece_name)

# %% [markdown]
# ---
#
# ## score_1361-Medrosa-Anacleto_de_Medeiros

# %%
piece_name = "score_1361-Medrosa-Anacleto_de_Medeiros"
timeline = load_timeline(piece_name)
timeline.flow_control

# %%
sections_table(timeline)

# %%
timeline.flow_control.breaks

# %%
timeline.flow_control.jumps

# %%
timeline.flow_control.jump_targets

# %%
conventions_table(timeline)

# %%
# Export all flows for this piece to {DATA_DIR}/flows/{piece_name}.csv.
export_flows(timeline, piece_name)

# %% [markdown]
# ---
#
# ## score_133-Digitalis-Irineu_de_Almeida

# %%
piece_name = "score_133-Digitalis-Irineu_de_Almeida"
timeline = load_timeline(piece_name)
timeline.flow_control

# %%
sections_table(timeline)

# %%
timeline.flow_control.breaks

# %%
timeline.flow_control.jumps

# %%
timeline.flow_control.jump_targets

# %%
conventions_table(timeline)

# %%
# Export all flows for this piece to {DATA_DIR}/flows/{piece_name}.csv.
export_flows(timeline, piece_name)

# %% [markdown]
# ---
#
# ## What playing found
#
# Playing the default flow reports, without raising, what looks wrong: a flow
# that never ends (`endless_loop`), a flow ending at a Break that states no
# condition before the end (`flow_ends_at_break`), two destinations of one
# Jump holding at once (`ambiguous_targets`), a destination nobody can check
# (`undetermined_target`), and a stretch never played (`never_played`). Each
# list below holds one sentence per finding; an empty list means the default
# flow found nothing. Nothing is repaired: a reading that corrects the source
# rejects the wrong elements and adds the right ones.

# %%
findings(load_timeline("score_1988-Ciume_e_brincadeira"))

# %%
findings(load_timeline("score_1361-Medrosa-Anacleto_de_Medeiros"))

# %%
findings(load_timeline("score_133-Digitalis-Irineu_de_Almeida"))

# %% [markdown]
# ---
#
# ## Demo: FlowMaps
#
# Swap `demo_name` to inspect any of the three pieces above. The cells below
# write the piece's atomic sections and its flows as FlowMaps, then convert
# positions through them with `unfold()` and `fold()`.

# %%
demo_name = "score_1988-Ciume_e_brincadeira"  # or score_1361-... / score_133-...

notes_demo, measures_demo = make_paths(demo_name)
loader_demo = Ms3Loader.from_file(notes_demo, measures_demo)
score_tl = loader_demo.create_timeline()
computed = score_tl.create_flow_maps_from_flow_control()
default_map = computed.flow_maps[0]
resolved = default_map.resolve(score_tl)
{
    "demo_name": demo_name,
    "n_sections": len(computed.sections),
    "played_length": resolved.length,
    "attached_maps": score_tl.list_flow_maps(),
}

# %%
# A folded coord inside the first section appears at as many unfolded
# positions as the section is played.
first = computed.sections[0]
coord = (first.start.value + first.end.value) / 2
unfolded = resolved.unfold(coord)
{
    "source_coord": coord,
    "unfolded_positions": unfolded,
    "n_appearances": len(unfolded),
}

# %%
# Every unfolded position folds back to the same source coord.
[(position, resolved.fold(position)) for position in unfolded]

# %%
# The same folded coord behaves differently under each computed FlowMap.
pd.DataFrame(
    {
        "map_id": flow_map.id,
        "convention": str(flow_map.convention),
        "n_appearances": len(flow_map.resolve(score_tl).unfold(coord)),
    }
    for flow_map in computed.flow_maps
)
