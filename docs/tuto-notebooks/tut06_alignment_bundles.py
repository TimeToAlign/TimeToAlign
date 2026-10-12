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
# # Alignment Bundles and MatchClaims
#
# ## What you will build
#
# You will build a small {{< glossary AlignmentBundle >}}
# that connects a score, its tick grid, and an irregular performance. Its
# alignment records stated evidence, so you can query one position across every
# connected {{< glossary Timeline >}} and see whether the answer is exact or
# interpolated.
#
# ## Before you start
#
# Complete [Timeline Groups](tut05_timeline_groups.ipynb),
# which introduced interpolation within a group.

# %%
import warnings
from collections import Counter
from fractions import Fraction

from timetoalign import (
    AlignmentBundle,
    ClaimType,
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    Coordinate,
    DiscreteLogicalTimeline,
    MatchClaim,
    MatchfileLoader,
    MatchLine,
    MatchStamp,
    SecondsToSamples,
    TimeUnit,
    WarpMap,
)
from timetoalign.core import SupportPolicy
from timetoalign.testdata import ensure_data

vienna_data = ensure_data("vienna_1x22")

# %% [markdown]
# ## Interpolation is a guess; a claim is evidence
#
# The previous tutorial stretched one timeline onto another at a constant
# ratio. Real performances change pace, so we need to assert that **this**
# {{< glossary Coordinate >}} corresponds to **that** one and interpolate only
# between those observations.

# %%
score = ContinuousLogicalTimeline(
    length=Fraction(12), uid="edition_quarters", name="Edition"
)
performance = ContinuousPhysicalTimeline(
    length=8.0, uid="performance_seconds", name="Performance"
)

score_events = [
    {"id": "score-c1", "event_type": "Cue", "instant": Fraction(2)},
    {"id": "score-c2", "event_type": "Cue", "instant": Fraction(4)},
    {"id": "score-c3", "event_type": "Cue", "instant": Fraction(8)},
    {"id": "score-c4", "event_type": "Cue", "instant": Fraction(10)},
]
performance_events = [
    {"id": "perf-c1", "event_type": "Cue", "instant": 1.0},
    {"id": "perf-c2", "event_type": "Cue", "instant": 2.5},
    {"id": "perf-c3", "event_type": "Cue", "instant": 5.75},
    {"id": "perf-c4", "event_type": "Cue", "instant": 7.5},
    {"id": "perf-ornament", "event_type": "Ornament", "instant": 6.5},
]
score.add_events(score_events)
performance.add_events(performance_events)

score_cues = score.get_events(event_type="Cue")
performance_cues = performance.get_events(event_type="Cue")
score_cue_frame = score_cues.to_dataframe(coordinates=True)
performance_cue_frame = performance_cues.to_dataframe(coordinates=True)
observed_pairs = list(zip(score_cue_frame["start"], performance_cue_frame["start"]))
observed_pairs

# %% [markdown]
# Each pair contains unit-bearing coordinates from real {{< glossary Event >}}s.
# Their changing spacing is the evidence that a single constant ratio would be
# a poor model of this performance.

# %% [markdown]
# ## A {{< glossary MatchClaim >}}
#
# One claim says that an event or coordinate on one timeline corresponds to an
# event or coordinate on another. `create_match_claims()` accepts several such
# pairs and records who or what supplied the evidence.

# %%
claim_bundle = AlignmentBundle(name="Visible teaching claims")
claim_bundle.add_timeline(score)
claim_bundle.add_timeline(performance)

event_pairs = [
    (score_id, score.id, performance_id, performance.id)
    for score_id, performance_id in zip(
        score_cue_frame["id"], performance_cue_frame["id"]
    )
]
evidence_claims = claim_bundle.create_match_claims(
    event_pairs,
    agent="musicologist",
    agent_identifier="manual-cue-alignment",
)
shown_claim = evidence_claims[1]
shown_claim

# %% [markdown]
# The rendered claim names both events, both timelines, both coordinates, and
# its provenance. It is an assertion about one observed correspondence, not a
# rule for the whole performance.

# %%
single_event_claim = MatchClaim.from_events(
    score.get_event("score-c2"),
    score.id,
    performance.get_event("perf-c2"),
    performance.id,
    unit_a=score.unit,
    unit_b=performance.unit,
)
construction_comparison = {
    "standalone from_events claim": single_event_claim,
    "claims registered by create_match_claims": len(claim_bundle.get_match_claims()),
}
construction_comparison

# %% [markdown]
# Reach for `MatchClaim.from_events()` when you have one event pair and want a
# standalone claim; its `unit_a` and `unit_b` arguments make the two event
# conventions explicit. Use `create_match_claims()` when you have several
# pairs in a bundle: it gets the units from the registered timelines and adds
# all four claims here to the bundle, as the output shows.

# %% [markdown]
# ## The kinds of claim
#
# {{< glossary ClaimType >}} is derived from a claim's structure; it is never
# stored separately. The possibilities are `event_match`, `projection`,
# `anchor`, {{< glossary NOMATCH >}}, `conceptual`, and `implicit`.

# %%
anonymous_claim = claim_bundle.create_match_claims(
    [({"start": Fraction(3)}, score.id, {"start": 1.7}, performance.id)]
)[0]
projection_event = performance.get_event("perf-c2")
projection_claim = MatchClaim.from_projection(
    event=projection_event,
    source_tl_id=performance.id,
    target_tl_id=score.id,
    target_coord=Coordinate(Fraction(4), TimeUnit.quarters),
    source_unit=TimeUnit.seconds,
)
absence_claim = claim_bundle.create_match_claims(
    [(None, score.id, "perf-ornament", performance.id)]
)[0]
conceptual_claim = MatchClaim.nomatch(
    event={},
    source_tl_id=performance.id,
    target_tl_id=score.id,
    unit=TimeUnit.seconds,
)
implicit_claim = MatchClaim.implicit(
    tl_a_id=score.id,
    coord_a=Coordinate(Fraction(2), TimeUnit.quarters),
    tl_b_id=performance.id,
    coord_b=Coordinate(1.0, TimeUnit.seconds),
    source_claim=evidence_claims[0],
)

claim_examples = {
    ClaimType.event_match: evidence_claims[0],
    ClaimType.projection: projection_claim,
    ClaimType.anchor: anonymous_claim,
    ClaimType.nomatch: absence_claim,
    ClaimType.conceptual: conceptual_claim,
    ClaimType.implicit: implicit_claim,
}
derived_types = {kind.value: claim.claim_type for kind, claim in claim_examples.items()}
derived_types

# %% [markdown]
# A synchronous claim names two events, one event, or no events to become an
# event match, projection, or anonymous anchor. A non-synchronous claim with
# exactly one named event is a NOMATCH; here it points **from the performance**,
# which has `perf-ornament`, **to the score**, which lacks it. There is no
# dedicated conceptual constructor: passing an empty event to
# `MatchClaim.nomatch()` leaves no named event, so the claim becomes
# `conceptual`. `implicit` wins over every other discriminator because it marks
# a relationship inferred by graph extension rather than asserted.

# %% [markdown]
# ## The bundle
#
# A bundle organises commensurable timelines into {{< glossary TimelineGroup >}}s.
# Every timeline is registered under its own id. `as_group=` starts a group and
# gives it a name, while the group's id is minted (`group_1`, `group_2`, …);
# `grouped_with=` joins the group of a timeline that is already registered.

# %%
score_ticks = DiscreteLogicalTimeline(
    length=5760, uid="edition_ticks", name="Edition MIDI grid"
)
bundle = AlignmentBundle(name="Edition and performance")
bundle.add_timeline(score, as_group="edition")
bundle.add_timeline(score_ticks, grouped_with=score.id)
bundle.add_timeline(performance, as_group="performance")
bundle.add_match_claims(evidence_claims)

edition_group = bundle.get_group_for_timeline(score.id)
performance_group = bundle.get_group_for_timeline(performance.id)
bundle_structure = {
    "group_ids": bundle.group_ids,
    "group names": [edition_group.name, performance_group.name],
    "edition members": edition_group.timeline_ids,
    "performance members": performance_group.timeline_ids,
    "all timeline_ids": bundle.timeline_ids,
}
bundle_structure

# %% [markdown]
# The edition group contains quarters and integer ticks because those axes are
# perfectly commensurable. The performance begins a separate group; the claims,
# rather than group membership, connect it to the edition.

# %% [markdown]
# ## From claims to a map
#
# Claims between two groups accumulate into a {{< glossary MatchLine >}}. The
# line returns {{< glossary AlignmentAnchor >}}s: each one names both timelines
# and carries both coordinates, so an anchor says which position on which axis
# corresponds to which. A {{< glossary WarpMap >}} is built from that same line
# and interpolates positions between the anchors. The two views in the output
# below show the same four pairs: the asserted positions print as `Coordinate`
# objects, while the anchors use the library's display formatter, which states
# the unit once and drops a trailing zero — `@1 seconds` and
# `Coordinate(1.0, seconds)` are the same value written two ways.

# %%
bundle_claims = bundle.get_match_claims()
match_line = MatchLine.from_claims(bundle_claims, score.id)
line_anchors = match_line.get_alignment_anchors(performance.id)
warp_map = WarpMap.from_match_line(
    match_line,
    performance.id,
    source_unit=TimeUnit.quarters,
    target_unit=TimeUnit.seconds,
)
between_coordinate = Coordinate(Fraction(6), TimeUnit.quarters)
mapped_between = warp_map.get_coordinate_at(between_coordinate)
map_summary = {
    "asserted positions": observed_pairs,
    "ordered claim anchors": line_anchors,
    "map anchors": warp_map.n_anchors,
    "interpolated position": mapped_between,
}
map_summary

# %% [markdown]
# You have now met all three transfer mechanisms: exact offsets between parent
# and child, linear interpolation within a group, and warp maps across groups.

# %% [markdown]
# ## Querying
#
# `bundle.get_matchstamp_at(coord, timeline_id)` returns a
# {{< glossary MatchStamp >}}, the second and wider rung after the
# {{< glossary TimeStamp >}} that timelines and groups return. Both have the
# same coordinate and unit accessors.

# %%
claimed_coordinate = Coordinate(Fraction(4), TimeUnit.quarters)
exact_stamp = bundle.get_matchstamp_at(claimed_coordinate, score.id)
between_stamp = bundle.get_matchstamp_at(between_coordinate, score.id)
claim_stamp = shown_claim.get_matchstamp()

stamp_comparison = {
    "claimed": {
        "query": claimed_coordinate,
        "tick grid": exact_stamp.get_coordinate(score_ticks.id),
        "performance": exact_stamp.get_coordinate(performance.id),
        "is_interpolated": exact_stamp.is_interpolated,
    },
    "between claims": {
        "query": between_coordinate,
        "tick grid": between_stamp.get_coordinate(score_ticks.id),
        "performance": between_stamp.get_coordinate(performance.id),
        "is_interpolated": between_stamp.is_interpolated,
    },
    "claim getter agrees": (
        claim_stamp.get_coordinate(performance.id)
        == exact_stamp.get_coordinate(performance.id)
    ),
    "stamp class": isinstance(exact_stamp, MatchStamp),
}
stamp_comparison

# %% [markdown]
# `False` means the query itself was one of the asserted coordinates. `True`
# means the answer lies between assertions and came from the WarpMap. The tick
# coordinate is also present because the queried score belongs to the edition
# group.

# %% [markdown]
# ## In batches
#
# Batch queries start from coordinates on the query timeline, not from claim
# objects. `get_matchstamps_at` is the plural of `get_matchstamp_at` and returns
# stamps; `get_matchstamp_table` takes the same arguments and places those same
# cross-sections into columns.

# %%
query_coordinates = score_cue_frame.loc[1:2, "start"].tolist()
batch_stamps = bundle.get_matchstamps_at(query_coordinates, score.id)
batch_table = bundle.get_matchstamp_table(query_coordinates, score.id)
claim_table = bundle.get_matchstamp_table()
batch_summary = {
    "query coordinates": query_coordinates,
    "performance coordinates": [
        stamp.get_coordinate(performance.id) for stamp in batch_stamps
    ],
    "coordinate-query table shape": batch_table.shape,
    "one-row-per-claim table shape": claim_table.shape,
}
batch_summary

# %% [markdown]
# The two query coordinates come directly from the score's cue events and yield
# two full cross-sections. The no-argument table is a different view: it has one
# sparse row for each synchronous claim, which is why its row count follows the
# number of assertions rather than the number of query coordinates. These are
# PyArrow tables; later tutorials explain their columnar representation.

# %% [markdown]
# ## Converted units are opt-in here
#
# A bundle can span many timelines, each with several derived units. Matchstamp
# getters therefore omit such conversions by default and expose them only when
# `conversion_maps=True`.

# %%
samples_map = SecondsToSamples(sample_rate=48000)
performance.add_conversion_map(samples_map)
performance_query = between_stamp.get_coordinate(performance.id)
default_conversion_stamp = bundle.get_matchstamp_at(performance_query, performance.id)
converted_stamp = bundle.get_matchstamp_at(
    performance_query,
    performance.id,
    conversion_maps=True,
)
try:
    default_samples = default_conversion_stamp.get_unit(TimeUnit.samples)
except KeyError as missing_map:
    default_samples = f"KeyError: {missing_map}"
conversion_comparison = {
    "default": default_samples,
    "conversion_maps=True": converted_stamp.get_unit(TimeUnit.samples),
}
conversion_comparison

# %% [markdown]
# By default the conversion is not merely empty — it raises, because derived
# units were never requested and a missing map is reported rather than returned
# as a blank. With the opt-in, the sample coordinate arrives as an integer, as
# a discrete unit requires. Timeline stamps defaulted to showing conversions in
# the earlier tutorial; doing that across a large bundle would bury the
# alignment answer.

# %% [markdown]
# ## Outside the evidence
#
# {{< glossary SupportPolicy >}} controls queries beyond the first or last
# claim: `omit` is the default, while `clamp` and `extrapolate` retain the
# destination timeline in different ways.

# %%
before_evidence = Coordinate(Fraction(1), TimeUnit.quarters)
omitted_stamp = bundle.get_matchstamp_at(before_evidence, score.id)
clamped_stamp = bundle.get_matchstamp_at(
    before_evidence,
    score.id,
    support_policy=SupportPolicy.clamp,
)
extrapolated_stamp = bundle.get_matchstamp_at(
    before_evidence,
    score.id,
    support_policy=SupportPolicy.extrapolate,
)
support_comparison = {
    "bundle default": bundle.support_policy,
    "omit": (
        omitted_stamp.get_coordinate_for(performance.id)
        if performance.id in omitted_stamp.present_timelines
        else "absent from the stamp"
    ),
    "clamp": clamped_stamp.get_coordinate_for(performance.id),
    "extrapolate": extrapolated_stamp.get_coordinate_for(performance.id),
}
support_comparison

# %% [markdown]
# `omit` leaves the unsupported performance out of the stamp altogether, so it
# is not among `present_timelines` and asking for it raises; that is the
# conservative choice for analysis, and checking membership is how you ask
# without assuming. `clamp` is useful when an interface must remain at the
# nearest known boundary. Extrapolate only a short distance when the local
# tempo trend is itself a defensible assumption.

# %% [markdown]
# ## Loading an alignment instead of building one
#
# A {{< glossary MatchfileLoader >}} reads the same model from Vienna `.match`
# files. Here one score is connected to 22 performances.

# %%
match_files = sorted(vienna_data.glob("*.match"))
match_loader = MatchfileLoader()
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"Quarter duration .*",
        category=UserWarning,
        module=r"partitura\.utils\.music",
    )
    match_loader.load(*match_files)
vienna_bundle = match_loader.create_bundle()
vienna_diagram = vienna_bundle.diagram(max_standalone=4, depth=0)
print(vienna_diagram)

# %% [markdown]
# The diagram introduces the loaded bundle before any of its identifiers are
# used. It shows one score timeline, 22 performance timelines, and the claims
# that connect them. This is the same model built by hand above, only larger and
# populated by a loader.

# %% [markdown]
# ## Reading a loaded cross-section
#
# We can discover the score identifier from its registered group, choose a
# frequently matched score coordinate, and read that position across several
# performances.

# %%
vienna_score_group_id = vienna_bundle.group_ids[0]
vienna_score_group = vienna_bundle.get_group(vienna_score_group_id)
vienna_score_id = vienna_score_group.timeline_ids[0]
vienna_claims = vienna_bundle.get_match_claims()
vienna_score_coordinates = [
    claim.get_coordinate_for(vienna_score_id)
    for claim in vienna_claims
    if claim.is_synchronous and claim.connects(vienna_score_id)
]
coordinate_counts = Counter(vienna_score_coordinates)
vienna_coordinate = coordinate_counts.most_common(1)[0][0]
vienna_stamp = vienna_bundle.get_matchstamp_at(vienna_coordinate, vienna_score_id)
vienna_performance_ids = [
    timeline_id
    for timeline_id in vienna_stamp.present_timelines
    if timeline_id != vienna_score_id
][:4]
vienna_shown_ids = [vienna_score_id, *vienna_performance_ids]
vienna_cross_section = {
    timeline_id: vienna_stamp.get_coordinate(timeline_id)
    for timeline_id in vienna_shown_ids
}
vienna_result = {
    "selected claim coordinate": vienna_coordinate,
    "queried cross-section": vienna_cross_section,
}
vienna_result

# %% [markdown]
# The score identifier comes from the bundle rather than a hardcoded string.
# The bundle query preserves the claim's exact value: the selected coordinate
# and the score entry of the returned cross-section are the same
# `Fraction(65, 2)` quarters, because a quarters axis is exact whichever route
# a value reaches it by. The performance coordinates are integer ticks for the
# same reason — each timeline's entry follows that timeline's own declared
# type, so one cross-section legitimately mixes fractions and integers.

# %% [markdown]
# ## Merging
#
# `AlignmentBundle.from_bundles([...])` registers the source groups, timelines,
# and claims in a new bundle. It does not invent evidence between the sources;
# add bridge claims when you know how the two sides correspond.

# %%
facsimile = ContinuousGraphicalTimeline(
    length=1200.0, unit=TimeUnit.points, uid="facsimile_points", name="Facsimile"
)
facsimile_bundle = AlignmentBundle(name="Facsimile only")
facsimile_bundle.add_timeline(facsimile, as_group="facsimile")
merged_bundle = AlignmentBundle.from_bundles(
    [bundle, facsimile_bundle], name="Edition, performance, and facsimile"
)
merged_claims_before_bridge = merged_bundle.get_match_claims()
unbridged_stamp = merged_bundle.get_matchstamp_at(between_coordinate, score.id)
bridge_claims = merged_bundle.create_match_claims(
    [
        ({"start": Fraction(2)}, score.id, {"start": 200.0}, facsimile.id),
        ({"start": Fraction(10)}, score.id, {"start": 1000.0}, facsimile.id),
    ],
    agent="page-alignment",
    agent_identifier="manual-landmarks",
)
bridged_stamp = merged_bundle.get_matchstamp_at(between_coordinate, score.id)
merge_summary = {
    "registered groups": len(merged_bundle.group_ids),
    "registered timelines": len(merged_bundle.timeline_ids),
    "claims carried from sources": len(merged_claims_before_bridge),
    "bridge claims added": len(bridge_claims),
    "facsimile before bridge": (
        unbridged_stamp.get_coordinate_for(facsimile.id)
        if facsimile.id in unbridged_stamp.present_timelines
        else "unreachable from the score"
    ),
    "facsimile after bridge": bridged_stamp.get_coordinate_for(facsimile.id),
}
merge_summary

# %% [markdown]
# Before the bridge, a score query cannot reach the facsimile. Two page
# landmarks supply enough evidence for interpolation; afterwards the same
# query reaches the performance, tick grid, and facsimile position. Merging
# preserves knowledge, but only claims create new knowledge between bundles.

# %% [markdown]
# ## What you learned
#
# - You can replace a constant-ratio guess with explicit coordinate evidence.
# - You can create and inspect a MatchClaim with recorded provenance.
# - You can distinguish every derived ClaimType and orient a NOMATCH correctly.
# - You can start and extend timeline groups inside an AlignmentBundle.
# - You can turn claims into a MatchLine and a WarpMap.
# - You can tell an exact MatchStamp from an interpolated one.
# - You can query coordinates singly, in batches, or as a table of claims.
# - You can opt into converted units without crowding the default answer.
# - You can choose how out-of-support queries behave.
# - You can load a 22-performance alignment into the same object you built.
# - You can merge bundles and add only the bridge evidence you actually have.
#
# ## Next
#
# [Flow Control and Grids](tut07_flow_and_grids.ipynb)
#
# ## Go deeper
#
# [Create a note alignment](../howto/how03_create_note_alignment.ipynb),
# [load the Vienna corpus](../howto/how03_loading_vienna_corpus.ipynb), and
# [transfer annotations](../howto/how01_thoresen_annotation_transfer.ipynb).
