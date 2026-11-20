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
# # Pitch and Harmony across Formats
#
# ## What you will build
#
# You will build a typed comparison of pitch information from an ms3 table,
# two MusicXML readers, a MIDI performance, and a note-alignment export. By
# the end, you can state exactly which sources preserve a written spelling and
# which preserve only a sounding pitch, then apply the same distinction to
# harmony labels.
#
# ## Before you start
#
# Complete [The Data Model](tut08_data_model.ipynb) first; this notebook reuses
# its scalar, field, and protocol pattern.

# %% [markdown]
# ## Setup

# %%
import pandas as pd
import pyarrow as pa

from timetoalign import (
    Ms3Loader,
    Music21Loader,
    ParangonadaLoader,
    PartituraLoader,
    PerformanceMidiLoader,
)
from timetoalign.core import (
    DcmlHarmony,
    DcmlHarmonyField,
    EnharmonicPitch,
    EnharmonicPitchClass,
    EnharmonicPitchField,
    HarmonyLabel,
    MidiPitch,
    SpecificPitch,
    SpecificPitchClass,
    SpecificPitchField,
    build_struct_array,
)
from timetoalign.core.protocols import (
    DcmlHarmonyLike,
    EnharmonicPitchLike,
    GenericPitchLike,
    HarmonyLabelLike,
    RomanNumeralHarmonyLike,
    SpecificPitchClassLike,
    SpecificPitchLike,
    WesternTertianHarmonyLike,
)
from timetoalign.testdata import ensure_data

parangonar_data = ensure_data("parangonar")
vienna_data = ensure_data("vienna_1x22")

# %% [markdown]
# ## Faithfulness
#
# A typed view records what a source represented, not what another source
# you might infer. Begin with the two source facts that the rest of the
# notebook will compare.

# %%
source_facts = {
    "score notation": "C♯4",
    "MIDI performance": 61,
}
source_facts

# %% [markdown]
# The score fact includes a letter name and accidental; the MIDI fact is only
# MIDI pitch 61. Although both can describe the same key on a piano, the
# second fact does not contain enough information to choose C♯4 over D♭4.

# %% [markdown]
# ## Choose four source files
#
# To compare formats fairly, use the same Chopin work in each source. The
# mapping pairs every path with the public loader that understands it.

# %%
chopin_sources = {
    "ms3 TSV": (Ms3Loader, vienna_data / "ms3" / "chopin_op10_no3.notes.tsv"),
    "music21": (Music21Loader, vienna_data / "Chopin_op10_no3.musicxml"),
    "partitura": (PartituraLoader, vienna_data / "Chopin_op10_no3.musicxml"),
    "performance MIDI": (
        PerformanceMidiLoader,
        vienna_data / "Chopin_op10_no3_p01.mid",
    ),
}
chopin_sources

# %% [markdown]
# The dictionary shows one notes table, one MusicXML file read through two
# libraries, and one performance MIDI file. These paths feed the loaders in
# the next step.

# %% [markdown]
# ## Create the loaders
#
# Each loader uses the same public `from_file` construction route. Their short
# representations report how many source files and events were loaded.

# %%
chopin_loaders = {
    name: loader_class.from_file(path)
    for name, (loader_class, path) in chopin_sources.items()
}
chopin_loaders

# %% [markdown]
# The loader summaries show 498, 536, 547, and 3,874 loaded events. Next, ask
# each loader for its main event table, which narrows the score readers to
# their note data.

# %% [markdown]
# ## Retrieve the event tables
#
# The data-model tutorial introduced the loader-level `get_events` route. Here
# it gives us the {{< glossary Event >}} data from which every pitch view will
# be requested.

# %%
chopin_events = {name: loader.get_events() for name, loader in chopin_loaders.items()}
chopin_events

# %% [markdown]
# The representations show 498 rows for every symbolic reader and 3,874 for
# the `MidiEventData` table. The differing row counts mean that the selection
# below must not assume a common position.

# %% [markdown]
# ## Request a common pitch field
#
# `EnharmonicPitch` is the most informative pitch type shared by all four
# sources. Asking by scalar type returns the matching typed field.

# %%
enharmonic_fields = {
    name: events.get_field(EnharmonicPitch) for name, events in chopin_events.items()
}
enharmonic_fields

# %% [markdown]
# Every displayed value is an `EnharmonicPitchField`. The first three fields
# have length 498, while the MIDI field has length 3,874.

# %% [markdown]
# ## Define one selection rule
#
# A small helper makes the comparison reproducible: every field is searched
# for the first non-null value whose public `midi_number` attribute is 61.


# %%
def index_of_midi_pitch(pitch_field, midi_pitch):
    """Return the index of the first non-null value with this MIDI pitch."""
    for position, pitch in enumerate(pitch_field):
        if pitch is not None and pitch.midi_number == midi_pitch:
            return position
    raise ValueError(f"MIDI pitch {midi_pitch} does not occur")


# %% [markdown]
# The helper returns an integer row position and raises a clear `ValueError` if
# the requested MIDI pitch is absent. Apply exactly that rule four times.

# %% [markdown]
# ## Locate MIDI pitch 61
#
# The formats need not organise rows identically. Keeping their positions by
# source lets us retrieve the corresponding typed value from each field.

# %%
c_sharp_indices = {
    name: index_of_midi_pitch(field, source_facts["MIDI performance"])
    for name, field in enharmonic_fields.items()
}
c_sharp_indices

# %% [markdown]
# The first occurrences are at row 83 in the ms3 table, row 81 in both
# MusicXML readings, and row 931 in the MIDI stream. The positions are
# selection details, not musical claims.

# %% [markdown]
# ## Compare the common view
#
# Indexing each typed field at its saved position gives the four scalar values
# that the formats can be compared on without inventing a spelling.

# %%
c_sharp_by_format = {
    name: enharmonic_fields[name][position]
    for name, position in c_sharp_indices.items()
}
c_sharp_by_format

# %% [markdown]
# Every loader returns `EP(C♯/D♭4)` for MIDI pitch 61. The occurrence
# indices differ because these sources organise their rows differently, but
# their `EnharmonicPitch` fields agree on the sounding pitch without choosing
# a spelling.

# %% [markdown]
# ## Where they diverge
#
# The symbolic sources also represented spelling, so they afford
# `SpecificPitch`. The MIDI performance did not; request the same field
# deliberately and retain the resulting refusal as the cell value.

# %%
symbolic_sources = ("ms3 TSV", "music21", "partitura")
specific_fields = {
    name: chopin_events[name].get_field(SpecificPitch) for name in symbolic_sources
}
midi_specific_error = None
try:
    chopin_events["performance MIDI"].get_field(SpecificPitch)
except KeyError as exc:
    midi_specific_error = exc
midi_specific_error

# %% [markdown]
# The three assignments in `specific_fields` succeed. The displayed `KeyError`
# says that the MIDI table has no `SpecificPitchField`: refusing the request is
# the feature, because returning C♯4 or D♭4 would invent information.

# %% [markdown]
# ## What MIDI does afford
#
# MIDI does afford a numeric pitch view. The scalar exposes both its stored
# `midi_number` and its zero-based `pitch_class`.

# %%
midi_pitch_field = chopin_events["performance MIDI"].get_field(MidiPitch)
numeric_c_sharp = midi_pitch_field[c_sharp_indices["performance MIDI"]]
numeric_pitch_view = {
    "MIDI scalar": numeric_c_sharp,
    "midi_number": numeric_c_sharp.midi_number,
    "pitch_class": numeric_c_sharp.pitch_class,
}
numeric_pitch_view

# %% [markdown]
# `MP(61)` is the same datum as `EP(C♯/D♭4)` with a numeric face.
# Its public attributes show MIDI pitch 61 and pitch class 1.

# %% [markdown]
# ## Recognise pitch fields
#
# Scalar values live in typed columns. Checking the two field classes makes
# the scalar-versus-column distinction explicit.

# %%
pitch_field_checks = {
    "enharmonic field": isinstance(
        enharmonic_fields["performance MIDI"], EnharmonicPitchField
    ),
    "specific field": isinstance(specific_fields["ms3 TSV"], SpecificPitchField),
}
pitch_field_checks

# %% [markdown]
# Both checks are true. `EnharmonicPitchField` and `SpecificPitchField` are the
# PyArrow-backed columnar counterparts of their scalar types; indexing either
# field returns a typed scalar.

# %% [markdown]
# ## The pitch ladder
#
# Pitch types differ by how much they promise to know. Obtain the scalars
# from public fields, except for the octave-free spelling constructed with the
# documented `from_string` factory.

# %%
enharmonic_pitch = c_sharp_by_format["ms3 TSV"]
midi_pitch = numeric_pitch_view["MIDI scalar"]
specific_pitch = specific_fields["ms3 TSV"][c_sharp_indices["ms3 TSV"]]
specific_pitch_class = SpecificPitchClass.from_string("C♯")
enharmonic_pitch_class_field = enharmonic_fields["ms3 TSV"].convert_to(
    EnharmonicPitchClass
)
enharmonic_pitch_class = enharmonic_pitch_class_field[c_sharp_indices["ms3 TSV"]]
pitch_ladder = {
    "enharmonic pitch": enharmonic_pitch,
    "MIDI pitch": midi_pitch,
    "specific pitch": specific_pitch,
    "specific pitch class": specific_pitch_class,
    "enharmonic pitch class": enharmonic_pitch_class,
}
pitch_ladder

# %% [markdown]
# `EP(C♯/D♭4)` deliberately displays both spellings: `EnharmonicPitch`
# knows the octave and semitone but does not commit to an accidental.
# `MidiPitch` presents the same stored `midi_number` numerically, while
# `SpecificPitch` and `SpecificPitchClass` preserve the written spelling.
# `EnharmonicPitchClass` drops the octave as well as the spelling.

# %% [markdown]
# ## Protocols encode the ladder
#
# Protocols let an analysis require only the knowledge it needs. Testing the
# scalar values against all four protocols makes their guarantees explicit.

# %%
pitch_protocols = {
    "GenericPitchLike": GenericPitchLike,
    "EnharmonicPitchLike": EnharmonicPitchLike,
    "SpecificPitchClassLike": SpecificPitchClassLike,
    "SpecificPitchLike": SpecificPitchLike,
}
protocol_values = {
    "EnharmonicPitch": pitch_ladder["enharmonic pitch"],
    "SpecificPitchClass": pitch_ladder["specific pitch class"],
    "SpecificPitch": pitch_ladder["specific pitch"],
}
protocol_checks = {
    value_name: {
        protocol_name: isinstance(value, protocol)
        for protocol_name, protocol in pitch_protocols.items()
    }
    for value_name, value in protocol_values.items()
}
protocol_checks

# %% [markdown]
# All three values expose a pitch class. The octave-bearing enharmonic value
# satisfies `EnharmonicPitchLike`, and the spelled pitch class satisfies
# `SpecificPitchClassLike`; only the fully spelled, octave-bearing value
# satisfies `SpecificPitchLike`.

# %% [markdown]
# ## Side by side
#
# A compact table now separates the common enharmonic view from the extra
# spelling supplied by symbolic notation and the numeric face used for MIDI.

# %%
comparison_rows = [
    {
        "format": name,
        "enharmonic view": c_sharp_by_format[name],
        "specific view": specific_fields[name][c_sharp_indices[name]],
        "numeric view": "—",
    }
    for name in symbolic_sources
]
comparison_rows.append(
    {
        "format": "performance MIDI",
        "enharmonic view": c_sharp_by_format["performance MIDI"],
        "specific view": "not represented",
        "numeric view": numeric_c_sharp,
    }
)
four_format_comparison = pd.DataFrame(comparison_rows)
four_format_comparison

# %% [markdown]
# The enharmonic column visibly retains the two-spelling form `C♯/D♭4` in
# every row. Only the three symbolic rows contain `C♯4`; the performance row
# states that spelling was not represented rather than displaying an
# unexplained missing value.

# %% [markdown]
# ## A fifth source
#
# A parangonada export is a note alignment, rather than a score or a
# recording. Begin by loading the export directory so that its short summary is
# visible before asking it to build anything.

# %%
eroica_path = parangonar_data / "Beethoven_Eroica_op35-cpjku"
parangonada_loader = ParangonadaLoader.from_file(eroica_path)
parangonada_loader

# %% [markdown]
# The loader found five performers and 1,275 alignment claims. Those claims
# are ready to be assembled into the structure introduced earlier.

# %% [markdown]
# ## Build the alignment bundle
#
# `create_bundle` turns the loaded export into an earlier
# {{< glossary AlignmentBundle >}} without changing its pitch representation.

# %%
eroica_bundle = parangonada_loader.create_bundle()
eroica_bundle

# %% [markdown]
# The bundle summary shows 12 timelines arranged in six groups. The named
# score timeline is the source needed for the comparison.

# %% [markdown]
# ## Select the score timeline
#
# The score {{< glossary Timeline >}} carries the aligned score notes in
# quarters. Selecting it keeps the performance timelines out of this example.

# %%
eroica_score_timeline = eroica_bundle.get_timeline("score:clt_1")
eroica_score_timeline

# %% [markdown]
# The timeline is 64 quarters long and reports 251 events. Its representation
# also makes the quarters unit visible.

# %% [markdown]
# ## Retrieve the aligned score events
#
# The timeline's event table is the same public boundary used for the four
# Chopin sources.

# %%
eroica_events = eroica_score_timeline.get_events()
eroica_events

# %% [markdown]
# The event table confirms the same 251 rows, rational quarters, and no empty
# result. You can now request its two faithful pitch views.

# %% [markdown]
# ## Request the fifth source's pitch fields
#
# The alignment stores MIDI pitches, so it affords both enharmonic and numeric
# views but does not claim a written spelling.

# %%
eroica_pitch_fields = {
    "enharmonic": eroica_events.get_field(EnharmonicPitch),
    "numeric": eroica_events.get_field(MidiPitch),
}
eroica_pitch_fields

# %% [markdown]
# Both outputs are typed fields of length 251 over the same source column.
# Indexing them at one saved position will preserve their different displays.

# %% [markdown]
# ## Select an aligned pitch
#
# MIDI pitch 63 genuinely occurs in this Beethoven export. Reuse the same
# first-occurrence rule rather than pretending this is another copy of C♯4.

# %%
eroica_target_midi_pitch = 63
eroica_pitch_index = index_of_midi_pitch(
    eroica_pitch_fields["enharmonic"], eroica_target_midi_pitch
)
eroica_pitch_selection = {
    "row": eroica_pitch_index,
    "enharmonic view": eroica_pitch_fields["enharmonic"][eroica_pitch_index],
    "numeric view": eroica_pitch_fields["numeric"][eroica_pitch_index],
}
eroica_pitch_selection

# %% [markdown]
# Row 0 contains `EP(D♯/E♭4)` and `MP(63)`. The two renderings expose
# exactly what the shared source value supports.

# %% [markdown]
# ## Extend the comparison
#
# A final row places the note alignment beside the four earlier formats while
# stating plainly that no specific spelling was represented.

# %%
eroica_row = pd.DataFrame(
    [
        {
            "format": "parangonada note alignment",
            "enharmonic view": eroica_pitch_selection["enharmonic view"],
            "specific view": "not represented",
            "numeric view": eroica_pitch_selection["numeric view"],
        }
    ]
)
five_source_comparison = pd.concat(
    [four_format_comparison, eroica_row], ignore_index=True
)
five_source_comparison

# %% [markdown]
# The fifth row shows the same faithful choice as MIDI: alignment data carrying
# only MIDI pitches supports an enharmonic view, not a written spelling. The
# typed `MidiPitch` keeps its compact numeric rendering inside the table.

# %% [markdown]
# ## The same principle for harmony
#
# `HarmonyLabel` names the general harmony-label type. For a DCML label, the
# documented `DcmlHarmony.from_label` factory parses the encoded structure and
# gives us a scalar for testing the full harmony protocol chain.

# %%
parsed_harmony = DcmlHarmony.from_label("V65", globalkey="C")
harmony_protocol_checks = {
    "HarmonyLabelLike": isinstance(parsed_harmony, HarmonyLabelLike),
    "WesternTertianHarmonyLike": isinstance(parsed_harmony, WesternTertianHarmonyLike),
    "RomanNumeralHarmonyLike": isinstance(parsed_harmony, RomanNumeralHarmonyLike),
    "DcmlHarmonyLike": isinstance(parsed_harmony, DcmlHarmonyLike),
}
harmony_views = {
    "general label class": HarmonyLabel,
    "parsed DCML harmony": parsed_harmony,
    "label": parsed_harmony.label,
    "declared standard": parsed_harmony.standard,
    "parsed protocols": harmony_protocol_checks,
}
harmony_views

# %% [markdown]
# The output identifies the general class without constructing it through a
# Pydantic initializer. The parsed `V65` reports the `dcml` standard and
# satisfies all four protocols, from a labelled harmony to DCML-specific
# structure. Harmony uses `from_label`; pitch spelling uses `from_string`.

# %% [markdown]
# ## Building a harmony field
#
# Harmony uses the same bulk-construction route that the data-model tutorial
# used for pitch: build one Arrow struct array, then wrap it as the matching
# semantic field.

# %%
harmony_progression = [
    DcmlHarmony.from_label("I", globalkey="C"),
    DcmlHarmony.from_label("viio7", globalkey="C"),
    harmony_views["parsed DCML harmony"],
]
harmony_array = build_struct_array(DcmlHarmony, harmony_progression)
harmony_arrow_field = pa.field("harmony", harmony_array.type)
harmony_field = DcmlHarmonyField.from_field((harmony_array, harmony_arrow_field))
indexed_harmony = harmony_field[1]
harmony_field_result = {
    "is DcmlHarmonyField": isinstance(harmony_field, DcmlHarmonyField),
    "indexed scalar": indexed_harmony,
    "is DcmlHarmony": isinstance(indexed_harmony, DcmlHarmony),
}
harmony_field_result

# %% [markdown]
# The field retains the semantic type for the whole column, and indexing it
# returns the fully typed scalar `DcmlHarmony(label='viio7', key=C:I)`. No row
# dictionary or manual Arrow struct is needed.

# %% [markdown]
# ## What you learned
#
# - You can distinguish source representation from later musical inference.
# - You can pair four representations of one work with their public loaders.
# - You can construct those loaders through their shared `from_file` route.
# - You can retrieve the event table that each loader makes available.
# - You can request the common `EnharmonicPitchField` from every format.
# - You can define one explicit first-occurrence rule for MIDI pitches.
# - You can retain each format's row position instead of assuming shared rows.
# - You can compare four `EnharmonicPitch` scalars without inventing spelling.
# - You can interpret the absence of `SpecificPitch` from MIDI faithfully.
# - You can retrieve a `MidiPitch` and inspect its two numeric attributes.
# - You can recognise the field counterparts of typed pitch scalars.
# - You can obtain the pitch ladder through public fields and `from_string`.
# - You can use pitch protocols to express an analysis's requirements.
# - You can present represented and absent pitch views in one clear table.
# - You can load a parangonada note-alignment export.
# - You can assemble that export as an `AlignmentBundle`.
# - You can select its score timeline without mixing in performances.
# - You can retrieve the aligned score event table.
# - You can request enharmonic and numeric fields from alignment data.
# - You can select a genuine pitch from a second musical work.
# - You can add the alignment source without claiming a written spelling.
# - You can parse a DCML label and test the harmony protocol chain.
# - You can build and index a typed `DcmlHarmonyField` through the Arrow route.
#
# ## Next
#
# This concludes the tutorial series. Continue with the practical
# [How-to guide for loading the Vienna corpus](../howto/how03_loading_vienna_corpus.ipynb),
# then choose a guide that matches the format or alignment task in your own
# research.
#
# ## Go deeper
#
# [Load a parangonada note alignment](../howto/how03_parangonada.ipynb) and
# [load CSV or TSV data](../howto/how04_loading_csv.ipynb).
