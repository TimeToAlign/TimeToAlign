# TimeToAlign! -- Documentation & Notebooks

This directory contains the documentation homepage and the runnable
notebooks for the `timetoalign` library.

## Quick start

From the `timetoalign/` directory (the library root), install the
library with all dependencies needed to run the notebooks:

```bash
pip install -e ".[tutorial]"
```

## Documentation homepage

Open `docs/page/_site/index.html` in a browser. The homepage includes the
full conceptual specification, a glossary, API reference, and HTML
renderings of every notebook listed below.

## Notebooks

Notebooks are stored as Jupytext `py:percent` files (the `.py` file makes the notebooks version-controllable; paired `.ipynb` files are regenerated via
`jupytext --sync *.py`).

### Tutorials (`tuto-notebooks/`)

Step-by-step introductions to the library, from first principles to
complete alignment workflows.

| File | Title | Topics |
|------|-------|--------|
| `tut00_quickstart` | Quickstart | What the library is for, in five minutes: one section per tutorial that follows |
| `tut01_timelines_and_coordinates` | Timelines and Coordinates | Coordinates, units, domains, the six timeline classes, and conversion maps |
| `tut02_nesting_and_timestamps` | Nesting and Timestamps | Children, offsets, regions, segment lines, and the cross-section a timestamp gives you |
| `tut03_events` | Events on a Timeline | Adding events, EventData, the shared filter vocabulary, and one timestamp per event |
| `tut04_loading_data` | Loading Real Data | The two-phase loader contract, EventStores, timeline ids, and what a loader builds for you |
| `tut05_timeline_groups` | Timeline Groups | Commensurability, interpolation between peers, group timestamps, and partial alignment |
| `tut06_alignment_bundles` | Alignment Bundles and MatchClaims | Claims, claim types, MatchLines and WarpMaps, MatchStamps, and support policies |
| `tut07_flow_and_grids` | Flow Control and Grids | Folded and unfolded time, FlowMaps, apply_flow, and beat grids |
| `tut08_data_model` | The Data Model | Why PyArrow, the three layers, raw and semantic fields, blueprints, and Parquet metadata |
| `tut09_pitch_and_harmony` | Pitch and Harmony across Formats | What each format knows about a note, and why the library refuses to invent the rest |

### How-to notebooks (`howto-notebooks/`)

Focused recipes for specific tasks, including four of the five examples from
the manuscript (not Chorissimo from Figure 5).

| File | Title | Topics |
|------|-------|--------|
| `how01_coordinate_math` | Coordinate Math | Domains, TimeUnits, NumberType, arithmetic |
| `how01_advanced_cmaps` | Advanced Conversion Maps | ChainMap, PiecewiseMap, TableMap |
| `how01_advanced_timestamps` | Querying Timestamps | TimeStamp, TimeIntervalStamp, boundary tables |
| `how01_manual_timeline_construction` | Constructing Timelines Manually | Events via dict, parent/child hierarchies |
| `how01_loading_data` | Loading Data | Format-agnostic ingestion with Loaders |
| `how01_tabular_loaders` | Loading Tabular Data | CoordinateField, ComputedField |
| `how01_graphical_timelines` | Graphical Timelines | TimeAxisPath, image timelines, pixel-to-time conversion |
| `how01_beat_grids` | Building Beat Grids | BeatGrid, tempo segments, beat tables, annotation-tool export |
| **`how01_thoresen_annotation_transfer`** | **Transferring Annotations Between Graphical Analyses** | **SegmentLine, ConstantMap, MatchClaim, y-coordinate transfer (manuscript Section 3.3)** |
| **`how02_supra_piano_roll`** | **Aligning a Piano Roll (SUPRA)** | **IIIF images, ATON, MIDI, Audio, Score (manuscript Section 3.1)** |
| `how03_create_note_alignment` | Creating a Note Alignment | MatchClaims, AlignmentBundle, MatchLines (manuscript Section 3.4 prerequisite) |
| **`how03_beethoven_multimodal`** | **Aligning Multimodal Data (Beethoven)** | **AlignmentBundle, FlowMap, OMR, 23 timelines, 3 domains (manuscript Section 3.4)** |
| `how03_loading_vienna_corpus` | Loading the Vienna 4x22 Corpus | MatchfileLoader, 22 performances, AlignmentBundle |
| `how03_ieee1599` | How to Load an IEEE 1599 Document | Ieee1599Loader, spine/VTU, multimodal AlignmentBundle across score/pixels/audio |
| `how03_engraved_score_alignment` | How to Align an Engraved Score with a Recording and a Drawn Time Axis | VerovioLoader, MapsLoader, ScoreWarpLoader, css_pixels, create_calibrated_child |
| `how03_describe_the_work` | How to Describe What a Bundle and Its Timelines Stand For | Statements and provenance on a bundle, a timeline and a region; proposals and decisions, the inline context, MeiHeadLoader, MelodyLoader |
| **`how04_hendrix_song_genesis`** | **Encoding Song Genesis (Hendrix)** | **MatchClaim, NOMATCH, synchronous vs conceptual (manuscript Section 3.5)** |

Notebooks highlighted in bold correspond directly to examples in the
manuscript.

### Running a notebook

```bash
jupytext --sync docs/tuto-notebooks/tut00_quickstart.py
jupyter notebook docs/tuto-notebooks/tut00_quickstart.ipynb
```

Or open any of the pre-rendered HTML versions on the documentation
homepage.
