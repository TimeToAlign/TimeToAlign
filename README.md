# TimeToAlign!

[![DOI](https://img.shields.io/badge/DOI-10.5334%2Ftismir.296-blue)](https://doi.org/10.5334/tismir.296)
[![PyPI](https://img.shields.io/pypi/v/timetoalign)](https://pypi.org/project/timetoalign/)
[![Python versions](https://img.shields.io/pypi/pyversions/timetoalign)](https://pypi.org/project/timetoalign/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-timetoalign.github.io-blue)](https://timetoalign.github.io/)

This library is described in the TISMIR article *Time to Align! Modelling Musical Timelines for Music Information Retrieval and Digital Musicology* (2026): https://doi.org/10.5334/tismir.296.

A Python library for representing and aligning musical timelines.

**Documentation:** https://timetoalign.github.io/ · **Source:** https://github.com/timetoalign/timetoalign

## Installation

```bash
pip install timetoalign         # Core only — lightweight
```

The core install pulls in only PyArrow, pandas, NetworkX,
typing_extensions, and pydantic.  This gives you the full timeline / map /
alignment framework but no file-format-specific loaders.  If you need loaders,
plotting, or Jupyter support, install one of the [optional extras](#optional-dependencies)
described below.  To run the **tutorial notebooks** (see below), for example:

```bash
pip install "timetoalign[tutorial]"    # Loaders + plotting + Jupyter
```

### Optional Dependencies

TimeToAlign! organises its optional dependencies into *atomic* extras
(one concern each) and *composite* extras (convenience bundles that
include several atomic ones).  You can mix and match freely:
`pip install "timetoalign[midi,plot]"` is perfectly valid.

#### Atomic extras

Each atomic extra adds support for a single loader backend or feature.

| Extra | Packages | Purpose |
|---|---|---|
| `midi` | `mido` | MIDI file loading (`PerformanceMidiLoader`) |
| `partitura` | `partitura` | Score parsing via partitura (`PartituraLoader`, `ScoreMidiLoader`) |
| `music21` | `music21` | Score parsing via music21 (`Music21Loader`) |
| `ms3` | `ms3` | DCML TSV score parsing (`Ms3Loader`) |
| `audio` | `soundfile`, `mutagen` | Audio file loading + MP3/M4A metadata |
| `graphical` | `pymupdf`, `pillow` | PDF/image loading & drawing |
| `plot` | `matplotlib` | Visualisation |
| `delta` | `deltalake` | Delta Lake columnar storage (future) |
| `rdf` | `rdflib` | RDF / linked-data export (future) |
| `examples` | `pooch` | On-demand fetching of bundled corpora |

#### Composite extras

Composite extras are convenience bundles that pull in several atomic
extras at once.  Runtime levels include everything below them; `docs`
and `testing` target documentation builds and test-suite execution.

| Extra | Includes | Purpose |
|---|---|---|
| `scores` | `partitura`, `music21`, `ms3` | All score-loader backends |
| `loaders` | `midi`, `scores`, `audio`, `graphical` | Every loader dependency |
| `tutorial` | `loaders`, `plot`, `examples`, plus `jupytext`, `jupyter` | Everything needed for the tutorial notebooks |
| `docs` | `tutorial`, plus `quartodoc`, `griffe` | Documentation site building |
| `all` | `tutorial`, `delta`, `rdf` | All runtime features |
| `testing` | `all`, `examples`, plus `pytest`, `pytest-cov`, `pytest-xdist`, `pytest-benchmark`, `hypothesis` | Test-suite execution |

The inclusion chain is:

```
all  ⊃  tutorial  ⊃  loaders  ⊃  { midi, scores, audio, graphical }
                                 +  plot, examples, jupytext, jupyter
                   +  delta, rdf

docs     ⊃  tutorial, quartodoc, griffe
testing  ⊃  all, examples, pytest, pytest-cov, pytest-xdist, pytest-benchmark, hypothesis
```

#### Examples

```bash
pip install timetoalign                  # Core only
pip install "timetoalign[midi]"          # Core + MIDI loading
pip install "timetoalign[partitura]"     # Core + partitura score parsing
pip install "timetoalign[scores]"        # Core + all score-loader backends
pip install "timetoalign[loaders]"       # Core + every loader
pip install "timetoalign[tutorial]"      # Loaders + plotting + Jupyter
pip install "timetoalign[docs]"          # Tutorial stack + documentation build tools
pip install "timetoalign[testing]"       # Runtime features + test tools
pip install "timetoalign[all]"           # All runtime features
```

### Development install

Contributing or running the test suite requires an editable, source
install with the `dev` extra:

```bash
git clone https://github.com/timetoalign/timetoalign.git
pip install -e "./timetoalign[dev]"      # Editable install + everything + dev tooling
cd timetoalign && pip install --group conformance   # Test oracles (pip >= 25.1)
```

## Quick Start

```python
import timetoalign as tta

# Create a 60-second audio timeline
audio = tta.ContinuousPhysicalTimeline(length=60.0, uid="audio", name="Piano Recording")

# Add beats and notes -- IDs and temporal_type are inferred automatically
audio.add_events([
    {"event_type": "Beat", "instant": 0.0},
    {"event_type": "Beat", "instant": 0.5},
    {"event_type": "Beat", "instant": 1.0},
    {"event_type": "Note", "start": 0.0, "end": 0.5},
    {"event_type": "Note", "start": 8.0, "end": 12.0},   # spans Intro -> Verse
])

# Nest children (sections of the same recording)
intro  = audio.create_child(length=10.0, offset=0.0,  uid="intro",  name="Intro")
verse  = audio.create_child(length=20.0, offset=10.0, uid="verse",  name="Verse")
chorus = audio.create_child(length=15.0, offset=30.0, uid="chorus", name="Chorus")

# Attach ConversionMaps (seconds -> milliseconds, seconds -> samples at 48 kHz)
audio.add_conversion_map(tta.ScalarMap(scalar=1000, source_unit="seconds", target_unit="milliseconds"))
audio.add_conversion_map(tta.SecondsToSamples(sample_rate=48000))

# The diagram shows the hierarchy at a glance
print(audio.diagram())
```

```
ContinuousPhysicalTimeline[audio] (5 events, 3 children, 2 cmaps)
                      0 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 60 seconds
  ├─ Intro            0 ~~~~~                               10
  ├─ Verse           10      ~~~~~~~~~~~~                   30
  └─ Chorus          30                  ~~~~~~~~~          45
```

```python
# A TimeStamp is a cross-section through the entire hierarchy at a given coordinate.
# Only children whose span covers the coordinate appear:
ts = audio.get_timestamp(25.0)
print(ts)
```

```
TimeStamp @25 seconds
  audio         25 seconds
  verse         15 seconds
  milliseconds  25000 milliseconds
  samples       1200000 samples
```

```python
# A TimeIntervalStamp gives the cross-section for a [start, end) range.
# The note at [8, 12) straddles Intro [0, 10) and Verse [10, 30):
# its start falls in Intro, its end in Verse.
tis = audio.get_interval_stamp(8.0, 12.0)
print(tis)
```

```
TimeIntervalStamp [8, 12) seconds
                 start     end
  audio              8      12 seconds
  intro              8       - seconds
  verse              -       2 seconds
  milliseconds    8000   12000
  samples       384000  576000
```

## Tutorial Notebooks

The `docs/tuto-notebooks/` and `docs/howto-notebooks/` directories contain
Jupytext percent-script notebooks covering the library from first principles
to full alignment workflows.
Make sure you have installed the `tutorial` extra (see
[Installation](#installation)), then generate the paired `.ipynb` files
and launch Jupyter:

```bash
jupytext --sync docs/*-notebooks/*.py
jupyter notebook docs/tuto-notebooks/
```

## Development

```bash
# Run the test suite
pytest

# Run linting / pre-commit hooks
tox -e lint
```

## Glossary

| Term | Definition |
|---|---|
| AlignmentAnchor | A set of TimeStamps corresponding to matched events, representing their temporal equivalence across timelines. From a MatchClaim for TimeIntervalEvents, the library derives a start anchor and end anchor. |
| AlignmentBundle | The primary container object that manages a collection of timelines, their groupings, and transfers between them. |
| BeatGrid | An ordered set of tempo segments and the beats they generate on a seconds axis. Each BeatGridSegment states an anchor instant, a tempo, a BeatPolicy and the anchor's beat-in-bar index, and generates beats without end; the grid bounds each segment by its successor and labels the beats with measure and beat numbers. Not a Timeline: nothing is stored per beat. |
| Break | A control event that voids contiguity at its Instant. TimeIntervals cannot span a Break, and Breaks cannot be inserted into existing TimeIntervals. |
| ChainMap | A composed ConversionMap that applies multiple ConversionMaps in sequence, creating a conversion path from source to target unit. |
| Child | A timeline nested within a parent timeline, sharing the same measuring unit. `add_child()` locks a child by default to prevent side effects from modifications; an unlocked child may grow, contained by its parent. |
| CombinationMap | A ConversionMap that yields outputs from multiple ConversionMaps simultaneously, such as (x, y) coordinate pairs. |
| Commensurable | Two timelines are commensurable when connected by a match path or ConversionMap chain, enabling coordinate translation between them. |
| Composite map | An umbrella concept for ConversionMaps composed from other maps, including ChainMap, CombinationMap, and PiecewiseMap. |
| Contiguity | A TimeInterval is contiguous if it monotonically spans all coordinates between its start and end. An Instant is contiguous with a TimeInterval if it is synchronous with its EndInstant. |
| Control Event | An event affecting flow control: either a Break (voiding contiguity) or a Jump (creating new contiguity). |
| ConversionMap | A function mapping any coordinate to at most one value (another coordinate, specifier, or constant). Also called C-map. |
| Coordinate | A position on a timeline, expressed as the distance from the origin in the timeline's measuring unit. |
| Discrete/Continuous | A timeline is discrete when coordinates exist only at discrete points (e.g., pixels, samples); continuous otherwise (e.g., seconds, quarters). |
| Domain | One of three temporal categories: Graphical (visual/spatial), Logical (symbolic/musical), or Physical (audio/sound). |
| Event | Anything associated with a timeline via Instants. An InstantEvent has zero duration; a TimeIntervalEvent has duration defined by start and end coordinates. |
| FlowControlElement | A taxonomy of control events distinguishing between structural markers (e.g., *repeat_start*, *double_barline*) and jump instructions (e.g., *dal_segno*, *to_coda*). |
| FlowMap | A sequence of TimeIntervals representing a specific traversal path through a timeline, handling Jumps and repeats. |
| IdCoordinate | A coordinate specification that explicitly includes the unique identifier of the timeline to which it belongs, preventing ambiguity in multi-timeline contexts. |
| Instant | Associates a coordinate with a signification such as "start of event e". Instants sharing a coordinate are synchronous. |
| InterpolationMap | A *ConversionMap* that performs coordinate conversion via linear interpolation between a set of known correspondence points. |
| Inverse map | The reverse transformation of a bijective ConversionMap. |
| Jump | A control event with JumpFrom and JumpTo Instants. When active, makes events at JumpTo contiguous with those ending at JumpFrom (e.g., repeats, *dal segno*). |
| Length | The distance between a timeline's origin and its last Instant. A locked timeline cannot extend its length. |
| LinearMap | A *ConversionMap* implementing an affine transformation, f(x) = ax + b. Includes *ScalarMap* (b=0) and *ShiftMap* (a=1) as special cases. |
| MatchClaim | A claim — issued by a human or algorithmic agent — that events from disparate timelines are synchronous or equivalent, carrying provenance metadata (agent, criteria, certainty) and comprising one (instant) or two (interval) AlignmentAnchors. |
| MatchLine | An ordered line that links MatchClaim objects through MatchStamps, enabling selection and alignment of temporal correspondences. |
| MatchMetadata | Structured provenance information attached to a *MatchClaim*, recording the agent, decision criteria, and certainty level of the match. |
| Match path | A graph traversal connecting matched events across multiple timelines, for example A -> B -> C. |
| MetricMap | A specialized *ConversionMap* that handles complex metrical conversions (e.g., quarter notes to measure count) accounting for time signatures and anacrusis. |
| Origin | The zero coordinate of a timeline, from which all positions are measured. |
| PiecewiseMap | A ConversionMap that applies different sub-maps to disjoint coordinate intervals. |
| Region | A named part of a timeline defined by a TimeInterval (e.g., "Chorus", "Verse"). Regions are not timelines and cannot hold events or maps directly. |
| RotationMap | A *ConversionMap* that implements periodic or cyclic transformations using modular arithmetic. |
| Segment | A Child timeline that is contiguous with its siblings. A SegmentLine is a parent containing only contiguous Segments. |
| Synchrony | Strict synchrony: Instants sharing identical coordinates. Pragmatic synchrony: Instants binned together based on a threshold (e.g., for quantisation). |
| Time interval | Defined by a StartInstant and EndInstant; left-inclusive and right-exclusive [s, e). The EndInstant's coordinate must be >= the StartInstant's. |
| Timeline | A positive coordinate axis minimally defined by its origin and measuring unit. Accommodates events and potentially Children. |
| TimelineGroup | An array of commensurable timelines that share the same start and end: every member's 0 coincides with every other's, and so does every member's length; positions in between convert linearly and exactly. A part of a timeline takes part as its Child. |
| TimeStamp | A cross-section through a timeline hierarchy, comprising the root coordinate, synchronous Child coordinates, and all ConversionMap results. |
| WarpMap | A derived timeline where coordinates are re-adjusted based on AlignmentAnchors to align with another timeline. |

## Citing

If you use TimeToAlign!, please cite:

Hentschel, Johannes; Berndt, Axel; Cancino-Chacón, Carlos; Dixon, Simon; Foo, Anne; Gotham, Mark; Hu, Patricia; Köster, Maik; Martins, Felipe D.; Mauro, Davide A.; Müller, Meinard; Neuwirth, Markus; Pacha, Alexander; Page, Kevin R.; Peter, Silvan; Polyakov, Egor; Pugin, Laurent; Weigl, David M.; Weiß, Christof; and Widmer, Gerhard. 2026. "Time to Align! Modelling Musical Timelines for Music Information Retrieval and Digital Musicology." *Transactions of the International Society for Music Information Retrieval* 9 (1): 384--404. https://doi.org/10.5334/tismir.296.

```bibtex
@article{Hentschel2026TimeToAlign,
  author  = {Hentschel, Johannes and Berndt, Axel and Cancino-Chacón, Carlos and Dixon, Simon and Foo, Anne and Gotham, Mark and Hu, Patricia and Köster, Maik and Martins, Felipe D. and Mauro, Davide A. and Müller, Meinard and Neuwirth, Markus and Pacha, Alexander and Page, Kevin R. and Peter, Silvan and Polyakov, Egor and Pugin, Laurent and Weigl, David M. and Weiß, Christof and Widmer, Gerhard},
  title   = {Time to Align! Modelling Musical Timelines for Music Information Retrieval and Digital Musicology},
  journal = {Transactions of the International Society for Music Information Retrieval},
  year    = {2026},
  volume  = {9},
  number  = {1},
  pages   = {384--404},
  doi     = {10.5334/tismir.296}
}
```

## License

MIT
