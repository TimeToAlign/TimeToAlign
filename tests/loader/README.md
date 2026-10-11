# Loader tests

This directory tests the loader package end-to-end, covering all formats
(``tabular/``, ``score/``, ``midi/``, ``graphical/``, ``format/``, and
``physical/``), the ``EventStore`` and ``EventData`` machinery,
and the shared loader infrastructure (base classes, matchfiles, schemas,
bundles, error handling).

## Ingestion contract

Loader fixtures return field-oriented arrays so the tests exercise the public
vectorized loader contract directly. Row dictionaries remain covered by the
storage constructor tests, but are intentionally not accepted by
`Loader.load()`.

## Categories

| File / Subdirectory | What it validates |
|---------------------|-------------------|
| `test_loader.py` | Generic loader smoke tests and contract checks |
| `test_base_loaders.py` | The `EventLoader` / `ManifestLoader` / `AlignmentLoader` ABCs |
| `test_bundle.py` | `AlignmentBundle` produced by alignment loaders |
| `test_error_handling.py` | Faulty-input behaviour across loaders |
| `test_interval_policy.py` | Half-open interval semantics on event ingestion. Duration recomputed from `start`/`end` under the `warn`/`strict` "both present" branch stays an exact `Fraction` (not a rounded float) when both endpoints carry exact ratios; falls back to float subtraction otherwise. All recomputed endpoints/durations are asserted with exact `==` (zero tolerance): the integer/half operands round-trip bit-exactly, and the one genuinely-float fallback (`3.0 - 1/3`, mixed exact/float operands) is pinned against the identical Python expression, so no `pytest.approx` remains in this file. |
| `test_matchfile_loader.py` | `MatchfileLoader` parity against gold standard, including quarters-to-seconds claim units while preserving all match values and counts; `id_pattern` includes both matching and no-match filters |
| `test_performance_precision_loader.py` | `PerformancePrecisionLoader` against the CAAMP Chopin Nocturne specimen — composes `SoloLoader` for the `.solo` score (2494 notes), builds the score timeline by resolving every `"<measure>+<offset>"` label to absolute quarters via the `MetricMap`, and emits one physical timeline + three granularities of `MatchClaim` per performer. `id_pattern` includes both matching and no-match filters. Zero-tolerance counts (see "Validation logic" below). |
| `test_parangonada_loader.py` | `ParangonadaLoader` against the parangonada CSV export of the `Beethoven_Eroica_op35-cpjku` dataset (5 performers). Builds one shared multimodal `AlignmentBundle` (1 score group + 5 performer groups) from `part.csv` / `ppart.csv` / `align.csv`, plus measured `Beat` / `Dynamics` feature events (`.beats` / `.dyn`) on each performance's seconds timeline. `id_pattern` includes both matching and no-match filters. Zero-tolerance counts (see "Validation logic" below). |
| `test_listen_here_loader.py` | `ListenHereLoader` against an inline synthetic Listen Here! alignment JSON (built in `tmp_path`). Parses many recordings of one work warped onto a shared equidistant reference grid into one audio-to-audio `AlignmentBundle`: one empty seconds timeline per recording (each its own group) and complete-topology pairwise synchronous claims held columnar in a `MatchClaimField`. Tests assert that both sides of every bulk-produced anchor materialise with seconds units, and that `id_pattern` returns an exact empty list for a no-match pattern. Zero-tolerance counts (see "Validation logic" below). |
| `test_parsing.py` | Format-agnostic parsing helpers |
| `test_schema.py` | Loader schema and field-spec resolution |
| `test_store.py` | `EventStore` low-level operations |
| `test_tilia_loader.py` | `TiliaJsonLoader` round-trip, including `create_group(uids=...)` keyword selection |
| `test_mixins.py` | `EventData` field-access mixins — three-strategy field discovery (metadata, default-column, shape-based `matches_pa_field`), `has_field`, `get_field`, `get_fields`, `get_raw`, and the convenience accessors (`get_pitch_field`, `get_harmony_field`). |
| `test_mixins_semantic_access.py` | `SemanticFieldAccessMixin` dispatch — `get_field(ScalarClass)` pydantic-scalar dispatch, `IdCoordinate` vs `Coordinate` discrimination via metadata (`matches_pa_field` rejection contracts), `MultipleFieldsError` on ambiguity + `name=` resolution, and `get_fields_satisfying(ProtocolClass)` Protocol-based grouping (covering `GenericPitchLike` and `TimeScalarLike`). |
| `test_field_parsers.py` | The :class:`FieldParser` hierarchy and `resolve_field_parser` universal-resolution dispatcher. Exercises the DataField blueprint mechanism: `IntField`, `FloatField`, `StringField`, `RedundantNumberField`, `DenominateNumberField`, and paired SemanticField subclasses all accept `name=` for blueprint construction and expose a uniform `emit(source, name=...)` materialisation. `CompositeFieldParser` (separator + regex strategies, dict + iterable parts) and `CallableFieldParser` (escape hatch) are exercised end-to-end. Resolution-table assertions: every entry (Python type, `pa.DataType`, raw / paired `DataField` subclass, blueprint instance, `FieldParser` instance, callable) routes to the correct producer. |
| `test_step2_field_specs.py` | Step 2 (`field_specs`) blueprint resolution. Builds a fixture `pa.Table` and a `TabularLoader` subclass with `field_specs = [...]`, verifies that each blueprint matches its declared `source_fields=` entry, that the resulting column receives the versioned metadata blob (`field_type` = paired class name), that atomic source columns are packed into single-field structs matching the target `pa_schema`, and that unresolvable references raise `KeyError`. Exercises the two currently-supported `source_fields=` shorthands (string for single-source promotion; explicit dict for multi-sub-field mapping) and the negative cases (list shorthand rejected by `resolve_source_fields` today; live-mode SemanticField instances rejected; multi-source dict spec raises `NotImplementedError` at loader-materialisation time; a target that cannot be built from a name alone — `CoordinateField` and its siblings carry a bound time unit — raises a `TypeError` naming that target and its storage shape, rather than letting the field's own constructor fail on an argument the caller never passed). |
| `test_get_events_properties.py` | The four shapes accepted by `Loader.get_events(properties=...)` — `True`, `False`, a tuple of property names, and the single-string shorthand that normalises to a one-element tuple. |
| `test_represent_pitch_once.py` | The **represent-pitch-once** contract across every pitch-bearing EventData and loader. Verifies (1) the keystone `from_dicts` / `add_events` struct-preservation fix (carried struct-dict columns become real `pa.struct` columns with `field_type` metadata, never JSON strings); (2) the uniform `_afforded_fields` mechanism that promotes a raw atomic column to its semantic view on request; (3) the per-source default pitch type (`get_pitch_field()` → SP for spelled, EP for number-only, EP for MSM with SPC additionally afforded); (4) represent-once (no EventData carries a redundant *default* pitch struct — score notes no longer afford a default EnharmonicPitch); (5) the scalar↔EventData contract for MIDI EventData; (6) on-request EP from a SP field via both routes (conversion + raw column); (7) the **multi-batch concat re-affordance** — the afforded pitch view survives `EventData.extend` / `Timeline.add_events` schema-promotion across batches (the cache is dropped so the affordance re-attaches over the concatenated table). See "Represent pitch once" below. |
| `tabular/` | CSV / TSV / Parquet loader specifics |
| `score/` | Music-notation loaders (Ms3, music21, Partitura) |
| `midi/` | Score and performance MIDI loaders |
| `graphical/` | PDF / image loaders |
| `format/` | Cross-format loaders (JSON, XML, TTL) |
| `physical/` | Audio loaders and time-coordinate ingestion |

## Exact rational coordinates

Loader schema tests import coordinate encoders and decoders from
`timetoalign.core.time`, the sole owner of the number storage struct, its
converter family, and the one builder that fills it. Parser outputs are
compared against that canonical type.

Coordinate columns arriving from a file are built by
`build_number_struct_array` under the representation the target unit declares.
Unreadable cells are the one place the two error policies differ and the tests
pin both: a loader parsing a coordinate column raises, because a malformed
position is a malformed file; a blueprint field parser records the cell as
missing, because a tolerant reader of a messy table should not lose the other
million rows to one bad one.

The score-side loader tests assert the numerator and denominator stored with
logical coordinates, not only their floating-point convenience values. MS3
measure starts, durations, and ends are compared with exact `Fraction`
values from the TSV data. Matchfile score events are checked the same way for
known normalized onsets, durations, and ends. These assertions ensure that
rhythmic positions remain exactly addressable after they are placed in
PyArrow event storage, while performance and recording coordinates retain
their native numeric representation.

### MS3 coordinate validation logic

MS3 notes and measures use the symbolic `duration` or `act_dur` whole-note
fraction, multiplied by four, as the authoritative exact duration, and the
tests assert that exact `Fraction` — no tolerance, no range. Where a derived
`duration_qb` disagrees with the symbolic value, the derived value is the one
stored, and it too is asserted exactly.

There is no denominator threshold above which a value stops being exact, and
no such thing as a coordinate stored "with null exact fields". Every non-null
cell carries the number on both sides of the struct, and which side is
authoritative comes from the field's declared `number_type`, not from
inspecting the row. Score coordinates are fraction-canonical, so the tests
assert the ratio members and check that the float member mirrors them.
Starts follow the same rule, and interval ends are asserted from the
resulting stored coordinates.

## Data conventions

Alignment-loader tests treat anchor units as part of the gold-standard meaning of
every coordinate: score positions use the unit of the score timeline (for example,
quarters), performance and recording positions use seconds, and discrete score
positions use their timeline's discrete unit. The migration assertions add unit and
exact-value checks without changing any established coordinate value, claim count,
or NOMATCH count.

Timeline identity assertions use the library's type-based grammar rather than
source labels. Matchfile score/performance timelines are stored as
``score:clt_1`` and ``perf:<source-stem>:dlt_1``; Performance Precision uses
``score:clt_1`` and ``perf:<performer-key>:cpt_1``. TiLiA timelines are numbered
``cpt_1``, ``cpt_2``, ... in source order. Source filenames, performer keys, and
TiLiA IDs/titles remain human-readable metadata and lookup inputs, so changing
these expected IDs tests identity semantics without changing parsed content.

The tabular schema tests retain ``Field`` and ``ComputedField`` because
``TabularLoader`` uses them for nested coordinate selection and computed
coordinates. ``parse_json_to_struct`` remains required because both live
descriptors use it when a selected nested column arrives as JSON text. Tests
for ``ConvertedField`` are removed because it had no production consumer after
the ``column_specs``/``field_specs`` pipeline became authoritative.
``TableSchema`` is likewise removed without replacement because no loader
constructed it.

Tests resolve corpus paths via ``timetoalign.testdata.ensure_data("<corpus>")``
(see ``tests/data/README.md``).  Hardcoded relative ``Path("tests/data/...")``
constants are forbidden — they break under ``jupytext --execute`` and in CI
container layouts.  See ``tests/data/README.md`` for the binding contract.

## `PerformancePrecisionLoader` validation logic

Corpus: ``performance_precision`` (CAAMP export of Chopin Nocturne Op. 9
No. 2 — one ``.solo`` score, one Verovio timemap ``.json``, and an
``Alignments/`` directory with three CSVs for each of 7 recordings).

The loader's correctness rests on a single coordinate resolver that turns
a ``"<measure>+<offset_in_whole_notes>"`` label into an absolute
quarter-note position:

```
abs_quarters(M, offset_wn) = measure_start_q[M] + offset_wn * 4
```

``measure_start_q[M]`` is taken from the ``MetricMap`` built from the
Verovio timemap for LABEL measure ``M >= 1``.  LABEL measure ``0`` is the
anacrusis, whose offsets are measured from a *virtual full-bar downbeat*
preceding the first sounding note: ``measure_start_q[0] = starts[1] −
first_meter_quarters``, where ``first_meter_quarters`` is the nominal
length of the first bar (``12/8`` → ``6`` quarters) read from the
timemap's first ``meterSig``.  On this specimen ``measure_start_q[0] ==
Fraction(-11, 2)`` (−5.5).

**Validated resolver anchors** (exact ``Fraction``):

| LABEL | abs quarters |
|-------|--------------|
| `0+11/8` | `0` |
| `1+0/1` | `1/2` |
| `2+0/1` | `13/2` |
| `3+0/1` | `25/2` |
| `32+3/4` | `379/2` |
| `37+3/2` | `425/2` |

**Composed `SoloLoader`:** 2494 events.

**Score timeline:** 2494 events, unit quarters, length `Fraction(425, 2)`
(212.5).  The pickup note (`note_id == "n1b8xktz"`) resolves to start
quarter `0`; a `1+0/1` note resolves to `1/2`.

**Per performer (identical for all 7):** the note-level file has 559 data
rows splitting 480 *aligned* (synchronous claims) + 79 *dangling*
(`TIME == "N"`, NOMATCH claims); the bar file has 32 rows (all
synchronous); the beat file has 376 rows (all synchronous).  These are the
true data-row counts; raw `wc -l` figures (560/33/377) include the
``LABEL,TIME,FRAME`` header.

**Totals across 7 performers:** 6769 MatchClaims = 6216 synchronous + 553
NOMATCH.  The bundle holds 8 timelines (1 score group + 7 standalone
performers).

**Coordinate spot-check:** Ashkenazy's bar-level claim for LABEL `1+0/1`
has source coordinate `Fraction(1, 2)` (0.5 quarters) and target
coordinate `10.272` seconds.

The 6 of 1247 note onsets that do not resolve exactly are inherent tuplet
rounding in the ``.solo``'s quantised representation; the
``.solo``/``MetricMap`` arithmetic is the faithful score representation
and the tests assert exact values only on the validated anchors above.

## `ParangonadaLoader` validation logic

Corpus: ``parangonar`` (the parangonada CSV export of the
``Beethoven_Eroica_op35-cpjku`` dataset — Beethoven Eroica Variations
op. 35, Var. XIV, 5 performers). The loader ingests the *existing*
alignments; it never runs an aligner.

### Dataset shape

The 5 performers live as subdirectories of
``match/match_transkun/``. Their directory names carry a *non-uniform*
suffix — only two end in ``_parangonada``:

| Performer key | Subdirectory name |
|---------------|-------------------|
| `1966_Szegedi` | `1966_Szegedi_parangonada` |
| `1970_Gould`   | `1970_Gould_parangonada` |
| `1971_Curzon`  | `1971_Curzon` |
| `1985_Brendel` | `1985_Brendel` |
| `2023_Hewitt`  | `2023_Hewitt` |

Performers are discovered by globbing the immediate subdirectories of
``match/match_transkun/`` (not the sibling ``.csv`` / ``.match`` files
that also live there) and requiring each to contain ``part.csv``,
``ppart.csv``, and ``align.csv``. The key is the directory name with a
trailing ``_parangonada`` stripped. They are sorted by key, which is
chronological: Szegedi, Gould, Curzon, Brendel, Hewitt.

Three CSV families:

* ``part.csv`` — the shared score. **Byte-identical across all 5**
  performers (same md5), so it is parsed once. Header
  ``onset_beat,duration_beat,onset_quarter,duration_quarter,onset_div,
  duration_div,pitch,voice,id``; **251** data rows; ``id`` unique;
  ``pitch`` is a MIDI integer; ``onset_quarter`` a value parsed as an
  exact ``Fraction`` (the score has an anacrusis at ``-1/2``);
  ``onset_div`` an int. Two arithmetic invariants hold for all 251 rows
  (0 exceptions): ``onset_div == 32 * onset_quarter + 16`` and
  ``duration_div == 32 * duration_quarter``.
* ``ppart.csv`` — per-performer performance notes in seconds. Header
  ``onset_sec,duration_sec,pitch,velocity,track,channel,id``; ``id``
  like ``n0``; ``onset_sec`` a float. Data-row counts: Szegedi 232,
  Gould 253, Curzon 246, Brendel 249, Hewitt 244.
* ``align.csv`` — per-performer correspondences. Header
  ``idx,matchtype,partid,ppartid``. The "no counterpart" sentinel is
  the literal string ``undefined``. ``matchtype`` ``0`` = match (both
  ids present), ``1`` = score-only / insertion (``ppartid ==
  "undefined"``), ``2`` = performance-only / deletion (``partid ==
  "undefined"``).

The ``score.mei`` (partitura raises ``KeyError: 'dur'`` on it),
``feature.csv`` (all-zero placeholder), ``zalign.csv`` (byte-duplicate
of ``align.csv``), ``majority_match/*.match``, and the
``transcribed_midi/`` tree are deliberately **not** parsed.

The per-performer ``.beats`` / ``.dyn`` files under the dataset's
``features/`` directory **are** parsed (see "Feature events" below).

### Bundle structure

One shared **score group** ``"score"`` (the same 251 notes in two
logical units):

* ``score:clt_1`` — `ContinuousLogicalTimeline`, quarters, 251 note
  events with ``start = onset_quarter`` (exact `Fraction`), carrying
  ``pitch`` (MIDI int) and ``voice`` (int).
* ``score:dlt_1`` — `DiscreteLogicalTimeline`, ticks (``divs``), the
  same 251 notes with ``start = onset_div`` (int). A divs→quarters
  C-Map ``LinearMap(scalar=Fraction(1,32), offset=Fraction(-1,2))`` is
  attached (``scalar`` / ``offset`` are the constructor's real
  parameter names); it reproduces every ``onset_quarter`` from
  ``onset_div`` exactly (asserted for all 251 rows, exact `Fraction`
  equality).

Per performer, a **performance group** ``perf:<key>`` (the same
performed notes in two physical units):

* ``perf:<key>:cpt_1`` — `ContinuousPhysicalTimeline`, seconds, one note
  event per ``ppart.csv`` row with ``start = onset_sec`` (float),
  carrying ``pitch`` and ``velocity``; **plus** the measured beat
  features (see "Feature events" below) as ``Beat`` and ``Dynamics``
  events on the same timeline.
* ``perf:<key>:dpt_1`` — `DiscretePhysicalTimeline`, samples, the same
  notes with ``start = round(onset_sec * sample_rate)``. A
  ``SamplesToSeconds(sample_rate=44100)`` C-Map is attached
  (``sample_rate`` read from the performer's 44100 Hz stereo ``.wav``
  via `AudioLoader`).

### Feature events

Each performance carries a pair of measured beat-feature files under
the dataset's ``features/`` directory, tab-separated. They are
discovered by **globbing** ``<key>_measure*.{beats,dyn}`` — the stems
are non-uniform: Szegedi's are ``1966_Szegedi_measure_fix.{beats,dyn}``,
the other four are ``<key>_measure.{beats,dyn}``. All ten files have
**63 data rows**.

* ``.beats`` (7 columns) — header
  ``measure_number  beat  onset_beat  onset_sec  BPM  interp  swap``.
  Each row becomes a ``Beat`` event on ``perf:<key>:cpt_1`` at
  ``start = onset_sec``, carrying ``measure_number`` / ``beat`` /
  ``bpm`` (from ``BPM``) / ``interp`` / ``swap``. Event id
  ``<key>:beat:<i>``.
* ``.dyn`` (6 columns) — header
  ``measure_number  beat  onset_beat  velocity_mean  velocity_max
  interp`` (**no** ``onset_sec``, **no** ``swap``). Each row becomes a
  ``Dynamics`` event on ``perf:<key>:cpt_1`` carrying ``measure_number``
  / ``beat`` / ``velocity_mean`` / ``velocity_max`` / ``interp``. Event
  id ``<key>:dyn:<i>``.

Because the ``.dyn`` rows have no onset, each is joined to the
``.beats`` row sharing its ``(measure_number, beat)`` key to recover
its ``onset_sec``. The two files are **1:1 on ``(measure_number,
beat)``**: the keys are unique within each file and the two key sets
are identical, so the join yields exactly 63 matches (asserted). These
are *measured* per-beat features (variable tempo + dynamics from the
recording); no tempo curve is synthesised (no ``BeatGrid.from_tempo``).

The events are added to the **existing** seconds timeline (not a new
timeline, not on ``dpt_1``) via ``add_events(rows,
allow_expansion=True)``. The ``cpt_1`` schema already holds ``Note``
columns (``pitch`` / ``velocity``); the ``Beat`` / ``Dynamics`` rows
introduce new columns (``bpm`` / ``measure_number`` / ``beat`` /
``velocity_mean`` / …). Successive ``add_events`` calls grow the schema
via ``pyarrow.concat_tables(promote_options="default")``, so the
heterogeneous ``Note`` / ``Beat`` / ``Dynamics`` table round-trips with
absent columns null-filled. After ingestion each ``perf:<key>:cpt_1``
holds its ppart ``Note`` events (the per-performer count above) **plus
63 ``Beat`` and 63 ``Dynamics``** events.

The feature events do **not** change the bundle totals — the claim
counts, timeline count (12), and group count (6) are derived from
``part.csv`` / ``ppart.csv`` / ``align.csv`` only and are unchanged.

**Feature spot-checks (Szegedi):**

* ``.beats`` row 0 (``1  1  0.000000  0.796354  83.660126  0  0``) → a
  ``Beat`` event at ``start == 0.796354`` with ``measure_number == 1``,
  ``beat == 1``, ``bpm == 83.660126``.
* ``.dyn`` row 0 (``1  1  0.000000  48.500000  58.000000  0``) → a
  ``Dynamics`` event at ``start == 0.796354`` (onset joined from the
  ``.beats`` row with the same ``(1, 1)`` key) with
  ``velocity_mean == 48.5``, ``velocity_max == 58.0``.

### Cross-group MatchClaims

One claim per ``align.csv`` row, per performer, faithfully (no
deduplication):

* **matchtype 0** → `MatchClaim.from_projection` from ``score:clt_1`` to
  ``perf:<key>:cpt_1``: source coordinate = the part note's
  ``onset_quarter``, target coordinate = the ppart note's
  ``onset_sec``.
* **matchtype 1** (score-only) → `MatchClaim.nomatch` with source
  ``score:clt_1`` and target ``perf:<key>:cpt_1`` (source coordinate =
  the part note's ``onset_quarter``).
* **matchtype 2** (performance-only) → `MatchClaim.nomatch` with source
  ``perf:<key>:cpt_1`` and target ``score:clt_1`` (source coordinate =
  the ppart note's ``onset_sec``).

Each performer's claims share a
``MatchMetadata`` whose agent is
``Agent(name="parangonada", type=AgentType.software, identifier=<key>)``
with ``certainty=1.0`` (the per-performer key now lives in the agent's
``identifier``).

### Zero-tolerance counts

| Performer | ppart events | align total | mt0 (sync) | mt1+mt2 (NOMATCH) |
|-----------|--------------|-------------|------------|--------------------|
| Szegedi   | 232 | 256 | 227 | 29 |
| Gould     | 253 | 257 | 247 | 10 |
| Curzon    | 246 | 255 | 242 | 13 |
| Brendel   | 249 | 253 | 249 | 4 |
| Hewitt    | 244 | 254 | 243 | 11 |
| **Total** |     | **1275** | **1208** | **67** |

The bundle holds **12** timelines and **6** groups (1 ``score`` + 5
``perf:<key>``). ``score:clt_1`` and ``score:dlt_1`` each carry 251
events.

**Coordinate spot-check:** Szegedi's ``align.csv`` row at ``idx 0`` is
``0,0,ngx1f26,n149``. It produces a synchronous claim from
``score:clt_1`` to ``perf:1966_Szegedi:cpt_1`` whose source coordinate is
``40.0`` (``score_q_by_id["ngx1f26"]``, an exact quarter) and whose
target coordinate is ``38.183334`` seconds (``perf_sec_by_id["n149"]``).

**Known data quirk (faithfully preserved, NOT a bug):** Brendel and
Hewitt each contain exactly one *duplicated* matchtype-0 row — Brendel
``n1qocvw7 ↔ n65`` at ``idx`` 74 and 75; Hewitt ``n1qocvw7 ↔ n64`` at
``idx`` 71 and 72. The loader emits one claim per row, so the duplicate
survives (Brendel's 253 and Hewitt's 254 totals include it). One test
asserts the duplicate is present in the raw ``align.csv`` so a future
reader knows it is data, not a loader fault. The loader does **not**
deduplicate.

**C-Map round-trip:** ``SamplesToSeconds(44100)`` on a ``dpt_1`` timeline
converts ``44100`` samples to ``1.0`` seconds (and ``sample / 44100``
generally).

## `MpmLoader` validation logic

Corpus: ``mpm_toolbox`` (two MPM-Toolbox projects, each a sibling
``.msm`` / ``.mpm`` / ``.mpr`` triple in its own subdirectory). Tests are
parametrized over both specimens:

- **Beethoven** — directory ``MPRproject_1971Curzon_VariationXIV/``, stem
  ``Beethoven_op35_1971Curzon_Var14only``. Recording: a 44100 Hz ``.wav``.
- **Reger** — directory ``Max Reger - Moment Musical (MPM Toolbox
  Tutorial)/``, stem ``Reger - Moment Musical op 13 no 4``. Recording: a
  48000 Hz ``.wav``.

### What the loader builds

``MpmLoader.from_file(mpr_path).create_bundle()`` produces one
``AlignmentBundle`` with **5 timelines** in **2 groups** (both specimens
carry a spectrogram, so the graphical axis is always present here):

- a shared ``"score"`` group:
  - ``score:dlt_1`` — a ``DiscreteLogicalTimeline`` in ticks holding the
    MSM ``Note`` events *and* every MPM markup event (``Tempo`` /
    ``Dynamics`` / ``Articulation`` / ``Asynchrony`` / any other map
    type), carrying a ticks→quarters ``TicksToQuarters(ppq)`` map;
  - ``score:clt_1`` — a ``ContinuousLogicalTimeline`` in quarters holding
    the same ``Note`` events (onset ``date / ppq`` as an exact
    ``Fraction``), carrying a modelled quarters→seconds ``TableMap``;
- a ``"perf"`` group (3 timelines):
  - ``perf:cpt_1`` — a ``ContinuousPhysicalTimeline`` in seconds, one
    ``Note`` event per observed onset (``milliseconds.date / 1000``);
  - ``perf:dpt_1`` — a ``DiscretePhysicalTimeline`` in samples, the same
    onsets scaled by the sample rate, carrying ``SamplesToSeconds``;
  - ``perf:dgt_1`` — a ``DiscreteGraphicalTimeline`` in pixels: the
    spectrogram's frame-column x-axis (see *Spectrogram graphical axis*
    below). It carries **no events** — it is a graphical axis, not an
    event timeline — and a px→seconds ``ScalarMap``;
- one synchronous cross-group ``MatchClaim`` per score note, projecting
  ``score:clt_1`` (quarters) onto ``perf:cpt_1`` (seconds).

The bundle therefore spans all **three domains**: logical (quarters /
ticks), physical (seconds / samples), and graphical (pixels). A test
collects the units across every timeline (via ``bundle.get_timeline(uid)``
and ``timeline.unit.domain``) and asserts that all three ``Domain``
members are present.

### Parsing notes

The three files are parsed with ``lxml`` using
``etree.XMLParser(recover=True, collect_ids=False)``. ``collect_ids`` is
**off** because the MSM / MPM / MPR reuse ``xml:id`` values across files;
a default id-collecting parser raises ``XMLSyntaxError: ID … already
defined``. The ``.mpm`` uses the CEMFI MPM default namespace, so its
elements are matched by local name; the ``.msm`` and ``.mpr`` are plain
XML. All numeric attributes are float-strings, parsed ``int(float(x))``
for integers (ticks / pitch / octave) and ``float(x)`` for reals.

The MSM ``midi.pitch`` integer is stored verbatim as the pitch; the
spelling attributes (``pitchname`` / ``accidentals`` / ``octave``) are
carried as-is. No ``SpecificPitch`` is constructed — the MSM octave
numbering is inconsistent with ``midi.pitch`` under scientific notation,
so interpreting it would be inference rather than faithful
representation.

The ``.mpr`` carries two ``<note ref …>`` blocks: a ``<score><page>``
block of score-image 2-D coordinates (**not** parsed here — a later
concern) and an ``<alignment>`` block (under ``<audios>/<audio>``) of
observed onsets (``ref`` / ``midi.pitch`` / ``milliseconds.date`` /
``velocity``). Only the alignment block is read for the onset events.

### Spectrogram graphical axis

The ``.mpr`` also carries a ``<spectrogram windowFunction hopSize
minFrequency maxFrequency binsPerSemitone normalize file>`` element under
``<audios>/<audio>``. Its x-axis is time in *frame columns*: each column
advances by ``hopSize`` audio samples. The XML has **no width/height**, so
the number of frame columns equals the **pixel width of the referenced
``.png``** (``file``, resolved relative to the ``.mpr``'s parent
directory). The width is read directly from the PNG IHDR header (bytes
16:20 are the big-endian ``uint32`` width) — no image-decoding dependency
is added. The pinned widths are **Beethoven 26469** and **Reger 1587**
frame columns.

``perf:dgt_1`` is built as ``DiscreteGraphicalTimeline(length=<png width>,
unit=TimeUnit.pixels)`` and carries **0 events** (the columns are an axis,
not events). Its only attachment is a px→seconds ``ScalarMap`` with scalar
``hopSize / sample_rate`` (so ``seconds = px * hopSize / sample_rate``,
``map(0) == 0``):

- **Beethoven**: ``hopSize`` 128, sample rate 44100, scalar
  ``128 / 44100``; ``map(1) == 128 / 44100`` and ``map(26469) ==
  26469 * 128 / 44100`` (the audio duration in seconds).
- **Reger**: ``hopSize`` 512, sample rate 48000, scalar ``512 / 48000``;
  ``map(1) == 512 / 48000``.

The map is pulled off the timeline via
``timeline.get_conversion_map(TimeUnit.seconds)`` and asserted to be a
``ScalarMap`` with the expected scalar. The per-note score-image
``<score><page><note ref x y>`` 2-D coordinates remain unparsed (a later
concern — per-event 2-D graphical storage does not exist yet).

When a project carries no ``<spectrogram>`` (or its ``.png`` is missing),
``perf:dgt_1`` is simply absent and the bundle holds 4 timelines; the
loader does not crash. Both specimens here have one.

#### No-spectrogram fallback

``_parse_spectrogram`` returns ``None`` whenever the project carries no
``<spectrogram>`` element (or its ``hopSize`` / ``file`` attribute is
missing, or the referenced ``.png`` is absent / unreadable). In that case
``perf:dgt_1`` is never built: the graphical timeline reference is left
``None`` and the bundle simply omits the graphical domain. Both corpus
specimens ship a complete spectrogram, so this branch never executes when
the loader is driven against the real files — it cannot be reached by
loading either specimen as-is. To pin it without inventing a synthetic
spectrogram-less corpus, the fallback tests replace ``_parse_spectrogram``
with a stub that returns ``None`` (a per-test ``monkeypatch`` of the
static method, so the substitution is local and parallel-safe), then load
the unmodified specimen ``.mpr``. Every other parse step runs exactly as
in the spectrogram case; only the graphical axis drops out.

The fallback tests are parametrized over both specimens (free loader
parity) and pin the exact 4-timeline shape:

- after ``load``, ``loader._perf_dgt is None``;
- ``create_bundle()`` yields ``n_timelines == 4``, ``n_groups == 2``,
  ``group_ids == ["score", "perf"]``, and ``timeline_ids ==
  [score:clt_1, score:dlt_1, perf:cpt_1, perf:dpt_1]`` (no ``perf:dgt_1``);
  the ``perf`` group holds exactly ``{perf:cpt_1, perf:dpt_1}``; the claim
  count is unchanged (one synchronous claim per note); and the bundle's
  timeline-unit domains are exactly ``{logical, physical}`` — the
  graphical domain is **absent** (the converse of the three-domain test);
- ``create_timelines()`` returns exactly the four non-graphical timelines
  by ``.id``;
- ``create_timeline("perf:dgt_1")`` raises ``KeyError`` (matching ``No
  timeline with uid``) because that timeline was never registered;
- ``_repr_html_()`` reports ``"4 in 2 group(s) (score, perf)"`` and never
  ``"5 in 2 group(s)"``.

### Performance selection and style resolution

The MPM holds several ``<performance>`` blocks; the **first** is the
default. ``load(mpr_path, performance=<name>)`` selects another.

Within the chosen performance, maps appear under both
``performance>global>dated`` and every ``performance>part>dated``
(``tempoMap`` / ``asynchronyMap`` are global, ``articulationMap`` is
per-part, ``dynamicsMap`` is either). Each map's leading ``<style
name.ref="…">`` child is *not* an entry; it names the active styleDef.
One markup event is emitted per remaining entry, ``event_type`` =
capitalised element local-name.

Tempo / dynamics values are inline numbers (audio performances) *or*
style names resolved against the performance's ``tempoDef`` / ``dynamicsDef``
(``value``). Articulation ``name.ref`` resolves against an
``articulationDef`` whose snake-cased numeric attributes
(``relative_duration`` / ``absolute_duration_ms`` / ``absolute_velocity``
/ ``absolute_velocity_change`` …) become event columns; a ``name.ref``
with no matching def carries the name only (no crash). Any unrecognised
map type is emitted generically (raw attributes carried verbatim), so
nothing is dropped.

### The central join

The alignment ``ref`` → MSM ``xml:id`` is a **perfect bijection** in both
specimens (zero orphan refs, zero unaligned MSM ids). Every score note
has exactly one observed onset, so every claim is **synchronous** (no
NOMATCH). Tests assert the bijection (0 orphans, 0 unaligned) and that
the claim count equals the MSM note count.

### Modelled quarters→seconds TableMap

Constant tempo per segment (``transition.to`` accelerando / ritardando
ramps are **ignored** for integration — stored only as a Tempo-event
attribute). For each ``tempoMap`` entry (sorted by date) with resolved
bpm ``B`` and beat-length ``L``, the segment's seconds-per-quarter is
``(0.25 / L) * (60 / B)``. Entry dates (ticks) become quarters
(``date / ppq``); the first entry sits at ``q = 0, s = 0``. Segments are
cumulatively integrated (``s_{i+1} = s_i + (q_{i+1} − q_i) · spq_i``); a
final anchor at the score's last quarter extends the last segment. The
result is a ``TableMap(x_values=[q…], y_values=[s…], kind="linear",
source_unit=quarters, target_unit=seconds)``.

Tests assert ``map(0) == 0.0`` and strict monotonic increase. For
**Reger** (single tempo entry, ``beatLength`` 0.0625, bpm 80) the spq is
``(0.25 / 0.0625) · (60 / 80) = 4 · 0.75 = 3.0``, so the map is
``s = 3·q`` and ``map(1) == 3.0`` (2 anchors). For **Beethoven** (7 tempo
entries → 8 anchors) the first segment (bpm 100, beatLength 0.25) has
``spq = 0.6``, so ``map(39.5) == 23.7``; the second entry (bpm 25, same
beat-length, spq 2.4) advances 0.5 quarter to ``map(40.0) == 24.9``.

### Zero-tolerance counts

| Quantity | Beethoven | Reger |
|----------|-----------|-------|
| PPQ | 720 | 720 |
| MSM notes | 251 | 92 |
| ``score:clt_1`` events (notes) | 251 | 92 |
| Tempo events (default perf) | 7 | 1 |
| Dynamics events (default perf) | 34 | 11 |
| Articulation events (default perf) | 207 | 80 |
| Asynchrony events (default perf) | 0 | 0 |
| Other markup (default perf) | 0 | 1 (Ornament) |
| ``score:dlt_1`` events (notes + markup) | 499 | 185 |
| Alignment notes | 251 | 92 |
| ``perf:cpt_1`` / ``perf:dpt_1`` events | 251 | 92 |
| Synchronous claims | 251 | 92 |
| NOMATCH claims | 0 | 0 |
| Tempo-map anchors | 8 | 2 |
| Sample rate (Hz) | 44100 | 48000 |
| ``perf:dgt_1`` length (frame columns / pixels) | 26469 | 1587 |
| ``perf:dgt_1`` events | 0 | 0 |
| Spectrogram ``hopSize`` | 128 | 512 |
| px→seconds ``ScalarMap`` scalar | 128 / 44100 | 512 / 48000 |

The bundle holds **5 timelines** (``score:clt_1`` / ``score:dlt_1`` /
``perf:cpt_1`` / ``perf:dpt_1`` / ``perf:dgt_1``) and **2 groups**
(``score`` / ``perf``) for both specimens; the ``perf`` group holds 3
timelines. The timeline units across the bundle span all three
``Domain``s: logical, physical, and graphical.

**C-Map spot-check (``TicksToQuarters``):** on ``score:dlt_1``, tick
``360`` resolves to ``0.5`` quarters and ``720`` to ``1.0`` (queried via
``timeline.get_timestamp(tick).get_unit(TimeUnit.quarters)``).

**Onset spot-check:** Beethoven note ``nbwxzb1`` has
``milliseconds.date == 828.0190259247195``, so its ``perf:cpt_1`` onset
and its claim target coordinate are ``828.0190259247195 / 1000.0``
seconds (the loader divides by 1000; the test pins the same Python
expression, so the IEEE-754 double matches exactly). Its source
coordinate is ``360 / 720 == 0.5`` quarters.

**Note pitch spot-check:** Beethoven ``nbwxzb1`` → pitch ``75``,
pitchname ``"e"``, accidentals ``-1``, octave ``4``, date ``360`` ticks.

**Style-resolution spot-checks:** Beethoven Dynamics ``volume="p"`` →
``48.0``, Tempo ``bpm="Meno mosso."`` → ``100.0``, Articulation
``name.ref="staccato"`` → ``absolute_duration_ms == 160.0`` (and
``noteid`` stripped of its leading ``#``). Reger Dynamics
``volume="dolciss."`` → ``74.0``, Tempo ``bpm="Andantino"`` → ``80.0``.

**Performance selector:** Beethoven
``load(mpr, performance="Curzon_1971_DECCA-SXL6523_audio")`` reaches a
different performance whose inline-numeric bpm values resolve directly —
its ``score:dlt_1`` then carries **25** Asynchrony events (vs 0 in the
default) and **123** Tempo events.

## Represent pitch once — validation logic

`test_represent_pitch_once.py` pins the rule that **pitch is represented
exactly once** — by the single most-expressive semantic pitch type the
source *faithfully* supports — and that poorer / derived views are
reached on request, never stored a second time.  The only deterministic
derivations are downhill from SpecificPitch (`GP ← SP → EP`,
`GPC ← SPC → EPC`, and `Pitch → PitchClass`); every other step (above
all `EP → SP`) is inference and is forbidden as a faithful
representation.

**The keystone (struct-preservation in `from_dicts` / `add_events`).**
A carried struct-dict column (e.g. `pitch={"midi_number": 60}`) must
survive ingestion as a real `pa.struct` column — not be JSON-stringified.
The tests assert, on a plain `EventData.from_dicts` and on a
`Timeline.add_events` round-trip:

- a `pitch` dict becomes `struct<midi_number: int64>` (exact type), not a
  `string` column;
- the column carries `b"timetoalign"` metadata with `field_type ==
  "EnharmonicPitchField"` (so the affordance round-trips without relying
  on shape discovery);
- the sub-field order is the paired class's canonical order even when the
  carried dict keys were in another order (`{"alter": -1, "step": "E"}` →
  stored `struct<step, alter>`), because `pa.array` infers struct fields
  alphabetically and the loader reorders to canonical;
- `events.get_field(EnharmonicPitch)[i]` reconstructs the exact scalar.

A heterogeneous sequence of `add_events` calls (a `Note` batch carrying
the `pitch` struct, then a markup/feature batch without it) preserves the
struct column and null-fills the absent rows — the affordance is intact
after the concat.

**The uniform affordance mechanism (`_afforded_fields`).** An EventData
subclass declares `{raw_column: PairedField}`.  `get_fields` promotes the
raw atomic column to a live field via the paired class's `emit()` only
when asked (4th discovery strategy, after metadata / default-name /
shape), and caches it.  Tested invariants:

- `MidiEventData` / `ScoreMidiEventData` declare `{"pitch":
  EnharmonicPitchField}`; their raw `pitch` column is `int64` (uniform
  with the `{midi_number: int64}` view it materialises into);
- `NoteEventData` declares `{"midi": EnharmonicPitchField}` — a
  *non-default* affordance over the raw MIDI number, alongside the
  default `specific_pitch` struct;
- a declared affordance is surfaced by `get_field(<ScalarClass>)`,
  `has_field`, `get_fields_satisfying(GenericPitchLike)`, and `get_pitch_field`;
- the two number-only views are interchangeable at lookup time: every raw
  MIDI-number affordance answers both `get_field(EnharmonicPitch)` and
  `get_field(MidiPitch)` from the same atomic column;
- the raw column is left untouched (still queryable as a plain int).

Blueprint parity is part of the same contract: a
``get_field(EnharmonicPitchField(source_fields="midi"))`` or
``MidiPitchField`` request must use that declared affordance's ``emit()``
promotion, not attempt to wrap the raw ``int64`` as a struct directly.  The
Vienna Chopin notes fixture pins the resulting 498-row fields and exact first
MIDI number ``59`` for both blueprint and class paths.  A source that cannot
be promoted (such as a non-numeric string column) remains a ``TypeError``.

**Per-source default pitch type (zero-tolerance affordance pins).**
`get_pitch_field()` returns the single most-expressive default:

| Producer | `get_pitch_field()` type | Also afforded on request | NOT afforded |
|----------|--------------------------|--------------------------|--------------|
| `NoteEventData` (ms3 / music21 / partitura) | `SpecificPitchField` | `EnharmonicPitch` and `MidiPitch` from raw `midi` | a *default* `EnharmonicPitch` struct (represent-once) |
| `MidiEventData` / `ScoreMidiEventData` | `EnharmonicPitchField` | `MidiPitch` display view; `EnharmonicPitchClass` | `SpecificPitch` (inference) |
| Parangonada score / perf timelines | `EnharmonicPitchField` | `MidiPitch` | `SpecificPitch` |
| `PerformancePrecisionLoader` score timeline | `EnharmonicPitchField` | `MidiPitch` | `SpecificPitch` |
| MSM (`MpmLoader`) score timelines | `EnharmonicPitchField` | `SpecificPitchClass` from `pitchname`+`accidentals` | `SpecificPitch` (octave inconsistent with `midi.pitch`) |

**Represent-once assertion.** For `NoteEventData`, exactly one default
semantic pitch field exists (`specific_pitch`, a `SpecificPitchField`);
`get_field(SpecificPitchField)` resolves it, and there is **no**
`midi_pitch` struct column.  The MIDI number lives only as the raw `midi`
int (no `field_type` metadata).

**Scalar↔EventData contract (MIDI).** Every pitch-bearing loader affords both
number-only views over its one raw MIDI-number source. Thus
`get_field(EnharmonicPitch)` and `get_field(MidiPitch)` produce identical MIDI
numbers for note rows (and nulls for control rows), while number-only sources
still refuse `SpecificPitch`. The Vienna Chopin notes TSV pins 498 rows and
first-note MIDI number 59 for both views.

**On-request EP from a SP field (both routes).** On `NoteEventData`:

- *conversion route* — `sp_field.convert_to(MidiPitch)[i].midi_number`
  equals the note's MIDI number computed from `(octave+1)*12 + step_semi
  + alter` (the scalar's own `midi_number`); `SpecificPitch.to` does not
  support `EnharmonicPitch` directly, so the test uses the `MidiPitch`
  thin alias as the documented path;
- *raw-column route* — `get_field(EnharmonicPitch)[i].midi_number` equals
  the raw `midi` value, and (zero-tolerance) the two routes agree
  element-wise on the spelled-source fixture.

**Multi-batch concat re-affordance (`TestMultiBatchConcatAffordance`).**
This validates the key represent-once model risk: a
number-only source stores pitch as a bare `int64` and *affords* the
`EnharmonicPitch` view over it on demand, rather than materialising a
pitch struct.  The view must therefore survive the multi-batch ingestion
path, where `EventData.extend` replaces the table in place with a
`pa.concat_tables(..., promote_options="default")` result (schema
promotion: extra columns null-fill).  A cached field wraps the array as
it was at first access, so an interleaved query (afford → extend → query
again) would otherwise expose only the pre-extend rows.  `extend` drops
the field cache (`_invalidate_field_cache`) so the affordance re-attaches
over the concatenated table.

The tests pin, with exact `midi_number` values and no ranges:

- **EventData level.** Build a `MidiEventData` carrying a bare-int
  `pitch` (batch 1), query `get_field(EnharmonicPitch)` (populating the
  cache), `extend` a second batch, query again — the field now spans
  batches 1+2 with the exact values; a third, *heterogeneous* batch
  (a Control-Change row with no `pitch`, plus a brand-new column,
  exercising `promote_options="default"` null-fill) keeps the affordance
  with `None` for the pitch-less row.  `get_pitch_field()` stays an
  `EnharmonicPitchField` throughout.
- **Robustness / negative.** After every concat the raw `pitch` column
  stays `int64` — it is never silently promoted to a struct nor
  stringified — and carries no `field_type` metadata (the affordance is
  materialised on read, not stamped onto the column).
- **`Timeline.add_events` level.** A timeline whose events store is a
  `MidiEventData` (built via `SingleStore(midi_data).create_timeline()`,
  which preserves the concrete EventData class through `prefix_ids`)
  receives note events with bare-int pitch across **multiple**
  `add_events` calls — each routing through `_add_events_unchecked` →
  `from_dicts` → `extend`.  After each call the timeline's
  `events.get_pitch_field()` / `get_field(EnharmonicPitch)` resolve with
  the exact accumulated `midi_number` values, proving "a pitch view
  survives ingestion into a timeline" across batches.

## `ListenHereLoader` validation logic

### What the loader builds

A Listen Here! alignment JSON describes many recordings of one work warped
onto a shared **equidistant reference grid**.  Per recording, ``body.audio``
holds a ``times`` array whose ``i``-th entry is that recording's clock-time
(seconds) at reference-grid column ``i``; the arrays are parallel and equal
length, so together they are a dense alignment matrix.  ``ListenHereLoader``
reads it into one audio-to-audio ``AlignmentBundle``:

* one empty seconds ``ContinuousPhysicalTimeline`` (``<stem>:cpt_1``) per
  recording, each in its **own** group named by the recording stem;
  ``length`` is that recording's stored ``duration``;
* the **complete pairwise topology** — for every unordered pair of recordings
  ``(a, b)`` and every grid column ``i``, one synchronous instant claim
  ``a@times_a[i] ↔ b@times_b[i]`` — held columnar in a ``MatchClaimField``,
  built vectorized via ``MatchClaimField.from_columns`` (never one
  ``MatchClaim`` object per row).

The recordings carry **no symbolic events**; all alignment lives in the claim
field.  The reference named by ``header.ref`` is just another recording, not a
privileged hub.  The loader reads the existing alignment — it never runs an
aligner.

### Public field access — the uniform ``get_field`` API

The claim field is reached through the uniform field API, not a bespoke
property: ``loader.get_field(MatchClaim) -> MatchClaimField`` (the paired
``MatchClaimField`` class resolves identically).  There is **no
``loader.claim_field`` property** — the tests assert it is gone
(``not hasattr(loader, "claim_field")``).  Any other selector raises
``TypeError``.

### Reference recording in the repr

Reading the reference is essential to interpreting the bundle, so both
``repr(loader)`` and ``loader._repr_html_()`` name it.  The ``__repr__`` is
``ListenHereLoader(recordings=3, reference='rec-ref.mp3', claims=15)``; the
HTML carries a ``Reference`` row with the ``header.ref`` string.

### Affordance HTML card — Claims, not "Events: 0"

The three whole-directory / single-file alignment loaders (Parangonada,
MPM, Listen Here!) keep their payload in assembled claims/timelines, not the
per-source ``AlignmentStore``.  Their ``_repr_html_`` cards therefore label the
payload-count row **Claims** (via the ``_repr_count_row`` hook) and never show
the base ``Events`` row that read the empty store as ``0``.  Tests assert the
exact ``<tr><td><b>Claims</b></td><td>N</td></tr>`` row, the absence of any
``Events`` text, and the loader-specific rows:

- Listen Here!: ``Recordings`` (3), ``File``, and ``… in 3 group(s)``.
- MPM: ``Project``, ``Performance``, and ``… in 2 group(s) (score, perf)``.
- Parangonada: ``Performers`` (5), ``Dataset``, ``Performer keys``, and
  ``… in 6 group(s)``.

### Columnar bundle query — the field is never exploded

``create_bundle()`` hands the ``MatchClaimField`` to the bundle via
``add_match_claim_field`` (the columnar store), **not** ``add_match_claims``.
Consequently the bundle's Python-list ``cross_group_claims`` stays **empty**
(length 0) while ``cross_group_claim_fields`` holds the one field.
``get_matchstamp_at`` answers by filtering the field's struct column vectorized
(exact-equality boolean mask on ``timeline_a/b_id`` + ``coordinate_a/b``) and
materialising only the matched rows — a whole-work field is never turned into a
million Python claims.  Tests pin this by asserting
``len(bundle.cross_group_claims) == 0`` and
``len(bundle.cross_group_claim_fields) == 1`` while the matchstamp read still
succeeds.

Which store a bundle uses must not change **what** it answers, only how fast.
So the loader-produced bundle is also tested on the readers that a columnar
bundle previously left unanswered, all with exact values:

| Test | Validates |
|------|-----------|
| `test_bundle_claim_count` | `n_cross_group_claims == 15` (spans both stores) |
| `test_bundle_get_match_claims` | 15 claims, all synchronous |
| `test_bundle_get_match_claims_filtered` | `timeline_id="rec-a:cpt_1"` → 10; `between=("rec-a:cpt_1","rec-b:cpt_1")` → 5 |
| `test_bundle_get_claim_fields` | the vectorized accessor selects `[15]` / `[10]`, and `[]` for an unknown UID |
| `test_bundle_matchstamp_table_per_claim` | 15 rows × 3 columns, exactly two filled cells per row |
| `test_bundle_matchstamp_table_from_graph` | the 15 pairwise rows collapse into the 5 reference-grid cross-sections, exact coordinates, ordered by the `rec-a` coordinate |
| `test_bundle_are_commensurable` | all recording pairs commensurable; an absent ID is not |
| `test_bundle_transfer` | `transfer(0.02, "rec-a:cpt_1", "rec-b:cpt_1") == 0.01` — the `MatchLine` → `WarpMap` path over the columnar store |
| `test_bundle_diagram_reports_claim_count` | `diagram()` reports `MatchClaims: 15` |

The general two-store parity contract is documented in
``tests/alignment/README.md`` ("Claim-Store Parity").

### The synthetic specimen

Committed tests use an **inline synthetic** ``alignment.json`` written to
``tmp_path`` (no pooch corpus, parallel-safe).  The canonical specimen:

```json
{
  "header": {"ref": "rec-ref.mp3", "createdBy": "Listen Here! v0.20.0"},
  "body": {"audio": {
    "rec-a.mp3":   {"times": [0.00, 0.02, 0.04, 0.06, 0.08], "peaks": [0.1,0.2], "duration": 0.08},
    "rec-b.mp3":   {"times": [-0.01, 0.01, 0.03, 0.05, 0.07], "peaks": [0.1,0.2], "duration": 0.10},
    "rec-ref.mp3": {"times": [0.00, 0.025, 0.045, 0.065, 0.085], "peaks": [0.1,0.2], "duration": 0.085}
  }}
}
```

Sorted keys → ``["rec-a.mp3", "rec-b.mp3", "rec-ref.mp3"]``; stems ``rec-a`` /
``rec-b`` / ``rec-ref``; timeline uids ``rec-a:cpt_1`` / ``rec-b:cpt_1`` /
``rec-ref:cpt_1``; pairs ``(a, b)``, ``(a, ref)``, ``(b, ref)``.  ``rec-b``'s
first column is ``-0.01`` — a **negative** pre-onset warp coordinate that the
loader keeps faithfully (never clamped or dropped).  ``peaks`` is
waveform-display payload and is ignored.

### Zero-tolerance counts

* After ``load``: ``len(loader) == 15`` and
  ``len(loader.get_field(MatchClaim)) == 15`` (C(3,2) = 3 pairs × 5 columns);
  ``loader.recording_keys == ["rec-a", "rec-b", "rec-ref"]``;
  ``loader.reference == "rec-ref.mp3"``.
* ``loader.get_field(MatchClaim)`` is a ``MatchClaimField``;
  ``loader.get_field(MatchClaimField)`` returns the same field.
* ``get_field(MatchClaim).timeline_ids == {"rec-a:cpt_1", "rec-b:cpt_1",
  "rec-ref:cpt_1"}``.
* **Faithfulness:** ``pc.min`` over both coordinate columns of the field is
  exactly ``-0.01`` (negatives kept).
* ``bundle = loader.create_bundle()``: exactly **3** timelines and **3**
  groups; timeline uids equal the set above; each timeline holds **0** events;
  ``rec-b:cpt_1`` length ``== 0.10``; the claim field is added columnar, so
  ``len(bundle.cross_group_claims) == 0`` and
  ``len(bundle.cross_group_claim_fields) == 1``.
* ``bundle.get_matchstamp_at(0.045, "rec-ref:cpt_1")``: ``stamp.n_timelines ==
  3``; ``stamp.get_coordinate("rec-a:cpt_1") == 0.04``;
  ``stamp.get_coordinate("rec-b:cpt_1") == 0.03``;
  ``stamp.get_coordinate("rec-ref:cpt_1") == 0.045``.  The exact float literals
  from the specimen are reused so the equality is exact.  Because the bundle's
  Python claim list is empty, this read proves the columnar path resolves the
  matchstamp without exploding the field.
* **Bare-array form:** a second fixture where one entry is a bare list
  ``[0.0, 0.02, 0.04]`` (no ``peaks`` / ``duration``) parses, and that
  recording's timeline ``length == max(times)``.
* **Metadata:** the field's first materialised claim (``get_field(MatchClaim)[0]``)
  has ``.metadata.agent.name == "Listen Here! v0.20.0"`` (from ``header.createdBy``)
  and ``.metadata.agent.identifier == "dtw_chroma_alignment"`` (the alignment
  method).

### Error cases

``load`` raises ``ValueError`` when:

* the per-recording ``times`` arrays differ in length (the message names the
  differing lengths);
* ``header.ref`` is absent from ``body.audio``;
* fewer than two recordings are present.
