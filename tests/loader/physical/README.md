# Physical Loader Tests

Tests for the physical-domain loaders in `timetoalign.loader.physical`.

## Test Files

### `test_audio.py` (36 tests)

Tests for `AudioLoader` against minimal in-memory WAV files and one MP3 specimen.

Durations are asserted **exactly**, with zero tolerance: `AudioLoader`
computes `duration_seconds = n_samples / sample_rate` and `SamplesToSeconds`
performs the same division, so every duration in these tests is bit-identical
to the value expression and is checked with `==` (no `pytest.approx`). The
integer-sample fixtures divide out cleanly (44100/44100 = `1.0`, 96000/48000 =
`2.0`, 22050/44100 = `0.5`); the 10-sample edge case is pinned as `10 / 44100`,
the same double the C-map returns. No approx assertions remain in this file.

**Two timelines per file.** A file's native axis is its sample count, so
`create_timeline()` must give a `DiscretePhysicalTimeline` in `samples`, `int`,
whose length is the sample count, and attach a `SamplesToSeconds` C-Map at the
file's sample rate; `create_continuous_timeline()` must give the
`ContinuousPhysicalTimeline` in `seconds`, `float`, whose length is
`duration_seconds`, and attach a `SecondsToSamples` C-Map. What proves each map
is attached at the right rate is its value at the timeline's own end: the
samples map must take the sample count to the duration, the seconds map the
duration back to the sample count — a wrong rate, a missing map or a map in the
wrong direction cannot produce both. The seconds unit declares `float`, so the
samples map answers with the double `n_samples / sample_rate`, pinned with
`==` and its type (a `Fraction` there would mean the axis's number type was
not applied); the samples map answers with an `int`. `attach_cmap=False`
attaches no map to either timeline (`get_conversion_map` of the other unit is
`None`). The default id and name come from the file name for both.

These facts are pinned on the WAV fixtures and once on a real specimen read by
the soundfile or mutagen backend (`TestAudioLoaderSpecimen`, corpus `audio`,
`hard_techno/Bass Kick.mp3`): 10190592 samples at 44100 Hz, 2 channels,
`duration_seconds == 231.0791836734694`. The samples timeline's map takes
10190592 to `231.0791836734694` (a `float`); the seconds timeline's map takes
`231.0791836734694` to `10190592` (an `int`). A manifest round trip
(`to_dict` then `Timeline.from_dict`) of either timeline must keep its class,
unit and length, and its one C-Map's source unit, target unit and scalar
(`Fraction(1, 44100)` and `44100`); the read map converts the end exactly as
the attached one did, and the round-tripped `to_dict` equals the original.

**Backend-probe silence** (`TestAudioLoaderProbeSilence`, 3 tests). The loader
tries backends in order and hands an unreadable file to the next one, so a
failing probe is a normal step, not an event a user should be told about. The
decoders behind libsndfile do not agree: they write their diagnostics straight
to file descriptor 2, where neither `warnings` nor `logging` can reach them —
so proving the loader is quiet means capturing the *descriptor*, not
`sys.stderr`. These tests therefore use pytest's `capfd`, never `capsys`, and
the fixture is a file whose extension lies: 70,000 zero bytes named `.mp3`,
past the 65,536-byte window the MP3 decoder searches before it reports failure.

What must hold: probing that file produces **empty** captured stderr and
stdout while still raising `ValueError` (the diagnosis reaches the caller as an
exception, not as terminal noise); a native `os.write(2, ...)` immediately
afterwards **is** captured, since a muting that leaked would silence the rest
of the session; and a WAV file that loads successfully is equally silent and
still reports its exact sample count, which is what distinguishes muting the
probe from muting the loader.

### `test_repovizz_loader.py` (27 tests)

Tests for `RepoVizzLoader`, a manifest-style loader for RepoVizz XML manifests
and legacy 2-line CSV sensor files from the EEP (Expressive Ensemble Performance)
dataset.

**Test Classes:**

| Class | Tests | Focus |
|-------|-------|-------|
| `TestCatalogueEntry` | 3 | Frozen dataclass for catalogue entries |
| `TestRepovizzDictStore` | 2 | DictStore subclass with category properties |
| `TestRepoVizzLoaderXmlMode` | 8 | XML manifest parsing and timeline creation |
| `TestRepoVizzLoaderCsvMode` | 7 | Legacy CSV backwards compatibility |
| `TestRepoVizzLoaderErrors` | 4 | Error handling and edge cases |
| `TestRepoVizzLoaderIntegration` | 2 | Cross-recording content and group iteration |

**Exact counts (zero tolerance).** Both the `Normal` and `Mechanical`
recordings catalogue exactly **376** entries, of which **232** become
timelines (**144** audio, **4** score entries), and both expose the same
group list `["audio", "descriptors", "mocap", "score"]`. Durations are exact:
a signal entry's `duration_seconds` is `n_samples / sample_rate` (asserted with
`==`, not `approx`) — the 441000-sample / 44100 Hz `CatalogueEntry` gives
exactly `10.0`, and the legacy-CSV `duration_seconds` equals
`n_samples / frame_rate` bit-for-bit. `TestRepoVizzLoaderIntegration.test_multiple_recordings`
no longer merely checks "both load"; it pins the counts above **and** the fact
that the two takes carry distinct audio content: the shared cardioid-ambient
entry (`ROOT0_Audi1_Audi0_Ambi0`, "Cardioid microphone", 44100 Hz) has
**11753638** samples in the Normal recording and **12426696** in the
Mechanical one.

**Test Specimens:**

| Specimen | Location | Validation Focus |
|----------|----------|------------------|
| `StringQuartetEEP_I_Normal.xml` | `tests/data/score/beethoven_op18-4iv_multimodal/` | XML manifest with 376 catalogue entries |
| `StringQuartetEEP_I_Mechanical.xml` | `tests/data/score/beethoven_op18-4iv_multimodal/` | Alternative recording XML |
| `vln1_bow_vel.csv` | `tests/data/score/beethoven_op18-4iv_multimodal/` | Legacy 2-line RepoVizz CSV |

**Key Validations:**

- XML manifest creates correct catalogue entries by group (audio, score, descriptors, mocap)
- Timeline creation with `SamplesToSeconds` conversion map attached
- Backwards compatibility: CSV files still load with legacy behaviour
- Group filtering and TimelineGroup creation, including explicit catalogue selection via the `entries` keyword
- Catalogue entry lookup by ID, name, or partial match

**Specimen Provenance:**

The EEP multimodal dataset contains string quartet recordings with:
- Audio recordings (ambient + pickup microphones)
- Essentia audio descriptors (tonal, lowlevel, rhythm)
- Bowing gesture descriptors (9 per instrument)
- MoCap position data (48 markers x 3 axes)
- Score alignment files (.notes)

Each recording directory has an XML manifest cataloguing all 332+ data files.

---

## Test Results

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_repovizz_loader.py` | 27 | All pass |
| `test_audio.py` | 36 | All pass |
| **Total** | **63** | **All pass** |

All tests pass in parallel mode (`pytest-xdist -n auto`).
