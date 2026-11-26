# Test-data corpora

The contents of this directory are **not tracked in git**. They are downloaded
on demand by `timetoalign.testdata` (a thin wrapper around
[pooch](https://www.fatiando.org/pooch/)) from a release of the
[`TimeToAlign/tta_test_data`](https://github.com/TimeToAlign/tta_test_data)
repository, which holds the raw corpora and a GitHub Action that bundles
them into release assets on every `testdata-v*` tag push.

## How it works

Each top-level subdirectory of this folder corresponds to one `.tar.gz`
archive in the test-data release. The first time a test or notebook calls
`ensure_data("<name>")`, the matching archive is downloaded, its SHA256 is
verified, and its contents are extracted in place. A sentinel
`.tta_testdata_hash` file is written so subsequent runs skip the re-extract.

The cached `.tar.gz` archives live in pooch's standard cache directory
(`%LOCALAPPDATA%\timetoalign-testdata` on Windows,
`~/.cache/timetoalign-testdata` on Linux,
`~/Library/Caches/timetoalign-testdata` on macOS). Override with the
`TTA_TESTDATA_CACHE` environment variable. Override the extraction location
with `TTA_TESTDATA_DIR`.

## Available corpora

| Name | Contents |
|------|----------|
| `audio` | Audio files used for `AudioLoader`/`BeatGrid` how-tos. |
| `audiolabs_omr` | AudioLabs OMR JSON exports (Wagner WWV086). |
| `fixtures` | Synthetic fixtures (corrupt files, ATON minimals, .lab samples). |
| `hendrix` | TiLiA hierarchy timelines for the *All Along the Watchtower* genesis study. |
| `midi` | Performance and score MIDI specimens. |
| `mtd` | Musical Theme Dataset (MTD) sample — four themes in the MTD archive's layout (`MTD_sample.zip`). |
| `performance_precision` | Performance-precision benchmark inputs. |
| `score` | Score corpora — Beethoven WoO 71, Op. 18 multimodal, Bruckner 5, Couperin, Rachmaninoff, Wagner, *Out of the Flow* experience. The big one (~750 MB extracted). |
| `supra` | SUPRA piano-roll annotations and audio. |
| `tabular` | CSV/TSV samples for tabular loaders. |
| `target_flows` | Reference flow CSVs used by the score-parsing matrix. |
| `thoresen` | Lasse Thoresen *Sound-Objects* / *Form-Building Patterns* figures. |
| `vienna_1x22` | Vienna 1x22 corpus — Chopin Op. 10 No. 3, 22 performances. |

The canonical list (with SHA256 digests) is `REGISTRY` in
`timetoalign/testdata/__init__.py`.

## Fetching from code

```python
from timetoalign.testdata import ensure_data

DATA_DIR = ensure_data("vienna_1x22")            # single corpus
score_dir, midi_dir = ensure_data("score", "midi")  # multiple
```

## Publishing a new test-data release

The raw corpora live in the sibling `tta_test_data` repository. Editing
them and pushing a tag is enough — CI does the packaging and the release.

1. In `tta_test_data/`, edit files under `data/<name>/`, commit, push.
2. Tag and push:

       cd tta_test_data
       git tag testdata-vN
       git push origin testdata-vN

   The `Publish test data` workflow tarballs `data/<name>/` into
   `<name>.tar.gz`, creates a GitHub release with the assets, and writes a
   `REGISTRY = {...}` block into the release notes.

3. In this repository, bump `RELEASE_TAG` and paste the new `REGISTRY` in
   `timetoalign/testdata/__init__.py`.
