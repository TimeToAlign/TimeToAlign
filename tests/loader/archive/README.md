# Loader tests: sources inside ZIP archives

Every loader reads a source given as a filesystem path (`Path` or `str`)
or as a member of a ZIP archive (`zipfile.Path`), without extracting the
archive. A loader whose parser takes a file object reads the member in
place; one whose parser needs a file on disk (Partitura, music21, Verovio,
the reader of Verovio timemaps) reads a copy that `materialize` makes once
in a cache folder. These tests check both ways and the cache itself.

`helpers.py` builds the archives (`pack`, file entries only, so every
directory inside is implied) and the same files on disk (`lay_out`);
`conftest.py` points `TTA_CACHE_DIR` at the test's own folder.

## What proves an archive source right

An archive source is right **if and only if loading the member gives exactly
what loading the same bytes from the filesystem gives.** Nothing else is a
fair witness: the archive adds no information and must lose none. So each
test packs files into a ZIP archive in `tmp_path`, loads the member, loads
the very same bytes from the filesystem, and compares the two results with
exact equality:

- event loaders: the events table (`pa.Table.equals`) and the created
  timeline (`to_dict()`);
- score loaders: the store's facet tables and the created timeline (its
  events, its measure map), so ms3 companion discovery inside an archive is
  proven by the companions' tables arriving, not by a file count;
- alignment and project loaders: the bundle (`create_bundle().to_dict()`),
  or the timelines, and the members they read;
- payload loaders: the parsed payload (rows, holes, tables, paths, image
  pixels), field by field, except the one field that names the source,
  which must name the member.

No expected value is computed by the code under test: the filesystem
reading is the oracle, and the loaders' own test files pin that reading
against the formats. Inputs are test-data files resolved with
`ensure_data(...)` where a corpus holds the format; where none does, the
test writes a small specimen and puts the same bytes into the archive and
on disk (named below as *inline*).

## What is covered

`test_archive_sources.py`, one loader per family: `CsvLoader`; `Ms3Loader`
on a measures TSV and with `auto_discover` (flat siblings and DCML facet
folders inside the archive); `PerformanceMidiLoader`; `ScoreMidiLoader` on
Partitura; `XmlLoader`; `RekordboxLoader`; `MeasureMapLoader`;
`IIIFManifestLoader`; `AudioLoader` (a WAV); `MatchfileLoader` (Partitura
reads a copy); `PerformancePrecisionLoader`, a directory member searched in
place, its `.solo` read in place and its Verovio timemap read from a copy
by `MetricMap.from_verovio_timemap`.

`test_archive_loaders.py`, one test per further read path: `VerovioLoader`
(MEI), `PartituraLoader` and `Music21Loader` (MusicXML), each read from a
copy; `MpmLoader` (the `.msm`/`.mpm` siblings, the recursive `*.wav`
search with the recording it picks, the spectrogram `.png` header);
`ListenHereLoader` (inline JSON; it reads no recording);
`ParangonadaLoader`; `RepoVizzLoader` on its XML manifest with the `.notes`
files it reads and one signal CSV, and on a CSV alone; `Ieee1599Loader`,
with its single-source refusal from an archive; `MeiFacsimileLoader`
(inline); `MeiHeadLoader` (inline); `ScoreWarpLoader` (inline SVG);
`DmosLoader` (inline); `ATONLoader`; `ImageSource.from_image_file` (a PNG,
decoded from the member's bytes); `LabLoader`; `SoloLoader`; `JsonLoader`;
`MapsLoader` (inline rows, aligned to an inline MEI read by Verovio); the
wave and RIFF header readers of `AudioLoader` on a WAV written by the test.

`MelodyLoader` (inline Turtle; no corpus holds any) is covered in
`test_archive_sources.py`, see below.

Not covered by a test of their own, because they share a tested read path:
`TsvLoader` (`CsvLoader`'s), `EepNotesLoader` (read through
`RepoVizzLoader`), `GraphicalLoader.add_image` (`ImageSource`'s).

## Archive members never become path strings

`str` of a `zipfile.Path` joins archive and member into a path that does not
exist (`dcml.zip/repo/WoO71.measures.tsv`). For every loader that writes
its source into stored metadata (`Ms3Loader` on measures and notes,
`MeasureMapLoader`, `VerovioLoader`, `PartituraLoader`, `Music21Loader`,
`RepoVizzLoader`'s manifest metadata), no string anywhere in the loader's
and the store's metadata, keys and values at any depth, contains `".zip/"`;
the source is recorded as the archive's absolute path plus `member`, and a
filesystem source keeps exactly `str(path)` with no `member` key.

## Provenance

A source's entry in `loader.metadata["sources"]` names where it came from:
for a member, `path` is the archive's absolute path and `member` the
member's POSIX path inside it; a filesystem source keeps `path` as given
and has no `member` key. `loader.sources` holds the `zipfile.Path` itself.
The loaders' own test files pin the filesystem entries of `XmlLoader`,
`JsonLoader`, `ATONLoader` and `Ieee1599Loader`.

## Reading in place

- No audio backend needs a copy: after an archived WAV, an MPM project,
  a Parangonada dataset or a Listen Here! export (which reads no
  recording) is read, the cache folder holds no file, and the audio
  information equals the disk reading.
- MeLODy Turtle is parsed in place. A relative IRI in a member resolves
  against the member's URI, `jar:<archive file URI>!/<member>`
  (`<other.ttl>` in `repo/works/1.ttl` names
  `jar:file:///…/melo%20dy.zip!/repo/works/other.ttl`); on disk it
  resolves against the file's `file:` URI exactly as before. A directory
  member reads the same files as the directory on disk (index folders
  left out), and its graphs equal the disk graphs once the one base is
  put for the other.

## The cache (`materialize`)

- A filesystem path comes back unchanged.
- A file member is copied once to `archives/<key>/<member path>`; the copy's
  bytes equal the member's.
- A second call for the same member copies nothing: the copy keeps its inode
  and its modification time.
- An archive rewritten with other bytes (a new size and modification time)
  gets a fresh copy under another key, holding the new bytes.
- A member rewritten with other bytes of the same length, the archive
  keeping its size and its modification time, gets a fresh copy too: the
  key holds the member's CRC-32 and size.
- A directory member is copied with everything below it, and nothing else.
- No `*.part` temporary file is left behind.

## Navigation and order

- `find_sources` matches as `glob`/`rglob` do, directories that the archive
  holds only by implication included.
- Paths and members of one directory come in the same order, by parts
  compared as strings (`C.wav`, `a-2.wav`, `a.wav`, `a_1.wav`, `b.wav`);
  filesystem paths come before members.
- `join_source` resolves `./` and `..`; a member's parent stays inside the
  archive.

## Display and errors

- A member is shown as `<archive> member <name>`: `describe_source`, the
  loaders' event summary and the single-source refusal (exact message).
- A parser failing on the copy of a member raises its own exception with a
  note naming the archive and the member.
- A member the archive does not hold raises `FileNotFoundError` naming both
  the archive and the member.
- A string such as `"x.zip/inner.tsv"` is a filesystem path: it raises the
  loader's `FileNotFoundError` for that path even when `x.zip` exists and
  holds `inner.tsv`.
- A directory member given to a loader of single files raises the error the
  same loader raises for a filesystem directory.
- A member naming a path outside the archive (`../`) is refused by
  `materialize` with `ValueError`.

## Parallel safety

Every archive is built in the test's own `tmp_path` from read-only test
data; no test shares state with another.
