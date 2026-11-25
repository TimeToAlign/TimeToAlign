# Loader tests: sources inside ZIP archives

Every loader reads a source given as a filesystem path (`Path` or `str`)
or as a member of a ZIP archive (`zipfile.Path`), without extracting the
archive. A loader whose parser takes a file object reads the member in
place; one whose parser needs a file on disk (Partitura, music21, Verovio,
the audio backends, rdflib) reads a copy that `materialize` makes once in a
cache folder. These tests check both ways and the cache itself.

## What proves an archive source right

An archive source is right **if and only if loading the member gives exactly
what loading the same bytes from the filesystem gives.** Nothing else is a
fair witness: the archive adds no information and must lose none. So each
test packs real test-data files (resolved with `ensure_data(...)`) into a
ZIP archive in `tmp_path`, loads the member, loads the very same file from
the test data, and compares the two results with exact equality:

- event loaders: the events table (`pa.Table.equals`), the created
  timeline's id, length and event table;
- score loaders: the store's facet tables and the created timeline (its
  events, its measure map), so ms3 companion discovery inside an archive is
  proven by the companions' tables arriving, not by a file count;
- alignment loaders: the timelines' ids and event tables, and the
  MatchClaim table of the bundle;
- manifest-like loaders (audio, IIIF): the parsed info, field by field,
  except the one field that names the source, which must name the member.

No expected value is computed by the code under test: the filesystem
reading is the oracle, and the loaders' own test files pin that reading
against the formats. The families covered are the ones the loaders form:
tabular CSV (`CsvLoader`), `Ms3Loader` on a measures TSV and with
`auto_discover` (flat siblings and DCML facet folders inside the archive),
MIDI (`PerformanceMidiLoader`, `ScoreMidiLoader` on Partitura), XML
(`XmlLoader`, `RekordboxLoader`), JSON (`MeasureMapLoader`,
`IIIFManifestLoader`), audio (`AudioLoader`, read from a copy), and
alignment (`MatchfileLoader`, read by Partitura from a copy;
`PerformancePrecisionLoader`, a directory member searched in place).

## Provenance

A source's entry in `loader.metadata["sources"]` names where it came from:
for a member, `path` is the archive's absolute path and `member` the
member's POSIX path inside it; a filesystem source keeps `path` as given
and has no `member` key. `loader.sources` holds the `zipfile.Path` itself.

## The cache (`materialize`)

Every test that may copy sets `TTA_CACHE_DIR` to `tmp_path` (monkeypatch),
so no test writes to the home cache and parallel workers never share one.

- A filesystem path comes back unchanged.
- A file member is copied once to `archives/<key>/<member path>`; the copy's
  bytes equal the member's.
- A second call for the same member copies nothing: the copy keeps its inode
  and its modification time.
- An archive rewritten with other bytes (a new size and modification time)
  gets a fresh copy under another key, holding the new bytes.
- A directory member is copied with everything below it, and nothing else.
- No `*.part` temporary file is left behind.

## Errors

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
