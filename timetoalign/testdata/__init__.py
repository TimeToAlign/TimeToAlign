"""On-demand fetching of bundled test/example corpora.

The corpora live as ``.tar.gz`` release assets in the
``TimeToAlign/tta_test_data`` GitHub repository. They are downloaded the
first time a corpus is requested, verified by SHA256, and extracted into a
predictable directory tree.

Public API
----------

* :func:`ensure_data` — fetch and extract one or more corpora, return their
  on-disk paths.
* :func:`get_data_dir` — single-corpus convenience wrapper.
* :data:`DATA_DIR` — directory under which corpora are extracted.
* :data:`REGISTRY` — mapping of archive filename → expected SHA256 digest.

Usage
-----

::

    from timetoalign.testdata import ensure_data

    HENDRIX = ensure_data("hendrix")
    VIENNA = ensure_data("vienna_1x22")
    audio_dir = ensure_data("audio") / "hard_techno"

Where the data lands
--------------------

By default :data:`DATA_DIR` is resolved as follows:

1. If the ``TTA_TESTDATA_DIR`` environment variable is set, it is used verbatim.
2. Otherwise, if this module appears to be running from a source checkout
   (a sibling ``tests/data/`` directory exists), that directory is used so
   that the existing ``Path(...) / "data" / "<name>"`` constants in
   ``conftest.py`` files keep working unchanged.
3. Otherwise (e.g. ``pip install timetoalign``), pooch's per-user cache
   directory is used.

The cached ``.tar.gz`` archives live in pooch's standard cache (override
with ``TTA_TESTDATA_CACHE``).

Atomic extraction and the sentinel
----------------------------------

Each archive is unpacked into a temporary sibling directory under
:data:`DATA_DIR` and then moved into place with a single rename, so a
process killed mid-extraction leaves either the previous state or an
orphaned temp directory — never a half-written corpus under a trusted
sentinel.

Completeness is recorded in a two-line ``.tta_testdata_hash`` sentinel that
:func:`ensure_data` writes *after* the rename::

    <archive SHA256 hex digest>
    <count of payload files under the corpus dir, excluding the sentinel>

A directory is reused without re-extracting only when the sentinel's digest
matches the expected release digest *and* at least the recorded number of
files are on disk; a tree with fewer files is a partial extraction and
self-heals on the next call, while extra files a test may have written
alongside the corpus are tolerated. Any ``.tta_testdata_hash`` entry shipped
inside an archive is stripped during extraction — the sentinel is authored
solely by :func:`ensure_data`. Legacy single-line sentinels (digest only)
carry no count and force one re-extraction that upgrades them to the
two-line form.

Publishing a new test-data release
----------------------------------

Edit corpora under ``tta_test_data/data/<name>/`` in the
``TimeToAlign/tta_test_data`` repository, commit, and push a tag::

    git tag testdata-v2
    git push origin testdata-v2

The release workflow in ``tta_test_data`` packages the corpora, publishes a
release with the tarballs as assets, and writes the matching ``REGISTRY``
block into the release notes. Update :data:`RELEASE_TAG` and :data:`REGISTRY`
below to point at the new release.
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
import threading
from pathlib import Path

_LOG = logging.getLogger(__name__)

#: Name of the per-corpus completeness sentinel authored by :func:`ensure_data`.
_SENTINEL_NAME = ".tta_testdata_hash"

try:
    import pooch
except ImportError as exc:  # pragma: no cover - tested via integration
    raise ImportError(
        "timetoalign.testdata requires the 'pooch' package. "
        "Install it via `pip install timetoalign[examples]` "
        "(or `pip install pooch`)."
    ) from exc


# ---------------------------------------------------------------------------
# Release coordinates — update together when republishing the test data
# ---------------------------------------------------------------------------

#: GitHub release tag in the ``tta_test_data`` repository.
RELEASE_TAG = "testdata-v7"

#: Base URL where the release assets are hosted.
BASE_URL = (
    f"https://github.com/TimeToAlign/tta_test_data/releases/download/{RELEASE_TAG}/"
)

#: ``filename -> "sha256:<hex>"`` for every archive in the release.
#:
#: The canonical values are produced by the ``Publish test data`` GitHub
#: Action in ``TimeToAlign/tta_test_data`` and copied from the release notes.
#: ``tarfile`` is not reproducible by default (mtimes / entry order leak into
#: the gzip bytes), so a local rebuild will not match the CI-produced
#: archives — always use the digests emitted by the workflow.
REGISTRY = {
    "audio.tar.gz": "sha256:fe2d402828579e48124e9c9cbec2579c7945f336002061b0c471ab27e06bbf92",
    "audiolabs_omr.tar.gz": "sha256:92560e241a592572af28aa04a2f7286d92613df98946201c20f449d65563c591",
    "fixtures.tar.gz": "sha256:30e57a44c6ccd505e5722791c4d242dedb01401ee1bbbe0e7d896f6d32c656fc",
    "hendrix.tar.gz": "sha256:0f8c18413451912df932fbeb7e8f9b5eb2c85c22920e11201af36ffa9e38021e",
    "ieee1599.tar.gz": "sha256:bd36de84560cf29bdf79abf72e14b79d2b00425ebeb4d488cf8ab68101de9e36",
    "layerit_clair_de_lune.tar.gz": "sha256:4c36ad623f3cb13dab37c87795b0598087445ffcad5a72f154c12c8fe66aef57",
    "midi.tar.gz": "sha256:0dac0ee900272657d6be48dade982159cc9fdbe740f1e6f32bec9b85ae3f417b",
    "mpm_toolbox.tar.gz": "sha256:9f550e63da196a166ffd95a2ddf7be3807439c39d0a4bee18ed25a0b92a40755",
    "mtd.tar.gz": "sha256:ada876ea06504e499fa6fd9355cd3a4806fee35c51902e2cac6b7a89c3a5f6bf",
    "parangonar.tar.gz": "sha256:d6969d36d5387a5c3d4472e7d61c8785a5b048b2c8a87e4e8549f811543cf9cd",
    "performance_precision.tar.gz": "sha256:f1b117a2963d562f87ef1267e6c985180a64fd2b872a5104994fded1455b1574",
    "score.tar.gz": "sha256:8e06d5861fda6637422d8d4ebbfee8050ed5c04da58121b8508ee801511d34ed",
    "supra.tar.gz": "sha256:4f9eab707cbab32e975567dbb2a929e2ec8b42651e409654726c3b10c21223b6",
    "tabular.tar.gz": "sha256:7c8423d4360229a0db928f8a6efc9e5a35129b9891f2d0010901f3dcb5d888fd",
    "target_flows.tar.gz": "sha256:6c4c54acf8497daca51cdbf12077b72c2cc22f028b9cfb2412f6ed1afd9ae5b8",
    "thoresen.tar.gz": "sha256:b7fef4054e26f6d3ccb6e818e71161a25647a0f19430d57fbea2d3834a1a2c77",
    "vienna_1x22.tar.gz": "sha256:3f410adaab402d51a7dc7e0f1c10d2a3f3d45268da410c57174243834489bb69",
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _default_data_dir() -> Path:
    """Resolve the directory under which corpora are extracted."""
    override = os.environ.get("TTA_TESTDATA_DIR")
    if override:
        return Path(override).expanduser().resolve()

    # ``timetoalign/testdata/__init__.py`` -> repo_root in an editable install
    # is two ``parent`` hops away.
    pkg_parent = Path(__file__).resolve().parents[2]
    source_tests = pkg_parent / "tests" / "data"
    if (pkg_parent / "tests").is_dir():
        return source_tests

    return Path(pooch.os_cache("timetoalign-testdata")) / "data"


#: Directory under which corpora are extracted.
DATA_DIR: Path = _default_data_dir()

_CACHE_DIR = Path(
    os.environ.get("TTA_TESTDATA_CACHE") or pooch.os_cache("timetoalign-testdata")
)

_POOCH = pooch.create(
    path=_CACHE_DIR,
    base_url=BASE_URL,
    registry=REGISTRY,
    env="TTA_TESTDATA_CACHE",
)

_EXTRACT_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def available_corpora() -> tuple[str, ...]:
    """Return the names of every corpus the registry knows about."""
    return tuple(sorted(name.removesuffix(".tar.gz") for name in REGISTRY))


def ensure_data(*names: str) -> tuple[Path, ...]:
    """Make sure each named corpus is extracted under :data:`DATA_DIR`.

    Downloads the archive via pooch (verifying its SHA256) on first use,
    unpacks it into a temporary sibling directory, and moves it into
    ``DATA_DIR / <name>`` with a single rename. Any ``.tta_testdata_hash``
    entry shipped inside the archive is stripped; the sentinel is authored
    here, last, as two lines — the archive digest and the count of extracted
    payload files. Subsequent calls with the same name return immediately
    while that sentinel's digest still matches and at least the recorded
    number of files remain, so a partial or hand-truncated tree self-heals.

    Args:
        *names: Corpus names (top-level subdirectory names under
            :data:`DATA_DIR`), e.g. ``"midi"``, ``"vienna_1x22"``.

    Returns:
        The resolved ``DATA_DIR / <name>`` paths in the same order. A single
        ``Path`` (not a tuple) is returned when called with exactly one name,
        to support ``DATA_DIR = ensure_data("midi")``.

    Raises:
        KeyError: If a name is not present in :data:`REGISTRY`.
    """
    paths = tuple(_ensure_one(name) for name in names)
    if len(paths) == 1:
        return paths[0]  # type: ignore[return-value]
    return paths


def get_data_dir(name: str) -> Path:
    """Return ``DATA_DIR / <name>`` after ensuring it is extracted."""
    return _ensure_one(name)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _ensure_one(name: str) -> Path:
    archive_name = f"{name}.tar.gz"
    if archive_name not in REGISTRY:
        raise KeyError(
            f"Unknown test-data corpus '{name}'. "
            f"Known: {', '.join(available_corpora())}"
        )

    target_dir = DATA_DIR / name
    expected_digest = REGISTRY[archive_name].removeprefix("sha256:")
    marker = target_dir / _SENTINEL_NAME

    if _looks_ready(target_dir, marker, expected_digest):
        return target_dir

    with _EXTRACT_LOCK:
        if _looks_ready(target_dir, marker, expected_digest):
            return target_dir

        archive_path = Path(_POOCH.fetch(archive_name))
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Unpack into a private temp dir on the same filesystem, then swap it
        # into place. A crash before the swap leaves the old tree (or nothing)
        # plus an orphan temp dir — never a partial tree under a valid sentinel.
        staging = Path(tempfile.mkdtemp(dir=DATA_DIR, prefix=f".{name}.staging-"))
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                _safe_extractall(tar, staging)

            extracted = staging / name
            if not extracted.is_dir():
                raise RuntimeError(
                    f"Archive {archive_name} did not produce expected directory "
                    f"{name!r}. Did the tarball layout change?"
                )

            if target_dir.exists():
                shutil.rmtree(target_dir)
            os.replace(extracted, target_dir)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        # Sentinel authored LAST, once the tree is fully in place: the digest
        # plus the payload-file count that later calls check for completeness.
        marker.write_text(
            f"{expected_digest}\n{_count_payload_files(target_dir, marker)}\n",
            encoding="utf-8",
        )

    return target_dir


def _count_payload_files(target_dir: Path, marker: Path) -> int:
    """Count regular files under ``target_dir``, excluding the sentinel itself.

    This is the completeness measure recorded on the sentinel's second line
    and re-verified by :func:`_looks_ready`. Directories are not counted;
    hidden files are.
    """
    count = 0
    for root, _dirs, files in os.walk(target_dir):
        for fname in files:
            if Path(root) / fname != marker:
                count += 1
    return count


def _looks_ready(target_dir: Path, marker: Path, expected_digest: str) -> bool:
    """Return True iff ``target_dir`` should be trusted without re-extracting.

    A directory is trusted in one of two cases:

    * **Sentinel match (canonical case)** — the two-line sentinel ``marker``
      exists, its first line equals ``expected_digest``, and the live
      payload-file count (:func:`_count_payload_files`) is at least the
      recorded count. The count guards against *partial* extractions where
      the sentinel was written but data files are missing: without it, a
      downstream ``open()`` would fail with ``FileNotFoundError`` far from the
      cause. A tree with more files than recorded (a test wrote next to the
      corpus) is still complete and stays ready. A legacy single-line
      sentinel (digest only, no count) has no count line and is therefore
      *not* ready, forcing one re-extraction that upgrades it.
    * **Developer-checkout fallback** — the sentinel is missing but the
      directory exists and contains at least one non-marker file (developer
      checkout / migration window — assume the user placed correct data here
      on purpose). This path emits a single ``logging.WARNING`` (channel
      ``timetoalign.testdata``) advising the caller that contents are not
      verified against the release digest.

    Returns ``False`` when ``target_dir`` does not exist, is empty, holds
    only the sentinel, or carries a sentinel whose digest mismatches or whose
    recorded count exceeds what is on disk. Deleting ``target_dir`` is the
    way to force a re-fetch.
    """
    if not target_dir.is_dir():
        return False

    if marker.is_file():
        lines = marker.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2:
            # Legacy single-line (digest-only) sentinel — no completeness
            # record, so re-extract once and upgrade to the two-line form.
            return False
        if lines[0].strip() != expected_digest:
            return False
        try:
            recorded_count = int(lines[1].strip())
        except ValueError:
            return False
        live_count = _count_payload_files(target_dir, marker)
        # Completeness means "every recorded file is present", i.e. at least
        # the recorded count. A tree with FEWER files is a partial extraction
        # and must be repaired; a tree with MORE (a test wrote alongside the
        # corpus) is still complete and must NOT trigger a re-extraction —
        # otherwise a stray sibling file would race concurrent readers into a
        # destructive rebuild. The count guards against under-population only.
        return live_count >= recorded_count and live_count > 0

    if not any(p for p in target_dir.iterdir() if p.name != marker.name):
        return False

    _LOG.warning(
        "%s has no .tta_testdata_hash sentinel — trusting existing contents; "
        "rerun ensure_data() after deleting the directory if you suspect drift",
        target_dir,
    )
    return True


def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract ``tar`` into ``dest`` while rejecting members that escape it.

    Any member whose basename is the sentinel name is dropped: the
    completeness sentinel is authored by :func:`ensure_data`, never trusted
    from the archive payload.
    """
    dest_resolved = dest.resolve()
    members = []
    for member in tar.getmembers():
        if Path(member.name).name == _SENTINEL_NAME:
            continue
        member_path = (dest_resolved / member.name).resolve()
        if member_path != dest_resolved and dest_resolved not in member_path.parents:
            raise RuntimeError(f"Refusing to extract unsafe path: {member.name!r}")
        members.append(member)
    tar.extractall(dest, members=members)


__all__ = [
    "BASE_URL",
    "DATA_DIR",
    "REGISTRY",
    "RELEASE_TAG",
    "available_corpora",
    "ensure_data",
    "get_data_dir",
]
