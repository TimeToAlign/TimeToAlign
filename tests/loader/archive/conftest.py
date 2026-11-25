"""Every archive test writes its copies below its own folder."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every copy goes below this test's own folder, never to the home cache."""
    cache = tmp_path / "cache"
    monkeypatch.setenv("TTA_CACHE_DIR", str(cache))
    return cache
