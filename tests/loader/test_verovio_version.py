"""Verovio release suffixes must not prevent real MEI ingestion."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

verovio = pytest.importorskip("verovio")

from timetoalign.core.time import struct_to_rational  # noqa: E402
from timetoalign.loader.score.verovio import VerovioLoader  # noqa: E402

_MEI = """<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.1">
  <meiHead><fileDesc><titleStmt><title>Version test</title></titleStmt>
    <pubStmt/></fileDesc></meiHead>
  <music><body><mdiv xml:id="movement"><score>
    <scoreDef meter.count="1" meter.unit="4"><staffGrp>
      <staffDef n="1" lines="5" clef.shape="G" clef.line="2"/>
    </staffGrp></scoreDef>
    <section><measure xml:id="measure1" n="1"><staff n="1"><layer n="1">
      <note xml:id="n1" pname="c" oct="4" dur="8"/>
      <note xml:id="n2" pname="d" oct="4" dur="8"/>
    </layer></staff></measure></section>
  </score></mdiv></body></music>
</mei>
"""


@pytest.fixture
def mei_file(tmp_path: Path) -> Path:
    source = tmp_path / "version.mei"
    source.write_text(_MEI, encoding="utf-8")
    return source


def _report_version(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    real_toolkit = verovio.toolkit

    class VersionedToolkit:
        def __init__(self) -> None:
            self._toolkit = real_toolkit()

        def getVersion(self) -> str:
            return version

        def __getattr__(self, name: str) -> Any:
            return getattr(self._toolkit, name)

    monkeypatch.setattr(verovio, "toolkit", VersionedToolkit)


@pytest.mark.parametrize(
    "version",
    ["6.4.1", "6.4.1-fbe8f46", "6.4.1[undefined]", "4.4.0[undefined]"],
)
def test_supported_version_loads_mei(
    monkeypatch: pytest.MonkeyPatch, mei_file: Path, version: str
) -> None:
    _report_version(monkeypatch, version)

    loader = VerovioLoader.from_file(mei_file)
    timeline = loader.create_timeline()
    events = {
        row["id"]: (struct_to_rational(row["start"]), struct_to_rational(row["end"]))
        for row in timeline.get_events(include_children=True)
    }

    assert events == {
        "n1": (Fraction(0), Fraction(1, 2)),
        "n2": (Fraction(1, 2), Fraction(1)),
    }
    assert loader.store.metadata["verovio_version"] == version


@pytest.mark.parametrize("version", ["4.3.1", "4.3.1[undefined]"])
def test_old_version_raises_minimum_version_error(
    monkeypatch: pytest.MonkeyPatch, mei_file: Path, version: str
) -> None:
    _report_version(monkeypatch, version)

    with pytest.raises(ImportError) as exc_info:
        VerovioLoader.from_file(mei_file)

    assert str(exc_info.value) == (
        "VerovioLoader requires verovio>=4.4 (exact timemap positions); "
        f"found {version}"
    )
