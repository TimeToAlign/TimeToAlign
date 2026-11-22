"""A loader whose package is missing is reported naming the extra to install.

Strategy: ``tests/loader/README.md``, "Optional dependencies".
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from timetoalign.loader.graphical.source import ImageMetadata, ImageSource
from timetoalign.loader.score import Ms3Loader

MISSING = [
    # (package, loader, cached modules to drop, dependency, extra)
    (
        "timetoalign.loader.score",
        "PartituraLoader",
        ["partitura"],
        "partitura",
        "partitura",
    ),
    ("timetoalign.loader.score", "Music21Loader", ["music21"], "music21", "music21"),
    ("timetoalign.loader.score", "VerovioLoader", ["verovio"], "verovio", "verovio"),
    (
        "timetoalign.loader.midi",
        "PerformanceMidiLoader",
        ["performance", "base"],
        "mido",
        "midi",
    ),
    ("timetoalign.loader.midi", "ScoreMidiLoader", ["score"], "partitura", "partitura"),
]


def _without(
    monkeypatch: pytest.MonkeyPatch, package: str, modules: list[str], dependency: str
) -> None:
    """Make *dependency* unimportable and drop the cached modules that imported it."""
    monkeypatch.setitem(sys.modules, dependency, None)
    for module in modules:
        monkeypatch.delitem(sys.modules, f"{package}.{module}")


@pytest.mark.parametrize(
    ("package", "name", "modules", "dependency", "extra"),
    MISSING,
    ids=[entry[1] for entry in MISSING],
)
def test_missing_package_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
    package: str,
    name: str,
    modules: list[str],
    dependency: str,
    extra: str,
) -> None:
    exports = importlib.import_module(package)
    _without(monkeypatch, package, modules, dependency)
    monkeypatch.delattr(exports, name)
    with pytest.raises(ImportError) as excinfo:
        getattr(exports, name)
    assert (
        str(excinfo.value)
        == f'{name} needs {dependency}: pip install "timetoalign[{extra}]"'
    )


def test_from_import_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    exports = importlib.import_module("timetoalign.loader.score")
    _without(monkeypatch, "timetoalign.loader.score", ["music21"], "music21")
    monkeypatch.delattr(exports, "Music21Loader")
    with pytest.raises(ImportError) as excinfo:
        from timetoalign.loader.score import Music21Loader  # noqa: F401
    assert (
        str(excinfo.value)
        == 'Music21Loader needs music21: pip install "timetoalign[music21]"'
    )


def test_ms3_load_names_the_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "piece.notes.tsv"
    source.write_text("mc\tquarterbeats\n1\t0\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "ms3", None)
    with pytest.raises(ImportError) as excinfo:
        Ms3Loader().load(source)
    assert str(excinfo.value) == 'Ms3Loader needs ms3: pip install "timetoalign[ms3]"'


GRAPHICAL = 'pip install "timetoalign[graphical]"'


def test_an_image_source_without_pymupdf_names_the_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "pymupdf", None)
    with pytest.raises(ImportError) as excinfo:
        ImageSource.from_image_file(tmp_path / "page.png")
    assert str(excinfo.value) == f"ImageSource needs pymupdf: {GRAPHICAL}"


@pytest.mark.parametrize(
    "draw",
    [
        lambda image: image.draw_rectangle(0, 0, 1, 1),
        lambda image: image.draw_line(0, 0, 1, 1),
        lambda image: image.draw_path(None),
    ],
    ids=["draw_rectangle", "draw_line", "draw_path"],
)
def test_drawing_without_pillow_names_the_extra(
    monkeypatch: pytest.MonkeyPatch, draw
) -> None:
    import pymupdf

    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 2, 2), False)
    image = ImageSource(pixmap, ImageMetadata(width=2, height=2, source_type="file"))
    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(ImportError) as excinfo:
        draw(image)
    assert str(excinfo.value) == f"Drawing on an ImageSource needs PIL: {GRAPHICAL}"


@pytest.mark.parametrize(
    ("package", "names"),
    [
        (
            "timetoalign.loader.score",
            [
                "MeasureMapLoader",
                "ScoreStore",
                "ScoreEventType",
                "PartituraLoader",
                "Music21Loader",
                "Ms3Loader",
                "VerovioLoader",
            ],
        ),
        (
            "timetoalign.loader.midi",
            [
                "CC_PURPOSE",
                "MidiEventData",
                "MidiEventType",
                "MidiLoader",
                "MidiStore",
                "ScoreMidiEventData",
                "PerformanceMidiLoader",
                "ScoreMidiLoader",
            ],
        ),
    ],
)
def test_installed_exports_are_listed(package: str, names: list[str]) -> None:
    exports = importlib.import_module(package)
    assert exports.__all__ == names
    assert all(hasattr(exports, name) for name in names)


def test_unknown_name_is_an_attribute_error() -> None:
    exports = importlib.import_module("timetoalign.loader.score")
    with pytest.raises(AttributeError) as excinfo:
        exports.NoSuchLoader  # noqa: B018
    assert str(excinfo.value) == (
        "module 'timetoalign.loader.score' has no attribute 'NoSuchLoader'"
    )
