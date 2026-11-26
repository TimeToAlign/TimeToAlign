"""A loader whose package is missing is reported naming the extra to install.

Strategy: ``tests/loader/README.md``, "Optional dependencies".
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
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
    ("timetoalign.loader.midi", "ScoreMidiLoader", ["score"], "partitura", "partitura"),
    (
        "timetoalign.loader.alignment",
        "MatchfileLoader",
        ["matchfile"],
        "partitura",
        "partitura",
    ),
    ("timetoalign.loader.alignment", "MpmLoader", ["mpm"], "lxml", "mpm"),
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


def test_performance_midi_load_names_the_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from timetoalign.loader.midi import PerformanceMidiLoader

    source = tmp_path / "piece.mid"
    source.write_bytes(b"")
    monkeypatch.setitem(sys.modules, "mido", None)
    with pytest.raises(ImportError) as excinfo:
        PerformanceMidiLoader().load(source)
    assert str(excinfo.value) == (
        'PerformanceMidiLoader needs mido: pip install "timetoalign[midi]"'
    )


SUBPACKAGES = [
    "timetoalign.loader",
    "timetoalign.loader.score",
    "timetoalign.loader.midi",
    "timetoalign.loader.alignment",
    "timetoalign.loader.graphical",
    "timetoalign.loader.tabular",
    "timetoalign.loader.physical",
    "timetoalign.loader.metadata",
    "timetoalign.loader.format",
]

USES = {
    "partitura": "from timetoalign.loader import MatchfileLoader",
    "scipy": "from timetoalign.loader import MatchfileLoader",
    "lxml": "from timetoalign.loader.alignment import MpmLoader",
    "mido": "PerformanceMidiLoader().load(source)",
}


@pytest.mark.parametrize(
    ("hidden", "message"),
    [
        (
            "partitura",
            'MatchfileLoader needs partitura: pip install "timetoalign[partitura]"',
        ),
        (
            "scipy",
            'MatchfileLoader needs scipy.interpolate: pip install "timetoalign[partitura]"',
        ),
        ("lxml", 'MpmLoader needs lxml: pip install "timetoalign[mpm]"'),
        ("mido", 'PerformanceMidiLoader needs mido: pip install "timetoalign[midi]"'),
    ],
)
def test_an_installation_without_the_package_imports_every_loader_subpackage(
    tmp_path: Path, hidden: str, message: str
) -> None:
    source = tmp_path / "piece.mid"
    source.write_bytes(b"")
    script = textwrap.dedent(f"""
        import importlib
        import sys
        from pathlib import Path

        sys.modules[{hidden!r}] = None
        for name in {SUBPACKAGES!r}:
            importlib.import_module(name)
            print(name)
        from timetoalign.loader.midi import PerformanceMidiLoader
        from timetoalign.loader.tabular import CsvLoader

        print(CsvLoader.__name__)
        source = Path({str(source)!r})
        try:
            {USES[hidden]}
        except ImportError as error:
            print(error)
        """)
    completed = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [*SUBPACKAGES, "CsvLoader", message]


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
        (
            "timetoalign.loader.alignment",
            [
                "Ieee1599Loader",
                "ListenHereLoader",
                "MapsLoader",
                "MtdLoader",
                "MatchfileLoader",
                "MpmLoader",
                "ParangonadaLoader",
                "PerformancePrecisionLoader",
                "TiliaDictStore",
                "TiliaJsonLoader",
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
