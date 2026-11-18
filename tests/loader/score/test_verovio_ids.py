"""VerovioLoader's minted timeline ids beside a given root id (validation logic: README.md)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("verovio")

from timetoalign.loader.score import VerovioLoader  # noqa: E402


def _movement(n: int, pname: str) -> str:
    """One movement of one 2/4 measure holding one half note."""
    return (
        f'<mdiv xml:id="mdiv{n}" n="{n}"><score><scoreDef xml:id="sd{n}" '
        f'meter.count="2" meter.unit="4"><staffGrp xml:id="sg{n}"><staffDef '
        f'xml:id="sdef{n}" n="1" lines="5" clef.shape="G" clef.line="2"/></staffGrp>'
        f'</scoreDef><section xml:id="s{n}"><measure xml:id="m{n}" n="1">'
        f'<staff xml:id="m{n}s" n="1"><layer xml:id="m{n}l" n="1">'
        f'<note xml:id="m{n}n" pname="{pname}" oct="4" dur="2"/></layer></staff>'
        "</measure></section></score></mdiv>"
    )


MEI = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.1">'
    "<meiHead><fileDesc><titleStmt><title>Two movements</title></titleStmt>"
    "<pubStmt/></fileDesc></meiHead><music><body>"
    + _movement(1, "c")
    + _movement(2, "d")
    + "</body></music></mei>"
)


@pytest.fixture
def mei(tmp_path: Path) -> Path:
    path = tmp_path / "two_movements.mei"
    path.write_text(MEI, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("uid", "root", "movements"),
    [
        (None, "score:clt_1", ["score:clt_2", "score:clt_3"]),
        ("score:clt_2", "score:clt_2", ["score:clt_3", "score:clt_4"]),
        ("score", "score", ["score:clt_2", "score:clt_3"]),
    ],
    ids=["minted", "given-minted-shape", "given-other-shape"],
)
def test_movements_never_take_the_root_id(
    mei: Path, uid: str | None, root: str, movements: list[str]
) -> None:
    timeline = VerovioLoader.from_file(mei).create_timeline(uid=uid)
    assert timeline.id == root
    assert timeline.list_children() == movements


def test_rendered_movements_never_take_the_root_id(mei: Path) -> None:
    timeline = VerovioLoader.from_file(mei).create_physical_timeline(
        uid="rendered:cpt_2"
    )
    assert timeline.id == "rendered:cpt_2"
    assert timeline.list_children() == ["rendered:cpt_3", "rendered:cpt_4"]
