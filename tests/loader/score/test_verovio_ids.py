"""VerovioLoader's minted timeline ids, given root or not (validation logic: README.md)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("verovio")

from tests.loader.minting import minted  # noqa: E402
from timetoalign.loader.score import VerovioLoader  # noqa: E402


def _movement(n: int, pname: str) -> str:
    """One named movement of one 2/4 measure holding one half note."""
    return (
        f'<mdiv xml:id="mdiv{n}" n="{n}" label="{"I" * n}"><score><scoreDef xml:id="sd{n}" '
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


def test_minted_root_then_movements(mei: Path) -> None:
    timeline = VerovioLoader.from_file(mei, seed=0).create_timeline()
    root, first, second = (f"score:{id_}" for id_ in minted(0, "clt", "clt", "clt"))
    assert timeline.id == root
    assert timeline.list_children() == [first, second]


@pytest.mark.parametrize("uid", ["score", "perf:clt_2", "anything"])
def test_a_given_root_draws_no_id(mei: Path, uid: str) -> None:
    timeline = VerovioLoader.from_file(mei, seed=0).create_timeline(uid=uid)
    assert timeline.id == uid
    assert timeline.list_children() == [
        f"score:{id_}" for id_ in minted(0, "clt", "clt")
    ]


def test_rendered_movements_follow_the_cmap(mei: Path) -> None:
    timeline = VerovioLoader.from_file(mei, seed=0).create_physical_timeline(
        uid="rendered:x"
    )
    _, first, second = minted(0, "cmap", "cpt", "cpt")
    assert timeline.id == "rendered:x"
    assert timeline.list_children() == [f"rendered:{first}", f"rendered:{second}"]


def test_the_seed_comes_from_the_bytes_read(mei: Path, tmp_path: Path) -> None:
    once = VerovioLoader.from_file(mei).create_timeline()
    again = VerovioLoader.from_file(mei).create_timeline()
    assert [once.id, *once.list_children()] == [again.id, *again.list_children()]
    other = tmp_path / "other.mei"
    other.write_text(MEI.replace("Two movements", "Two Movements"), encoding="utf-8")
    changed = VerovioLoader.from_file(other).create_timeline()
    assert changed.id != once.id
