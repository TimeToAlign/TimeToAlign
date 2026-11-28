"""Event files beside a manifest, and packages of manifests sharing their tables.

Values in tests/alignment/README.md, section "Event Files and Packages".
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import timetoshare
from timetoshare.tables import (
    PackageProblem,
    TableError,
    fixture_path,
    non_event_rows,
    package_fixture_path,
    package_problems,
    read_table,
)

from timetoalign.alignment import AlignmentBundle, MatchClaim, Package
from timetoalign.alignment import package as package_module
from timetoalign.alignment import resolve
from timetoalign.core import (
    BoundingBox,
    Box,
    Coordinate,
    Event,
    EventReference,
    NumberType,
    Projection,
    Provenance,
    TimeUnit,
)
from timetoalign.storage import EventData, Parse
from timetoalign.timelines import Timeline
from timetoalign.timelines.manifest import (
    PARQUET_MEDIA_TYPE,
    Resource,
    _write_package_file,
    content_in_memory,
    event_tables,
    pre_parsed_schema,
    resource_path,
)
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    DiscretePhysicalTimeline,
)

#: TimeToShare's package fixture of two manifests sharing their table folders.
_SHARED = package_fixture_path("shared")

#: A file name the package writer mints.
_PART = re.compile(r"part_[0-9a-z]{13}\.parquet")

#: The rows each manifest of the shared fixture reads, as its expected.yaml states them.
_EXPECTED = {
    "bach.tts.yaml": {
        "MatchClaim": ["claim_1", "claim_2", "claim_5"],
        "Note": ["note_1", "note_2", "note_3", "note_4"],
        "Projection": ["projection_1", "projection_2", "projection_7", "projection_9"],
    },
    "handel.tts.yaml": {
        "MatchClaim": ["claim_3", "claim_4"],
        "Note": ["note_5", "note_6", "note_7"],
        "Projection": [
            "projection_3",
            "projection_4",
            "projection_5",
            "projection_6",
            "projection_8",
        ],
    },
}


def _number(
    numerator: int | None = None,
    denominator: int | None = None,
    value: float | None = None,
) -> dict[str, Any]:
    """A number as a package writes it: once, the other members null."""
    return {"numerator": numerator, "denominator": denominator, "value": value}


def _a() -> AlignmentBundle:
    """The bundle ``a`` of the README."""
    score = ContinuousLogicalTimeline(length=Fraction(16), uid="a_score")
    bar = ContinuousLogicalTimeline(length=Fraction(4), uid="a_bar")
    score.add_child(bar, Fraction(8))
    score.add_events(
        [
            {
                "id": "a_n1",
                "event_type": "Note",
                "start": Fraction(0),
                "duration": Fraction(1),
                "midi": 60,
                "name": "C4",
            },
            {
                "id": "a_n2",
                "event_type": "Note",
                "start": Fraction(3, 2),
                "midi": 62,
                "staff": 1,
            },
            {"id": "a_b1", "event_type": "Beat", "start": Fraction(1, 3)},
        ]
    )
    bar.add_events(
        [{"id": "a_n3", "event_type": "Note", "start": Fraction(1, 2), "midi": 64}]
    )
    pcm = DiscretePhysicalTimeline(length=705600, uid="a_pcm")
    pcm.add_events([{"id": "a_o1", "event_type": "Note", "start": 22050}])
    seg = DiscreteGraphicalTimeline(
        length=867,
        uid="a_seg",
        meta={"box": {"x": 7, "y": 0, "width": 867, "height": 270}},
    )
    seg.add_events(
        Box.from_shape(
            BoundingBox.from_xywh(385, 46, 139, 20, unit="pixels"),
            origin=7,
            id="a_rect",
        )
    )
    bundle = AlignmentBundle(id="a")
    for timeline in (score, pcm, seg):
        bundle.add_timeline(timeline)
    bundle.create_match_claims(
        [({"start": 0, "end": 4}, "a_score", "a_o1", "a_pcm")],
        provenance=Provenance(method="imported", service="https://example.org/aligner"),
    )
    return bundle


def _b() -> AlignmentBundle:
    """The bundle ``b`` of the README."""
    score = ContinuousLogicalTimeline(length=Fraction(8), uid="b_score")
    score.add_events(
        [
            {"id": "b_n1", "event_type": "Note", "start": Fraction(0), "midi": 67},
            {"id": "b_b1", "event_type": "Beat", "start": Fraction(1, 2)},
        ]
    )
    bundle = AlignmentBundle(id="b")
    bundle.add_timeline(score)
    return bundle


def _cell(
    value: float, numerator: int, denominator: int, unit: str, number_type: str
) -> dict[str, Any]:
    """A coordinate cell as memory holds it: both sides, with its axis."""
    return {
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "unit": unit,
        "number_type": number_type,
    }


def _ids(table: pa.Table) -> list[str]:
    """The ``id`` column of *table*, every row's ID a string."""
    return [str(value) for value in table.column("id").to_pylist()]


def _length(timeline: Timeline) -> Any:
    """The value of a timeline's known length."""
    length = timeline.length
    assert length is not None
    return length.value


def _rows(bundle: AlignmentBundle) -> dict[str, dict[str, dict[str, Any]]]:
    """Every timeline's event rows of *bundle*, by timeline and event ID, each
    row its cells holding a value: a column a timeline's family reads from a
    pre-parsed table spanning other timelines, null on this one, is no value
    of its events."""
    return {
        timeline_id: {
            row["id"]: {key: value for key, value in row.items() if value is not None}
            for row in timeline.events.table.to_pylist()
        }
        for timeline_id, timeline in sorted(bundle._known_timelines().items())
    }


def _files(directory: Path) -> dict[str, bytes]:
    """Every file under *directory* by its path from it, with its bytes."""
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _tool(root: str) -> str:
    """The ID of the parse a family of *root* mints for events made in code."""
    return Parse.of_tool(scope=root).id


def _event_rows(registry: pa.Table) -> list[dict[str, Any]]:
    """The event rows of a registry (its rows of a position), in its order."""
    return [row for row in registry.to_pylist() if row["start"] is not None]


def _non_events(registry: pa.Table) -> list[tuple[Any, ...]]:
    """The non-event rows of a registry: ``id``, ``kind``, ``location``,
    ``timeline_id``, sorted."""
    return sorted(
        (row["id"], row["kind"], row["location"], row["timeline_id"])
        for row in registry.to_pylist()
        if row["start"] is None
    )


def non_event_rows_of_lone(directory: Path, name: str) -> pa.Table:
    """TimeToShare's non-event rows of the manifest *name* lying in *directory*
    with its registry, Parse table and MatchClaim table, read as a
    package of one manifest (a copy given a ``package.yaml`` listing them)."""
    copy = Path(
        shutil.copytree(directory, directory.parent / f"{directory.name}_as_package")
    )
    (copy / "package.yaml").write_text(
        timetoshare.dumps(
            {
                "timetoshare": "0.2.0",
                "manifests": [{"url": name}],
                "tables": [
                    {"type": "Registry", "url": "registry.parquet"},
                    {"type": "Parse", "url": "parses.parquet"},
                    {"type": "MatchClaim", "url": "match_claims.parquet"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return non_event_rows(copy)


def _parts(directory: Path, kind: str) -> list[str]:
    """The names of the files of a package's folder of *kind*."""
    return sorted(path.name for path in (directory / "tables" / kind).iterdir())


class TestTablesBesideAManifest:
    def test_every_event_is_written_the_manifest_naming_none(
        self, tmp_path: Path
    ) -> None:
        bundle = _a()

        path = bundle.to_file(tmp_path / "a.yaml", events=True)

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "Note.parquet",
            "a.yaml",
            "boxes.parquet",
            "match_claims.parquet",
            "parses.parquet",
            "registry.parquet",
        ]
        manifest = timetoshare.loads(path.read_text(encoding="utf-8"))
        assert "events" not in str(manifest)
        assert [(r.type, r.timelines, r.url) for r in bundle.resources] == [
            ("Registry", ("a_bar", "a_pcm", "a_score", "a_seg"), "registry.parquet"),
            ("MatchClaim", ("a_pcm", "a_score"), "match_claims.parquet"),
        ]
        (claim,) = bundle.get_match_claims()
        fraction = {"unit": "quarters", "number_type": "fraction"}
        rows = {
            _tool("a_score"): [
                ("a_n3", "Note", "a_bar", fraction, _number(17, 2), None),
                (
                    "a_n1",
                    "Note",
                    "a_score",
                    fraction,
                    _number(0, 1),
                    _number(1, 1),
                ),
                (
                    claim.event_a_id,
                    "Projection",
                    "a_score",
                    fraction,
                    _number(0, 1),
                    _number(4, 1),
                ),
                ("a_b1", "Beat", "a_score", fraction, _number(1, 3), None),
                ("a_n2", "Note", "a_score", fraction, _number(3, 2), None),
            ],
            _tool("a_pcm"): [
                (
                    "a_o1",
                    "Note",
                    "a_pcm",
                    {"unit": "samples", "number_type": "int"},
                    _number(22050),
                    None,
                )
            ],
            _tool("a_seg"): [
                (
                    "a_rect",
                    "Box",
                    "a_seg",
                    {"unit": "pixels", "number_type": "int"},
                    _number(378),
                    _number(139),
                )
            ],
        }
        # Beats and Projections hold nothing but their IDs: no pre-parsed table.
        locations = {"Note": "Note.parquet", "Box": "boxes.parquet"}
        of = {claim.event_a_id: {"timeline_id": "a_pcm", "event_id": "a_o1"}}
        registry = pq.read_table(tmp_path / "registry.parquet")
        assert _event_rows(registry) == [
            {
                "id": event_id,
                "kind": kind,
                "location": locations.get(kind),
                "timeline_id": timeline_id,
                **axis,
                "start": start,
                "duration": duration,
                "parse_id": parse_id,
                "locator": None,
                "vanished_in": None,
                "projection_of": of.get(event_id),
            }
            for parse_id in sorted(rows)
            for event_id, kind, timeline_id, axis, start, duration in rows[parse_id]
        ]
        assert _non_events(registry) == _non_events(
            non_event_rows_of_lone(tmp_path, "a.yaml")
        )
        notes = pq.read_table(tmp_path / "Note.parquet")
        assert notes.column_names == ["id", "name", "midi", "staff"]
        # In the registry's order: the parses by ID, a_score's or a_pcm's first.
        score_notes = [
            {"id": "a_n3", "name": None, "midi": 64, "staff": None},
            {"id": "a_n1", "name": "C4", "midi": 60, "staff": None},
            {"id": "a_n2", "name": None, "midi": 62, "staff": 1},
        ]
        pcm_notes = [{"id": "a_o1", "name": None, "midi": None, "staff": None}]
        assert notes.to_pylist() == (
            score_notes + pcm_notes
            if _tool("a_score") < _tool("a_pcm")
            else pcm_notes + score_notes
        )
        assert [
            Parse.from_row(row)
            for row in pq.read_table(tmp_path / "parses.parquet").to_pylist()
        ] == sorted(bundle.registry.parses, key=lambda item: item.id)

    def test_the_events_read_back_the_same(self, tmp_path: Path) -> None:
        bundle = _a()
        (tmp_path / "first").mkdir()
        (tmp_path / "second").mkdir()
        bundle.to_file(tmp_path / "first" / "a.yaml", events=True)

        read = AlignmentBundle.from_file(tmp_path / "first" / "a.yaml")

        assert _rows(read) == _rows(bundle)
        assert _rows(read)["a_bar"]["a_n3"]["start"] == {
            "value": 0.5,
            "numerator": 1,
            "denominator": 2,
        }
        assert read.match_claim_table == bundle.match_claim_table
        assert event_tables(read.registry, events=True).rows.equals(
            event_tables(bundle.registry, events=True).rows
        )
        assert sorted(read.registry.parses, key=lambda item: item.id) == sorted(
            bundle.registry.parses, key=lambda item: item.id
        )
        read.to_file(tmp_path / "second" / "a.yaml")
        assert _files(tmp_path / "second") == _files(tmp_path / "first")

    def test_without_events_the_tables_alone_are_written(self, tmp_path: Path) -> None:
        _a().to_file(tmp_path / "a.yaml")

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "a.yaml",
            "boxes.parquet",
            "match_claims.parquet",
            "parses.parquet",
            "registry.parquet",
        ]
        assert [
            row["kind"]
            for row in _event_rows(pq.read_table(tmp_path / "registry.parquet"))
        ] == (
            ["Projection", "Box"]
            if _tool("a_score") < _tool("a_seg")
            else ["Box", "Projection"]
        )

    def test_the_sort_is_exact(self) -> None:
        floats = ContinuousPhysicalTimeline(length=10.0, uid="t_float")
        floats.add_events(
            [
                {"id": "q", "event_type": "Beat", "start": 0.5},
                {"id": "p", "event_type": "Beat", "start": 0.25},
            ]
        )
        fractions = ContinuousLogicalTimeline(length=Fraction(4), uid="t_frac")
        below_a_third = Fraction(6004799503160661, 2**54)
        assert float(below_a_third) == float(Fraction(1, 3))
        assert below_a_third < Fraction(1, 3)
        fractions.add_events(
            [
                {"id": "a", "event_type": "Beat", "start": Fraction(1, 3)},
                {"id": "b", "event_type": "Beat", "start": below_a_third},
                {"id": "c", "event_type": "Beat", "start": Fraction(1, 3)},
                {"id": "d", "event_type": "Beat", "start": Fraction(0)},
            ]
        )
        ticks = DiscreteLogicalTimeline(length=960, uid="t_int")
        ticks.add_events(
            [
                {"id": "z", "event_type": "Beat", "start": 480},
                {"id": "y", "event_type": "Beat", "start": 0},
                {"id": "x", "event_type": "Beat", "start": 480},
            ]
        )
        big = ContinuousLogicalTimeline(length=Fraction(4), uid="t_big")
        m = Fraction(4862121164571850412, 2331674737005167863)
        n = Fraction(2431060582285925452, 1165837368502584049)
        assert m < n
        assert float(m.numerator) / float(m.denominator) > float(n.numerator) / float(
            n.denominator
        )
        big.add_events(
            [
                {"id": "n", "event_type": "Beat", "start": n},
                {"id": "m", "event_type": "Beat", "start": m},
            ]
        )
        bundle = AlignmentBundle(id="sorts")
        for timeline in (ticks, fractions, floats, big):
            bundle.add_timeline(timeline)

        rows = event_tables(bundle.registry, events=True).rows.to_pylist()

        by_timeline: dict[str, list[str]] = {}
        for row in rows:
            by_timeline.setdefault(row["timeline_id"], []).append(row["id"])
        assert by_timeline == {
            "t_float": ["p", "q"],
            "t_frac": ["d", "b", "a", "c"],
            "t_int": ["y", "x", "z"],
            "t_big": ["m", "n"],
        }
        assert [row["parse_id"] for row in rows] == sorted(
            row["parse_id"] for row in rows
        )
        (b,) = [row for row in rows if row["id"] == "b"]
        assert b["start"] == _number(6004799503160661, 2**54)

    def test_a_further_coordinate_cell_is_written_once(self, tmp_path: Path) -> None:
        score = ContinuousLogicalTimeline(length=Fraction(8), uid="t")
        score.add_events(
            [
                {
                    "id": "n1",
                    "event_type": "Note",
                    "start": Fraction(0),
                    "alt": _cell(0.5, 1, 2, "seconds", "float"),
                },
                {
                    "id": "n2",
                    "event_type": "Note",
                    "start": Fraction(1),
                    "alt": _cell(0.75, 3, 4, "quarters", "fraction"),
                },
            ]
        )
        bundle = AlignmentBundle(id="alt")
        bundle.add_timeline(score)

        bundle.to_file(tmp_path / "alt.yaml", events=True)

        table = pq.read_table(tmp_path / "Note.parquet")
        assert table.column("alt").to_pylist() == [
            {**_number(value=0.5), "unit": "seconds", "number_type": "float"},
            {**_number(3, 4), "unit": "quarters", "number_type": "fraction"},
        ]
        assert content_in_memory(
            table, TimeUnit.quarters, NumberType.fraction, "Note table"
        ).column("alt").to_pylist() == [
            _cell(0.5, 1, 2, "seconds", "float"),
            _cell(0.75, 3, 4, "quarters", "fraction"),
        ]

    def test_projections_are_read_as_a_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundle = _a()
        bundle.to_file(tmp_path / "a.yaml", events=True)

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a Projection built row by row")

        with monkeypatch.context() as patch:
            patch.setattr(Projection, "from_row", classmethod(refuse))
            read = AlignmentBundle.from_file(tmp_path / "a.yaml")
            assert _rows(read) == _rows(bundle)

        # A manifest stating a_score's rows inline as well.
        data = timetoshare.loads((tmp_path / "a.yaml").read_text(encoding="utf-8"))
        inline = bundle.to_dict(events=True)
        (score,) = [t for t in inline["timelines"] if t["id"] == "#a_score"]
        (written,) = [t for t in data["timelines"] if t["id"] == "#a_score"]
        written["events"] = [
            row for row in score["events"] if row["event_type"] == "Projection"
        ]
        assert len(written["events"]) == 1
        (tmp_path / "a.yaml").write_text(timetoshare.dumps(data), encoding="utf-8")

        read = AlignmentBundle.from_file(tmp_path / "a.yaml")

        assert _rows(read) == _rows(bundle)

        # The inline Projection elsewhere under the registry row's ID.
        (row,) = written["events"]
        written["events"] = [{**row, "start": [7, 2], "duration": [1, 2]}]
        (tmp_path / "a.yaml").write_text(timetoshare.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError) as refused:
            AlignmentBundle.from_file(tmp_path / "a.yaml")
        assert str(refused.value) == (
            f"Event id '{row['id']}' is already held by an event of timeline "
            "'a_score'"
        )

    def test_a_user_segment_is_an_event(self, tmp_path: Path) -> None:
        line = ContinuousLogicalTimeline(length=Fraction(8), uid="seg_t")
        line.add_child(
            ContinuousLogicalTimeline(length=Fraction(2), uid="seg_c"), Fraction(4)
        )
        line.add_events(
            [
                {
                    "id": "s1",
                    "event_type": "Segment",
                    "start": Fraction(1),
                    "duration": Fraction(1),
                },
                {"id": "n1", "event_type": "Note", "start": Fraction(0)},
            ]
        )
        bundle = AlignmentBundle(id="seg")
        bundle.add_timeline(line)

        bundle.to_file(tmp_path / "seg.yaml", events=True)

        # Both hold nothing but their IDs: their registry rows alone.
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "parses.parquet",
            "registry.parquet",
            "seg.yaml",
        ]
        assert sorted(
            (row["id"], row["kind"], row["location"])
            for row in _event_rows(pq.read_table(tmp_path / "registry.parquet"))
        ) == [("n1", "Note", None), ("s1", "Segment", None)]
        read = AlignmentBundle.from_file(tmp_path / "seg.yaml")
        timeline = read.get_timeline("seg_t")
        assert sorted(_ids(timeline.events.table)) == ["n1", "s1", "seg_c"]
        assert timeline.get_child("seg_c").id == "seg_c"
        assert _rows(read) == _rows(bundle)
        inline = Timeline.from_dict(line.to_dict(events=True))
        assert sorted(_ids(inline.events.table)) == ["n1", "s1", "seg_c"]

    def test_rows_read_obey_the_length_rule(self, tmp_path: Path) -> None:
        _b().to_file(tmp_path / "b.yaml", events=True)
        text = (tmp_path / "b.yaml").read_text(encoding="utf-8")
        assert text.count("length: [8, 1]") == 1
        (tmp_path / "b.yaml").write_text(
            text.replace("length: [8, 1]", "length: [1, 4]"), encoding="utf-8"
        )

        read = AlignmentBundle.from_file(tmp_path / "b.yaml")

        assert _length(read.get_timeline("b_score")) == Fraction(1, 2)
        (tmp_path / "b.yaml").write_text(
            text.replace("length: [8, 1]", "length: [1, 4]\n    locked: true"),
            encoding="utf-8",
        )
        with pytest.raises(ValueError) as refused:
            AlignmentBundle.from_file(tmp_path / "b.yaml")
        assert str(refused.value) == (
            "Adding content ending at 1/2 exceeds timeline length 1/4. Timeline "
            "is locked. Use allow_expansion=True to override."
        )

        score = ContinuousLogicalTimeline(length=Fraction(8), uid="t")
        score.add_events([{"id": "n1", "event_type": "Note", "start": Fraction(3)}])
        data = score.to_dict(events=True)
        data["timelines"][0]["length"] = [1, 1]
        assert _length(Timeline.from_dict(data)) == Fraction(3)
        data["timelines"][0]["locked"] = True
        with pytest.raises(ValueError) as refused:
            Timeline.from_dict(data)
        assert str(refused.value) == (
            "Adding content ending at 3 exceeds timeline length 1. Timeline is "
            "locked. Use allow_expansion=True to override."
        )

    def test_a_table_of_other_rows_is_carried(self, tmp_path: Path) -> None:
        (tmp_path / "first").mkdir()
        (tmp_path / "second").mkdir()
        features = tmp_path / "first" / "features.parquet"
        pq.write_table(pa.table({"x": [1, 2]}), features)
        bundle = _b()
        bundle.resources.append(
            Resource("features", "Features", ("b_score",), "features.parquet")
        )
        bundle.to_file(tmp_path / "first" / "b.yaml", events=True)

        read = AlignmentBundle.from_file(tmp_path / "first" / "b.yaml")

        assert _rows(read) == _rows(bundle)
        (carried,) = [r for r in read.resources if r.id == "features"]
        assert (carried.type, carried.url) == ("Features", "features.parquet")
        shutil.copy(features, tmp_path / "second" / "features.parquet")
        read.to_file(tmp_path / "second" / "b.yaml", events=True)
        manifest = timetoshare.loads(
            (tmp_path / "second" / "b.yaml").read_text(encoding="utf-8")
        )
        assert [r for r in manifest["resources"] if r["id"] == "#features"] == [
            {
                "id": "#features",
                "type": "Features",
                "timelines": ["#b_score"],
                "url": "features.parquet",
            }
        ]
        assert (tmp_path / "second" / "features.parquet").read_bytes() == (
            features.read_bytes()
        )

    def test_a_folder_of_a_type_not_written_is_carried(self, tmp_path: Path) -> None:
        bundle = _b()
        scans = Resource("scans", "Scan", ("b_score",), "scans/")
        bundle.resources.append(scans)

        bundle.to_file(tmp_path / "b.yaml", events=True)

        read = AlignmentBundle.from_file(tmp_path / "b.yaml")
        assert scans in read.resources
        assert _rows(read) == _rows(bundle)
        assert read._carried_tables == {}

        # The folder lying there, holding no Parquet file: no rows.
        (tmp_path / "scans").mkdir()
        (tmp_path / "scans" / "page1.png").write_bytes(b"\x89PNG")
        read = AlignmentBundle.from_file(tmp_path / "b.yaml")
        assert scans in read.resources
        assert _rows(read) == _rows(bundle)
        assert list(read._carried_tables) == ["scans"]

    def test_the_file_declares_its_type_version_order_and_index(
        self, tmp_path: Path
    ) -> None:
        _a().to_file(tmp_path / "lone" / "a.yaml", events=True)
        package = Package()
        package.write_bundle(_a(), "a.tts.yaml")
        package.to_file(tmp_path / "package" / "package.yaml")

        assert pq.read_schema(tmp_path / "package" / "registry.parquet").metadata == {
            b"timetoshare:type": b"Registry",
            b"timetoshare:version": b"0.2.0",
        }
        for path, kind, sorting in (
            (tmp_path / "lone" / "registry.parquet", b"Registry", ()),
            (tmp_path / "lone" / "Note.parquet", b"Note", ()),
        ):
            file = pq.ParquetFile(path)
            assert file.schema_arrow.metadata == {
                b"timetoshare:type": kind,
                b"timetoshare:version": b"0.2.0",
            }
            assert file.metadata.num_row_groups == 1
            group = file.metadata.row_group(0)
            assert group.sorting_columns == sorting
            chunks = [group.column(index) for index in range(group.num_columns)]
            assert {chunk.compression for chunk in chunks} == {"ZSTD"}
            assert all(chunk.is_stats_set for chunk in chunks)
            assert all(chunk.has_offset_index for chunk in chunks)
            assert all(chunk.has_column_index for chunk in chunks)

    def test_row_groups_hold_at_most_65536_rows(self, tmp_path: Path) -> None:
        n = 65_537
        table = pa.table(
            {"id": pa.array([f"beat_{index}" for index in range(n)])},
            schema=pre_parsed_schema("Beat"),
        )

        _write_package_file(table, tmp_path / "beats.parquet", ())

        metadata = pq.ParquetFile(tmp_path / "beats.parquet").metadata
        assert [
            metadata.row_group(index).num_rows
            for index in range(metadata.num_row_groups)
        ] == [65_536, 1]

    def test_a_table_without_its_type_is_refused(self, tmp_path: Path) -> None:
        table = pre_parsed_schema("Beat").empty_table().replace_schema_metadata(None)

        with pytest.raises(ValueError) as refused:
            _write_package_file(table, tmp_path / "beats.parquet", ())

        assert str(refused.value) == (
            "A package table names its type in its metadata ('timetoshare:type')."
        )
        assert list(tmp_path.iterdir()) == []

    def test_a_column_of_two_types_is_refused(self, tmp_path: Path) -> None:
        bundle = AlignmentBundle(id="staves")
        for uid, staff in (("s1", 1), ("s2", "I")):
            score = ContinuousLogicalTimeline(length=Fraction(4), uid=uid)
            score.add_events(
                [
                    {
                        "id": f"{uid}_n",
                        "event_type": "Note",
                        "start": Fraction(0),
                        "staff": staff,
                    }
                ]
            )
            bundle.add_timeline(score)
        (tmp_path / "with").mkdir()
        (tmp_path / "without").mkdir()

        with pytest.raises(ValueError) as refused:
            bundle.to_file(tmp_path / "with" / "staves.yaml", events=True)

        assert str(refused.value) == (
            "The Note table cannot hold the column 'staff' of timeline 's2' "
            "(string): timeline 's1' holds it as int64, and a column has one type "
            "in a table."
        )
        assert list((tmp_path / "with").iterdir()) == []
        bundle.to_file(tmp_path / "without" / "staves.yaml")
        assert sorted(p.name for p in (tmp_path / "without").iterdir()) == [
            "staves.yaml"
        ]


def _velocity(uid: str, value: Any, type_: pa.DataType) -> Timeline:
    """A ticks timeline *uid* made from an event table (as a MIDI loader
    makes it), holding the Note ``<uid>_n`` at 0 with ``velocity`` *value*
    of *type_*."""
    data = EventData.from_arrays(
        {
            "id": pa.array([f"{uid}_n"]),
            "event_type": pa.array(["Note"]),
            "start": pa.array([0]),
            "velocity": pa.array([value], type_),
        },
        TimeUnit.ticks,
        NumberType.int,
        extra_fields=[pa.field("velocity", type_)],
    )
    return data.create_timeline(uid=uid)


def _velocities(table: pa.Table) -> dict[str, Any]:
    """Each Note's ``velocity`` of a pre-parsed Note table, by ID."""
    return dict(
        zip(
            table.column("id").to_pylist(),
            table.column("velocity").to_pylist(),
            strict=True,
        )
    )


def _sourced(uid: str, source: Path | zipfile.Path) -> tuple[Timeline, Parse]:
    """The root *uid* (quarters) holding the Note ``<uid>_n`` made by the
    parse of *source*, at the locator ``row:0``, and that parse."""
    parse = Parse.of_source(source, parser="TestParser", parser_version="1")
    score = ContinuousLogicalTimeline(length=Fraction(4), uid=uid)
    score.add_events(
        [
            {
                "id": f"{uid}_n",
                "event_type": "Note",
                "start": Fraction(0),
                "parse_id": parse.id,
                "locator": "row:0",
            }
        ],
        parses=[parse],
    )
    return score, parse


def _bundle_of(*timelines: Timeline) -> AlignmentBundle:
    """A bundle ``piece`` of *timelines*."""
    bundle = AlignmentBundle(id="piece")
    for timeline in timelines:
        bundle.add_timeline(timeline)
    return bundle


#: The name the package writer mints for ``e``'s folder ``scans/``, whose
#: name ``b``'s copy holds (seeded from the type ``Scan`` and the folder's
#: content; probed once).
_MINTED_SCANS = "scans_2uozflji0q0pn"


def _two_bundle_package(
    directory: Path,
) -> tuple[Path, AlignmentBundle, AlignmentBundle]:
    """``a`` and ``b`` written into a new package in *directory*."""
    a, b = _a(), _b()
    package = Package(id="https://example.org/pieces", name="Pieces")
    package.write_bundle(a, "a.tts.yaml")
    package.write_bundle(b, "b.tts.yaml")
    return package.to_file(directory / "package.yaml"), a, b


def _carrying_scans(
    directory: Path, bundle: AlignmentBundle, data: bytes
) -> AlignmentBundle:
    """*bundle* given a resource ``scans`` (``Scan``) naming the folder
    ``scans/`` holding ``page1.png`` of *data*, written into *directory* with
    its events and read back: a bundle carrying that folder."""
    (directory / "scans").mkdir(parents=True)
    (directory / "scans" / "page1.png").write_bytes(data)
    (timeline,) = bundle.timelines
    bundle.resources.append(Resource("scans", "Scan", (timeline,), "scans/"))
    bundle.to_file(directory / f"{bundle.id}.tts.yaml", events=True)
    read = AlignmentBundle.from_file(directory / f"{bundle.id}.tts.yaml")
    assert list(read._carried_tables) == ["scans"]
    return read


def _carrying(
    directory: Path, bundle: AlignmentBundle, values: list[int]
) -> AlignmentBundle:
    """*bundle* given a resource ``features`` (``Features``) naming a Parquet
    file of the column ``x`` (*values*), written into *directory* with its
    events and read back: a bundle carrying that table."""
    directory.mkdir(parents=True)
    pq.write_table(pa.table({"x": values}), directory / "features.parquet")
    (timeline,) = bundle.timelines
    bundle.resources.append(
        Resource("features", "Features", (timeline,), "features.parquet")
    )
    bundle.to_file(directory / f"{bundle.id}.tts.yaml", events=True)
    read = AlignmentBundle.from_file(directory / f"{bundle.id}.tts.yaml")
    assert list(read._carried_tables) == ["features"]
    return read


#: The name the package writer mints for ``b``'s ``beats.parquet``, whose
#: name ``a``'s copy holds (seeded from the type ``Beat`` and the file's
#: content; probed once).
_MINTED_BEATS = "beats_2dggumq9r5gv4.parquet"


def _carrying_beats(
    directory: Path, bundle: AlignmentBundle, event_id: str, label: str
) -> AlignmentBundle:
    """*bundle* given a resource ``beats`` (``Beat``) naming ``beats.parquet``,
    a Parquet file of one row: *event_id*, the bundle's root timeline,
    *label*; written into *directory* with its events and read back: a
    bundle carrying that table."""
    directory.mkdir(parents=True)
    (timeline,) = [t for t in bundle.timelines if t.endswith("_score")]
    pq.write_table(
        pa.table({"id": [event_id], "timeline_id": [timeline], "label": [label]}),
        directory / "beats.parquet",
    )
    bundle.resources.append(Resource("beats", "Beat", (timeline,), "beats.parquet"))
    bundle.to_file(directory / f"{directory.name}.tts.yaml", events=True)
    read = AlignmentBundle.from_file(directory / f"{directory.name}.tts.yaml")
    assert list(read._carried_tables) == ["beats"]
    return read


def _own_rows(root: Path, manifest: str) -> list[dict[str, Any]]:
    """The rows of the ``Beat`` table the manifest at *manifest* names, read
    with the timelines its resource ``beats`` lists."""
    (resource,) = [
        r
        for r in timetoshare.loads((root / manifest).read_text(encoding="utf-8"))[
            "resources"
        ]
        if r["id"] == "#beats"
    ]
    folder = (root / manifest).parent
    return read_table(
        resource_path(folder, resource["url"]),
        timelines=[timeline.removeprefix("#") for timeline in resource["timelines"]],
        type="Beat",
    ).to_pylist()


class TestWritersOfLoneManifests:
    """Values in tests/alignment/README.md, "The tables beside a lone
    manifest" (widening, locators, the rewrite)."""

    def test_a_column_held_in_two_widths_is_widened(self, tmp_path: Path) -> None:
        bundle = AlignmentBundle(id="velocities")
        bundle.add_timeline(_velocity("s1", 100, pa.int8()))
        bundle.add_timeline(_velocity("s2", 300, pa.int64()))

        bundle.to_file(tmp_path / "v.yaml", events=True)

        notes = pq.read_table(tmp_path / "Note.parquet")
        assert notes.schema.field("velocity").type == pa.int64()
        assert _velocities(notes) == {"s1_n": 100, "s2_n": 300}
        read = AlignmentBundle.from_file(tmp_path / "v.yaml")
        assert [
            read.get_timeline(uid).events.table.column("velocity").to_pylist()
            for uid in ("s1", "s2")
        ] == [[100], [300]]

    def test_a_widening_that_may_change_a_value_is_refused(
        self, tmp_path: Path
    ) -> None:
        bundle = AlignmentBundle(id="velocities")
        bundle.add_timeline(_velocity("s1", 100, pa.int64()))
        bundle.add_timeline(_velocity("s2", 100.5, pa.float64()))

        with pytest.raises(ValueError) as refused:
            bundle.to_file(tmp_path / "v.yaml", events=True)

        assert str(refused.value) == (
            "The Note table cannot hold the column 'velocity' of timeline 's2' "
            "(double): timeline 's1' holds it as int64, and a column has one type "
            "in a table."
        )
        assert list(tmp_path.iterdir()) == []

    def test_a_locator_repeated_across_families_is_refused(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "notes.csv").write_bytes(b"a\n1\n")
        first, parse = _sourced("l1", tmp_path / "notes.csv")
        second, _ = _sourced("l2", tmp_path / "notes.csv")
        bundle = _bundle_of(first, second)
        (tmp_path / "out").mkdir()

        with pytest.raises(ValueError) as refused:
            bundle.to_file(tmp_path / "out" / "l.yaml", events=True)

        assert str(refused.value) == (
            f"The locator 'row:0' of the parse '{parse.id}' names an "
            "event on the timeline 'l1' and one on 'l2': a locator names one event "
            "of its parse's source; nothing is written."
        )
        assert list((tmp_path / "out").iterdir()) == []

    def test_a_rewrite_leaves_no_pre_parsed_file_of_a_kind_it_no_longer_holds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundle = _a()
        path = bundle.to_file(tmp_path / "a.yaml", events=True)
        before = _files(tmp_path)

        def refuse(*args: Any) -> Path:
            raise OSError("induced")

        with monkeypatch.context() as patch:
            patch.setattr(package_module, "_write_package_file", refuse)
            with pytest.raises(OSError, match="^induced$"):
                bundle.to_file(path)
        assert _files(tmp_path) == before

        bundle.to_file(path)

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "a.yaml",
            "boxes.parquet",
            "match_claims.parquet",
            "parses.parquet",
            "registry.parquet",
        ]


class TestParseSources:
    """Values in tests/alignment/README.md, "A source is written where the
    manifest lies" and "A source is written from the package's root"."""

    @pytest.mark.parametrize("where", ["beside", "elsewhere", "member"])
    def test_a_source_is_written_where_the_manifest_lies(
        self, tmp_path: Path, where: str
    ) -> None:
        folder = (tmp_path / "lone").resolve()
        (tmp_path / "elsewhere").mkdir()
        folder.mkdir()
        if where == "beside":
            source: Path | zipfile.Path = folder / "notes 1.csv"
            source.write_bytes(b"a\n1\n")
            written = "notes%201.csv"
        elif where == "elsewhere":
            source = (tmp_path / "elsewhere" / "notes.csv").resolve()
            source.write_bytes(b"a\n1\n")
            written = source.as_uri()
        else:
            with zipfile.ZipFile(folder / "archive.zip", "w") as archive:
                archive.writestr("dir/notes.csv", b"a\n1\n")
            source = zipfile.Path(
                zipfile.ZipFile(folder / "archive.zip"), at="dir/notes.csv"
            )
            written = "jar:archive.zip!/dir/notes.csv"
        score, parse = _sourced("score", source)

        _bundle_of(score).to_file(folder / "a.yaml", events=True)

        (row,) = pq.read_table(folder / "parses.parquet").to_pylist()
        assert (row["id"], row["source"]) == (parse.id, written)
        read = AlignmentBundle.from_file(folder / "a.yaml")
        assert read.registry.parses == [parse]

    def test_a_source_is_written_from_the_package_s_root(self, tmp_path: Path) -> None:
        root = (tmp_path / "package").resolve()
        (root / "sources").mkdir(parents=True)
        (root / "sources" / "notes.csv").write_bytes(b"a\n1\n")
        score, parse = _sourced("score", root / "sources" / "notes.csv")
        package = Package()
        package.write_bundle(_bundle_of(score), "score.tts.yaml")

        path = package.to_file(root / "package.yaml")

        table = root / "tables" / "parses.parquet"
        (row,) = pq.read_table(table).to_pylist()
        assert (row["id"], row["source"]) == (parse.id, "sources/notes.csv")
        opened = Package.from_file(path)
        read = opened.bundle("score.tts.yaml")
        assert read.registry.parses == [parse]
        before = table.read_bytes()
        opened.write_bundle(read, "score.tts.yaml")
        assert table.read_bytes() == before
        assert package_problems(path) == []


class TestPackage:
    def test_two_bundles_in_one_package(self, tmp_path: Path) -> None:
        path, a, b = _two_bundle_package(tmp_path)

        files = _files(tmp_path)
        parses = [_tool(root) for root in ("a_score", "a_pcm", "a_seg", "b_score")]
        assert [name for name in files if not name.startswith("tables/")] == [
            "a.tts.yaml",
            "b.tts.yaml",
            "package.yaml",
            "registry.parquet",
        ]
        parts = {kind: _parts(tmp_path, kind) for kind in ("Box", "MatchClaim", "Note")}
        assert {kind: len(names) for kind, names in parts.items()} == dict.fromkeys(
            parts, 1
        )
        assert all(_PART.fullmatch(name) for names in parts.values() for name in names)
        # Beats and Projections hold nothing but their IDs: no folder of theirs.
        assert sorted(p.name for p in (tmp_path / "tables").iterdir()) == [
            "Box",
            "MatchClaim",
            "Note",
            "parses.parquet",
        ]
        assert len(files) == 4 + 3 + 1
        assert package_problems(path) == []
        tables = [
            {
                "media_type": PARQUET_MEDIA_TYPE,
                "type": "Registry",
                "url": "registry.parquet",
            },
            {
                "media_type": PARQUET_MEDIA_TYPE,
                "type": "Parse",
                "url": "tables/parses.parquet",
            },
            *(
                {
                    "media_type": PARQUET_MEDIA_TYPE,
                    "type": kind,
                    "url": f"tables/{kind}/",
                }
                for kind in ("Note", "Box", "MatchClaim")
            ),
        ]
        assert timetoshare.loads(path.read_text(encoding="utf-8")) == {
            "timetoshare": "0.2.0",
            "id": "https://example.org/pieces",
            "name": "Pieces",
            "manifests": [
                {"media_type": "application/yaml", "url": "a.tts.yaml"},
                {"media_type": "application/yaml", "url": "b.tts.yaml"},
            ],
            "tables": tables,
        }
        opened = Package.from_file(path)
        assert opened.manifests == ["a.tts.yaml", "b.tts.yaml"]
        assert opened.tables == tables
        assert (opened.id, opened.name, opened.path) == (
            "https://example.org/pieces",
            "Pieces",
            path,
        )
        for bundle, url, expected in (
            (
                a,
                "a.tts.yaml",
                [
                    ("MatchClaim", ("a_pcm", "a_score"), "tables/MatchClaim/"),
                    (
                        "Registry",
                        ("a_bar", "a_pcm", "a_score", "a_seg"),
                        "registry.parquet",
                    ),
                ],
            ),
            (
                b,
                "b.tts.yaml",
                [("Registry", ("b_score",), "registry.parquet")],
            ),
        ):
            assert sorted(
                (r.type, r.timelines, r.url, r.media_type, r.size, r.integrity)
                for r in bundle.resources
            ) == [
                (kind, timelines, url_, PARQUET_MEDIA_TYPE, None, None)
                for kind, timelines, url_ in expected
            ]
            manifest = timetoshare.with_bare_ids(
                timetoshare.loads((tmp_path / url).read_text(encoding="utf-8"))
            )
            assert sorted(
                (Resource.from_dict(entry) for entry in manifest["resources"]),
                key=lambda resource: resource.id,
            ) == sorted(bundle.resources, key=lambda resource: resource.id)
            read = opened.bundle(url)
            assert _rows(read) == _rows(bundle)
            assert read.match_claim_table == bundle.match_claim_table
        for name in files:
            if name.startswith("tables/") or name == "registry.parquet":
                metadata = pq.read_schema(tmp_path / name).metadata
                kind = (
                    "Registry"
                    if name == "registry.parquet"
                    else (
                        "Parse"
                        if name == "tables/parses.parquet"
                        else name.split("/")[1]
                    )
                )
                assert metadata[b"timetoshare:type"] == kind.encode()
                assert metadata[b"timetoshare:version"] == b"0.2.0"
        registry = pq.read_table(tmp_path / "registry.parquet")
        events = _event_rows(registry)
        # The non-event rows first, then the event rows.
        starts = registry.column("start").to_pylist()
        assert starts[: registry.num_rows - len(events)] == [None] * (
            registry.num_rows - len(events)
        )
        assert _non_events(registry) == _non_events(non_event_rows(tmp_path))
        (projection,) = [row["id"] for row in events if row["kind"] == "Projection"]
        assert [(row["timeline_id"], row["id"]) for row in events] == [
            ("a_bar", "a_n3"),
            ("a_pcm", "a_o1"),
            ("a_score", "a_n1"),
            ("a_score", projection),
            ("a_score", "a_b1"),
            ("a_score", "a_n2"),
            ("a_seg", "a_rect"),
            ("b_score", "b_n1"),
            ("b_score", "b_b1"),
        ]
        root_of = {"a_bar": "a_score"}
        assert [row["parse_id"] for row in events] == [
            _tool(root_of.get(row["timeline_id"], row["timeline_id"])) for row in events
        ]
        assert events[0]["start"] == _number(17, 2)
        assert sorted(
            pq.read_table(tmp_path / "tables" / "parses.parquet")
            .column("id")
            .to_pylist()
        ) == sorted(parses)
        (claims,) = parts["MatchClaim"]
        claim_file = pq.read_table(tmp_path / "tables" / "MatchClaim" / claims)
        assert b"timetoshare:provenance" not in claim_file.schema.metadata
        assert claim_file.column("provenance").to_pylist() == [
            {
                "basis": None,
                "beliefs": None,
                "date": None,
                "match": None,
                "method": "imported",
                "note": None,
                "service": "https://example.org/aligner",
                "stated_by": None,
            }
        ]

    def test_a_kind_holding_nothing_but_ids_has_no_table(self, tmp_path: Path) -> None:
        take = ContinuousPhysicalTimeline(length=10.0, uid="o_take")
        take.add_events(
            [
                {"id": "o1", "event_type": "Onset", "start": 0.5},
                {"id": "o2", "event_type": "Onset", "start": 1.25, "duration": 0.25},
            ]
        )
        bundle = AlignmentBundle(id="onsets")
        bundle.add_timeline(take)
        package = Package()
        package.write_bundle(bundle, "o.tts.yaml")

        path = package.to_file(tmp_path / "package.yaml")

        assert sorted(_files(tmp_path)) == [
            "o.tts.yaml",
            "package.yaml",
            "registry.parquet",
            "tables/parses.parquet",
        ]
        assert [
            (table["type"], table["url"])
            for table in timetoshare.loads(path.read_text(encoding="utf-8"))["tables"]
        ] == [("Registry", "registry.parquet"), ("Parse", "tables/parses.parquet")]
        registry = pq.read_table(tmp_path / "registry.parquet")
        assert [
            (row["id"], row["kind"], row["location"], row["start"], row["duration"])
            for row in _event_rows(registry)
        ] == [
            ("o1", "Onset", None, _number(value=0.5), None),
            ("o2", "Onset", None, _number(value=1.25), _number(value=0.25)),
        ]
        assert package_problems(path) == []
        opened = Package.from_file(path)
        read = opened.bundle("o.tts.yaml")
        assert _rows(read) == _rows(bundle)
        before = _files(tmp_path)
        opened.write_bundle(read, "o.tts.yaml")
        assert _files(tmp_path) == before

    def test_one_file_per_type_however_many_bundles(self, tmp_path: Path) -> None:
        score = ContinuousLogicalTimeline(length=Fraction(4), uid="e_score")
        score.add_events(
            [
                {"id": "e_n1", "event_type": "Note", "start": Fraction(0)},
                {"id": "e_b1", "event_type": "Beat", "start": Fraction(1, 4)},
            ]
        )
        e = AlignmentBundle(id="e")
        e.add_timeline(score)
        package = Package()
        package.write_bundle(e, "e.tts.yaml")
        package.write_bundle(_a(), "a.tts.yaml")
        package.write_bundle(_a(), "a.tts.yaml")
        package.write_bundle(_b(), "b.tts.yaml")

        path = package.to_file(tmp_path / "package.yaml")

        parts = {kind: _parts(tmp_path, kind) for kind in ("Box", "MatchClaim", "Note")}
        assert {kind: len(names) for kind, names in parts.items()} == dict.fromkeys(
            parts, 1
        )
        assert not (tmp_path / "tables" / "Beat").exists()
        assert not (tmp_path / "tables" / "Projection").exists()
        roots = ("a_score", "a_pcm", "a_seg", "b_score", "e_score")
        assert not (tmp_path / "registry").exists()
        registry = pq.read_table(tmp_path / "registry.parquet")
        assert sorted({row["parse_id"] for row in _event_rows(registry)}) == sorted(
            _tool(root) for root in roots
        )
        notes = pq.read_table(tmp_path / "tables" / "Note" / parts["Note"][0])
        # The bundles in the order staged, each in its registry's order; e's
        # Note holds nothing but its ID, so e writes no Note table.
        assert notes.column("id").to_pylist() == [
            *(
                ["a_n3", "a_n1", "a_n2", "a_o1"]
                if _tool("a_score") < _tool("a_pcm")
                else ["a_o1", "a_n3", "a_n1", "a_n2"]
            ),
            "b_n1",
        ]
        unstored = sorted(
            row["id"] for row in _event_rows(registry) if row["location"] is None
        )
        projection = next(
            row["id"] for row in _event_rows(registry) if row["kind"] == "Projection"
        )
        assert unstored == sorted(["a_b1", "b_b1", "e_b1", "e_n1", projection])
        assert package_problems(path) == []
        assert package.manifests == [
            "e.tts.yaml",
            "a.tts.yaml",
            "b.tts.yaml",
        ]

    def test_rewriting_one_bundle_keeps_the_other_s_rows(self, tmp_path: Path) -> None:
        path, a, b = _two_bundle_package(tmp_path)
        stated = timetoshare.loads(path.read_text(encoding="utf-8"))
        stated["manifests"][0].update(
            size=1, integrity={"algorithm": "SHA-256", "digest": "00"}
        )
        path.write_text(timetoshare.dumps(stated), encoding="utf-8")
        before = _files(tmp_path)
        # b's Beat holds nothing but its ID: b shares the Note file alone.
        shared = {name for name in before if name.startswith("tables/Note/")}

        def b_rows() -> list[dict[str, Any]]:
            registry = pq.read_table(tmp_path / "registry.parquet")
            return [r for r in _event_rows(registry) if r["timeline_id"] == "b_score"]

        b_before = b_rows()
        package = Package.from_file(path)
        changed = package.bundle("a.tts.yaml")
        changed.get_timeline("a_score").add_events(
            [{"id": "a_n4", "event_type": "Note", "start": Fraction(2)}]
        )

        package.write_bundle(changed, "a.tts.yaml")

        after = _files(tmp_path)
        pre_parsed = {
            name
            for name in before
            if name.startswith("tables/") and name != "tables/parses.parquet"
        }
        assert len(pre_parsed) == 3
        assert len(shared) == 1
        assert shared <= after.keys()
        assert (pre_parsed - shared).isdisjoint(after)
        assert sorted(_ids(pq.read_table(tmp_path / name))[0] for name in shared) == [
            "b_n1"
        ]
        for name in shared:
            assert pq.read_table(tmp_path / name).num_rows == 1
        assert (
            len(
                [
                    name
                    for name in after
                    if name.startswith("tables/") and name != "tables/parses.parquet"
                ]
            )
            == 4
        )
        assert [row["id"] for row in b_before] == ["b_n1", "b_b1"]
        assert b_rows() == b_before
        registry = pq.read_table(tmp_path / "registry.parquet")
        assert [
            row["id"]
            for row in _event_rows(registry)
            if row["timeline_id"] == "a_score" and row["kind"] != "Projection"
        ] == ["a_n1", "a_b1", "a_n2", "a_n4"]
        assert sorted(
            pq.read_table(tmp_path / "tables" / "parses.parquet")
            .column("id")
            .to_pylist()
        ) == sorted(_tool(root) for root in ("a_score", "a_pcm", "a_seg", "b_score"))
        assert package_problems(path) == []
        data = (tmp_path / "a.tts.yaml").read_bytes()
        assert timetoshare.loads(path.read_text(encoding="utf-8"))["manifests"][0] == {
            "url": "a.tts.yaml",
            "media_type": "application/yaml",
            "size": len(data),
            "integrity": {
                "algorithm": "SHA-256",
                "digest": hashlib.sha256(data).hexdigest(),
            },
        }
        reopened = Package.from_file(path)
        assert _rows(reopened.bundle("b.tts.yaml")) == _rows(b)
        rows = _rows(reopened.bundle("a.tts.yaml"))
        assert rows["a_score"]["a_n4"]["start"] == {
            "value": 2.0,
            "numerator": 2,
            "denominator": 1,
        }
        expected = _rows(a)
        del rows["a_score"]["a_n4"]
        assert rows == expected

    def test_a_shared_file_is_rewritten_its_other_rows_kept(
        self, tmp_path: Path
    ) -> None:
        copy = shutil.copytree(_SHARED, tmp_path / "shared")

        def rows_of(parse: str) -> list[dict[str, Any]]:
            registry = pq.read_table(copy / "registry.parquet")
            return [row for row in _event_rows(registry) if row["parse_id"] == parse]

        handel_rows = rows_of("parse_2")
        package = Package.from_file(copy / "package.yaml")
        bach = package.bundle("bach.tts.yaml")

        package.write_bundle(bach, "bach.tts.yaml")

        def ids(name: str) -> list[str]:
            return _ids(pq.read_table(copy / name))

        assert [row["id"] for row in rows_of("parse_1")] == [
            "projection_1",
            "projection_2",
            "projection_9",
            "projection_7",
        ]
        assert [row["id"] for row in rows_of("parse_3")] == [
            "note_1",
            "note_2",
            "note_4",
            "note_3",
        ]
        assert len(handel_rows) == 8
        assert rows_of("parse_2") == handel_rows
        # The Projections lie in the registry alone; the Note files keep
        # handel's rows in their order.
        assert not (copy / "tables" / "Projection").exists()
        assert ids("tables/Note/part_a.parquet") == ["note_5", "note_6"]
        assert ids("tables/Note/part_b.parquet") == ["note_7"]
        assert ids("tables/MatchClaim/part_a.parquet") == ["claim_3", "claim_4"]
        assert ids("tables/parses.parquet") == ["parse_1", "parse_2", "parse_3"]
        claims = pq.read_table(copy / "tables" / "MatchClaim" / "part_a.parquet")
        assert b"timetoshare:provenance" not in claims.schema.metadata
        assert (
            claims.column("provenance").to_pylist()
            == [
                {
                    "basis": None,
                    "beliefs": None,
                    "date": "2026-11-20",
                    "match": None,
                    "method": "imported",
                    "note": None,
                    "service": "https://example.org/aligner",
                    "stated_by": None,
                }
            ]
            * 2
        )
        for kind, rows in (("MatchClaim", 3), ("Note", 4)):
            (new,) = [n for n in _parts(copy, kind) if _PART.fullmatch(n)]
            assert pq.read_table(copy / "tables" / kind / new).num_rows == rows
        assert package_problems(copy / "package.yaml") == []
        _assert_expected_reads(Package.from_file(copy / "package.yaml"))

    def test_the_fixture_opened_as_a_package(self) -> None:
        package = Package.from_file(_SHARED / "package.yaml")

        assert package.manifests == ["bach.tts.yaml", "handel.tts.yaml"]
        assert [(table["type"], table["url"]) for table in package.tables] == [
            ("Registry", "registry.parquet"),
            ("Parse", "tables/parses.parquet"),
            ("MatchClaim", "tables/MatchClaim/"),
            ("Note", "tables/Note/"),
        ]
        _assert_expected_reads(package)

    def test_a_shared_registry_is_filtered(self, tmp_path: Path) -> None:
        copy = shutil.copytree(_SHARED, tmp_path / "shared")
        assert package_problems(copy / "package.yaml") == []

        _assert_expected_reads(Package.from_file(copy / "package.yaml"))
        # The manifest and the registry alone, no package listing them.
        alone = tmp_path / "alone"
        alone.mkdir()
        shutil.copy(copy / "bach.tts.yaml", alone / "bach.tts.yaml")
        shutil.copy(copy / "registry.parquet", alone / "registry.parquet")
        with pytest.raises(TableError) as refused:
            AlignmentBundle.from_file(alone / "bach.tts.yaml")
        assert refused.value.problem.code == "unknown_timeline"

    def test_a_projection_of_another_manifest_s_timeline_is_read(self) -> None:
        directory = package_fixture_path("acceptances", "timeline_in_two_manifests")
        bundle = Package.from_file(directory / "package.yaml").bundle("handel.tts.yaml")
        assert "clt_1" not in bundle._known_timelines()
        events = bundle._known_timelines()["dpt_1"].events.table
        assert events.select(["id", "event_type", "projection_of"]).to_pylist() == [
            {
                "id": f"projection_{number}",
                "event_type": "Projection",
                "projection_of": {"timeline_id": "clt_1", "event_id": f"note_{note}"},
            }
            for number, note in ((1, 1), (2, 2), (9, 4), (7, 3))
        ]

    def test_a_projection_of_a_timeline_no_manifest_holds_is_refused(
        self, tmp_path: Path
    ) -> None:
        source = package_fixture_path("acceptances", "timeline_in_two_manifests")
        copy = shutil.copytree(source, tmp_path / "package")
        _project_to(copy / "registry.parquet", "projection_1", "nowhere")
        for url in ("bach.tts.yaml", "handel.tts.yaml"):
            package = Package.from_file(copy / "package.yaml")
            with pytest.raises(TableError) as refused:
                package.bundle(url)
            assert (
                str(refused.value)
                == "no manifest of the package has the timeline 'nowhere'"
            )
            assert refused.value.problem.code == "unknown_timeline"
            assert refused.value.problem.column == ("projection_of", "timeline_id")

    def test_a_lone_manifest_refuses_a_projection_of_a_timeline_it_lacks(
        self, tmp_path: Path
    ) -> None:
        copy = shutil.copytree(
            fixture_path("tables.tts.yaml").parent, tmp_path / "lone"
        )
        AlignmentBundle.from_file(copy / "tables.tts.yaml")
        _project_to(copy / "registry.parquet", "projection_1", "nowhere")
        with pytest.raises(TableError) as refused:
            AlignmentBundle.from_file(copy / "tables.tts.yaml")
        assert str(refused.value) == "the manifest has no timeline 'nowhere'"
        assert refused.value.problem.code == "unknown_timeline"
        assert refused.value.problem.column == ("projection_of", "timeline_id")

    def test_an_id_held_twice_is_refused_before_any_file(self, tmp_path: Path) -> None:
        a = _a()
        package = Package(name="Pieces")
        package.write_bundle(a, "a.tts.yaml")
        package.to_file(tmp_path / "package.yaml")
        before = _files(tmp_path)
        note_file = "registry.parquet"
        (claim_file,) = _parts(tmp_path, "MatchClaim")
        (claim,) = a.get_match_claims()

        def c(*, note: str = "c_n1", pcm: str = "c_pcm", claim_id: str | None = None):
            score = ContinuousLogicalTimeline(length=Fraction(4), uid="c_score")
            score.add_events([{"id": note, "event_type": "Note", "start": Fraction(0)}])
            samples = DiscretePhysicalTimeline(length=44100, uid=pcm)
            samples.add_events([{"id": "c_o1", "event_type": "Note", "start": 0}])
            bundle = AlignmentBundle(id="c")
            bundle.add_timeline(score)
            bundle.add_timeline(samples)
            if claim_id is not None:
                bundle.add_match_claims(
                    [
                        MatchClaim.from_events(
                            {"id": note, "start": Fraction(0)},
                            score,
                            {"id": "c_o1", "start": 0},
                            samples,
                            id=claim_id,
                        )
                    ]
                )
            return bundle

        for bundle, message in (
            (
                c(note="a_n1"),
                "The event 'a_n1' of timeline 'c_score' of the manifest "
                "'c.tts.yaml' cannot join the package: its ID is held by the "
                f"event of timeline 'a_score' in '{note_file}'; nothing "
                "is renamed.",
            ),
            (
                c(pcm="a_pcm"),
                "timeline 'a_pcm' of the manifest 'c.tts.yaml' cannot join the "
                "package: its id 'a_pcm' is already held by a timeline of the "
                "manifest 'a.tts.yaml'",
            ),
            (
                c(claim_id=claim.id),
                f"The MatchClaim '{claim.id}' of timeline 'c_score' of the manifest "
                "'c.tts.yaml' cannot join the package: its ID is held by the "
                f"MatchClaim of timeline 'a_score' in 'tables/MatchClaim/{claim_file}'; "
                "nothing is renamed.",
            ),
        ):
            with pytest.raises(ValueError) as refused:
                package.write_bundle(bundle, "c.tts.yaml")
            assert str(refused.value) == message
            assert _files(tmp_path) == before
            assert package.manifests == ["a.tts.yaml"]

        fresh = Package()
        fresh.write_bundle(_a(), "a.tts.yaml")
        with pytest.raises(ValueError) as refused:
            fresh.write_bundle(c(note="a_n1"), "c.tts.yaml")
        assert str(refused.value) == (
            "The event 'a_n1' of timeline 'c_score' of the manifest "
            "'c.tts.yaml' cannot join the package: its ID is held by the "
            "event of timeline 'a_score' in the manifest 'a.tts.yaml'; nothing "
            "is renamed."
        )
        assert fresh.manifests == ["a.tts.yaml"]
        assert fresh.path is None

    def test_a_manifest_s_match_claims_are_kept(self, tmp_path: Path) -> None:
        path, a, _ = _two_bundle_package(tmp_path)
        before = _files(tmp_path)
        package = Package.from_file(path)
        read = package.bundle("a.tts.yaml")
        without = AlignmentBundle(id="a")
        for timeline in read.timelines.values():
            without.add_timeline(timeline)
        (claim,) = a.get_match_claims()

        with pytest.raises(ValueError) as refused:
            package.write_bundle(without, "a.tts.yaml")

        assert str(refused.value) == (
            f"The new MatchClaim table lacks the MatchClaim '{claim.id}' of the "
            "existing one: a MatchClaim is never removed; record the belief "
            "'false' in it instead (AlignmentBundle.add_match_claim_belief)."
        )
        assert _files(tmp_path) == before

    def test_a_refused_write_restores_the_package(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path, _, _ = _two_bundle_package(tmp_path / "opened")
        before = _files(tmp_path / "opened")
        package = Package.from_file(path)
        manifests, tables = package.manifests, package.tables
        changed = package.bundle("a.tts.yaml")
        changed.get_timeline("a_score").add_events(
            [{"id": "a_n4", "event_type": "Note", "start": Fraction(2)}]
        )
        problem = PackageProblem("duplicate_id", "a.tts.yaml", 0, (), "induced")
        written = package_module.package_problems

        with monkeypatch.context() as patch:
            patch.setattr(package_module, "package_problems", lambda path: [problem])
            with pytest.raises(ValueError) as refused:
                package.write_bundle(changed, "a.tts.yaml")
            assert str(refused.value) == (
                f"The package check refuses the package written at '{path}': "
                "duplicate_id in 'a.tts.yaml', row 0: induced; the package "
                "is restored as it was."
            )
            assert _files(tmp_path / "opened") == before
            assert sorted(
                p.relative_to(tmp_path / "opened").as_posix()
                for p in (tmp_path / "opened").rglob("*")
                if p.is_file()
            ) == sorted(before)
            assert (package.manifests, package.tables) == (manifests, tables)

            new = Package()
            new.write_bundle(_a(), "a.tts.yaml")
            new.write_bundle(_b(), "b.tts.yaml")
            (tmp_path / "new").mkdir()
            with pytest.raises(ValueError):
                new.to_file(tmp_path / "new" / "package.yaml")
            assert list((tmp_path / "new").iterdir()) == []
            assert new.path is None
        assert written is package_module.package_problems
        assert new.to_file(tmp_path / "new" / "package.yaml") == (
            tmp_path / "new" / "package.yaml"
        )
        assert package_problems(tmp_path / "new" / "package.yaml") == []

        # An opened package of another version (the reader refuses one
        # today, so the document read is changed in memory).
        package._document["timetoshare"] = "0.1.0"
        with pytest.raises(ValueError) as refused:
            package.write_bundle(changed, "a.tts.yaml")
        assert str(refused.value) == (
            f"The package at '{path}' is of the format version '0.1.0', and the "
            "library writes '0.2.0': a package holds files of one version; "
            "nothing is written."
        )
        assert _files(tmp_path / "opened") == before

    @pytest.mark.parametrize(
        "url",
        [
            "../a.yaml",
            "tables/a.yaml",
            "package.yaml",
            "a/",
            "https://example.org/a.yaml",
        ],
    )
    def test_a_manifest_lies_in_the_package(self, url: str) -> None:
        with pytest.raises(ValueError) as refused:
            Package().write_bundle(_b(), url)

        assert str(refused.value) == (
            "A manifest of a package lies at a file path relative to its root, "
            f"outside tables/, not at {url!r}."
        )

    def test_a_column_two_manifests_hold_in_two_widths_is_widened(
        self, tmp_path: Path
    ) -> None:
        package = Package()
        package.write_bundle(_bundle_of(_velocity("s1", 100, pa.int8())), "v1.tts.yaml")
        package.write_bundle(
            _bundle_of(_velocity("s2", 300, pa.int64())), "v2.tts.yaml"
        )

        path = package.to_file(tmp_path / "package.yaml")

        (part,) = _parts(tmp_path, "Note")
        notes = pq.read_table(tmp_path / "tables" / "Note" / part)
        assert notes.schema.field("velocity").type == pa.int64()
        assert _velocities(notes) == {"s1_n": 100, "s2_n": 300}
        assert package_problems(path) == []

    def test_staging_many_bundles_checks_each_against_them_all(self) -> None:
        def staged(index: int, note: str | None = None) -> AlignmentBundle:
            score = ContinuousLogicalTimeline(length=Fraction(4), uid=f"s{index}_score")
            score.add_events(
                [
                    {
                        "id": note or f"s{index}_n1",
                        "event_type": "Note",
                        "start": Fraction(0),
                    }
                ]
            )
            bundle = AlignmentBundle(id=f"s{index}")
            bundle.add_timeline(score)
            return bundle

        def refusal(held: int) -> str:
            return (
                f"The event 's{held}_n1' of timeline 's6_score' of the manifest "
                "'s6.tts.yaml' cannot join the package: its ID is held by the "
                f"event of timeline 's{held}_score' in the manifest "
                f"'s{held}.tts.yaml'; nothing is renamed."
            )

        package = Package()
        for index in range(6):
            package.write_bundle(staged(index), f"s{index}.tts.yaml")
        for held in (0, 5):
            with pytest.raises(ValueError) as refused:
                package.write_bundle(staged(6, f"s{held}_n1"), "s6.tts.yaml")
            assert str(refused.value) == refusal(held)

        package.write_bundle(staged(0), "s0.tts.yaml")

        with pytest.raises(ValueError) as refused:
            package.write_bundle(staged(6, "s0_n1"), "s6.tts.yaml")
        assert str(refused.value) == refusal(0)
        assert package.manifests == [f"s{index}.tts.yaml" for index in range(6)]

    def test_where_a_package_is_written(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError) as refused:
            Package().to_file(tmp_path / "pieces.yaml")
        assert str(refused.value) == (
            "A package's file is named 'package.yaml', not 'pieces.yaml'."
        )
        path = Package(name="Pieces").to_file(tmp_path / "package.yaml")
        with pytest.raises(ValueError) as refused:
            Package().to_file(path)
        assert str(refused.value) == (
            f"'{path}' exists already: a new package is written into a folder "
            "holding none; open that one with Package.from_file."
        )
        elsewhere = tmp_path / "other" / "package.yaml"
        with pytest.raises(ValueError) as refused:
            Package.from_file(path).to_file(elsewhere)
        assert str(refused.value) == (
            f"The package lies at '{path}' and is written there, not at "
            f"'{elsewhere}': its manifests and tables lie beside it."
        )
        with pytest.raises(KeyError) as missing:
            Package.from_file(path).bundle("a.tts.yaml")
        assert missing.value.args[0] == (
            "The package lists no manifest 'a.tts.yaml'. Available " "manifests: []"
        )

    def test_a_table_of_other_rows_is_a_problem_of_the_package(
        self, tmp_path: Path
    ) -> None:
        read = _carrying(tmp_path / "source", _b(), [1, 2])
        package = Package()
        package.write_bundle(read, "b.tts.yaml")

        for folder in ("package", "elsewhere"):
            path = tmp_path / folder / "package.yaml"
            with pytest.raises(ValueError) as refused:
                package.to_file(path)

            assert str(refused.value) == (
                f"The package check refuses the package written at '{path}': "
                "foreign_table in 'tables/Features/': the table "
                "'tables/Features/' of type 'Features' has no 'id' "
                "column: a package's tables hold the registry, the Parse "
                "table, MatchClaim tables and pre-parsed tables of events keyed by "
                "'id'; the package is restored as it was."
            )
            assert _files(tmp_path / folder) == {}
            assert package.path is None
            assert package.manifests == ["b.tts.yaml"]

    def test_a_manifest_alone_carries_a_table_of_other_rows(
        self, tmp_path: Path
    ) -> None:
        read = _carrying(tmp_path / "source", _b(), [1, 2])
        source = tmp_path / "source" / "features.parquet"
        data = source.read_bytes()

        read.to_file(tmp_path / "source" / "b.tts.yaml")

        assert source.read_bytes() == data
        again = AlignmentBundle.from_file(tmp_path / "source" / "b.tts.yaml")
        (carried,) = [r for r in again.resources if r.id == "features"]
        assert (carried.type, carried.url) == ("Features", "features.parquet")
        assert list(again._carried_tables) == ["features"]

    def test_a_folder_without_parquet_files_is_copied_and_listed(
        self, tmp_path: Path
    ) -> None:
        read = _carrying_scans(tmp_path / "source", _b(), b"\x89PNG")
        package = Package()
        package.write_bundle(read, "b.tts.yaml")

        path = package.to_file(tmp_path / "package" / "package.yaml")

        assert (
            tmp_path / "package" / "tables" / "Scan" / "scans" / "page1.png"
        ).read_bytes() == b"\x89PNG"
        assert [
            (table["type"], table["url"])
            for table in timetoshare.loads(path.read_text(encoding="utf-8"))["tables"]
        ] == [
            ("Registry", "registry.parquet"),
            ("Parse", "tables/parses.parquet"),
            ("Note", "tables/Note/"),
            ("Scan", "tables/Scan/"),
        ]
        manifest = timetoshare.loads(
            (tmp_path / "package" / "b.tts.yaml").read_text(encoding="utf-8")
        )
        assert [r for r in manifest["resources"] if r["id"] == "#scans"] == [
            {
                "id": "#scans",
                "type": "Scan",
                "timelines": ["#b_score"],
                "url": "tables/Scan/",
            }
        ]
        assert package_problems(path) == []
        again = Package.from_file(path).bundle("b.tts.yaml")
        assert again._carried_tables == {
            "scans": Path(os.path.abspath(tmp_path / "package" / "tables" / "Scan"))
        }
        assert _rows(again) == _rows(_b())

        # Written back through the opened package, the copy stays.
        before = _files(tmp_path / "package")
        opened = Package.from_file(path)
        opened.write_bundle(opened.bundle("b.tts.yaml"), "b.tts.yaml")
        after = _files(tmp_path / "package")
        assert [name for name in after if "Scan" in name] == [
            "tables/Scan/scans/page1.png"
        ]
        assert len(after) == len(before)
        assert [table["url"] for table in opened.tables].count("tables/Scan/") == 1
        assert package_problems(path) == []

    def test_a_carried_folder_s_name_taken(self, tmp_path: Path) -> None:
        score = ContinuousLogicalTimeline(length=Fraction(4), uid="e_score")
        score.add_events([{"id": "e_n1", "event_type": "Note", "start": Fraction(0)}])
        e = AlignmentBundle(id="e")
        e.add_timeline(score)
        f = AlignmentBundle(id="f")
        f.add_timeline(ContinuousLogicalTimeline(length=Fraction(4), uid="f_score"))
        package = Package()
        package.write_bundle(
            _carrying_scans(tmp_path / "b", _b(), b"\x89PNG b"), "b.tts.yaml"
        )
        package.write_bundle(
            _carrying_scans(tmp_path / "e", e, b"\x89PNG e"), "e.tts.yaml"
        )
        package.write_bundle(
            _carrying_scans(tmp_path / "f", f, b"\x89PNG b"), "f.tts.yaml"
        )

        path = package.to_file(tmp_path / "package" / "package.yaml")

        root = tmp_path / "package"
        assert _parts(root, "Scan") == ["scans", _MINTED_SCANS]
        assert (root / "tables/Scan/scans/page1.png").read_bytes() == b"\x89PNG b"
        assert (root / "tables/Scan" / _MINTED_SCANS / "page1.png").read_bytes() == (
            b"\x89PNG e"
        )
        resources = {
            url: [
                (r["url"], r["timelines"])
                for r in timetoshare.loads((root / url).read_text(encoding="utf-8"))[
                    "resources"
                ]
                if r["id"] == "#scans"
            ]
            for url in package.manifests
        }
        assert resources == {
            "b.tts.yaml": [("tables/Scan/", ["#b_score"])],
            "e.tts.yaml": [("tables/Scan/", ["#e_score"])],
            "f.tts.yaml": [("tables/Scan/", ["#f_score"])],
        }
        assert [
            (t["type"], t["url"]) for t in package.tables if t["type"] == "Scan"
        ] == [("Scan", "tables/Scan/")]
        assert package_problems(path) == []

    def test_two_carried_parquet_tables_are_parts_of_one_entry(
        self, tmp_path: Path
    ) -> None:
        a_beats = _carrying_beats(tmp_path / "a", _a(), "a_b1", "one")
        b_beats = _carrying_beats(tmp_path / "b", _b(), "b_b1", "two")
        package = Package()
        package.write_bundle(a_beats, "a.tts.yaml")
        package.write_bundle(b_beats, "b.tts.yaml")

        path = package.to_file(tmp_path / "package" / "package.yaml")

        root = tmp_path / "package"
        assert [t for t in package.tables if t["type"] == "Beat"] == [
            {"type": "Beat", "url": "tables/Beat/"}
        ]
        assert _parts(root, "Beat") == ["beats.parquet", _MINTED_BEATS]
        assert (root / "tables/Beat/beats.parquet").read_bytes() == (
            tmp_path / "a" / "beats.parquet"
        ).read_bytes()
        assert (root / "tables/Beat" / _MINTED_BEATS).read_bytes() == (
            tmp_path / "b" / "beats.parquet"
        ).read_bytes()
        for url, timeline in (
            ("a.tts.yaml", "a_score"),
            ("b.tts.yaml", "b_score"),
        ):
            (resource,) = [
                r
                for r in timetoshare.loads((root / url).read_text(encoding="utf-8"))[
                    "resources"
                ]
                if r["id"] == "#beats"
            ]
            assert resource == {
                "id": "#beats",
                "type": "Beat",
                "timelines": [f"#{timeline}"],
                "url": "tables/Beat/",
            }
        assert _own_rows(root, "a.tts.yaml") == [
            {"id": "a_b1", "timeline_id": "a_score", "label": "one"}
        ]
        assert _own_rows(root, "b.tts.yaml") == [
            {"id": "b_b1", "timeline_id": "b_score", "label": "two"}
        ]
        assert package_problems(path) == []
        opened = Package.from_file(path)
        for url, written in (
            ("a.tts.yaml", a_beats),
            ("b.tts.yaml", b_beats),
        ):
            again = opened.bundle(url)
            assert again._carried_tables == {
                "beats": Path(os.path.abspath(root / "tables" / "Beat"))
            }
            assert _rows(again) == _rows(written)

    def test_a_type_held_as_a_file_moves_into_its_folder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        package = Package()
        package.write_bundle(
            _carrying_beats(tmp_path / "a", _a(), "a_b1", "one"), "a.tts.yaml"
        )
        root = tmp_path / "package"
        path = package.to_file(root / "package.yaml")
        # The type listed as one file, beats.parquet at the package's root, by hand.
        (root / "tables/Beat/beats.parquet").rename(root / "beats.parquet")
        (root / "tables/Beat").rmdir()
        document = timetoshare.loads(path.read_text(encoding="utf-8"))
        index = document["tables"].index({"type": "Beat", "url": "tables/Beat/"})
        document["tables"][index] = {"type": "Beat", "url": "beats.parquet"}
        path.write_text(timetoshare.dumps(document), encoding="utf-8")
        manifest_path = root / "a.tts.yaml"
        manifest = timetoshare.loads(manifest_path.read_text(encoding="utf-8"))
        (beats,) = [r for r in manifest["resources"] if r["id"] == "#beats"]
        beats["url"] = "beats.parquet"
        manifest_path.write_text(timetoshare.dumps(manifest), encoding="utf-8")
        assert package_problems(path) == []
        before = _files(root)
        opened = Package.from_file(path)
        tables = opened.tables
        b_beats = _carrying_beats(tmp_path / "b", _b(), "b_b1", "two")

        problem = PackageProblem("duplicate_id", "a.tts.yaml", 0, (), "induced")
        with monkeypatch.context() as patch:
            patch.setattr(package_module, "package_problems", lambda path: [problem])
            with pytest.raises(ValueError):
                opened.write_bundle(b_beats, "b.tts.yaml")
        assert _files(root) == before
        assert opened.tables == tables

        opened.write_bundle(b_beats, "b.tts.yaml")

        assert not (root / "beats.parquet").exists()
        assert _parts(root, "Beat") == ["beats.parquet", _MINTED_BEATS]
        assert (root / "tables/Beat/beats.parquet").read_bytes() == before[
            "beats.parquet"
        ]
        assert [t for t in opened.tables if t["type"] == "Beat"] == [
            {"type": "Beat", "url": "tables/Beat/"}
        ]
        assert opened.tables[index] == {"type": "Beat", "url": "tables/Beat/"}
        rewritten = timetoshare.loads(manifest_path.read_text(encoding="utf-8"))
        (beats,) = [r for r in rewritten["resources"] if r["id"] == "#beats"]
        assert beats == {
            "id": "#beats",
            "type": "Beat",
            "timelines": ["#a_score"],
            "url": "tables/Beat/",
        }
        assert [r for r in rewritten["resources"] if r["id"] != "#beats"] == [
            r for r in manifest["resources"] if r["id"] != "#beats"
        ]
        assert _own_rows(root, "a.tts.yaml") == [
            {"id": "a_b1", "timeline_id": "a_score", "label": "one"}
        ]
        assert _own_rows(root, "b.tts.yaml") == [
            {"id": "b_b1", "timeline_id": "b_score", "label": "two"}
        ]
        assert package_problems(path) == []

    def test_a_table_the_writer_has_no_file_of_is_refused(self, tmp_path: Path) -> None:
        bundle = _b()
        bundle.resources.append(
            Resource("features", "Features", ("b_score",), "features.parquet")
        )
        package = Package()
        with pytest.raises(ValueError) as refused:
            package.write_bundle(bundle, "b.tts.yaml")
        assert str(refused.value) == (
            "The resource 'features' of the bundle 'b' names the table "
            "'features.parquet' of the type Features, of which the bundle holds no "
            "rows, and the bundle was not read from it (built in code, or its file "
            "missing when it was read): the package writer copies such a table "
            "into the package from the file the bundle was read with and has none "
            "to copy; nothing is written."
        )
        assert package.manifests == []

        read = _carrying(tmp_path / "source", _b(), [1, 2])
        source = tmp_path / "source" / "features.parquet"
        source.unlink()
        with pytest.raises(ValueError) as refused:
            package.write_bundle(read, "b.tts.yaml")
        assert str(refused.value) == (
            "The resource 'features' of the bundle 'b' names the table "
            f"'features.parquet' of the type Features, which the bundle carries "
            f"from '{source}', where no file lies any more: the package writer has "
            "nothing to copy into the package; nothing is written."
        )
        assert (package.manifests, package.path) == ([], None)

    def test_a_write_cut_short_leaves_no_hidden_file(self, tmp_path: Path) -> None:
        package = Package()
        package.write_bundle(_a(), "a.tts.yaml")
        path = package.to_file(tmp_path / "package.yaml")
        before = _files(tmp_path)
        hidden = [
            tmp_path / "tables" / "Note" / ".part_0000000000001.parquet",
            tmp_path / "tables" / "Chord" / ".part_0000000000002.parquet",
            tmp_path / ".registry.parquet",
            tmp_path / "..registry.parquet",
        ]
        for stray in hidden:
            stray.parent.mkdir(parents=True, exist_ok=True)
            stray.write_bytes(b"cut short")

        Package.from_file(path)

        assert _files(tmp_path) == before
        assert package_problems(path) == []

    def test_a_bundle_of_the_package_is_written_through_it(
        self, tmp_path: Path
    ) -> None:
        path, _, written = _two_bundle_package(tmp_path / "package")
        read = Package.from_file(path).bundle("b.tts.yaml")
        merged = AlignmentBundle.from_bundles([read], id="merged")
        (tmp_path / "alone").mkdir()
        before = _files(tmp_path)

        for bundle in (read, written, merged):
            with pytest.raises(NotImplementedError) as refused:
                bundle.to_file(
                    tmp_path / "alone" / f"{bundle.id}.tts.yaml", events=True
                )

            (first,) = bundle.resources
            assert str(refused.value) == (
                f"The resource '{first.id}' names the package table "
                f"'registry.parquet' for its {first.type} rows: "
                "AlignmentBundle.to_file writes a manifest's tables as single files "
                "beside it and does not write into a package's tables, which other "
                "manifests may share; write the bundle into its package with "
                "Package.write_bundle."
            )
            assert _files(tmp_path) == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ["alone", "package"]


def _project_to(registry: Path, projection_id: str, timeline_id: str) -> None:
    """Rewrite the registry file *registry* with the Projection *projection_id*
    standing for an event of *timeline_id*, its file metadata kept."""
    table = pq.read_table(registry)
    index = table.schema.get_field_index("projection_of")
    field = table.schema.field(index)
    projected = table.column(index).to_pylist()
    row = table.column("id").to_pylist().index(projection_id)
    projected[row] = {**projected[row], "timeline_id": timeline_id}
    pq.write_table(
        table.set_column(index, field, pa.array(projected, field.type)), registry
    )


def _assert_expected_reads(package: Package) -> None:
    """Each manifest of the shared fixture's package, read through *package*,
    holds the MatchClaims, Notes and Projections ``expected.yaml`` states."""
    for url, expected in _EXPECTED.items():
        bundle = package.bundle(url)
        assert sorted(_ids(bundle.match_claim_table.table)) == (expected["MatchClaim"])
        for kind in ("Note", "Projection"):
            held = sorted(
                row["id"]
                for timeline in bundle._known_timelines().values()
                for row in timeline.events.table.to_pylist()
                if row["event_type"] == kind
            )
            assert held == expected[kind]


class TestResolve:
    """Values in tests/alignment/README.md, "Resolving an ID"."""

    def test_every_resolution_of_the_fixture(self) -> None:
        cases = timetoshare.loads(
            (_SHARED / "expected.yaml").read_text(encoding="utf-8")
        )["resolutions"]
        samples = TimeUnit.samples
        expected: dict[tuple[str, str], list[tuple[Any, ...]]] = {
            ("bach.tts.yaml", "dpt_1"): [("DiscretePhysicalTimeline", "dpt_1")],
            ("bach.tts.yaml", "projection_1"): [
                (
                    "Projection",
                    "projection_1",
                    Coordinate(0, samples),
                    None,
                    EventReference(timeline_id="clt_1", event_id="note_1"),
                )
            ],
            ("bach.tts.yaml", "projection_3"): [],
            ("bach.tts.yaml", "claim_1"): [("MatchClaim", "claim_1")],
            ("bach.tts.yaml", "parse_1"): [
                (
                    "Parse",
                    "parse_1",
                    (_SHARED.resolve() / "bach" / "alignment.csv").as_posix(),
                )
            ],
            ("bach.tts.yaml", "parse_2"): [],
            ("handel.tts.yaml", "projection_5"): [_PROJECTION_5],
            (".", "resource_1"): [
                ("Resource", "resource_1", ("clt_1", "dpt_1")),
                ("Resource", "resource_1", ("clt_2", "dpt_2")),
            ],
            (".", "projection_5"): [_PROJECTION_5],
            (".", "ghost"): [],
        }
        assert [(case["path"], case["id"]) for case in cases] == list(expected)

        for case in cases:
            found = resolve(_SHARED / case["path"], case["id"])
            assert len(found) == len(case["found"])
            assert [_described(item) for item in found] == expected[
                (case["path"], case["id"])
            ]

    def test_a_lone_manifest_a_path_or_a_file_url(self, tmp_path: Path) -> None:
        bundle = _a()
        path = bundle.to_file(tmp_path / "a.yaml", events=True)
        (claim,) = bundle.get_match_claims()

        (event,) = resolve(path, "a_n3")
        assert (type(event), event.id, event.start) == (
            Event,
            "a_n3",
            Coordinate(Fraction(17, 2), TimeUnit.quarters),
        )
        assert resolve(path.as_uri(), "a_n3") == [event]
        (timeline,) = resolve(str(path), "#a_score")
        assert timeline.id == "a_score"
        assert [found.id for found in resolve(path, claim.id)] == [claim.id]
        with pytest.raises(ValueError) as refused:
            resolve("https://example.org/a.yaml", "a_n3")
        assert str(refused.value) == (
            "'https://example.org/a.yaml' is a https: URL; resolve reads a local "
            "manifest or package root, a path or a file: URL"
        )


#: ``projection_5`` on handel's child ``clt_3``, at 9/2 on its root's axis.
_PROJECTION_5 = (
    "Projection",
    "projection_5",
    Coordinate(Fraction(9, 2), TimeUnit.quarters),
    Fraction(1, 2),
    EventReference(timeline_id="dpt_2", event_id="onset_1"),
)


def _described(item: Any) -> tuple[Any, ...]:
    """What a test compares of an object ``resolve`` answers with."""
    if isinstance(item, Projection):
        return (
            "Projection",
            item.id,
            item.start,
            None if item.duration is None else item.duration.value,
            item.projection_of,
        )
    if isinstance(item, Parse):
        return ("Parse", item.id, item.source)
    if isinstance(item, Resource):
        return ("Resource", item.id, item.timelines)
    if isinstance(item, MatchClaim):
        return ("MatchClaim", item.id)
    return (type(item).__name__, item.id)
