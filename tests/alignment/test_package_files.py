"""Event files beside a manifest, and packages of manifests sharing their tables.

Values in tests/alignment/README.md, section "Event Files and Packages".
"""

from __future__ import annotations

import hashlib
import re
import shutil
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
    package_fixture_path,
    package_problems,
)

from timetoalign.alignment import AlignmentBundle, MatchClaim, Package
from timetoalign.alignment import package as package_module
from timetoalign.core import BoundingBox, Box, Projection, Provenance
from timetoalign.timelines import Timeline
from timetoalign.timelines.manifest import (
    PARQUET_MEDIA_TYPE,
    Resource,
    event_table,
    event_table_schema,
    read_event_table,
    write_event_table,
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
    "bach/manifest.yaml": {
        "MatchClaim": ["claim_1", "claim_2", "claim_5"],
        "Projection": ["projection_1", "projection_2", "projection_7", "projection_9"],
    },
    "handel/manifest.yaml": {
        "MatchClaim": ["claim_3", "claim_4"],
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
    """Every timeline's event rows of *bundle*, by timeline and event ID."""
    return {
        timeline_id: {row["id"]: row for row in timeline.events.table.to_pylist()}
        for timeline_id, timeline in sorted(bundle._known_timelines().items())
    }


def _files(directory: Path) -> dict[str, bytes]:
    """Every file under *directory* by its path from it, with its bytes."""
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _parts(directory: Path, kind: str) -> list[str]:
    """The names of the files of a package's folder of *kind*."""
    return sorted(path.name for path in (directory / "tables" / kind).iterdir())


class TestEventFiles:
    def test_every_event_type_is_written_the_manifest_naming_none(
        self, tmp_path: Path
    ) -> None:
        bundle = _a()

        path = bundle.to_file(tmp_path / "a.yaml", events=True)

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "Beat.parquet",
            "Note.parquet",
            "a.yaml",
            "boxes.parquet",
            "match_claims.parquet",
            "projections.parquet",
        ]
        manifest = timetoshare.loads(path.read_text(encoding="utf-8"))
        assert "events" not in str(manifest)
        assert sorted((r.type, r.timelines, r.url) for r in bundle.resources) == [
            ("Beat", ("a_score",), "Beat.parquet"),
            ("Box", ("a_seg",), "boxes.parquet"),
            ("MatchClaim", ("a_pcm", "a_score"), "match_claims.parquet"),
            ("Note", ("a_bar", "a_pcm", "a_score"), "Note.parquet"),
            ("Projection", ("a_score",), "projections.parquet"),
        ]
        notes = pq.read_table(tmp_path / "Note.parquet")
        assert notes.column_names == [
            "id",
            "timeline_id",
            "unit",
            "number_type",
            "start",
            "duration",
            "midi",
            "name",
            "staff",
        ]
        fraction = {"unit": "quarters", "number_type": "fraction"}
        assert notes.to_pylist() == [
            {
                "id": "a_n3",
                "timeline_id": "a_bar",
                **fraction,
                "start": _number(1, 2),
                "duration": None,
                "midi": 64,
                "name": None,
                "staff": None,
            },
            {
                "id": "a_o1",
                "timeline_id": "a_pcm",
                "unit": "samples",
                "number_type": "int",
                "start": _number(22050),
                "duration": None,
                "midi": None,
                "name": None,
                "staff": None,
            },
            {
                "id": "a_n1",
                "timeline_id": "a_score",
                **fraction,
                "start": _number(0, 1),
                "duration": _number(1, 1),
                "midi": 60,
                "name": "C4",
                "staff": None,
            },
            {
                "id": "a_n2",
                "timeline_id": "a_score",
                **fraction,
                "start": _number(3, 2),
                "duration": None,
                "midi": 62,
                "name": None,
                "staff": 1,
            },
        ]

    def test_the_events_read_back_the_same(self, tmp_path: Path) -> None:
        bundle = _a()
        (tmp_path / "first").mkdir()
        (tmp_path / "second").mkdir()
        bundle.to_file(tmp_path / "first" / "a.yaml", events=True)

        read = AlignmentBundle.from_file(tmp_path / "first" / "a.yaml")

        assert _rows(read) == _rows(bundle)
        assert read.match_claim_table == bundle.match_claim_table
        read.to_file(tmp_path / "second" / "a.yaml")
        assert _files(tmp_path / "second") == _files(tmp_path / "first")

    def test_without_events_the_tables_alone_are_written(self, tmp_path: Path) -> None:
        _a().to_file(tmp_path / "a.yaml")

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "a.yaml",
            "boxes.parquet",
            "match_claims.parquet",
            "projections.parquet",
        ]

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

        table = event_table([ticks, fractions, floats, big], "Beat")

        assert table.column("id").to_pylist() == [
            "m",
            "n",
            "p",
            "q",
            "d",
            "b",
            "a",
            "c",
            "y",
            "x",
            "z",
        ]
        assert table.column("start").to_pylist()[5] == _number(6004799503160661, 2**54)

    def test_a_further_coordinate_cell_is_written_once(self) -> None:
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

        table = event_table([score], "Note")

        assert table.column("alt").to_pylist() == [
            {**_number(value=0.5), "unit": "seconds", "number_type": "float"},
            {**_number(3, 4), "unit": "quarters", "number_type": "fraction"},
        ]
        assert read_event_table(table)["t"].column("alt").to_pylist() == [
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

        # The inline Projection of other content under the file row's ID.
        (row,) = written["events"]
        written["events"] = [{**row, "start": [7, 2], "duration": [1, 2]}]
        (tmp_path / "a.yaml").write_text(timetoshare.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError) as refused:
            AlignmentBundle.from_file(tmp_path / "a.yaml")
        assert str(refused.value) == (
            f"Event id '{row['id']}' is already held by another event of timeline "
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

        assert pq.read_table(tmp_path / "Segment.parquet").column("id").to_pylist() == [
            "s1"
        ]
        read = AlignmentBundle.from_file(tmp_path / "seg.yaml")
        timeline = read.get_timeline("seg_t")
        assert sorted(_ids(timeline.events.table)) == [
            "n1",
            "s1",
            "seg_c",
        ]
        assert timeline.get_child("seg_c").id == "seg_c"
        assert _rows(read) == _rows(bundle)
        inline = Timeline.from_dict(line.to_dict(events=True))
        assert sorted(_ids(inline.events.table)) == [
            "n1",
            "s1",
            "seg_c",
        ]

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
        table = event_table(_a()._known_timelines().values(), "Note")

        write_event_table(table, tmp_path / "one.parquet")
        write_event_table(table, tmp_path / "two.parquet")

        assert (tmp_path / "one.parquet").read_bytes() == (
            tmp_path / "two.parquet"
        ).read_bytes()
        file = pq.ParquetFile(tmp_path / "one.parquet")
        assert file.schema_arrow.metadata == {
            b"timetoshare:type": b"Note",
            b"timetoshare:version": b"0.2.0",
        }
        assert file.metadata.num_row_groups == 1
        group = file.metadata.row_group(0)
        assert group.sorting_columns == (pq.SortingColumn(1),)
        chunks = [group.column(index) for index in range(group.num_columns)]
        assert {chunk.compression for chunk in chunks} == {"ZSTD"}
        assert all(chunk.is_stats_set for chunk in chunks)
        assert all(chunk.has_offset_index for chunk in chunks)
        assert all(chunk.has_column_index for chunk in chunks)

    def test_row_groups_hold_at_most_65536_rows(self, tmp_path: Path) -> None:
        n = 65_537
        schema = event_table_schema("Beat")
        table = pa.table(
            {
                "id": pa.array([f"beat_{index}" for index in range(n)]),
                "timeline_id": pa.repeat(pa.scalar("t"), n),
                "unit": pa.repeat(pa.scalar("ticks"), n),
                "number_type": pa.repeat(pa.scalar("int"), n),
                "start": pa.StructArray.from_arrays(
                    [
                        pa.array(range(n), pa.int64()),
                        pa.nulls(n, pa.int64()),
                        pa.nulls(n, pa.float64()),
                    ],
                    fields=list(schema.field("start").type),
                ),
                "duration": pa.nulls(n, schema.field("duration").type),
            },
            schema=schema,
        )

        write_event_table(table, tmp_path / "beats.parquet")

        metadata = pq.ParquetFile(tmp_path / "beats.parquet").metadata
        assert [
            metadata.row_group(index).num_rows
            for index in range(metadata.num_row_groups)
        ] == [65_536, 1]

    def test_a_table_without_its_type_is_refused(self, tmp_path: Path) -> None:
        table = event_table_schema("Beat").empty_table().replace_schema_metadata(None)

        with pytest.raises(ValueError) as refused:
            write_event_table(table, tmp_path / "beats.parquet")

        assert str(refused.value) == (
            "A package table names its type in its metadata ('timetoshare:type'), "
            "as event_table writes it."
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


def _two_bundle_package(
    directory: Path,
) -> tuple[Path, AlignmentBundle, AlignmentBundle]:
    """``a`` and ``b`` written into a new package in *directory*."""
    a, b = _a(), _b()
    package = Package(id="https://example.org/pieces", name="Pieces")
    package.write_bundle(a, "a/manifest.yaml")
    package.write_bundle(b, "b/manifest.yaml")
    return package.to_file(directory / "package.yaml"), a, b


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
    bundle.to_file(directory / "manifest.yaml", events=True)
    read = AlignmentBundle.from_file(directory / "manifest.yaml")
    assert list(read._carried_tables) == ["features"]
    return read


class TestPackage:
    def test_two_bundles_in_one_package(self, tmp_path: Path) -> None:
        path, a, b = _two_bundle_package(tmp_path)

        files = _files(tmp_path)
        assert [name for name in files if not name.startswith("tables/")] == [
            "a/manifest.yaml",
            "b/manifest.yaml",
            "package.yaml",
        ]
        parts = {
            kind: _parts(tmp_path, kind)
            for kind in ("Beat", "Box", "MatchClaim", "Note", "Projection")
        }
        assert {kind: len(names) for kind, names in parts.items()} == {
            "Beat": 1,
            "Box": 1,
            "MatchClaim": 1,
            "Note": 1,
            "Projection": 1,
        }
        assert all(_PART.fullmatch(name) for names in parts.values() for name in names)
        assert len(files) == 3 + 5
        assert package_problems(path) == []
        tables = [
            {"media_type": PARQUET_MEDIA_TYPE, "type": kind, "url": f"tables/{kind}/"}
            for kind in ("Projection", "Box", "Beat", "Note", "MatchClaim")
        ]
        assert timetoshare.loads(path.read_text(encoding="utf-8")) == {
            "timetoshare": "0.2.0",
            "id": "https://example.org/pieces",
            "name": "Pieces",
            "manifests": [
                {"media_type": "application/yaml", "url": "a/manifest.yaml"},
                {"media_type": "application/yaml", "url": "b/manifest.yaml"},
            ],
            "tables": tables,
        }
        opened = Package.from_file(path)
        assert opened.manifests == ["a/manifest.yaml", "b/manifest.yaml"]
        assert opened.tables == tables
        assert (opened.id, opened.name, opened.path) == (
            "https://example.org/pieces",
            "Pieces",
            path,
        )
        for bundle, url, expected in (
            (
                a,
                "a/manifest.yaml",
                [
                    ("Beat", ("a_score",)),
                    ("Box", ("a_seg",)),
                    ("MatchClaim", ("a_pcm", "a_score")),
                    ("Note", ("a_bar", "a_pcm", "a_score")),
                    ("Projection", ("a_score",)),
                ],
            ),
            (b, "b/manifest.yaml", [("Beat", ("b_score",)), ("Note", ("b_score",))]),
        ):
            assert sorted(
                (r.type, r.timelines, r.url, r.media_type, r.size, r.integrity)
                for r in bundle.resources
            ) == [
                (kind, timelines, f"../tables/{kind}/", PARQUET_MEDIA_TYPE, None, None)
                for kind, timelines in expected
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
            if name.startswith("tables/"):
                metadata = pq.read_schema(tmp_path / name).metadata
                assert metadata[b"timetoshare:type"] == name.split("/")[1].encode()
                assert metadata[b"timetoshare:version"] == b"0.2.0"
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
        package.write_bundle(e, "e/manifest.yaml")
        package.write_bundle(_a(), "a/manifest.yaml")
        package.write_bundle(_a(), "a/manifest.yaml")
        package.write_bundle(_b(), "b/manifest.yaml")

        path = package.to_file(tmp_path / "package.yaml")

        parts = {
            kind: _parts(tmp_path, kind)
            for kind in ("Beat", "Box", "MatchClaim", "Note", "Projection")
        }
        assert {kind: len(names) for kind, names in parts.items()} == dict.fromkeys(
            parts, 1
        )
        notes = pq.read_table(tmp_path / "tables" / "Note" / parts["Note"][0])
        assert list(
            zip(
                notes.column("id").to_pylist(),
                notes.column("timeline_id").to_pylist(),
                strict=True,
            )
        ) == [
            ("a_n3", "a_bar"),
            ("a_o1", "a_pcm"),
            ("a_n1", "a_score"),
            ("a_n2", "a_score"),
            ("b_n1", "b_score"),
            ("e_n1", "e_score"),
        ]
        beats = pq.read_table(tmp_path / "tables" / "Beat" / parts["Beat"][0])
        assert beats.column("id").to_pylist() == ["a_b1", "b_b1", "e_b1"]
        assert package_problems(path) == []
        assert package.manifests == [
            "e/manifest.yaml",
            "a/manifest.yaml",
            "b/manifest.yaml",
        ]

    def test_rewriting_one_bundle_keeps_the_other_s_rows(self, tmp_path: Path) -> None:
        path, a, b = _two_bundle_package(tmp_path)
        stated = timetoshare.loads(path.read_text(encoding="utf-8"))
        stated["manifests"][0].update(
            size=1, integrity={"algorithm": "SHA-256", "digest": "00"}
        )
        path.write_text(timetoshare.dumps(stated), encoding="utf-8")
        before = _files(tmp_path)
        shared = {
            name for name in before if name.startswith(("tables/Beat/", "tables/Note/"))
        }
        package = Package.from_file(path)
        changed = package.bundle("a/manifest.yaml")
        changed.get_timeline("a_score").add_events(
            [{"id": "a_n4", "event_type": "Note", "start": Fraction(2)}]
        )

        package.write_bundle(changed, "a/manifest.yaml")

        after = _files(tmp_path)
        tables_before = {name for name in before if name.startswith("tables/")}
        assert len(tables_before) == 5
        assert len(shared) == 2
        assert shared <= after.keys()
        assert (tables_before - shared).isdisjoint(after)
        assert sorted(_ids(pq.read_table(tmp_path / name))[0] for name in shared) == [
            "b_b1",
            "b_n1",
        ]
        for name in shared:
            assert pq.read_table(tmp_path / name).num_rows == 1
        assert len([name for name in after if name.startswith("tables/")]) == 7
        assert package_problems(path) == []
        data = (tmp_path / "a" / "manifest.yaml").read_bytes()
        assert timetoshare.loads(path.read_text(encoding="utf-8"))["manifests"][0] == {
            "url": "a/manifest.yaml",
            "media_type": "application/yaml",
            "size": len(data),
            "integrity": {
                "algorithm": "SHA-256",
                "digest": hashlib.sha256(data).hexdigest(),
            },
        }
        reopened = Package.from_file(path)
        assert _rows(reopened.bundle("b/manifest.yaml")) == _rows(b)
        rows = _rows(reopened.bundle("a/manifest.yaml"))
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
        package = Package.from_file(copy / "package.yaml")
        bach = package.bundle("bach/manifest.yaml")

        package.write_bundle(bach, "bach/manifest.yaml")

        def ids(name: str) -> list[str]:
            return _ids(pq.read_table(copy / "tables" / name))

        assert ids("Projection/part_a.parquet") == [
            "projection_5",
            "projection_3",
            "projection_4",
        ]
        assert ids("Projection/part_b.parquet") == ["projection_8", "projection_6"]
        assert ids("MatchClaim/part_a.parquet") == ["claim_3", "claim_4"]
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
        for kind, rows in (("MatchClaim", 3), ("Projection", 4)):
            (new,) = [n for n in _parts(copy, kind) if _PART.fullmatch(n)]
            assert pq.read_table(copy / "tables" / kind / new).num_rows == rows
        assert package_problems(copy / "package.yaml") == []
        _assert_expected_reads(Package.from_file(copy / "package.yaml"))

    def test_the_fixture_opened_as_a_package(self) -> None:
        package = Package.from_file(_SHARED / "package.yaml")

        assert package.manifests == ["bach/manifest.yaml", "handel/manifest.yaml"]
        assert [(table["type"], table["url"]) for table in package.tables] == [
            ("MatchClaim", "tables/MatchClaim/"),
            ("Projection", "tables/Projection/"),
        ]
        _assert_expected_reads(package)

    def test_a_single_shared_file_is_filtered(self, tmp_path: Path) -> None:
        copy = shutil.copytree(_SHARED, tmp_path / "shared")
        for kind in ("MatchClaim", "Projection"):
            folder = copy / "tables" / kind
            joined = pa.concat_tables(
                [pq.read_table(file) for file in sorted(folder.iterdir())]
            )
            if kind == "Projection":
                joined = joined.replace_schema_metadata(
                    pq.read_schema(folder / "part_a.parquet").metadata
                )
            pq.write_table(joined, copy / "tables" / f"{kind}.parquet")
            shutil.rmtree(folder)
        for name in ("package.yaml", "bach/manifest.yaml", "handel/manifest.yaml"):
            text = (copy / name).read_text(encoding="utf-8")
            for kind in ("MatchClaim", "Projection"):
                text = text.replace(f"tables/{kind}/", f"tables/{kind}.parquet")
            (copy / name).write_text(text, encoding="utf-8")
        assert package_problems(copy / "package.yaml") == []

        _assert_expected_reads(Package.from_file(copy / "package.yaml"))
        with pytest.raises(TableError) as refused:
            AlignmentBundle.from_file(copy / "bach" / "manifest.yaml")
        assert refused.value.problem.code == "unknown_timeline"

    def test_an_id_held_twice_is_refused_before_any_file(self, tmp_path: Path) -> None:
        a = _a()
        package = Package(name="Pieces")
        package.write_bundle(a, "a/manifest.yaml")
        package.to_file(tmp_path / "package.yaml")
        before = _files(tmp_path)
        (note_file,) = _parts(tmp_path, "Note")
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
                "'c/manifest.yaml' cannot join the package: its ID is held by the "
                f"event of timeline 'a_score' in 'tables/Note/{note_file}'; nothing "
                "is renamed.",
            ),
            (
                c(pcm="a_pcm"),
                "timeline 'a_pcm' of the manifest 'c/manifest.yaml' cannot join the "
                "package: its id 'a_pcm' is already held by a timeline of the "
                "manifest 'a/manifest.yaml'",
            ),
            (
                c(claim_id=claim.id),
                f"The MatchClaim '{claim.id}' of timeline 'c_score' of the manifest "
                "'c/manifest.yaml' cannot join the package: its ID is held by the "
                f"MatchClaim of timeline 'a_score' in 'tables/MatchClaim/{claim_file}'; "
                "nothing is renamed.",
            ),
        ):
            with pytest.raises(ValueError) as refused:
                package.write_bundle(bundle, "c/manifest.yaml")
            assert str(refused.value) == message
            assert _files(tmp_path) == before
            assert package.manifests == ["a/manifest.yaml"]

        fresh = Package()
        fresh.write_bundle(_a(), "a/manifest.yaml")
        with pytest.raises(ValueError) as refused:
            fresh.write_bundle(c(note="a_n1"), "c/manifest.yaml")
        assert str(refused.value) == (
            "The event 'a_n1' of timeline 'c_score' of the manifest "
            "'c/manifest.yaml' cannot join the package: its ID is held by the "
            "event of timeline 'a_score' in the manifest 'a/manifest.yaml'; nothing "
            "is renamed."
        )
        assert fresh.manifests == ["a/manifest.yaml"]
        assert fresh.path is None

    def test_a_manifest_s_match_claims_are_kept(self, tmp_path: Path) -> None:
        path, a, _ = _two_bundle_package(tmp_path)
        before = _files(tmp_path)
        package = Package.from_file(path)
        read = package.bundle("a/manifest.yaml")
        without = AlignmentBundle(id="a")
        for timeline in read.timelines.values():
            without.add_timeline(timeline)
        (claim,) = a.get_match_claims()

        with pytest.raises(ValueError) as refused:
            package.write_bundle(without, "a/manifest.yaml")

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
        changed = package.bundle("a/manifest.yaml")
        changed.get_timeline("a_score").add_events(
            [{"id": "a_n4", "event_type": "Note", "start": Fraction(2)}]
        )
        problem = PackageProblem("duplicate_id", "a/manifest.yaml", 0, (), "induced")
        written = package_module.package_problems

        with monkeypatch.context() as patch:
            patch.setattr(package_module, "package_problems", lambda path: [problem])
            with pytest.raises(ValueError) as refused:
                package.write_bundle(changed, "a/manifest.yaml")
            assert str(refused.value) == (
                f"The package check refuses the package written at '{path}': "
                "duplicate_id in 'a/manifest.yaml', row 0: induced; the package "
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
            new.write_bundle(_a(), "a/manifest.yaml")
            new.write_bundle(_b(), "b/manifest.yaml")
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
            package.write_bundle(changed, "a/manifest.yaml")
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
            Package.from_file(path).bundle("a/manifest.yaml")
        assert missing.value.args[0] == (
            "The package lists no manifest 'a/manifest.yaml'. Available "
            "manifests: []"
        )

    def test_a_carried_table_is_copied_into_the_package_and_listed(
        self, tmp_path: Path
    ) -> None:
        read = _carrying(tmp_path / "source", _b(), [1, 2])
        source = tmp_path / "source" / "features.parquet"
        package = Package()
        package.write_bundle(read, "b/manifest.yaml")

        path = package.to_file(tmp_path / "package" / "package.yaml")

        copy = tmp_path / "package" / "tables" / "Features" / "features.parquet"
        assert copy.read_bytes() == source.read_bytes()
        assert [
            (table["type"], table["url"])
            for table in timetoshare.loads(path.read_text(encoding="utf-8"))["tables"]
        ] == [
            ("Beat", "tables/Beat/"),
            ("Note", "tables/Note/"),
            ("Features", "tables/Features/features.parquet"),
        ]
        manifest = timetoshare.loads(
            (tmp_path / "package" / "b" / "manifest.yaml").read_text(encoding="utf-8")
        )
        assert [r for r in manifest["resources"] if r["id"] == "#features"] == [
            {
                "id": "#features",
                "type": "Features",
                "timelines": ["#b_score"],
                "url": "../tables/Features/features.parquet",
            }
        ]
        assert package_problems(path) == []
        again = Package.from_file(path).bundle("b/manifest.yaml")
        (carried,) = [r for r in again.resources if r.id == "features"]
        assert carried.url == "../tables/Features/features.parquet"
        assert list(again._carried_tables) == ["features"]
        assert _rows(again) == _rows(_b())

        # Written back through the opened package, the copy stays.
        before = _files(tmp_path / "package")
        opened = Package.from_file(path)
        opened.write_bundle(opened.bundle("b/manifest.yaml"), "b/manifest.yaml")
        after = _files(tmp_path / "package")
        assert [name for name in after if "Features" in name] == [
            "tables/Features/features.parquet"
        ]
        assert after["tables/Features/features.parquet"] == source.read_bytes()
        assert len(after) == len(before)
        assert [table["url"] for table in opened.tables].count(
            "tables/Features/features.parquet"
        ) == 1
        assert package_problems(path) == []

    def test_a_folder_without_parquet_files_is_copied_and_listed(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "source" / "scans").mkdir(parents=True)
        (tmp_path / "source" / "scans" / "page1.png").write_bytes(b"\x89PNG")
        bundle = _b()
        bundle.resources.append(Resource("scans", "Scan", ("b_score",), "scans/"))
        bundle.to_file(tmp_path / "source" / "b.yaml", events=True)
        read = AlignmentBundle.from_file(tmp_path / "source" / "b.yaml")
        package = Package()
        package.write_bundle(read, "b/manifest.yaml")

        path = package.to_file(tmp_path / "package" / "package.yaml")

        assert (
            tmp_path / "package" / "tables" / "Scan" / "scans" / "page1.png"
        ).read_bytes() == b"\x89PNG"
        assert {"type": "Scan", "url": "tables/Scan/scans/"} in package.tables
        (scans,) = [r for r in read.resources if r.id == "scans"]
        assert scans.url == "../tables/Scan/scans/"
        assert package_problems(path) == []
        again = Package.from_file(path).bundle("b/manifest.yaml")
        assert list(again._carried_tables) == ["scans"]
        assert _rows(again) == _rows(_b())

    def test_a_carried_table_s_name_taken(self, tmp_path: Path) -> None:
        score = ContinuousLogicalTimeline(length=Fraction(4), uid="e_score")
        score.add_events([{"id": "e_n1", "event_type": "Note", "start": Fraction(0)}])
        e = AlignmentBundle(id="e")
        e.add_timeline(score)
        f = AlignmentBundle(id="f")
        f.add_timeline(ContinuousLogicalTimeline(length=Fraction(4), uid="f_score"))
        package = Package()
        package.write_bundle(_carrying(tmp_path / "b", _b(), [1, 2]), "b/manifest.yaml")
        package.write_bundle(_carrying(tmp_path / "e", e, [3]), "e/manifest.yaml")
        package.write_bundle(_carrying(tmp_path / "f", f, [1, 2]), "f/manifest.yaml")

        path = package.to_file(tmp_path / "package" / "package.yaml")

        root = tmp_path / "package"
        assert _parts(root, "Features") == [
            "features.parquet",
            "features_0eyvk1ozczb0s.parquet",
        ]
        assert (
            root / "tables/Features/features_0eyvk1ozczb0s.parquet"
        ).read_bytes() == ((tmp_path / "e" / "features.parquet").read_bytes())
        urls = {
            url: [
                r["url"]
                for r in timetoshare.loads((root / url).read_text(encoding="utf-8"))[
                    "resources"
                ]
                if r["id"] == "#features"
            ]
            for url in package.manifests
        }
        assert urls == {
            "b/manifest.yaml": ["../tables/Features/features.parquet"],
            "e/manifest.yaml": ["../tables/Features/features_0eyvk1ozczb0s.parquet"],
            "f/manifest.yaml": ["../tables/Features/features.parquet"],
        }
        assert [t["url"] for t in package.tables if t["type"] == "Features"] == [
            "tables/Features/features.parquet",
            "tables/Features/features_0eyvk1ozczb0s.parquet",
        ]
        assert package_problems(path) == []

    def test_a_table_the_writer_has_no_file_of_is_refused(self, tmp_path: Path) -> None:
        bundle = _b()
        bundle.resources.append(
            Resource("features", "Features", ("b_score",), "features.parquet")
        )
        package = Package()
        with pytest.raises(ValueError) as refused:
            package.write_bundle(bundle, "b/manifest.yaml")
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
            package.write_bundle(read, "b/manifest.yaml")
        assert str(refused.value) == (
            "The resource 'features' of the bundle 'b' names the table "
            f"'features.parquet' of the type Features, which the bundle carries "
            f"from '{source}', where no file lies any more: the package writer has "
            "nothing to copy into the package; nothing is written."
        )
        assert (package.manifests, package.path) == ([], None)

    def test_a_write_cut_short_leaves_no_hidden_file(self, tmp_path: Path) -> None:
        package = Package()
        package.write_bundle(_a(), "a/manifest.yaml")
        path = package.to_file(tmp_path / "package.yaml")
        before = _files(tmp_path)
        hidden = [
            tmp_path / "tables" / "Note" / ".part_0000000000001.parquet",
            tmp_path / "tables" / "Chord" / ".part_0000000000002.parquet",
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
        path, _, _ = _two_bundle_package(tmp_path / "package")
        bundle = Package.from_file(path).bundle("b/manifest.yaml")
        (tmp_path / "alone").mkdir()

        with pytest.raises(NotImplementedError) as refused:
            bundle.to_file(tmp_path / "alone" / "manifest.yaml")

        first = sorted(bundle.resources, key=lambda resource: resource.id)[0]
        assert str(refused.value) == (
            f"The resource '{first.id}' names the folder '../tables/{first.type}/' "
            f"for its {first.type} rows: AlignmentBundle.to_file writes a "
            "manifest's tables as single files beside it and does not write into "
            "a folder, whose files other manifests may share; write the bundle "
            "into its package with Package.write_bundle."
        )
        assert list((tmp_path / "alone").iterdir()) == []


def _assert_expected_reads(package: Package) -> None:
    """Each manifest of the shared fixture's package, read through *package*,
    holds the MatchClaims and Projections ``expected.yaml`` states."""
    for url, expected in _EXPECTED.items():
        bundle = package.bundle(url)
        assert sorted(_ids(bundle.match_claim_table.table)) == (expected["MatchClaim"])
        held = sorted(
            row["id"]
            for timeline in bundle._known_timelines().values()
            for row in timeline.events.table.to_pylist()
            if row["event_type"] == "Projection"
        )
        assert held == expected["Projection"]
