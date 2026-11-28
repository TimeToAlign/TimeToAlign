"""The event registry: root positions, local events, views, updates, refusals."""

from __future__ import annotations

import io
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

import timetoalign
from timetoalign.alignment import AlignmentBundle
from timetoalign.core import Coordinate, IdCoordinate, NumberType, TimeUnit
from timetoalign.storage import EventData, EventRegistry, Extraction
from timetoalign.storage.registry import (
    EXTRACTION_SCHEMA,
    REGISTRY_SCHEMA,
    extraction_table,
)
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteLogicalTimeline,
    Timeline,
    TimelineGroup,
)

ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
SAME = "among a root timeline and its descendants"


def _quarters(uid: str, length: Any, number_type: str = "fraction") -> Timeline:
    return ContinuousLogicalTimeline(
        length=length, unit="quarters", number_type=number_type, uid=uid
    )


def _family() -> tuple[Timeline, Timeline, Timeline, Timeline]:
    """``clt_root`` ⊃ ``clt_a`` (at 2) ⊃ ``clt_g`` (at 1), and ``clt_b`` (at 4)."""
    root = _quarters("clt_root", 8)
    root.add_events([{"id": "r1", "event_type": "Beat", "start": Fraction(1, 2)}])
    a = _quarters("clt_a", 4)
    a.add_events(
        [
            {"id": "a1", "event_type": "Beat", "start": Fraction(1, 3)},
            {"id": "a2", "event_type": "Beat", "start": 0},
        ]
    )
    g = _quarters("clt_g", 2)
    g.add_events(
        [
            {
                "id": "g1",
                "event_type": "Note",
                "start": Fraction(1, 4),
                "duration": Fraction(1, 2),
                "pitch": 60,
            }
        ]
    )
    a.add_child(g, offset=1)
    root.add_child(a, offset=2)
    b = _quarters("clt_b", 4)
    b.add_events([{"id": "b1", "event_type": "Beat", "start": Fraction(1, 10)}])
    root.add_child(b, offset=4)
    return root, a, g, b


def _ids(table: pa.Table) -> list[str]:
    return table.column("id").to_pylist()


def _exact(cells: list[dict[str, Any] | None]) -> list[Fraction | None]:
    """Each cell's exact ratio, None for a null cell."""
    return [
        None if cell is None else Fraction(cell["numerator"], cell["denominator"])
        for cell in cells
    ]


def _starts(table: pa.Table) -> list[Fraction | None]:
    return _exact(table.column("start").to_pylist())


def _values(table: pa.Table, column: str = "start") -> list[float | None]:
    return [
        None if cell is None else cell["value"]
        for cell in table.column(column).to_pylist()
    ]


def _registry_rows(rows: list[dict[str, Any]], unit: str, number_type: str) -> pa.Table:
    """Registry rows on a root's axis, positions given as numbers."""

    def cell(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        exact = Fraction(value)
        return {
            "value": float(value),
            "numerator": exact.numerator,
            "denominator": exact.denominator,
        }

    return pa.Table.from_pylist(
        [
            {
                **row,
                "unit": unit,
                "number_type": number_type,
                "start": cell(row["start"]),
                "duration": cell(row.get("duration")),
            }
            for row in rows
        ],
        schema=pa.schema(
            [
                REGISTRY_SCHEMA.field(name)
                for name in (
                    "id",
                    "kind",
                    "location",
                    "timeline_id",
                    "unit",
                    "number_type",
                    "start",
                    "duration",
                    "extraction_id",
                )
            ]
        ),
    )


def _package_extraction() -> Extraction:
    return Extraction(
        id="extraction_p",
        source="score.mei",
        source_integrity={"algorithm": "SHA-256", "digest": ABC_SHA256},
        parser="TestParser",
        parser_version="1",
        date="2026-11-27T00:00:00Z",
    )


def _audio() -> Timeline:
    audio = ContinuousPhysicalTimeline(length=10.0, uid="audio")
    audio.add_events([{"id": "o1", "event_type": "Onset", "start": 1.0}])
    return audio


def _bundle(root: Timeline | None = None) -> AlignmentBundle:
    """A bundle of the family's root and ``audio`` (seconds) holding ``o1`` at 1."""
    bundle = AlignmentBundle(id="b")
    bundle.add_timeline(_family()[0] if root is None else root)
    bundle.add_timeline(_audio())
    return bundle


class TestRootPositions:
    def test_the_root_holds_every_row_on_its_axis(self) -> None:
        root, *_ = _family()
        table = root.registry.table

        assert table.schema.names == REGISTRY_SCHEMA.names
        assert _ids(table) == ["r1", "a1", "a2", "g1", "b1"]
        assert table.column("kind").to_pylist() == [
            "Beat",
            "Beat",
            "Beat",
            "Note",
            "Beat",
        ]
        assert table.column("timeline_id").to_pylist() == [
            "clt_root",
            "clt_a",
            "clt_a",
            "clt_g",
            "clt_b",
        ]
        assert set(table.column("unit").to_pylist()) == {"quarters"}
        assert set(table.column("number_type").to_pylist()) == {"fraction"}
        assert _starts(table) == [
            Fraction(1, 2),
            Fraction(7, 3),
            2,
            Fraction(13, 4),
            Fraction(41, 10),
        ]
        assert _exact(table.column("duration").to_pylist()) == [
            None,
            None,
            None,
            Fraction(1, 2),
            None,
        ]

    def test_the_family_holds_the_segment_rows(self) -> None:
        root, *_ = _family()
        table = root._registry._family.table()

        assert _ids(table) == ["r1", "clt_a", "clt_b", "a1", "a2", "clt_g", "g1", "b1"]
        segments = table.filter(pa.compute.equal(table.column("kind"), "Segment"))
        assert _ids(segments) == ["clt_a", "clt_b", "clt_g"]
        assert _starts(segments) == [2, 4, 3]
        assert _exact(segments.column("duration").to_pylist()) == [4, 4, 2]
        assert segments.column("extraction_id").to_pylist() == [None, None, None]

    def test_every_event_names_the_familys_tool_extraction(self) -> None:
        root, *_ = _family()
        table = root.registry.table

        (tool,) = root.registry.extractions
        assert (tool.parser, tool.parser_version, tool.source) == (
            "timetoalign",
            timetoalign.__version__,
            None,
        )
        assert set(table.column("extraction_id").to_pylist()) == {tool.id}
        for name in ("location", "locator", "vanished_in"):
            assert table.column(name).null_count == len(table)

    def test_an_event_without_an_event_type_is_of_kind_event(self) -> None:
        root = _quarters("clt_z", 4)
        root.add_events([{"id": "z1", "start": 1}])

        assert root.registry.table.column("kind").to_pylist() == ["Event"]
        assert root.events.table.column("event_type").to_pylist() == [None]

    def test_each_timeline_reads_its_own_coordinates(self) -> None:
        _, a, g, b = _family()

        assert _starts(a.events.table) == [Fraction(1, 3), 0, 1]
        assert _starts(g.events.table) == [Fraction(1, 4)]
        assert _starts(b.events.table) == [Fraction(1, 10)]

    def test_an_int_axis_is_exact_both_ways(self) -> None:
        root = DiscreteLogicalTimeline(length=100, uid="dlt_root")
        child = DiscreteLogicalTimeline(length=20, uid="dlt_c")
        child.add_events([{"id": "t1", "event_type": "Tick", "start": 3}])
        root.add_child(child, offset=7)

        (row,) = root.registry.filter(kinds="Tick").table.to_pylist()
        assert row["start"] == {"value": 10.0, "numerator": 10, "denominator": 1}
        assert row["number_type"] == "int"
        assert _starts(child.events.table) == [3]

    def test_rows_read_from_a_package_are_local_to_their_timeline(self) -> None:
        root = _quarters("clt_p", 8)
        child = _quarters("clt_pc", 4)
        root.add_child(child, offset=2)
        rows = _registry_rows(
            [
                {
                    "id": "p1",
                    "kind": "Note",
                    "location": "tables/Note/part.parquet",
                    "timeline_id": "clt_pc",
                    "start": Fraction(5, 2),
                    "duration": Fraction(1, 2),
                    "extraction_id": "extraction_p",
                }
            ],
            "quarters",
            "fraction",
        )
        content = pa.table({"id": ["p1"], "pitch": [62]})

        root._registry.add_rows(rows, {"Note": content}, [_package_extraction()])

        (event,) = child.events.table.to_pylist()
        assert (event["id"], event["event_type"], event["temporal_type"]) == (
            "p1",
            "Note",
            "interval",
        )
        assert _exact([event["start"], event["end"], event["duration"]]) == [
            Fraction(1, 2),
            1,
            Fraction(1, 2),
        ]
        assert event["pitch"] == 62
        (row,) = root.registry.filter(kinds="Note").table.to_pylist()
        assert _exact([row["start"]]) == [Fraction(5, 2)]
        assert row["location"] == "tables/Note/part.parquet"

    def test_rows_read_from_a_package_on_an_int_axis(self) -> None:
        root = DiscreteLogicalTimeline(length=100, uid="dlt_p")
        child = DiscreteLogicalTimeline(length=20, uid="dlt_pc")
        root.add_child(child, offset=7)
        rows = _registry_rows(
            [
                {
                    "id": "p1",
                    "kind": "Tick",
                    "timeline_id": "dlt_pc",
                    "start": 10,
                    "extraction_id": "extraction_p",
                }
            ],
            "ticks",
            "int",
        )

        root._registry.add_rows(rows, {}, [_package_extraction()])

        assert _starts(child.events.table) == [3]
        assert child.events.table.column("temporal_type").to_pylist() == ["instant"]

    def test_rows_read_from_a_package_keep_their_order(self, tmp_path: Path) -> None:
        path = tmp_path / "score.txt"
        path.write_bytes(b"abc")
        first = Extraction(
            **{
                **Extraction.of_source(path, parser="P", parser_version="1").__dict__,
                "id": "extraction_first",
            }
        )
        timeline = _quarters("clt_t", 8)
        rows = _registry_rows(
            [
                {"id": "n1", "kind": "Note", "timeline_id": "clt_t", "start": 1},
                {"id": "b1", "kind": "Beat", "timeline_id": "clt_t", "start": 2},
                {"id": "n2", "kind": "Note", "timeline_id": "clt_t", "start": 3},
            ],
            "quarters",
            "fraction",
        )
        rows = rows.set_column(
            rows.schema.get_field_index("extraction_id"),
            "extraction_id",
            pa.array(["extraction_first"] * 3),
        ).append_column("locator", pa.array(["L1", "L2", "L3"]))
        contents = {
            "Note": pa.table({"id": ["n1", "n2"], "pitch": [60, 64]}),
            "Beat": pa.table({"id": ["b1"], "strength": [1]}),
        }

        timeline._registry.add_rows(rows, contents, [first], registered=False)

        assert _ids(timeline.events.table) == ["n1", "b1", "n2"]
        assert _ids(timeline.registry.table) == ["n1", "b1", "n2"]
        second = Extraction(**{**first.__dict__, "id": "extraction_second"})
        again = EventData.from_dicts(
            [
                {
                    "id": "q1",
                    "event_type": "Note",
                    "start": 1,
                    "locator": "L1",
                    "pitch": 61,
                },
                {
                    "id": "q3",
                    "event_type": "Note",
                    "start": 3,
                    "locator": "L3",
                    "pitch": 65,
                },
            ],
            TimeUnit.quarters,
            NumberType.fraction,
        )

        timeline._registry.update("clt_t", again, second)

        assert _ids(timeline.events.table) == ["n1", "b1", "n2"]
        assert _ids(timeline.registry.table) == ["n1", "b1", "n2"]
        assert timeline.events.table.column("pitch").to_pylist() == [61, None, 65]
        assert timeline.registry.table.column("vanished_in").to_pylist() == [
            None,
            "extraction_second",
            None,
        ]


class TestTheFloatCase:
    def test_built_in_code_both_positions_are_held(self) -> None:
        root = ContinuousPhysicalTimeline(length=10.0, uid="cpt_root")
        child = ContinuousPhysicalTimeline(length=1.0, uid="cpt_c")
        child.add_events([{"id": "f1", "event_type": "Onset", "start": 0.1}])
        root.add_child(child, offset=0.2)

        assert _values(root.registry.filter(kinds="Onset").table) == [
            0.30000000000000004
        ]
        assert _values(child.events.table) == [0.1]

    def test_read_from_a_package_the_local_position_is_binary64s(self) -> None:
        root = ContinuousPhysicalTimeline(length=10.0, uid="cpt_root")
        child = ContinuousPhysicalTimeline(length=1.0, uid="cpt_c")
        root.add_child(child, offset=0.2)
        rows = _registry_rows(
            [
                {
                    "id": "f1",
                    "kind": "Onset",
                    "timeline_id": "cpt_c",
                    "start": 0.3,
                    "extraction_id": "extraction_p",
                }
            ],
            "seconds",
            "float",
        )

        root._registry.add_rows(rows, {}, [_package_extraction()])

        assert _values(child.events.table) == [0.09999999999999998]
        assert _values(root.registry.filter(kinds="Onset").table) == [0.3]


class TestColumnWise:
    def test_no_row_takes_the_per_row_shift(self, monkeypatch) -> None:
        def refused(*_: Any, **__: Any) -> Any:
            raise AssertionError("a row was shifted one by one")

        monkeypatch.setattr(
            "timetoalign.timelines.engines.events.shift_coordinate", refused
        )
        monkeypatch.setattr("timetoalign.core.time.shift_coordinate", refused)
        root, *_ = _family()

        events = root.get_events()

        assert _starts(root.registry.table)[3] == Fraction(13, 4)
        assert _ids(events.table) == ["r1", "a1", "a2", "g1", "b1"]
        assert events.table.column("source_timeline").to_pylist() == [
            None,
            "clt_a",
            "clt_a",
            "clt_g",
            "clt_b",
        ]
        assert _starts(events.table) == [
            Fraction(1, 2),
            Fraction(7, 3),
            2,
            Fraction(13, 4),
            Fraction(41, 10),
        ]


class TestViews:
    def test_a_view_follows_the_rows(self) -> None:
        _, a, _, _ = _family()
        view = a.registry
        assert _ids(view.table) == ["a1", "a2", "g1"]

        a.add_events([{"id": "a3", "event_type": "Beat", "start": 2}])

        assert _ids(view.table) == ["a1", "a2", "a3", "g1"]

    def test_filters(self) -> None:
        root, *_ = _family()
        registry = root.registry

        beats = registry.filter(kinds="Beat")
        assert _ids(beats.table) == ["r1", "a1", "a2", "b1"]
        assert _ids(beats.filter(within="clt_a").table) == ["a1", "a2"]
        assert _ids(registry.filter(min_coord=2, max_coord=4).table) == [
            "a1",
            "a2",
            "g1",
        ]
        seconds = registry.filter(min_coord=Coordinate(1.0, TimeUnit.seconds))
        assert _ids(seconds.table) == []

    def test_a_bounds_unit_selects_the_families_in_that_unit(self) -> None:
        bundle = _bundle()

        quarters = bundle.registry.filter(min_coord=Coordinate(1, TimeUnit.quarters))
        seconds = bundle.registry.filter(min_coord=Coordinate(1.0, TimeUnit.seconds))

        assert _ids(quarters.table) == ["a1", "a2", "g1", "b1"]
        assert _ids(seconds.table) == ["o1"]

    def test_an_id_coordinate_lies_on_the_timeline_it_names(self) -> None:
        root, *_ = _family()
        bundle = _bundle(root)

        def on(value: Any, timeline_id: str) -> IdCoordinate:
            return IdCoordinate(
                value=value, unit=TimeUnit.quarters, timeline_id=timeline_id
            )

        registry = root.registry
        assert _ids(registry.filter(min_coord=on(Fraction(1, 3), "clt_a")).table) == [
            "a1",
            "g1",
            "b1",
        ]
        assert _ids(registry.filter(min_coord=on(1, "clt_g")).table) == ["b1"]
        assert _ids(registry.filter(max_coord=on(Fraction(1, 3), "clt_a")).table) == [
            "r1",
            "a2",
        ]
        assert _ids(bundle.registry.filter(min_coord=on(0, "clt_a")).table) == [
            "a1",
            "a2",
            "g1",
            "b1",
        ]
        assert _ids(bundle.registry.filter(min_coord=on(0, "clt_nowhere")).table) == []

    def test_counts(self) -> None:
        root, *_ = _family()
        registry = root.registry

        assert registry.counts().to_pylist() == [
            {"kind": "Beat", "count": 4},
            {"kind": "Note", "count": 1},
        ]
        assert len(registry) == 5
        assert registry.counts("timeline_id").to_pylist() == [
            {"timeline_id": "clt_a", "count": 2},
            {"timeline_id": "clt_b", "count": 1},
            {"timeline_id": "clt_g", "count": 1},
            {"timeline_id": "clt_root", "count": 1},
        ]
        with pytest.raises(KeyError) as refusal:
            registry.counts("start")
        assert refusal.value.args[0] == (
            "No registry column 'start' to count by. Available columns: ['id', "
            "'kind', 'location', 'timeline_id', 'unit', 'number_type', 'name', "
            "'extraction_id', 'locator', 'vanished_in']"
        )

    def test_concat_keeps_each_row_once(self) -> None:
        root, a, _, _ = _family()

        joined = EventRegistry.concat(a.registry, root.registry.filter(kinds="Note"))

        assert _ids(joined.table) == ["a1", "a2", "g1"]

    def test_a_bundles_registry_holds_every_family(self) -> None:
        root, *_ = _family()
        bundle = _bundle(root)

        table = bundle.registry.table

        assert _ids(table) == [*_ids(root.registry.table), "o1"]
        assert table.column("unit").to_pylist()[-1] == "seconds"
        assert table.column("number_type").to_pylist()[-1] == "float"

    def test_a_groups_registry_holds_its_members_and_their_descendants(self) -> None:
        _, a, _, _ = _family()
        group = TimelineGroup(id="paired", timelines=[a, _audio()])

        assert _ids(group.registry.table) == ["a1", "a2", "g1", "o1"]

    def test_content(self) -> None:
        root, *_ = _family()

        assert root.registry.content("Note").to_pylist() == [{"id": "g1", "pitch": 60}]
        assert root.registry.content("Beat", ids=["a1"]).to_pylist() == [{"id": "a1"}]
        assert _ids(root.registry.content("Beat")) == ["r1", "a1", "a2", "b1"]

    def test_a_segment_names_no_tool_extraction(self, tmp_path: Path) -> None:
        path = tmp_path / "score.txt"
        path.write_bytes(b"abc")
        extraction = Extraction.of_source(path, parser="P", parser_version="1")
        root = _quarters("clt_r", 8)
        child = _quarters("clt_c", 4)
        root.add_child(child, offset=1)
        data = EventData.from_dicts(
            [{"id": "n1", "event_type": "Note", "start": 0, "locator": "m0"}],
            TimeUnit.quarters,
            NumberType.fraction,
        )

        child._registry.add("clt_c", data, extraction=extraction)

        assert root.registry.extractions == [extraction]
        assert root._registry._family.tool is None


class TestTablesSpanTimelines:
    def test_a_timelines_events_are_keyed_by_it_in_a_join(self) -> None:
        _, _, _, b = _family()
        audio = _audio()

        joined = b.events.concat(audio.events)

        assert joined.table.column("timeline_id").to_pylist() == ["clt_b", "audio"]
        assert _starts(joined.table) == [Fraction(1, 10), 1]
        assert _values(joined.table) == [0.1, 1.0]
        assert joined.filter(include_number_types={NumberType.float}).column_values(
            "id"
        ) == ["o1"]

    def test_a_table_spanning_number_types_joins_no_timeline(self) -> None:
        _, a, _, _ = _family()
        spanning = EventData.from_dicts(
            [{"id": "f1", "event_type": "Beat", "start": Fraction(1, 2)}],
            TimeUnit.quarters,
            NumberType.fraction,
        ).concat(
            EventData.from_dicts(
                [{"id": "f2", "event_type": "Beat", "start": 0.5}],
                TimeUnit.quarters,
                NumberType.float,
            )
        )

        with pytest.raises(ValueError) as refusal:
            a._registry.add("clt_a", spanning)

        assert str(refusal.value) == (
            "the rows of timeline 'clt_a' lie on its axis, quarters in fraction; "
            "these lie on quarters in float and quarters in fraction"
        )
        assert _ids(a.events.table) == ["a1", "a2", "clt_g"]

        a._registry.add("clt_a", spanning.to_number_type(NumberType.fraction))

        assert _ids(a.events.table) == ["a1", "a2", "clt_g", "f1", "f2"]
        assert "timeline_id" not in a.events.table.column_names

    def test_a_timelines_number_type_filter_is_all_or_nothing(self) -> None:
        root, *_ = _family()

        assert _ids(root.get_events(include_number_types={"fraction"}).table) == [
            "r1",
            "a1",
            "a2",
            "g1",
            "b1",
        ]
        assert _ids(root.get_events(include_number_types={"float"}).table) == []


class TestMergeOnAttach:
    def test_a_childs_rows_shift_once_onto_the_new_root(self) -> None:
        root = _quarters("clt_root", 8)
        root.add_events([{"id": "r1", "event_type": "Beat", "start": 0}])
        a = _quarters("clt_a", 4)
        a.add_events([{"id": "a1", "event_type": "Beat", "start": Fraction(1, 3)}])
        g = _quarters("clt_g", 2)
        g.add_events([{"id": "g1", "event_type": "Beat", "start": Fraction(1, 4)}])
        a.add_child(g, offset=1)
        view = a.registry
        (child_tool,) = view.extractions
        assert _ids(view.table) == ["a1", "g1"]
        assert _starts(view.table) == [Fraction(1, 3), Fraction(5, 4)]

        root.add_child(a, offset=2)

        assert _starts(view.table) == [Fraction(7, 3), Fraction(13, 4)]
        (tool,) = root.registry.extractions
        assert tool.id != child_tool.id
        assert set(root.registry.table.column("extraction_id").to_pylist()) == {tool.id}
        a.add_events([{"id": "a3", "event_type": "Beat", "start": 2}])
        assert _starts(root.registry.filter(within="clt_a", kinds="Beat").table) == [
            Fraction(7, 3),
            4,
        ]

    def test_a_root_of_no_events_adopts_the_childs_tool_extraction(self) -> None:
        root = _quarters("clt_root", 8)
        a = _quarters("clt_a", 4)
        a.add_events([{"id": "a1", "event_type": "Beat", "start": 0}])
        (child_tool,) = a.registry.extractions

        root.add_child(a, offset=2)

        assert root.registry.extractions == [child_tool]


class TestUpdate:
    @staticmethod
    def _source(tmp_path: Path) -> Path:
        path = tmp_path / "score.txt"
        path.write_bytes(b"abc")
        return path

    @staticmethod
    def _run(
        path: Path, rows: list[tuple[str, str, Any, int]]
    ) -> tuple[Extraction, EventData]:
        extraction = Extraction.of_source(path, parser="TestParser", parser_version="1")
        data = EventData.from_dicts(
            [
                {
                    "id": event_id,
                    "event_type": "Note",
                    "start": start,
                    "pitch": pitch,
                    "locator": locator,
                }
                for event_id, locator, start, pitch in rows
            ],
            TimeUnit.quarters,
            NumberType.fraction,
        )
        return extraction, data

    def test_rows_matched_by_locator_keep_their_ids(self, tmp_path: Path) -> None:
        path = self._source(tmp_path)
        timeline = _quarters("clt_u", 8)
        first, data = self._run(
            path, [("n1", "m0", 0, 60), ("n2", "m1", 1, 62), ("n3", "m2", 2, 64)]
        )
        timeline._registry.add("clt_u", data, extraction=first)
        assert first.source_integrity == {"algorithm": "SHA-256", "digest": ABC_SHA256}
        assert "locator" not in timeline.events.table.column_names
        assert timeline.registry.table.column("locator").to_pylist() == [
            "m0",
            "m1",
            "m2",
        ]
        timeline.add_events([{"id": "k1", "event_type": "Beat", "start": 5}])

        second, again = self._run(
            path,
            [
                ("x0", "m0", 0, 61),
                ("x2", "m2", Fraction(5, 2), 64),
                ("x3", "m3", 3, 65),
            ],
        )
        second = Extraction(**{**second.__dict__, "id": "extraction_second"})
        timeline._registry.update("clt_u", again, second)

        table = timeline.registry.table
        assert _ids(table) == ["n1", "n2", "n3", "k1", "x3"]
        assert timeline.events.table.column("pitch").to_pylist() == [
            61,
            62,
            64,
            None,
            65,
        ]
        assert _starts(table) == [0, 1, Fraction(5, 2), 5, 3]
        tool = timeline._registry._family.tool
        assert tool is not None
        assert table.column("extraction_id").to_pylist() == [
            "extraction_second",
            first.id,
            "extraction_second",
            tool.id,
            "extraction_second",
        ]
        assert table.column("vanished_in").to_pylist() == [
            None,
            "extraction_second",
            None,
            None,
            None,
        ]
        assert _ids(timeline.registry.filter(vanished=True).table) == ["n2"]

    def test_an_update_without_locators_is_refused(self, tmp_path: Path) -> None:
        path = self._source(tmp_path)
        timeline = _quarters("clt_u", 8)
        extraction = Extraction.of_source(path, parser="TestParser", parser_version="1")
        data = EventData.from_dicts(
            [{"id": "n1", "event_type": "Note", "start": 0}],
            TimeUnit.quarters,
            NumberType.fraction,
        )

        with pytest.raises(ValueError) as refusal:
            timeline._registry.update("clt_u", data, extraction)

        assert str(refusal.value) == (
            "an update matches rows by their locator: the rows give none (a column "
            "'locator')"
        )
        assert len(timeline.events) == 0

    def _first(self, tmp_path: Path) -> tuple[Timeline, Extraction]:
        """A timeline holding ``n1``, ``n2``, ``n3`` of ``extraction_first``."""
        first, data = self._run(
            self._source(tmp_path),
            [("n1", "m0", 0, 60), ("n2", "m1", 1, 62), ("n3", "m2", 2, 64)],
        )
        first = Extraction(**{**first.__dict__, "id": "extraction_first"})
        timeline = _quarters("clt_u", 8)
        timeline._registry.add("clt_u", data, extraction=first)
        return timeline, first

    def test_a_locator_given_twice_in_a_run_is_refused(self, tmp_path: Path) -> None:
        timeline, first = self._first(tmp_path)
        before = timeline.registry.table
        second, again = self._run(
            self._source(tmp_path), [("x1", "m0", 0, 61), ("x2", "m0", 1, 62)]
        )
        second = Extraction(**{**second.__dict__, "id": "extraction_second"})

        with pytest.raises(ValueError) as refusal:
            timeline._registry.update("clt_u", again, second)

        assert str(refusal.value) == (
            "locator 'm0' is given to 2 events of extraction 'extraction_second'; "
            "a locator names one event of its source"
        )
        assert timeline.registry.table.equals(before)
        assert _ids(timeline.events.table) == ["n1", "n2", "n3"]

    def test_a_locator_held_by_an_event_of_its_extraction_is_refused(
        self, tmp_path: Path
    ) -> None:
        timeline, first = self._first(tmp_path)
        before = timeline.registry.table
        message = (
            "locator 'm0' is given to 2 events of extraction 'extraction_first'; "
            "a locator names one event of its source"
        )
        _, again = self._run(self._source(tmp_path), [("y1", "m0", 3, 61)])

        with pytest.raises(ValueError) as refusal:
            timeline._registry.add("clt_u", again, extraction=first)
        assert str(refusal.value) == message
        rows = _registry_rows(
            [
                {
                    "id": "y1",
                    "kind": "Note",
                    "timeline_id": "clt_u",
                    "start": 3,
                    "extraction_id": "extraction_first",
                }
            ],
            "quarters",
            "fraction",
        ).append_column("locator", pa.array(["m0"]))
        with pytest.raises(ValueError) as refusal:
            timeline._registry.add_rows(rows, registered=False)
        assert str(refusal.value) == message
        assert timeline.registry.table.equals(before)

        timeline._registry.add("clt_u", again, extraction=_package_extraction())

        assert _ids(timeline.registry.table) == ["n1", "n2", "n3", "y1"]


class TestExtractions:
    def test_a_source_file_is_named_by_its_digest(self, tmp_path: Path) -> None:
        path = tmp_path / "score.txt"
        path.write_bytes(b"abc")

        first = Extraction.of_source(
            path, parser="P", parser_version="2", parameters={"b": [2], "a": 1}
        )
        second = Extraction.of_source(
            path, parser="P", parser_version="2", parameters={"a": 1, "b": [2]}
        )

        assert first.source_integrity == {"algorithm": "SHA-256", "digest": ABC_SHA256}
        assert first.id == second.id
        assert first.id.startswith("extraction_")
        row = first.to_row()
        assert row["parameters"] == '{"a":1,"b":[2]}'
        assert Extraction.from_row(row) == first
        assert extraction_table([first]).schema == EXTRACTION_SCHEMA

    def test_the_id_never_depends_on_where_the_file_lies(self, tmp_path: Path) -> None:
        (tmp_path / "one").mkdir()
        (tmp_path / "two").mkdir()
        here, there = tmp_path / "one" / "score.txt", tmp_path / "two" / "copy.txt"
        here.write_bytes(b"abc")
        there.write_bytes(b"abc")

        runs = [
            Extraction.of_source(
                path,
                source=source,
                parser="P",
                parser_version="2",
                date="2026-11-28T00:00:00Z",
            )
            for path, source in (
                (here, None),
                (there, None),
                (here, "score.txt"),
                (there, "score.txt"),
            )
        ]

        assert len({run.id for run in runs}) == 1
        assert [run.source for run in runs] == [
            here.as_posix(),
            there.as_posix(),
            "score.txt",
            "score.txt",
        ]
        assert hash(runs[0]) == hash(runs[0].id)
        assert {runs[2], runs[3]} == {runs[2]}

    def test_a_member_of_a_zip_archive_is_read_in_place(self, tmp_path: Path) -> None:
        (tmp_path / "one").mkdir()
        disk = tmp_path / "one" / "score.txt"
        disk.write_bytes(b"abc")
        archive = tmp_path / "set.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("inner/score.txt", b"abc")
        run = {"parser": "P", "parser_version": "2", "date": "2026-11-28T00:00:00Z"}

        member = Extraction.of_source(
            zipfile.Path(archive, at="inner/score.txt"), **run
        )
        named = Extraction.of_source(
            zipfile.Path(archive, at="inner/score.txt"), source="score.txt", **run
        )

        assert member.source_integrity == {
            "algorithm": "SHA-256",
            "digest": ABC_SHA256,
        }
        assert member.id == Extraction.of_source(disk, **run).id
        assert member.source == f"jar:{archive.as_uri()}!/inner/score.txt"
        assert named.source == "score.txt"
        with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as from_object:
            inside = zipfile.Path(from_object, at="inner/score.txt")
            with pytest.raises(ValueError, match="opened from a file object"):
                Extraction.of_source(inside, **run)
            assert Extraction.of_source(inside, source="score.txt", **run) == named

    def test_a_table_carries_its_extractions_to_its_timeline(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "score.txt"
        path.write_bytes(b"abc")
        e1 = Extraction.of_source(path, parser="P", parser_version="1")
        e2 = Extraction.of_source(path, parser="P", parser_version="2")
        data = EventData.from_dicts(
            [
                {
                    "id": event_id,
                    "event_type": "Note",
                    "start": start,
                    "extraction_id": e1.id,
                    "locator": locator,
                }
                for event_id, start, locator in (("n1", 0, "m0"), ("n2", 1, "m1"))
            ],
            TimeUnit.quarters,
            NumberType.fraction,
        ).with_extractions(e1)

        timeline = data.create_timeline(uid="clt_e")

        assert timeline.registry.extractions == [e1]
        table = timeline.registry.table
        assert table.column("extraction_id").to_pylist() == [e1.id, e1.id]
        assert table.column("locator").to_pylist() == ["m0", "m1"]
        for name in ("extraction_id", "locator"):
            assert name not in timeline.events.table.column_names
        other = EventData.empty(TimeUnit.quarters, NumberType.fraction)
        joined = data.concat(other.with_extractions(e2, e1))
        assert joined.extractions == (e1, e2)

    def test_a_row_naming_an_extraction_nobody_gives_is_refused(self) -> None:
        data = EventData.from_dicts(
            [
                {
                    "id": "n1",
                    "event_type": "Note",
                    "start": 0,
                    "extraction_id": "extraction_nowhere",
                }
            ],
            TimeUnit.quarters,
            NumberType.fraction,
        )
        message = (
            "the rows name the extraction 'extraction_nowhere', which this family "
            "does not hold: give it with the rows (EventData.with_extractions)"
        )
        with pytest.raises(ValueError) as refusal:
            data.create_timeline(uid="clt_e")
        assert str(refusal.value) == message
        timeline = _quarters("clt_e", 4)

        with pytest.raises(ValueError) as refusal:
            timeline._registry.add("clt_e", data)

        assert str(refusal.value) == message
        assert len(timeline.events) == 0
        assert timeline._registry._family.extractions == {}

    def test_a_source_without_its_digest_is_refused(self) -> None:
        with pytest.raises(ValueError) as refusal:
            Extraction(
                id="extraction_x", source="a.mid", parser="P", parser_version="1"
            )

        assert str(refusal.value) == (
            "extraction 'extraction_x' names a source file and its digest together "
            "or neither"
        )

    def test_content_given_as_a_callable_is_read_once_when_needed(self) -> None:
        root = _quarters("clt_p", 8)
        calls: list[int] = []

        def content() -> pa.Table:
            calls.append(1)
            return pa.table({"id": ["p1"], "pitch": [62]})

        rows = _registry_rows(
            [
                {
                    "id": "p1",
                    "kind": "Note",
                    "timeline_id": "clt_p",
                    "start": 1,
                    "extraction_id": "extraction_p",
                }
            ],
            "quarters",
            "fraction",
        )

        root._registry.add_rows(rows, {"Note": content}, [_package_extraction()])
        assert calls == []
        assert root.events.table.column("pitch").to_pylist() == [62]
        assert root.events.table.column("pitch").to_pylist() == [62]
        assert calls == [1]


class TestOneRuleJoinsColumns:
    def test_a_struct_widens_in_concat_and_in_content(self) -> None:
        root = _quarters("clt_root", 8)
        root.add_events(
            [{"id": "r1", "event_type": "Beat", "start": 0, "tag": {"midi": 60}}]
        )
        a = _quarters("clt_a", 4)
        a.add_events(
            [
                {
                    "id": "a1",
                    "event_type": "Beat",
                    "start": 0,
                    "tag": {"midi": 60, "name": "C4"},
                }
            ]
        )
        root.add_child(a, offset=0)
        widened = pa.struct(
            [pa.field("midi", pa.int64()), pa.field("name", pa.string())]
        )

        joined = root.events.concat(a.events)
        content = root.registry.content("Beat")

        assert joined.table.schema.field("tag").type == widened
        assert joined.table.column("tag").to_pylist() == [
            {"midi": 60, "name": None},
            None,
            {"midi": 60, "name": "C4"},
        ]
        assert content.schema.field("tag").type == widened
        assert content.column("tag").to_pylist() == [
            {"midi": 60, "name": None},
            {"midi": 60, "name": "C4"},
        ]
        assert root.events.table.schema.field("tag").type == pa.struct(
            [pa.field("midi", pa.int64())]
        )

    def test_a_dictionary_column_beside_a_plain_one_reads_as_its_values(self) -> None:
        def tagged(event_id: str, tag: str) -> EventData:
            return EventData.from_dicts(
                [{"id": event_id, "event_type": "Beat", "start": 0, "tag": tag}],
                unit=TimeUnit.quarters,
                number_type=NumberType.fraction,
            )

        data = tagged("d1", "x")
        index = data.table.schema.get_field_index("tag")
        encoded = data._derived(
            data.table.set_column(
                index, "tag", pa.chunked_array([pa.array(["x"]).dictionary_encode()])
            )
        )

        joined = encoded.concat(tagged("e1", "y"))

        assert joined.table.schema.field("tag").type == pa.string()
        assert joined.table.column("tag").to_pylist() == ["x", "y"]

    def test_an_int_beside_a_string_has_no_one_table(self) -> None:
        root = _quarters("clt_root", 8)
        root.add_events(
            [{"id": "r1", "event_type": "Beat", "start": 0, "velocity": 80}]
        )
        a = _quarters("clt_a", 4)
        a.add_events([{"id": "a1", "event_type": "Beat", "start": 0, "velocity": "ff"}])
        root.add_child(a, offset=0)
        message = (
            "the column 'velocity' holds values of type int64 in one table and of "
            "type string in another, which no one type reads; one table holds one "
            "type per column"
        )

        with pytest.raises(ValueError) as refusal:
            root.events.concat(a.events)
        assert str(refusal.value) == message
        with pytest.raises(ValueError) as refusal:
            root.registry.content("Beat")
        assert str(refusal.value) == message
        assert a.events.table.column("velocity").to_pylist() == ["ff"]


class TestRefusals:
    def test_a_child_of_another_number_type(self) -> None:
        root, *_ = _family()
        floats = _quarters("clt_f", 2, "float")

        with pytest.raises(ValueError) as refusal:
            root.add_child(floats, offset=6)

        assert str(refusal.value) == (
            "Child 'clt_f' has the number type 'float', its parent 'clt_root' has "
            "'fraction': a child has its parent's unit and number type. Events of "
            "another number type reach a timeline only as Projections through a "
            "C-Map (add_projections)."
        )
        assert list(root._children) == ["clt_a", "clt_b"]

    def test_moving_a_child_holding_events(self) -> None:
        root, _, _, b = _family()
        other = _quarters("clt_o", 8)

        with pytest.raises(ValueError) as refusal:
            other.add_child(b, offset=0)

        assert str(refusal.value) == (
            "Timeline 'clt_b' is a child of 'clt_root' already: a timeline has one "
            "parent (it may be a member of many groups)"
        )
        assert other.n_children == 0
        assert _starts(root.registry.filter(within="clt_b").table) == [Fraction(41, 10)]

    @pytest.mark.parametrize("registered", [True, False])
    def test_attaching_a_family_read_from_a_package(self, registered: bool) -> None:
        read = _quarters("clt_p", 4)
        rows = _registry_rows(
            [
                {
                    "id": "p1",
                    "kind": "Note",
                    "timeline_id": "clt_p",
                    "start": 1,
                    "extraction_id": "extraction_p",
                }
            ],
            "quarters",
            "fraction",
        )
        read._registry.add_rows(
            rows, {}, [_package_extraction()], registered=registered
        )
        parent = _quarters("clt_parent", 8)

        if not registered:
            parent.add_child(read, offset=2)
            assert _starts(parent.registry.filter(kinds="Note").table) == [3]
            return
        with pytest.raises(ValueError) as refusal:
            parent.add_child(read, offset=2)
        assert str(refusal.value) == (
            "timeline 'clt_p' holds events read from a package, positioned on its "
            "own axis: it cannot become a child (that would move every position); "
            "relate the two through a group or MatchClaims"
        )
        assert parent.n_children == 0
        assert read._parent is None

    def test_rows_on_another_axis(self) -> None:
        _, a, _, _ = _family()
        floats = EventData.from_dicts(
            [{"id": "f1", "event_type": "Beat", "start": 0.5}],
            TimeUnit.quarters,
            NumberType.float,
        )

        with pytest.raises(ValueError) as refusal:
            a._registry.add("clt_a", floats)

        assert str(refusal.value) == (
            "the rows of timeline 'clt_a' lie on its axis, quarters in fraction; "
            "these lie on quarters in float"
        )
        assert _ids(a.events.table) == ["a1", "a2", "clt_g"]

    @pytest.mark.parametrize(
        ("row", "error", "message"),
        [
            (
                {"timeline_id": "clt_nowhere"},
                KeyError,
                "A registry row names the timeline 'clt_nowhere', which this family "
                "does not hold. Available timelines: ['clt_root', 'clt_a', "
                "'clt_g', 'clt_b']",
            ),
            (
                {"extraction_id": "extraction_unknown"},
                ValueError,
                "a registry row names the extraction 'extraction_unknown', which is "
                "given with none of the extractions",
            ),
            (
                {"id": "a1"},
                ValueError,
                "Event id 'a1' is already held by an event of timeline 'clt_a', so "
                f"it cannot join timeline 'clt_b': event ids are unique {SAME}",
            ),
        ],
        ids=["unknown-timeline", "unknown-extraction", "held-id"],
    )
    def test_registry_rows_refused(
        self, row: dict[str, Any], error: type[Exception], message: str
    ) -> None:
        root, *_ = _family()
        before = root.registry.table
        rows = _registry_rows(
            [
                {
                    "id": "p1",
                    "kind": "Note",
                    "timeline_id": "clt_b",
                    "start": 5,
                    "extraction_id": "extraction_p",
                    **row,
                }
            ],
            "quarters",
            "fraction",
        )

        with pytest.raises(error) as refusal:
            root._registry.add_rows(rows, {}, [_package_extraction()])

        text = refusal.value.args[0] if error is KeyError else str(refusal.value)
        assert text == message
        assert root.registry.table.equals(before)
        assert list(root._registry._family.extractions) == [
            root._registry._family.tool.id
        ]

    def test_an_extraction_is_held_only_with_its_rows(self) -> None:
        root = _quarters("clt_root", 8)
        root.add_events(
            [{"id": "r1", "event_type": "Beat", "start": 0, "velocity": 80}]
        )
        held = dict(root._registry._family.extractions)
        data = EventData.from_dicts(
            [{"id": "v2", "event_type": "Beat", "start": 1, "velocity": "ff"}],
            TimeUnit.quarters,
            NumberType.fraction,
        )

        with pytest.raises(ValueError):
            root._registry.add("clt_root", data, extraction=_package_extraction())

        assert root._registry._family.extractions == held
        assert _ids(root.events.table) == ["r1"]

    def test_a_child_holding_an_extraction_of_other_content(self) -> None:
        def run(parser: str) -> Extraction:
            return Extraction(
                id="extraction_x", parser=parser, parser_version="1", date="2026"
            )

        root = _quarters("clt_root", 8)
        root._registry.add(
            "clt_root",
            EventData.from_dicts(
                [{"id": "r1", "event_type": "Beat", "start": 0}],
                TimeUnit.quarters,
                NumberType.fraction,
            ),
            extraction=run("A"),
        )
        child = _quarters("clt_c", 2)
        child._registry.add(
            "clt_c",
            EventData.from_dicts(
                [{"id": "c1", "event_type": "Beat", "start": 0}],
                TimeUnit.quarters,
                NumberType.fraction,
            ),
            extraction=run("B"),
        )
        before = root._registry._family.table()

        with pytest.raises(ValueError) as refusal:
            root.add_child(child, offset=1)

        assert str(refusal.value) == (
            "extraction 'extraction_x' is held already, of other content"
        )
        assert root._registry._family.table().equals(before)
        assert root._registry._family.extractions["extraction_x"].parser == "A"
        assert child._parent is None
        assert _ids(child.registry.table) == ["c1"]

    def test_registry_rows_on_another_axis(self) -> None:
        root, *_ = _family()
        rows = _registry_rows(
            [
                {
                    "id": "p1",
                    "kind": "Note",
                    "timeline_id": "clt_b",
                    "start": 5,
                    "extraction_id": "extraction_p",
                }
            ],
            "quarters",
            "float",
        )

        with pytest.raises(ValueError) as refusal:
            root._registry.add_rows(rows, {}, [_package_extraction()])

        assert str(refusal.value) == (
            "event 'p1' lies on quarters in float, but the root of its timeline "
            "'clt_b' on quarters in fraction"
        )

    def test_rows_taken_in_through_a_view(self) -> None:
        root, *_ = _family()

        with pytest.raises(TypeError) as refusal:
            root.registry.add("clt_root", root.events)

        assert str(refusal.value) == (
            "rows join the registry of one family (a timeline's), not a view of it"
        )

    def test_an_id_of_the_root_added_to_a_child(self) -> None:
        _, a, _, _ = _family()

        with pytest.raises(ValueError) as refusal:
            a.add_events([{"id": "r1", "event_type": "Beat", "start": 1}])

        assert str(refusal.value) == (
            "Event id 'r1' is already held by an event of timeline 'clt_root', so "
            f"it cannot join timeline 'clt_a': event ids are unique {SAME}"
        )
        assert _ids(a.events.table) == ["a1", "a2", "clt_g"]

    def test_an_id_given_twice(self) -> None:
        _, a, _, _ = _family()

        with pytest.raises(ValueError) as refusal:
            a.add_events(
                [
                    {"id": "x", "event_type": "Beat", "start": 1},
                    {"id": "x", "event_type": "Beat", "start": 2},
                ]
            )

        assert str(refusal.value) == (
            f"Event id 'x' is given to 2 events joining timeline 'clt_a'; event ids "
            f"are unique {SAME}"
        )

    @pytest.mark.parametrize("held", ["r1", "clt_c"])
    def test_a_refused_child_changes_nothing(self, held: str) -> None:
        root, *_ = _family()
        before = root.registry.table
        c = _quarters("clt_c", 2)
        c.add_events([{"id": held, "event_type": "Beat", "start": 0}])
        own = c._registry

        with pytest.raises(ValueError):
            root.add_child(c, offset=6)

        assert list(root._children) == ["clt_a", "clt_b"]
        assert root.registry.table.equals(before)
        assert c._parent is None
        assert c._registry is own
        assert _ids(c.registry.table) == [held]

    def test_a_family_holds_the_timelines_it_holds(self) -> None:
        registry = EventRegistry("clt_1", ContinuousLogicalTimeline(length=1).events)

        with pytest.raises(KeyError) as refusal:
            registry._family.events_of("clt_2")

        assert refusal.value.args[0] == (
            "No timeline 'clt_2' in this family. Available timelines: ['clt_1']"
        )

    def test_events_are_read_only(self) -> None:
        _, a, _, _ = _family()

        with pytest.raises(AttributeError):
            setattr(a, "_events", a.events)
        with pytest.raises(AttributeError):
            setattr(a, "events", a.events)


class TestRename:
    def test_a_new_id_renames_the_rows(self) -> None:
        root, a, _, _ = _family()

        a._set_id("clt_x")

        table = root.registry.filter(within="clt_x").table
        assert _ids(table) == ["a1", "a2"]
        family = root._registry._family.table()
        assert family.filter(
            pa.compute.equal(family.column("timeline_id"), "clt_x")
        ).column("id").to_pylist() == ["a1", "a2", "clt_g"]
        assert root._registry._family.holder_of("a1") == "clt_x"
        assert _ids(a.events.table) == ["a1", "a2", "clt_g"]

    @pytest.mark.parametrize("new_id", ["clt_b", "clt_root", "r1"])
    def test_a_refused_id_changes_nothing(self, new_id: str) -> None:
        root, a, _, _ = _family()
        before = root.registry.table

        with pytest.raises(ValueError):
            a._set_id(new_id)

        assert list(root._children) == ["clt_a", "clt_b"]
        assert [offset.value for offset in root._child_offsets.values()] == [2, 4]
        assert root.registry.table.equals(before)
        assert a.id == "clt_a"
