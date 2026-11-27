"""The event registry: root positions, local events, views, updates, refusals."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

import timetoalign
from timetoalign.alignment import AlignmentBundle
from timetoalign.core import Coordinate, NumberType, TimeUnit
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


class TestRootPositions:
    def test_the_root_holds_every_row_on_its_axis(self) -> None:
        root, *_ = _family()
        table = root.registry.table

        assert table.schema.names == REGISTRY_SCHEMA.names
        assert _ids(table) == ["r1", "clt_a", "clt_b", "a1", "a2", "clt_g", "g1", "b1"]
        assert table.column("kind").to_pylist() == [
            "Beat",
            "Segment",
            "Segment",
            "Beat",
            "Beat",
            "Segment",
            "Note",
            "Beat",
        ]
        assert table.column("timeline_id").to_pylist() == [
            "clt_root",
            "clt_root",
            "clt_root",
            "clt_a",
            "clt_a",
            "clt_a",
            "clt_g",
            "clt_b",
        ]
        assert set(table.column("unit").to_pylist()) == {"quarters"}
        assert set(table.column("number_type").to_pylist()) == {"fraction"}
        assert _starts(table) == [
            Fraction(1, 2),
            2,
            4,
            Fraction(7, 3),
            2,
            3,
            Fraction(13, 4),
            Fraction(41, 10),
        ]
        assert _exact(table.column("duration").to_pylist()) == [
            None,
            4,
            4,
            None,
            None,
            2,
            Fraction(1, 2),
            None,
        ]

    def test_every_row_names_the_familys_tool_extraction(self) -> None:
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

        assert _starts(root.registry.table)[6] == Fraction(13, 4)
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
        assert _ids(view.table) == ["a1", "a2", "clt_g", "g1"]

        a.add_events([{"id": "a3", "event_type": "Beat", "start": 2}])

        assert _ids(view.table) == ["a1", "a2", "clt_g", "a3", "g1"]

    def test_filters(self) -> None:
        root, *_ = _family()
        registry = root.registry

        beats = registry.filter(kinds="Beat")
        assert _ids(beats.table) == ["r1", "a1", "a2", "b1"]
        assert _ids(beats.filter(timelines="clt_a").table) == ["a1", "a2"]
        assert _ids(registry.filter(min_coord=2, max_coord=4).table) == [
            "clt_a",
            "a1",
            "a2",
            "clt_g",
            "g1",
        ]
        seconds = registry.filter(min_coord=Coordinate(1.0, TimeUnit.seconds))
        with pytest.raises(ValueError) as refusal:
            seconds.table
        assert str(refusal.value) == (
            "a bound in seconds cannot be compared with the rows of timeline "
            "'clt_root''s family, on quarters"
        )

    def test_counts(self) -> None:
        root, *_ = _family()
        registry = root.registry

        assert registry.counts().to_pylist() == [
            {"kind": "Beat", "count": 4},
            {"kind": "Note", "count": 1},
            {"kind": "Segment", "count": 3},
        ]
        assert len(registry) == 8
        assert registry.counts("timeline_id").to_pylist() == [
            {"timeline_id": "clt_a", "count": 3},
            {"timeline_id": "clt_b", "count": 1},
            {"timeline_id": "clt_g", "count": 1},
            {"timeline_id": "clt_root", "count": 3},
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

        assert _ids(joined.table) == ["a1", "a2", "clt_g", "g1"]

    def test_a_bundles_registry_holds_every_family(self) -> None:
        root, *_ = _family()
        audio = ContinuousPhysicalTimeline(length=10.0, uid="audio")
        audio.add_events([{"id": "o1", "event_type": "Onset", "start": 1.0}])
        bundle = AlignmentBundle(id="b")
        bundle.add_timeline(root)
        bundle.add_timeline(audio)

        table = bundle.registry.table

        assert _ids(table) == [*_ids(root.registry.table), "o1"]
        assert table.column("unit").to_pylist()[-1] == "seconds"
        assert table.column("number_type").to_pylist()[-1] == "float"

    def test_content(self) -> None:
        root, *_ = _family()

        assert root.registry.content("Note").to_pylist() == [{"id": "g1", "pitch": 60}]
        assert root.registry.content("Beat", ids=["a1"]).to_pylist() == [{"id": "a1"}]


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
        assert _ids(view.table) == ["a1", "clt_g", "g1"]
        assert _starts(view.table) == [Fraction(1, 3), 1, Fraction(5, 4)]

        root.add_child(a, offset=2)

        assert _starts(view.table) == [Fraction(7, 3), 3, Fraction(13, 4)]
        (tool,) = root.registry.extractions
        assert tool.id != child_tool.id
        assert set(root.registry.table.column("extraction_id").to_pylist()) == {tool.id}
        a.add_events([{"id": "a3", "event_type": "Beat", "start": 2}])
        assert _starts(root.registry.filter(timelines="clt_a", kinds="Beat").table) == [
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
        assert _starts(root.registry.filter(timelines="clt_b").table) == [
            Fraction(41, 10)
        ]

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
            "attach its copy with new IDs, or relate the two through a group or "
            "MatchClaims"
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

        table = root.registry.filter(timelines="clt_x").table
        assert _ids(table) == ["a1", "a2", "clt_g"]
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
