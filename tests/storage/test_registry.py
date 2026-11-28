"""The event registry: root positions, local events, views, updates, refusals."""

from __future__ import annotations

import io
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from timetoshare import tables as format_tables

import timetoalign
from timetoalign.alignment import AlignmentBundle
from timetoalign.core import (
    BoundingBox,
    Box,
    Coordinate,
    IdCoordinate,
    NumberType,
    TimeUnit,
)
from timetoalign.loader import _sources
from timetoalign.storage import EventData, EventRegistry, Parse
from timetoalign.storage.registry import (
    PARSE_SCHEMA,
    REGISTRY_SCHEMA,
    member_uri,
    parse_table,
)
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
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
                    "parse_id",
                )
            ]
        ),
    )


def _package_parse() -> Parse:
    return Parse(
        id="parse_p",
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

        assert (
            table.schema.names
            == REGISTRY_SCHEMA.names
            == [
                "id",
                "kind",
                "location",
                "timeline_id",
                "unit",
                "number_type",
                "start",
                "duration",
                "parse_id",
                "locator",
                "vanished_in",
                "projection_of",
            ]
        )
        assert REGISTRY_SCHEMA.names == format_tables.REGISTRY_SCHEMA.names
        assert table.column("projection_of").null_count == 5
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
        assert segments.column("parse_id").to_pylist() == [None, None, None]

    def test_every_event_names_the_familys_tool_parse(self) -> None:
        root, *_ = _family()
        table = root.registry.table

        (tool,) = root.registry.parses
        assert (tool.parser, tool.parser_version, tool.source) == (
            "timetoalign",
            timetoalign.__version__,
            None,
        )
        assert set(table.column("parse_id").to_pylist()) == {tool.id}
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
                    "parse_id": "parse_p",
                }
            ],
            "quarters",
            "fraction",
        )
        content = pa.table({"id": ["p1"], "pitch": [62]})

        root._registry.add_rows(rows, {"Note": content}, [_package_parse()])

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

    def test_a_name_is_content_never_a_registry_column(self) -> None:
        root = _quarters("clt_n", 4)
        root.add_events(
            [{"id": "n1", "event_type": "Note", "start": Fraction(0), "name": "C4"}]
        )

        assert root.registry.table.column_names == REGISTRY_SCHEMA.names
        assert "name" not in root.registry.table.column_names
        assert root.registry.content("Note").to_pylist() == [{"id": "n1", "name": "C4"}]

    @pytest.mark.parametrize(
        ("content", "name"),
        [
            (pa.table({"id": ["p1"], "name": ["D4"], "pitch": [62]}), "D4"),
            (pa.table({"id": ["p1"], "pitch": [62]}), None),
        ],
        ids=["named", "unnamed"],
    )
    def test_a_name_read_from_a_package_is_the_contents(
        self, content: pa.Table, name: str | None
    ) -> None:
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
                    "parse_id": "parse_p",
                }
            ],
            "quarters",
            "fraction",
        )

        root._registry.add_rows(rows, {"Note": content}, [_package_parse()])

        (event,) = child.events.table.to_pylist()
        assert (event["id"], event["name"], event["pitch"]) == ("p1", name, 62)

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
                    "parse_id": "parse_p",
                }
            ],
            "ticks",
            "int",
        )

        root._registry.add_rows(rows, {}, [_package_parse()])

        assert _starts(child.events.table) == [3]
        assert child.events.table.column("temporal_type").to_pylist() == ["instant"]

    def test_rows_read_from_a_package_keep_their_order(self, tmp_path: Path) -> None:
        path = tmp_path / "score.txt"
        path.write_bytes(b"abc")
        first = Parse(
            **{
                **Parse.of_source(path, parser="P", parser_version="1").__dict__,
                "id": "parse_first",
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
            rows.schema.get_field_index("parse_id"),
            "parse_id",
            pa.array(["parse_first"] * 3),
        ).append_column("locator", pa.array(["L1", "L2", "L3"]))
        contents = {
            "Note": pa.table({"id": ["n1", "n2"], "pitch": [60, 64]}),
            "Beat": pa.table({"id": ["b1"], "strength": [1]}),
        }

        timeline._registry.add_rows(rows, contents, [first], registered=False)

        assert _ids(timeline.events.table) == ["n1", "b1", "n2"]
        assert _ids(timeline.registry.table) == ["n1", "b1", "n2"]
        second = Parse(**{**first.__dict__, "id": "parse_second"})
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

        timeline.registry.update(again, second, timeline_id="clt_t")

        assert _ids(timeline.events.table) == ["n1", "b1", "n2"]
        assert _ids(timeline.registry.table) == ["n1", "b1", "n2"]
        assert timeline.events.table.column("pitch").to_pylist() == [61, None, 65]
        assert timeline.registry.table.column("vanished_in").to_pylist() == [
            None,
            "parse_second",
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
                    "parse_id": "parse_p",
                }
            ],
            "seconds",
            "float",
        )

        root._registry.add_rows(rows, {}, [_package_parse()])

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
            "'kind', 'location', 'timeline_id', 'unit', 'number_type', "
            "'parse_id', 'locator', 'vanished_in']"
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

    def test_a_segment_names_no_tool_parse(self, tmp_path: Path) -> None:
        path = tmp_path / "score.txt"
        path.write_bytes(b"abc")
        parse = Parse.of_source(path, parser="P", parser_version="1")
        root = _quarters("clt_r", 8)
        child = _quarters("clt_c", 4)
        root.add_child(child, offset=1)
        data = EventData.from_dicts(
            [{"id": "n1", "event_type": "Note", "start": 0, "locator": "m0"}],
            TimeUnit.quarters,
            NumberType.fraction,
        )

        child._registry.add("clt_c", data, parse=parse)

        assert root.registry.parses == [parse]
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
        (child_tool,) = view.parses
        assert _ids(view.table) == ["a1", "g1"]
        assert _starts(view.table) == [Fraction(1, 3), Fraction(5, 4)]

        root.add_child(a, offset=2)

        assert _starts(view.table) == [Fraction(7, 3), Fraction(13, 4)]
        (tool,) = root.registry.parses
        assert tool.id != child_tool.id
        assert set(root.registry.table.column("parse_id").to_pylist()) == {tool.id}
        a.add_events([{"id": "a3", "event_type": "Beat", "start": 2}])
        assert _starts(root.registry.filter(within="clt_a", kinds="Beat").table) == [
            Fraction(7, 3),
            4,
        ]

    def test_a_root_of_no_events_adopts_the_childs_tool_parse(self) -> None:
        root = _quarters("clt_root", 8)
        a = _quarters("clt_a", 4)
        a.add_events([{"id": "a1", "event_type": "Beat", "start": 0}])
        (child_tool,) = a.registry.parses

        root.add_child(a, offset=2)

        assert root.registry.parses == [child_tool]


class TestUpdate:
    @staticmethod
    def _source(tmp_path: Path) -> Path:
        path = tmp_path / "score.txt"
        path.write_bytes(b"abc")
        return path

    @staticmethod
    def _run(
        path: Path, rows: list[tuple[str, str, Any, int]]
    ) -> tuple[Parse, EventData]:
        parse = Parse.of_source(path, parser="TestParser", parser_version="1")
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
        return parse, data

    def test_rows_matched_by_locator_keep_their_ids(self, tmp_path: Path) -> None:
        path = self._source(tmp_path)
        timeline = _quarters("clt_u", 8)
        first, data = self._run(
            path, [("n1", "m0", 0, 60), ("n2", "m1", 1, 62), ("n3", "m2", 2, 64)]
        )
        timeline._registry.add("clt_u", data, parse=first)
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
        second = Parse(**{**second.__dict__, "id": "parse_second"})
        timeline.registry.update(again, second, timeline_id="clt_u")

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
        assert table.column("parse_id").to_pylist() == [
            "parse_second",
            first.id,
            "parse_second",
            tool.id,
            "parse_second",
        ]
        assert table.column("vanished_in").to_pylist() == [
            None,
            "parse_second",
            None,
            None,
            None,
        ]
        assert _ids(timeline.registry.filter(vanished=True).table) == ["n2"]

    def test_an_update_without_locators_is_refused(self, tmp_path: Path) -> None:
        path = self._source(tmp_path)
        timeline = _quarters("clt_u", 8)
        parse = Parse.of_source(path, parser="TestParser", parser_version="1")
        data = EventData.from_dicts(
            [{"id": "n1", "event_type": "Note", "start": 0}],
            TimeUnit.quarters,
            NumberType.fraction,
        )

        with pytest.raises(ValueError) as refusal:
            timeline.registry.update(data, parse, timeline_id="clt_u")

        assert str(refusal.value) == (
            "an update matches rows by their locator: the rows give none (a column "
            "'locator')"
        )
        assert len(timeline.events) == 0

    def _first(self, tmp_path: Path) -> tuple[Timeline, Parse]:
        """A timeline holding ``n1``, ``n2``, ``n3`` of ``parse_first``."""
        first, data = self._run(
            self._source(tmp_path),
            [("n1", "m0", 0, 60), ("n2", "m1", 1, 62), ("n3", "m2", 2, 64)],
        )
        first = Parse(**{**first.__dict__, "id": "parse_first"})
        timeline = _quarters("clt_u", 8)
        timeline._registry.add("clt_u", data, parse=first)
        return timeline, first

    def test_a_locator_given_twice_in_a_run_is_refused(self, tmp_path: Path) -> None:
        timeline, first = self._first(tmp_path)
        before = timeline.registry.table
        second, again = self._run(
            self._source(tmp_path), [("x1", "m0", 0, 61), ("x2", "m0", 1, 62)]
        )
        second = Parse(**{**second.__dict__, "id": "parse_second"})

        with pytest.raises(ValueError) as refusal:
            timeline.registry.update(again, second, timeline_id="clt_u")

        assert str(refusal.value) == (
            "locator 'm0' is given to 2 events of parse 'parse_second'; "
            "a locator names one event of its source"
        )
        assert timeline.registry.table.equals(before)
        assert _ids(timeline.events.table) == ["n1", "n2", "n3"]

    def test_a_locator_held_by_an_event_of_its_parse_is_refused(
        self, tmp_path: Path
    ) -> None:
        timeline, first = self._first(tmp_path)
        before = timeline.registry.table
        message = (
            "locator 'm0' is given to 2 events of parse 'parse_first'; "
            "a locator names one event of its source"
        )
        _, again = self._run(self._source(tmp_path), [("y1", "m0", 3, 61)])

        with pytest.raises(ValueError) as refusal:
            timeline._registry.add("clt_u", again, parse=first)
        assert str(refusal.value) == message
        rows = _registry_rows(
            [
                {
                    "id": "y1",
                    "kind": "Note",
                    "timeline_id": "clt_u",
                    "start": 3,
                    "parse_id": "parse_first",
                }
            ],
            "quarters",
            "fraction",
        ).append_column("locator", pa.array(["m0"]))
        with pytest.raises(ValueError) as refusal:
            timeline._registry.add_rows(rows, registered=False)
        assert str(refusal.value) == message
        assert timeline.registry.table.equals(before)

        timeline._registry.add("clt_u", again, parse=_package_parse())

        assert _ids(timeline.registry.table) == ["n1", "n2", "n3", "y1"]

    def _under_a_root(self, tmp_path: Path) -> tuple[Timeline, Timeline, Parse]:
        """``clt_root`` with the child ``clt_u`` at 2 holding the first run's rows."""
        timeline, first = self._first(tmp_path)
        root = _quarters("clt_root", 16)
        root.add_child(timeline, offset=2)
        return root, timeline, first

    def _second(
        self, tmp_path: Path, rows: list[tuple[str, str, Any, int]]
    ) -> tuple[Parse, EventData]:
        second, again = self._run(self._source(tmp_path), rows)
        return Parse(**{**second.__dict__, "id": "parse_second"}), again

    def test_an_update_through_a_view_finds_its_timeline(self, tmp_path: Path) -> None:
        root, timeline, first = self._under_a_root(tmp_path)
        second, again = self._second(
            tmp_path,
            [
                ("x0", "m0", 0, 61),
                ("x2", "m2", Fraction(5, 2), 64),
                ("x3", "m3", 3, 65),
            ],
        )
        before = root.registry.table

        assert root.registry.update_target(again, second) == "clt_u"
        assert root.registry.table.equals(before)
        assert root.registry.update(again, second) == "clt_u"

        assert _ids(timeline.registry.table) == ["n1", "n2", "n3", "x3"]
        assert timeline.events.table.column("pitch").to_pylist() == [61, 62, 64, 65]
        assert _starts(timeline.events.table) == [0, 1, Fraction(5, 2), 3]
        assert _starts(timeline.registry.table) == [2, 3, Fraction(9, 2), 5]
        assert timeline.registry.table.column("vanished_in").to_pylist() == [
            None,
            "parse_second",
            None,
            None,
        ]

    def test_the_timeline_holding_most_matched_rows_is_updated(
        self, tmp_path: Path
    ) -> None:
        path = self._source(tmp_path)
        first, held_u = self._run(path, [("u2", "m2", 0, 60)])
        first = Parse(**{**first.__dict__, "id": "parse_first"})
        _, held_v = self._run(path, [("v0", "m0", 0, 60), ("v1", "m1", 1, 62)])
        root = _quarters("clt_root", 16)
        u = _quarters("clt_u", 4)
        v = _quarters("clt_v", 4)
        root.add_child(u, offset=0)
        root.add_child(v, offset=8)
        u._registry.add("clt_u", held_u, parse=first)
        v._registry.add("clt_v", held_v, parse=first)
        second, again = self._second(
            tmp_path, [("x0", "m0", 0, 61), ("x1", "m1", 1, 63), ("x2", "m2", 2, 65)]
        )

        assert root.registry.update(again, second) == "clt_v"

        assert _ids(v.registry.table) == ["v0", "v1", "x2"]
        assert _ids(u.registry.table) == ["u2"]

    def test_rows_of_another_source_fall_back_on_the_id(self, tmp_path: Path) -> None:
        root, timeline, _ = self._under_a_root(tmp_path)
        other = tmp_path / "other.txt"
        other.write_bytes(b"xyz")
        parse, again = self._run(other, [("z0", "m0", 4, 70)])

        assert root.registry.update_target(again, parse) is None
        before = root.registry.table
        with pytest.raises(ValueError) as refusal:
            root.registry.update(again, parse)
        assert str(refusal.value) == (
            f"the rows of parse {parse.id!r} ({parse.source}) match no "
            "timeline of the registry, by locator or by ID (None); nothing is updated"
        )
        assert root.registry.table.equals(before)

        assert root.registry.update(again, parse, timeline_id="clt_u") == "clt_u"

        assert _ids(timeline.registry.table) == ["n1", "n2", "n3", "z0"]
        assert timeline.registry.table.column("vanished_in").to_pylist() == [None] * 4


class TestWritten:
    @staticmethod
    def _with_a_segment_event() -> tuple[Timeline, Timeline, Timeline, Timeline]:
        root, a, g, b = _family()
        b.add_events([{"id": "s1", "event_type": "Segment", "start": 1}])
        return root, a, g, b

    def test_rows_contents_parses_families(self) -> None:
        root, a, _, _ = self._with_a_segment_event()

        written = root.registry.written()

        tool = root._registry._family.tool
        assert tool is not None
        assert _ids(written.rows) == ["r1", "a1", "a2", "g1", "b1", "s1"]
        assert written.rows.column("parse_id").to_pylist() == [tool.id] * 6
        assert _starts(written.rows) == [
            Fraction(1, 2),
            Fraction(7, 3),
            2,
            Fraction(13, 4),
            Fraction(41, 10),
            5,
        ]
        assert list(written.contents) == [
            ("clt_root", "Beat"),
            ("clt_a", "Beat"),
            ("clt_g", "Note"),
            ("clt_b", "Beat"),
            ("clt_b", "Segment"),
        ]
        assert written.contents[("clt_g", "Note")].to_pylist() == [
            {"id": "g1", "pitch": 60}
        ]
        assert written.contents[("clt_a", "Beat")].to_pylist() == [
            {"id": "a1"},
            {"id": "a2"},
        ]
        assert written.parses == (tool,)
        (family,) = written.families
        assert (family.root_id, family.unit, family.number_type, family.timelines) == (
            "clt_root",
            TimeUnit.quarters,
            NumberType.fraction,
            ("clt_root", "clt_a", "clt_g", "clt_b"),
        )
        assert written.family_of("clt_g") == family
        with pytest.raises(KeyError) as refusal:
            written.family_of("clt_nowhere")
        assert str(refusal.value) == (
            "\"No family holds the timeline 'clt_nowhere'. Available timelines: "
            "['clt_root', 'clt_a', 'clt_g', 'clt_b']\""
        )

    def test_kinds_and_views(self) -> None:
        root, a, _, _ = self._with_a_segment_event()

        notes = root.registry.written("Note")

        assert _ids(notes.rows) == ["g1"]
        assert list(notes.contents) == [("clt_g", "Note")]
        assert a.registry.written().rows.equals(root.registry.written().rows)

    def test_a_bundle_writes_each_family(self) -> None:
        root, *_ = _family()
        bundle = _bundle(root)

        written = bundle.registry.written()

        assert _ids(written.rows) == ["r1", "a1", "a2", "g1", "b1", "o1"]
        assert [
            (family.root_id, family.unit, family.number_type)
            for family in written.families
        ] == [
            ("clt_root", TimeUnit.quarters, NumberType.fraction),
            ("audio", TimeUnit.seconds, NumberType.float),
        ]
        assert written.family_of("audio").timelines == ("audio",)


class TestAddRows:
    @staticmethod
    def _timelines() -> tuple[Timeline, Timeline, Timeline, AlignmentBundle]:
        root = _quarters("clt_p", 4)
        child = _quarters("clt_pc", 2)
        root.add_child(child, offset=1)
        seconds = ContinuousPhysicalTimeline(length=1.0, uid="cpt_s", locked=False)
        bundle = AlignmentBundle(id="b")
        bundle.add_timeline(root)
        bundle.add_timeline(seconds)
        return root, child, seconds, bundle

    @staticmethod
    def _p1(start: Any = Fraction(5, 2)) -> pa.Table:
        return _registry_rows(
            [
                {
                    "id": "p1",
                    "kind": "Note",
                    "timeline_id": "clt_pc",
                    "start": start,
                    "duration": Fraction(1, 2),
                    "parse_id": "parse_p",
                }
            ],
            "quarters",
            "fraction",
        )

    @staticmethod
    def _o1(parse_id: str = "parse_p") -> pa.Table:
        return _registry_rows(
            [
                {
                    "id": "o1",
                    "kind": "Onset",
                    "timeline_id": "cpt_s",
                    "start": 2.5,
                    "parse_id": parse_id,
                }
            ],
            "seconds",
            "float",
        )

    def test_rows_join_their_families_through_a_view(self) -> None:
        _, child, seconds, bundle = self._timelines()
        rows = pa.concat_tables([self._p1(), self._o1()])

        bundle.registry.add_rows(
            rows,
            {"Note": pa.table({"id": ["p1"], "pitch": [62]})},
            [_package_parse()],
        )

        assert _ids(child.events.table) == ["p1"]
        assert _starts(child.events.table) == [Fraction(3, 2)]
        assert child.events.table.column("pitch").to_pylist() == [62]
        assert _values(seconds.events.table) == [2.5]
        assert child.length is not None and child.length.value == 2
        assert seconds.length is not None and seconds.length.value == 2.5
        assert child.registry.parses == [_package_parse()]
        assert seconds.registry.parses == [_package_parse()]

    def test_a_locked_timeline_refuses_a_row_past_its_length(self) -> None:
        locked = ContinuousLogicalTimeline(
            length=2, unit="quarters", number_type="fraction", uid="clt_l", locked=True
        )
        rows = self._p1()
        rows = rows.set_column(
            rows.schema.get_field_index("timeline_id"),
            "timeline_id",
            pa.array(["clt_l"]),
        )

        with pytest.raises(ValueError) as refusal:
            locked.registry.add_rows(rows, {}, [_package_parse()])

        assert str(refusal.value) == (
            "Adding content ending at 3 exceeds timeline length 2. Timeline is "
            "locked. Use allow_expansion=True to override."
        )
        assert len(locked.registry) == 0
        assert len(locked.events) == 0

    def test_a_row_stated_already_is_passed_over(self) -> None:
        root, child, _, _ = self._timelines()
        root.registry.add_rows(self._p1(), {}, [_package_parse()])
        before = root.registry.table

        root.registry.add_rows(self._p1(), {}, [_package_parse()])

        assert root.registry.table.equals(before)
        with pytest.raises(ValueError) as refusal:
            root.registry.add_rows(self._p1(start=3), {}, [_package_parse()])
        assert str(refusal.value) == (
            "Event id 'p1' is already held by an event of timeline 'clt_pc'"
        )
        assert root.registry.table.equals(before)

    def test_a_refusal_in_one_family_changes_none(self) -> None:
        _, _, seconds, bundle = self._timelines()
        rows = pa.concat_tables([self._o1(), self._p1()])
        rows = rows.set_column(
            rows.schema.get_field_index("parse_id"),
            "parse_id",
            pa.array(["parse_p", "parse_nowhere"]),
        )

        with pytest.raises(ValueError) as refusal:
            bundle.registry.add_rows(rows, {}, [_package_parse()])

        assert str(refusal.value) == (
            "a registry row names the parse 'parse_nowhere', which is "
            "given with none of the parses"
        )
        assert len(bundle.registry) == 0
        assert len(seconds.events) == 0

    def test_a_row_of_a_timeline_the_view_does_not_hold(self) -> None:
        _, _, _, bundle = self._timelines()
        rows = self._p1()
        rows = rows.set_column(
            rows.schema.get_field_index("timeline_id"),
            "timeline_id",
            pa.array(["clt_nowhere"]),
        )

        with pytest.raises(KeyError) as refusal:
            bundle.registry.add_rows(rows, {}, [_package_parse()])

        assert str(refusal.value) == (
            "\"A registry row names the timeline 'clt_nowhere', which no family of "
            "this registry holds. Available timelines: ['clt_p', 'clt_pc', 'cpt_s']\""
        )

    @staticmethod
    def _box_rows(start: int) -> tuple[pa.Table, dict[str, pa.Table], list[Parse]]:
        """The registry rows and Box content of ``rect_h2`` on ``seg_2``, its start *start*."""
        made = DiscreteGraphicalTimeline(
            length=867,
            uid="seg_2",
            meta={"box": {"x": 7, "y": 0, "width": 867, "height": 270}},
        )
        made.add_events(
            Box.from_shape(
                BoundingBox.from_xywh(385, 46, 139, 20, unit="pixels"),
                origin=7,
                id="rect_h2",
            )
        )
        rows = made.registry.table
        cells = pa.array(
            [{"value": float(start), "numerator": start, "denominator": 1}],
            REGISTRY_SCHEMA.field("start").type,
        )
        rows = rows.set_column(
            rows.schema.get_field_index("start"), REGISTRY_SCHEMA.field("start"), cells
        )
        return rows, {"Box": made.registry.content("Box")}, made.registry.parses

    def test_a_box_read_is_checked_against_its_shape(self) -> None:
        seg_2 = DiscreteGraphicalTimeline(
            length=867,
            uid="seg_2",
            meta={"box": {"x": 7, "y": 0, "width": 867, "height": 270}},
        )
        rows, contents, parses = self._box_rows(1)

        with pytest.raises(ValueError) as refusal:
            seg_2.registry.add_rows(rows, contents, parses)

        assert str(refusal.value) == (
            "Box 'rect_h2' starts at 1 on timeline 'seg_2', but its shape starts at "
            'x 385 and the timeline stands at x 7 of its image (meta["box"]): its '
            "start is 378."
        )
        assert len(seg_2.registry) == 0
        assert len(seg_2.events) == 0
        assert seg_2.registry.parses == []

        seg_2.registry.add_rows(*self._box_rows(378))

        assert _ids(seg_2.events.table) == ["rect_h2"]

    def test_a_registry_standing_alone_checks_no_box(self) -> None:
        rows, contents, parses = self._box_rows(1)
        registry = EventRegistry(
            "seg_2", EventData.empty(TimeUnit.pixels, NumberType.int)
        )

        registry.add_rows(rows, contents, parses)

        assert _ids(registry.table) == ["rect_h2"]


class TestAddEvents:
    @staticmethod
    def _parse(tmp_path: Path, version: str) -> Parse:
        path = tmp_path / "score.txt"
        path.write_bytes(b"abc")
        return Parse.of_source(path, parser="P", parser_version=version)

    @staticmethod
    def _rows(parse_id: str, *ids: str) -> list[dict[str, Any]]:
        return [
            {
                "id": event_id,
                "event_type": "Note",
                "start": index,
                "parse_id": parse_id,
                "locator": f"m{index}",
            }
            for index, event_id in enumerate(ids)
        ]

    def test_parses_are_held_with_the_rows(self, tmp_path: Path) -> None:
        e1 = self._parse(tmp_path, "1")
        e2 = self._parse(tmp_path, "2")
        timeline = _quarters("clt_e", 8)

        assert timeline.add_events(self._rows(e1.id, "n1", "n2"), parses=[e1, e2]) == [
            "n1",
            "n2",
        ]

        assert timeline.registry.parses == [e1]
        assert timeline.registry.table.column("parse_id").to_pylist() == [e1.id] * 2
        assert timeline.registry.table.column("locator").to_pylist() == ["m0", "m1"]
        assert "parse_id" not in timeline.events.table.column_names
        assert "locator" not in timeline.events.table.column_names
        assert e2.id not in timeline._registry._family.parses

    def test_refused_rows_hold_no_parse(self, tmp_path: Path) -> None:
        e1 = self._parse(tmp_path, "1")
        e3 = self._parse(tmp_path, "3")
        timeline = _quarters("clt_e", 8)
        timeline.add_events(self._rows(e1.id, "n1"), parses=[e1])

        with pytest.raises(ValueError):
            timeline.add_events(self._rows(e3.id, "n1"), parses=[e3])

        assert e3.id not in timeline._registry._family.parses
        assert timeline.registry.parses == [e1]

    def test_an_parse_neither_given_nor_held_is_refused(self) -> None:
        timeline = _quarters("clt_e", 8)

        with pytest.raises(ValueError) as refusal:
            timeline.add_events(self._rows("parse_nowhere", "n1"))

        assert str(refusal.value) == (
            "the rows name the parse 'parse_nowhere', which this family "
            "does not hold: give it with the rows (EventData.with_parses)"
        )
        assert len(timeline.events) == 0


class TestParses:
    def test_a_source_file_is_named_by_its_digest(self, tmp_path: Path) -> None:
        path = tmp_path / "score.txt"
        path.write_bytes(b"abc")

        first = Parse.of_source(
            path, parser="P", parser_version="2", parameters={"b": [2], "a": 1}
        )
        second = Parse.of_source(
            path, parser="P", parser_version="2", parameters={"a": 1, "b": [2]}
        )

        assert first.source_integrity == {"algorithm": "SHA-256", "digest": ABC_SHA256}
        assert first.id == second.id
        assert first.id.startswith("parse_")
        row = first.to_row()
        assert row["parameters"] == '{"a":1,"b":[2]}'
        assert Parse.from_row(row) == first
        assert parse_table([first]).schema == PARSE_SCHEMA

    def test_the_id_never_depends_on_where_the_file_lies(self, tmp_path: Path) -> None:
        (tmp_path / "one").mkdir()
        (tmp_path / "two").mkdir()
        here, there = tmp_path / "one" / "score.txt", tmp_path / "two" / "copy.txt"
        here.write_bytes(b"abc")
        there.write_bytes(b"abc")

        runs = [
            Parse.of_source(
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

        member = Parse.of_source(zipfile.Path(archive, at="inner/score.txt"), **run)
        named = Parse.of_source(
            zipfile.Path(archive, at="inner/score.txt"), source="score.txt", **run
        )

        assert member.source_integrity == {
            "algorithm": "SHA-256",
            "digest": ABC_SHA256,
        }
        assert member.id == Parse.of_source(disk, **run).id
        assert member.source == f"jar:{archive.as_uri()}!/inner/score.txt"
        assert named.source == "score.txt"
        with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as from_object:
            inside = zipfile.Path(from_object, at="inner/score.txt")
            with pytest.raises(ValueError, match="opened from a file object"):
                Parse.of_source(inside, **run)
            assert Parse.of_source(inside, source="score.txt", **run) == named

    def test_one_jar_uri(self, tmp_path: Path) -> None:
        archive = tmp_path / "set.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("inner/sc ore.txt", b"abc")
            handle.writestr("inner/folder/", b"")

        assert _sources.member_uri is member_uri
        assert member_uri(zipfile.Path(archive, at="inner/sc ore.txt")) == (
            f"jar:{archive.absolute().as_uri()}!/inner/sc%20ore.txt"
        )
        assert member_uri(zipfile.Path(archive, at="inner/folder/")) == (
            f"jar:{archive.absolute().as_uri()}!/inner/folder"
        )

    def test_a_table_carries_its_parses_to_its_timeline(self, tmp_path: Path) -> None:
        path = tmp_path / "score.txt"
        path.write_bytes(b"abc")
        e1 = Parse.of_source(path, parser="P", parser_version="1")
        e2 = Parse.of_source(path, parser="P", parser_version="2")
        data = EventData.from_dicts(
            [
                {
                    "id": event_id,
                    "event_type": "Note",
                    "start": start,
                    "parse_id": e1.id,
                    "locator": locator,
                }
                for event_id, start, locator in (("n1", 0, "m0"), ("n2", 1, "m1"))
            ],
            TimeUnit.quarters,
            NumberType.fraction,
        ).with_parses(e1)

        timeline = data.create_timeline(uid="clt_e")

        assert timeline.registry.parses == [e1]
        table = timeline.registry.table
        assert table.column("parse_id").to_pylist() == [e1.id, e1.id]
        assert table.column("locator").to_pylist() == ["m0", "m1"]
        for name in ("parse_id", "locator"):
            assert name not in timeline.events.table.column_names
        other = EventData.empty(TimeUnit.quarters, NumberType.fraction)
        joined = data.concat(other.with_parses(e2, e1))
        assert joined.parses == (e1, e2)

    def test_a_row_naming_an_parse_nobody_gives_is_refused(self) -> None:
        data = EventData.from_dicts(
            [
                {
                    "id": "n1",
                    "event_type": "Note",
                    "start": 0,
                    "parse_id": "parse_nowhere",
                }
            ],
            TimeUnit.quarters,
            NumberType.fraction,
        )
        message = (
            "the rows name the parse 'parse_nowhere', which this family "
            "does not hold: give it with the rows (EventData.with_parses)"
        )
        with pytest.raises(ValueError) as refusal:
            data.create_timeline(uid="clt_e")
        assert str(refusal.value) == message
        timeline = _quarters("clt_e", 4)

        with pytest.raises(ValueError) as refusal:
            timeline._registry.add("clt_e", data)

        assert str(refusal.value) == message
        assert len(timeline.events) == 0
        assert timeline._registry._family.parses == {}

    def test_a_source_without_its_digest_is_refused(self) -> None:
        with pytest.raises(ValueError) as refusal:
            Parse(id="parse_x", source="a.mid", parser="P", parser_version="1")

        assert str(refusal.value) == (
            "parse 'parse_x' names a source file and its digest together " "or neither"
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
                    "parse_id": "parse_p",
                }
            ],
            "quarters",
            "fraction",
        )

        root._registry.add_rows(rows, {"Note": content}, [_package_parse()])
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
                    "parse_id": "parse_p",
                }
            ],
            "quarters",
            "fraction",
        )
        read._registry.add_rows(rows, {}, [_package_parse()], registered=registered)
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
                {"parse_id": "parse_unknown"},
                ValueError,
                "a registry row names the parse 'parse_unknown', which is "
                "given with none of the parses",
            ),
            (
                {"id": "a1"},
                ValueError,
                "Event id 'a1' is already held by an event of timeline 'clt_a', so "
                f"it cannot join timeline 'clt_b': event ids are unique {SAME}",
            ),
        ],
        ids=["unknown-timeline", "unknown-parse", "held-id"],
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
                    "parse_id": "parse_p",
                    **row,
                }
            ],
            "quarters",
            "fraction",
        )

        with pytest.raises(error) as refusal:
            root._registry.add_rows(rows, {}, [_package_parse()])

        text = refusal.value.args[0] if error is KeyError else str(refusal.value)
        assert text == message
        assert root.registry.table.equals(before)
        assert list(root._registry._family.parses) == [root._registry._family.tool.id]

    def test_an_parse_is_held_only_with_its_rows(self) -> None:
        root = _quarters("clt_root", 8)
        root.add_events(
            [{"id": "r1", "event_type": "Beat", "start": 0, "velocity": 80}]
        )
        held = dict(root._registry._family.parses)
        data = EventData.from_dicts(
            [{"id": "v2", "event_type": "Beat", "start": 1, "velocity": "ff"}],
            TimeUnit.quarters,
            NumberType.fraction,
        )

        with pytest.raises(ValueError):
            root._registry.add("clt_root", data, parse=_package_parse())

        assert root._registry._family.parses == held
        assert _ids(root.events.table) == ["r1"]

    def test_a_child_holding_an_parse_of_other_content(self) -> None:
        def run(parser: str) -> Parse:
            return Parse(id="parse_x", parser=parser, parser_version="1", date="2026")

        root = _quarters("clt_root", 8)
        root._registry.add(
            "clt_root",
            EventData.from_dicts(
                [{"id": "r1", "event_type": "Beat", "start": 0}],
                TimeUnit.quarters,
                NumberType.fraction,
            ),
            parse=run("A"),
        )
        child = _quarters("clt_c", 2)
        child._registry.add(
            "clt_c",
            EventData.from_dicts(
                [{"id": "c1", "event_type": "Beat", "start": 0}],
                TimeUnit.quarters,
                NumberType.fraction,
            ),
            parse=run("B"),
        )
        before = root._registry._family.table()

        with pytest.raises(ValueError) as refusal:
            root.add_child(child, offset=1)

        assert str(refusal.value) == (
            "parse 'parse_x' is held already, of other content"
        )
        assert root._registry._family.table().equals(before)
        assert root._registry._family.parses["parse_x"].parser == "A"
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
                    "parse_id": "parse_p",
                }
            ],
            "quarters",
            "float",
        )

        with pytest.raises(ValueError) as refusal:
            root._registry.add_rows(rows, {}, [_package_parse()])

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
