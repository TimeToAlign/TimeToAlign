"""The tree store: the events of a timeline tree in one table held by its root."""

from __future__ import annotations

from fractions import Fraction

import pyarrow as pa
import pytest

from timetoalign.core import NumberType, TimeUnit
from timetoalign.core.fields import field_metadata
from timetoalign.storage import EventData
from timetoalign.storage.events import row_type_codes
from timetoalign.storage.tree import TreeStore
from timetoalign.timelines import ContinuousLogicalTimeline, Timeline


def _timeline(uid: str, length: int, number_type: str) -> ContinuousLogicalTimeline:
    return ContinuousLogicalTimeline(
        length=length, unit="quarters", number_type=number_type, uid=uid
    )


def _tree() -> tuple[Timeline, Timeline, Timeline]:
    """A fraction root with a fraction child at 0 and a float child at 4."""
    root = _timeline("clt_root", 8, "fraction")
    root.add_events([{"id": "r1", "event_type": "Beat", "start": Fraction(1, 2)}])
    a = _timeline("clt_a", 4, "fraction")
    a.add_events(
        [
            {"id": "a1", "event_type": "Beat", "start": Fraction(1, 3)},
            {"id": "a2", "event_type": "Beat", "start": 0},
        ]
    )
    b = _timeline("clt_b", 4, "float")
    b.add_events([{"id": "b1", "event_type": "Beat", "start": 0.1}])
    root.add_child(a, offset=0)
    root.add_child(b, offset=4)
    return root, a, b


def _ids(table: pa.Table) -> list[str]:
    return table.column("id").to_pylist()


class TestOneTableForTheTree:
    def test_the_root_holds_every_row_in_the_order_it_came(self) -> None:
        root, a, b = _tree()
        store = root._tree_store

        assert a._tree_store is store and b._tree_store is store
        (table,) = store.tables
        assert table.num_rows == 6
        assert pa.types.is_dictionary(table.schema.field("timeline_id").type)
        assert table.column("timeline_id").to_pylist() == [
            "clt_root",
            "clt_root",
            "clt_root",
            "clt_a",
            "clt_a",
            "clt_b",
        ]
        assert _ids(table) == ["r1", "clt_a", "clt_b", "a1", "a2", "b1"]
        assert _ids(a.events.table) == ["a1", "a2"]

    def test_positions_stay_child_local(self) -> None:
        _, _, b = _tree()

        assert b.events.table.column("start").to_pylist() == [
            {
                "value": 0.1,
                "numerator": 3602879701896397,
                "denominator": 36028797018963968,
            }
        ]

    def test_attached_stores_are_emptied(self) -> None:
        a = _timeline("clt_a", 4, "fraction")
        a.add_events([{"id": "a1", "event_type": "Beat", "start": 0}])
        own = a._tree_store
        assert own.timeline_ids == ("clt_a",)
        root = _timeline("clt_root", 8, "fraction")

        root.add_child(a, offset=0)

        assert own.timeline_ids == ()
        assert root._tree_store.timeline_ids == ("clt_root", "clt_a")
        assert _ids(a.events.table) == ["a1"]


class TestTheMapAndTheRowTypes:
    def test_one_entry_per_timeline_no_column(self) -> None:
        root, _, _ = _tree()
        store = root._tree_store

        assert store.axes == {
            "clt_root": (TimeUnit.quarters, NumberType.fraction),
            "clt_a": (TimeUnit.quarters, NumberType.fraction),
            "clt_b": (TimeUnit.quarters, NumberType.float),
        }
        (table,) = store.tables
        assert "number_type" not in table.column_names
        assert row_type_codes(table, store.axes).to_pylist() == [1, 1, 1, 1, 1, 2]


class TestTheSlice:
    def test_a_timelines_rows_copy_nothing(self) -> None:
        root, _, b = _tree()
        tree = root._tree_store.tables[0].column("start")
        own = b.events.table.column("start")

        def buffers(column: pa.ChunkedArray) -> list[tuple[int, int]]:
            held = [chunk.field("value").buffers()[1] for chunk in column.chunks]
            return [(buffer.address, buffer.size) for buffer in held if buffer]

        (sliced,) = buffers(own)
        assert sliced in buffers(tree)

    def test_the_slice_declares_its_timelines_axis(self) -> None:
        _, a, b = _tree()

        assert field_metadata(b.events.schema.field("start")) == {
            "unit": "quarters",
            "number_type": "float",
            "version": 1,
        }
        assert field_metadata(a.events.schema.field("start"))["number_type"] == (
            "fraction"
        )
        assert b.events.number_type is NumberType.float
        assert a.events.number_type is NumberType.fraction
        assert "timeline_id" not in b.events.table.column_names

    def test_the_same_slice_until_the_rows_change(self) -> None:
        _, a, _ = _tree()
        first = a.events

        assert a.events is first
        a.add_events([{"id": "a3", "event_type": "Beat", "start": 2}])
        assert a.events is not first
        assert _ids(a.events.table) == ["a1", "a2", "a3"]

    def test_another_timelines_change_leaves_the_rows_as_they_are(self) -> None:
        root, a, _ = _tree()
        first = a.events

        root.add_events([{"id": "r2", "event_type": "Beat", "start": 3}])
        c = _timeline("clt_c", 2, "fraction")
        c.add_events([{"id": "c1", "event_type": "Beat", "start": 0}])
        root.add_child(c, offset=6)

        assert a.events is first


class TestRowsChangeThroughTheStore:
    def test_events_are_read_only(self) -> None:
        _, a, _ = _tree()

        with pytest.raises(AttributeError):
            setattr(a, "_events", a.events)
        with pytest.raises(AttributeError):
            setattr(a, "events", a.events)

    def test_a_childs_new_event_joins_the_roots_table(self) -> None:
        root, a, _ = _tree()

        a.add_events([{"id": "a3", "event_type": "Beat", "start": 2}])

        assert root._tree_store.tables[0].num_rows == 7
        assert _ids(a.events.table) == ["a1", "a2", "a3"]


class TestAttach:
    def test_attaching_moves_the_rows(self) -> None:
        root = _timeline("clt_root", 8, "fraction")
        b = _timeline("clt_b", 4, "float")
        b.add_events([{"id": "b1", "event_type": "Beat", "start": 0.1}])
        assert _ids(b._tree_store.tables[0]) == ["b1"]
        own = b._tree_store

        root.add_child(b, offset=4)

        assert _ids(root._tree_store.tables[0]) == ["clt_b", "b1"]
        assert own.timeline_ids == ()

    def test_columns_typed_differently_share_the_tree(self) -> None:
        root = _timeline("clt_root", 8, "fraction")
        root.add_events(
            [{"id": "r1", "event_type": "Beat", "start": 0, "tag": {"midi": 60}}]
        )
        a = _timeline("clt_a", 4, "fraction")
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

        store = root._tree_store
        (table,) = store.tables
        assert table.schema.field("tag").type == pa.struct(
            [pa.field("midi", pa.int64()), pa.field("name", pa.string())]
        )
        assert table.column("tag").to_pylist() == [
            {"midi": 60, "name": None},
            None,
            {"midi": 60, "name": "C4"},
        ]
        assert root.events.table.schema.field("tag").type == pa.struct(
            [pa.field("midi", pa.int64())]
        )

    def test_a_plain_column_of_another_type_keeps_a_table_of_its_own(self) -> None:
        root = _timeline("clt_root", 8, "fraction")
        root.add_events(
            [{"id": "r1", "event_type": "Beat", "start": 0, "velocity": 80}]
        )
        a = _timeline("clt_a", 4, "fraction")
        a.add_events([{"id": "a1", "event_type": "Beat", "start": 0, "velocity": "ff"}])

        root.add_child(a, offset=0)

        store = root._tree_store
        first, second = store.tables
        assert first.column("timeline_id").to_pylist() == ["clt_root", "clt_root"]
        assert first.column("velocity").to_pylist() == [80, None]
        assert second.column("timeline_id").to_pylist() == ["clt_a"]
        assert second.column("velocity").to_pylist() == ["ff"]
        assert first.num_rows + second.num_rows == 3
        assert not hasattr(store, "table") and not hasattr(store, "data")
        assert a.events.table.column("velocity").to_pylist() == ["ff"]


class TestOneRuleJoinsColumns:
    def test_concat_widens_a_struct_as_the_tree_does(self) -> None:
        root = _timeline("clt_root", 8, "fraction")
        root.add_events(
            [{"id": "r1", "event_type": "Beat", "start": 0, "tag": {"midi": 60}}]
        )
        a = _timeline("clt_a", 4, "fraction")
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

        joined = root.events.concat(a.events)

        (tree,) = root._tree_store.tables
        widened = pa.struct(
            [pa.field("midi", pa.int64()), pa.field("name", pa.string())]
        )
        assert joined.table.schema.field("tag").type == widened
        assert tree.schema.field("tag").type == widened
        assert joined.table.column("tag").to_pylist() == [
            {"midi": 60, "name": None},
            None,
            {"midi": 60, "name": "C4"},
        ]
        assert tree.column("tag").to_pylist() == joined.table.column("tag").to_pylist()

    @staticmethod
    def _tagged(event_id: str, tag: str, number_type: NumberType) -> EventData:
        return EventData.from_dicts(
            [{"id": event_id, "event_type": "Beat", "start": 0, "tag": tag}],
            unit=TimeUnit.quarters,
            number_type=number_type,
        )

    def _encoded(self) -> EventData:
        data = self._tagged("d1", "x", NumberType.fraction)
        index = data.table.schema.get_field_index("tag")
        encoded = pa.chunked_array([pa.array(["x"]).dictionary_encode()])
        return data._derived(data.table.set_column(index, "tag", encoded))

    def test_a_dictionary_column_beside_a_plain_one_reads_as_its_values(
        self,
    ) -> None:
        encoded = self._encoded()
        assert pa.types.is_dictionary(encoded.table.schema.field("tag").type)

        joined = encoded.concat(self._tagged("e1", "y", NumberType.fraction))

        assert joined.table.schema.field("tag").type == pa.string()
        assert joined.table.column("tag").to_pylist() == ["x", "y"]

    def test_so_it_does_on_a_table_spanning_number_types(self) -> None:
        joined = self._encoded().concat(self._tagged("f1", "z", NumberType.float))

        assert joined.table.schema.field("tag").type == pa.string()
        assert joined.table.column("tag").to_pylist() == ["x", "z"]
        assert joined.table.column("timeline_id").to_pylist() == [
            "<quarters fraction>",
            "<quarters float>",
        ]

    def test_an_int_beside_a_string_has_no_one_table(self) -> None:
        root = _timeline("clt_root", 8, "fraction")
        root.add_events(
            [{"id": "r1", "event_type": "Beat", "start": 0, "velocity": 80}]
        )
        a = _timeline("clt_a", 4, "float")
        a.add_events([{"id": "a1", "event_type": "Beat", "start": 0, "velocity": "ff"}])

        with pytest.raises(ValueError) as refusal:
            root.events.concat(a.events)

        assert str(refusal.value) == (
            "the column 'velocity' holds values of type int64 in one table and of "
            "type string in another, which no one type reads; one table holds one "
            "type per column"
        )


class TestARefusedChildChangesNothing:
    @pytest.mark.parametrize("held", ["r1", "clt_c"])
    def test_every_check_runs_before_a_change(self, held: str) -> None:
        root, a, b = _tree()
        store = root._tree_store
        (rows,) = store.tables
        c = _timeline("clt_c", 2, "fraction")
        c.add_events([{"id": held, "event_type": "Beat", "start": 0}])
        own = c._tree_store

        with pytest.raises(ValueError):
            root.add_child(c, offset=6)

        assert root.n_children == 2
        assert list(root._children) == ["clt_a", "clt_b"]
        assert root._tree_store is store
        assert store.timeline_ids == ("clt_root", "clt_a", "clt_b")
        (after,) = store.tables
        assert after.equals(rows)
        assert c._parent is None
        assert c._tree_store is own
        assert own.timeline_ids == ("clt_c",)
        assert _ids(c.events.table) == [held]


class TestAJoinKeepsEachRowsType:
    def test_floats_stay_floats(self) -> None:
        _, a, b = _tree()

        joined = a.events.concat(b.events)

        assert joined.table.column("timeline_id").to_pylist() == [
            "clt_a",
            "clt_a",
            "clt_b",
        ]
        assert joined.table.column("start").to_pylist()[2] == (
            b.events.table.column("start").to_pylist()[0]
        )
        assert _ids(joined.filter(include_number_types={NumberType.float}).table) == [
            "b1"
        ]

    def test_the_cast_gives_the_exact_ratio(self) -> None:
        _, a, b = _tree()

        cast = a.events.concat(b.events).to_number_type("fraction")

        cell = cast.table.column("start").to_pylist()[2]
        assert (cell["numerator"], cell["denominator"]) == (
            3602879701896397,
            36028797018963968,
        )
        assert field_metadata(cast.schema.field("start"))["number_type"] == "fraction"
        assert cast.number_type is NumberType.fraction


class TestAJoinIsNeverRefusedForTheNumberType:
    @pytest.mark.parametrize("cast_first", [False, True])
    def test_rows_meet_their_own_cast(self, cast_first: bool) -> None:
        _, _, b = _tree()
        cast = b.events.to_number_type("fraction")
        joined = cast.concat(b.events) if cast_first else b.events.concat(cast)

        keys = (
            ["<quarters fraction>", "clt_b"]
            if cast_first
            else ["clt_b", "<quarters fraction>"]
        )
        assert joined.table.column("timeline_id").to_pylist() == keys
        assert joined._axes == {
            "clt_b": (TimeUnit.quarters, NumberType.float),
            "<quarters fraction>": (TimeUnit.quarters, NumberType.fraction),
        }
        float_rows = joined.filter(include_number_types={NumberType.float})
        fraction_rows = joined.filter(include_number_types={NumberType.fraction})
        assert float_rows.table.column("timeline_id").to_pylist() == ["clt_b"]
        assert fraction_rows.table.column("timeline_id").to_pylist() == [
            "<quarters fraction>"
        ]

    def test_a_fraction_timeline_meets_its_float_cast(self) -> None:
        _, a, _ = _tree()

        joined = a.events.concat(a.events.to_number_type("float"))

        assert joined.table.column("timeline_id").to_pylist() == [
            "clt_a",
            "clt_a",
            "<quarters float>",
            "<quarters float>",
        ]
        assert _ids(joined.table) == ["a1", "a2", "a1", "a2"]

    def test_the_cast_read_through_get_events_joins_too(self) -> None:
        _, a, _ = _tree()

        joined = a.get_events().to_number_type("float").concat(a.events)

        assert joined.table.column("timeline_id").to_pylist() == [
            "<quarters float>",
            "<quarters float>",
            "clt_a",
            "clt_a",
        ]

    def test_a_spanning_cast_keys_the_rows_it_changes_by_their_axis(self) -> None:
        _, a, b = _tree()
        cast = a.events.concat(b.events).to_number_type("float")
        assert cast.table.column("timeline_id").to_pylist() == [
            "<quarters float>",
            "<quarters float>",
            "clt_b",
        ]
        assert cast._axes == {
            "<quarters float>": (TimeUnit.quarters, NumberType.float),
            "clt_b": (TimeUnit.quarters, NumberType.float),
        }

        joined = cast.concat(a.events)

        assert joined.table.column("timeline_id").to_pylist() == [
            "<quarters float>",
            "<quarters float>",
            "clt_b",
            "clt_a",
            "clt_a",
        ]
        assert joined._axes == {
            "<quarters float>": (TimeUnit.quarters, NumberType.float),
            "clt_b": (TimeUnit.quarters, NumberType.float),
            "clt_a": (TimeUnit.quarters, NumberType.fraction),
        }
        assert _ids(
            joined.filter(include_number_types={NumberType.fraction}).table
        ) == [
            "a1",
            "a2",
        ]

    def test_a_cast_to_the_rows_own_type_keeps_its_timeline(self) -> None:
        _, _, b = _tree()

        joined = b.events.concat(b.events.to_number_type("float"))

        assert "timeline_id" not in joined.table.column_names
        assert joined._timeline_id == "clt_b"
        assert _ids(joined.table) == ["b1", "b1"]


_DOUBLE = (3602879701896397, 36028797018963968)
"""The exact ratio of the double nearest 0.1."""


def _starts(data: EventData) -> list[tuple[int, int]]:
    return [
        (cell["numerator"], cell["denominator"])
        for cell in data.table.column("start").to_pylist()
    ]


class TestACastJoinedAgainKeepsEachRowsType:
    def test_keys_of_axes(self) -> None:
        exact = EventData.from_dicts(
            [{"id": "a", "event_type": "Beat", "start": Fraction(1, 10)}],
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )
        measured = EventData.from_dicts(
            [{"id": "b", "event_type": "Beat", "start": 0.1}],
            unit=TimeUnit.quarters,
            number_type=NumberType.float,
        )
        span = exact.concat(measured)
        cast = span.to_number_type("float")
        assert cast.table.column("timeline_id").to_pylist() == [
            "<quarters float>",
            "<quarters float>",
        ]
        assert cast._axes == {"<quarters float>": (TimeUnit.quarters, NumberType.float)}

        joined = cast.concat(span)

        assert joined.table.column("timeline_id").to_pylist() == [
            "<quarters float>",
            "<quarters float>",
            "<quarters fraction>",
            "<quarters float>",
        ]
        assert joined._axes == {
            "<quarters float>": (TimeUnit.quarters, NumberType.float),
            "<quarters fraction>": (TimeUnit.quarters, NumberType.fraction),
        }
        fraction_rows = joined.filter(include_number_types={NumberType.fraction})
        float_rows = joined.filter(include_number_types={NumberType.float})
        assert fraction_rows.column_values("id") == ["a"]
        assert float_rows.column_values("id") == ["a", "b", "b"]
        assert _starts(joined.to_number_type("fraction")) == [
            _DOUBLE,
            _DOUBLE,
            (1, 10),
            _DOUBLE,
        ]

    def test_keys_of_timelines(self) -> None:
        root = _timeline("R", 4, "fraction")
        root.add_events([{"id": "a", "event_type": "Beat", "start": Fraction(1, 10)}])
        child = _timeline("C", 4, "float")
        child.add_events([{"id": "b", "event_type": "Beat", "start": 0.1}])
        root.add_child(child, offset=0)
        span = root.events.concat(child.events)
        assert span.table.column("timeline_id").to_pylist() == ["R", "R", "C"]

        joined = span.to_number_type("float").concat(span)

        assert joined.table.column("timeline_id").to_pylist() == [
            "<quarters float>",
            "<quarters float>",
            "C",
            "R",
            "R",
            "C",
        ]
        assert joined._axes == {
            "<quarters float>": (TimeUnit.quarters, NumberType.float),
            "C": (TimeUnit.quarters, NumberType.float),
            "R": (TimeUnit.quarters, NumberType.fraction),
        }
        fraction_rows = joined.filter(include_number_types={NumberType.fraction})
        assert fraction_rows.column_values("id") == ["a", "C"]
        assert _starts(joined.to_number_type("fraction")) == [
            _DOUBLE,
            (0, 1),
            _DOUBLE,
            (1, 10),
            (0, 1),
            _DOUBLE,
        ]


def test_the_extremes_of_a_spanning_table_are_read_row_by_row() -> None:
    exact = EventData.from_dicts(
        [{"id": "e", "event_type": "Beat", "start": Fraction(1, 2)}],
        unit=TimeUnit.quarters,
        number_type=NumberType.fraction,
    )
    measured = EventData.from_dicts(
        [{"id": "m", "event_type": "Beat", "start": 0.7}],
        unit=TimeUnit.quarters,
        number_type=NumberType.float,
    )

    low, high = exact.concat(measured).coordinate_range()

    assert (low, high) == (Fraction(1, 2), 0.7)
    assert type(low) is Fraction and type(high) is float
    assert Fraction(0.7) == Fraction(3152519739159347, 4503599627370496)


class TestTheNumberTypeFilter:
    def test_on_the_trees_timelines_joined(self) -> None:
        root, a, b = _tree()
        data = root.events.concat(a.events, b.events)

        assert _ids(data.filter(include_number_types={NumberType.float}).table) == [
            "b1"
        ]
        assert _ids(data.filter(include_number_types={NumberType.fraction}).table) == [
            "r1",
            "clt_a",
            "clt_b",
            "a1",
            "a2",
        ]

    def test_on_get_events(self) -> None:
        root, _, _ = _tree()

        assert root.get_events(include_number_types={NumberType.float}).column_values(
            "id"
        ) == ["b1"]
        assert root.get_events(
            include_number_types={NumberType.fraction}
        ).column_values("id") == ["r1", "a1", "a2"]


class TestUniqueEventIds:
    def test_an_id_of_the_root_added_to_a_child_is_refused(self) -> None:
        root, a, _ = _tree()

        with pytest.raises(ValueError) as refusal:
            a.add_events([{"id": "r1", "event_type": "Beat", "start": 1}])

        assert str(refusal.value) == (
            "Event id 'r1' is already held by an event of timeline 'clt_root', so "
            "it cannot join timeline 'clt_a': event ids are unique across a "
            "timeline tree"
        )
        assert _ids(a.events.table) == ["a1", "a2"]

    def test_an_id_given_twice_is_refused(self) -> None:
        _, a, _ = _tree()

        with pytest.raises(ValueError) as refusal:
            a.add_events(
                [
                    {"id": "x", "event_type": "Beat", "start": 1},
                    {"id": "x", "event_type": "Beat", "start": 2},
                ]
            )

        assert str(refusal.value) == (
            "Event id 'x' is given to 2 events joining timeline 'clt_a'; event ids "
            "are unique across a timeline tree"
        )
        assert _ids(a.events.table) == ["a1", "a2"]

    def test_a_child_holding_an_id_of_the_tree_is_refused(self) -> None:
        root, _, _ = _tree()
        c = _timeline("clt_c", 2, "fraction")
        c.add_events([{"id": "r1", "event_type": "Beat", "start": 0}])

        with pytest.raises(ValueError) as refusal:
            root.add_child(c, offset=6)

        assert str(refusal.value) == (
            "Event id 'r1' is already held by an event of timeline 'clt_root', so "
            "it cannot join timeline 'clt_c': event ids are unique across a "
            "timeline tree"
        )
        assert root.n_children == 2
        assert root._tree_store.tables[0].num_rows == 6
        assert c._tree_store.timeline_ids == ("clt_c",)
        assert _ids(c.events.table) == ["r1"]


class TestRename:
    def test_a_new_id_rewrites_no_row(self) -> None:
        root, a, _ = _tree()
        store = root._tree_store

        a._set_id("clt_x")

        assert store.timeline_ids == ("clt_root", "clt_x", "clt_b")
        assert _ids(a.events.table) == ["a1", "a2"]
        assert store.holder_of("a1") == "clt_x"

    @pytest.mark.parametrize("new_id", ["clt_b", "clt_root", "r1"])
    def test_a_refused_id_changes_nothing(self, new_id: str) -> None:
        root, a, _ = _tree()
        store = root._tree_store

        with pytest.raises(ValueError):
            a._set_id(new_id)

        assert list(root._children) == ["clt_a", "clt_b"]
        assert list(root._child_offsets) == ["clt_a", "clt_b"]
        assert [offset.value for offset in root._child_offsets.values()] == [0, 4]
        assert root.get_child("clt_a") is a
        assert _ids(root.events.table) == ["r1", "clt_a", "clt_b"]
        assert store.timeline_ids == ("clt_root", "clt_a", "clt_b")
        assert a.id == "clt_a"


def test_a_store_refuses_a_timeline_it_does_not_hold() -> None:
    store = TreeStore("clt_1", ContinuousLogicalTimeline(length=1).events)

    with pytest.raises(KeyError) as refusal:
        store.events_of("clt_2")

    assert refusal.value.args[0] == (
        "No timeline 'clt_2' in this tree. Available timelines: ['clt_1']"
    )
