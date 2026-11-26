"""The tree store: the events of a timeline tree in one table held by its root."""

from __future__ import annotations

from fractions import Fraction

import pyarrow as pa
import pytest

from timetoalign.core import NumberType, TimeUnit
from timetoalign.core.fields import field_metadata
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
    def test_the_root_holds_every_row_sorted_by_timeline_and_start(self) -> None:
        root, a, b = _tree()
        store = root._tree_store

        assert a._tree_store is store and b._tree_store is store
        table = store.table
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
        assert _ids(table) == ["clt_a", "r1", "clt_b", "a2", "a1", "b1"]
        assert _ids(a.events.table) == ["a2", "a1"]

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
        assert "number_type" not in store.table.column_names
        assert store.row_types().to_pylist() == [1, 1, 1, 1, 1, 2]


class TestTheSlice:
    def test_a_timelines_rows_copy_nothing(self) -> None:
        root, _, b = _tree()
        tree = root._tree_store.table.column("start")
        own = b.events.table.column("start")

        (tree_chunk,) = tree.chunks
        (own_chunk,) = own.chunks
        held = tree_chunk.field("value").buffers()[1]
        sliced = own_chunk.field("value").buffers()[1]
        assert held.address <= sliced.address
        assert sliced.address + sliced.size <= held.address + held.size

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
        assert _ids(a.events.table) == ["a2", "a1", "a3"]


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

        assert root._tree_store.table.num_rows == 7


class TestAttachAndSplit:
    def test_attaching_moves_the_rows(self) -> None:
        root = _timeline("clt_root", 8, "fraction")
        b = _timeline("clt_b", 4, "float")
        b.add_events([{"id": "b1", "event_type": "Beat", "start": 0.1}])
        assert _ids(b._tree_store.table) == ["b1"]
        own = b._tree_store

        root.add_child(b, offset=4)

        assert _ids(root._tree_store.table) == ["clt_b", "b1"]
        assert own.timeline_ids == ()

    def test_a_split_gives_the_subtree_a_store_of_its_own(self) -> None:
        root, _, _ = _tree()
        store = root._tree_store

        split = store.split(["clt_b"])

        assert split.timeline_ids == ("clt_b",)
        assert split.axes == {"clt_b": (TimeUnit.quarters, NumberType.float)}
        assert _ids(split.table) == ["b1"]
        assert _ids(split.events_of("clt_b").table) == ["b1"]
        assert store.timeline_ids == ("clt_root", "clt_a")
        assert store.table.num_rows == 5

    def test_the_root_does_not_split_off(self) -> None:
        root, _, _ = _tree()

        with pytest.raises(ValueError) as refusal:
            root._tree_store.split(["clt_root"])

        assert str(refusal.value) == (
            "the root 'clt_root' holds the tree; only a subtree leaves it"
        )


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


class TestTheNumberTypeFilter:
    def test_on_the_trees_table(self) -> None:
        root, _, _ = _tree()
        data = root._tree_store.data

        assert _ids(data.filter(include_number_types={NumberType.float}).table) == [
            "b1"
        ]
        assert _ids(data.filter(include_number_types={NumberType.fraction}).table) == [
            "clt_a",
            "r1",
            "clt_b",
            "a2",
            "a1",
        ]

    def test_on_get_events(self) -> None:
        root, _, _ = _tree()

        assert root.get_events(include_number_types={NumberType.float}).column_values(
            "id"
        ) == ["b1"]
        assert root.get_events(
            include_number_types={NumberType.fraction}
        ).column_values("id") == ["r1", "a2", "a1"]


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
        assert _ids(a.events.table) == ["a2", "a1"]

    def test_a_child_holding_an_id_of_the_tree_is_refused(self) -> None:
        root, _, _ = _tree()
        c = _timeline("clt_c", 2, "fraction")
        c.add_events([{"id": "r1", "event_type": "Beat", "start": 0}])

        with pytest.raises(ValueError) as refusal:
            root.add_child(c, offset=6)

        assert str(refusal.value) == (
            "Event id 'r1' of timeline 'clt_c' is already held by an event of "
            "timeline 'clt_root'; event ids are unique across a timeline tree"
        )
        assert root.n_children == 2
        assert root._tree_store.table.num_rows == 6
        assert c._tree_store.timeline_ids == ("clt_c",)
        assert _ids(c.events.table) == ["r1"]


class TestRename:
    def test_a_new_id_rewrites_no_row(self) -> None:
        root, a, _ = _tree()
        store = root._tree_store

        a._set_id("clt_x")

        assert store.timeline_ids == ("clt_root", "clt_x", "clt_b")
        assert _ids(a.events.table) == ["a2", "a1"]
        assert store.holder_of("a1") == "clt_x"


def test_a_store_refuses_a_timeline_it_does_not_hold() -> None:
    store = TreeStore("clt_1", ContinuousLogicalTimeline(length=1).events)

    with pytest.raises(KeyError) as refusal:
        store.events_of("clt_2")

    assert refusal.value.args[0] == (
        "No timeline 'clt_2' in this tree. Available timelines: ['clt_1']"
    )
